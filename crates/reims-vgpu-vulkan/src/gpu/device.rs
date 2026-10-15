//! `id<MTLDevice>`: core's `GpuDevice` over the runtime's `Device`.
//!
//! - **One object per device.** Every handle to one runtime device is a reference to one
//!   [`DeviceObject`] ([`adopt`]), so a device opened twice, and the `device` of every object made
//!   from it, have the same identity, as one `MTLDevice` does. The objects made from a device hold
//!   its handle.
//! - **Capabilities** are read once per device object ([`crate::gpu::caps`]) and shared.
//! - **The registry ID** is Metal's default device's for the runtime's system default device,
//!   asked of Metal at run time on macOS ([`registry_id`]), which is what a host names it by; the
//!   runtime's own on Linux.
//! - **A texture over an IOSurface** is a debug line under this module's target
//!   (`REIMS_VGPU_LOG=debug`, or `warn,reims_vgpu_vulkan::gpu::device=debug`), written as
//!   `newTextureWithDescriptor:iosurface:plane:` is called, before it answers or dies, but after
//!   a nil surface, which dies first ([`OnSurface`]): which formats a guest makes over which
//!   plane of which surfaces, at what size. The line is a diagnostic: it changes nothing the
//!   guest or the host sees, and with the target's debug lines off nothing formats it.
//! - **A buffer over an IOSurface** is a debug line under the same target, written as the buffer
//!   is made (`buffer_over_surface_at`): its selector and length, the memory the surface is in
//!   (its own, the guest's pages, or a mapping of its own of them, `crate::os::memory_kind`), and
//!   whether the buffer is imported at the surface's base address or lies over an alias of its
//!   allocation, at which offset, whether read-only for the GPU, and whether the alias is new,
//!   kept from an earlier buffer, or none. It too changes nothing the guest or the host sees.

use std::collections::BTreeMap;
use std::fmt;
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock, Weak};

use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::caps::DeviceCaps;
use reims_vgpu_core::platform::device::{FunctionResolver, GpuDevice, NoCopyDeallocator};
use reims_vgpu_core::platform::os::vm::HostVm;
use reims_vgpu_core::platform::types::{
    NsErrorText, PixelFormatInfo, ResourceAddressRange, SizeAndAlign, TextureLevelInfo,
    TextureLevelQuery,
};
use reims_vgpu_core::platform::values::{
    BufferDeallocator, BufferDesc, DepthStencilDesc, HeapDesc, HeapMade, IcbDesc, OptionSet,
    RenderPipelineDesc, SamplerDesc,
};
use reims_vgpu_core::platform::{Handle, HostRange};
use reims_vgpu_core::raise::{Kept, OnFault, RaiseResult, keep_forever};
use reims_vgpu_core::rt::{BlockEnv, Hold};
use reims_vgpu_core::sync::Mutex;

use crate::answer::{
    Nil, Policy, RUNTIME_DOMAIN, create, decide, get, get_describing, hearing, metal_aborts, plain,
    plain_at, send, unrepresentable,
};
use crate::gpu::backing::{self, Backing, HEAP_GRANULE};
use crate::gpu::caps;
use crate::gpu::command::{QueueObject, VulkanCommandQueue};
use crate::gpu::desc::{
    Maker, PipelineDescriptorObject, TextureFields, VulkanComputePipelineDescriptor,
    VulkanRenderPipelineDescriptor, VulkanTextureDescriptor, VulkanTileRenderPipelineDescriptor,
    format_refused,
};
use crate::gpu::event::{VulkanEvent, VulkanSharedEvent, VulkanSharedEventHandle};
use crate::gpu::icb::VulkanIcb;
use crate::gpu::metal_space::{self, Placing, SurfacePlace, TextureKind};
use crate::gpu::rate_map::{VulkanRateMap, VulkanRateMapDescriptor};
use crate::gpu::resource::{
    BufferObject, ComputePipelineObject, HeapObject, LibraryObject, TextureObject, Under,
    VulkanBuffer, VulkanComputePipeline, VulkanDepthStencilState, VulkanFence, VulkanFunction,
    VulkanHeap, VulkanLibrary, VulkanRenderPipeline, VulkanSamplerState, VulkanSharedTextureHandle,
    VulkanTexture,
};
use crate::gpu::serialized;
use crate::gpu::timing;
use crate::os::{OsSurface, OsVm};
use crate::platform::{Obj, VulkanPlatform};

/// `newTextureWithDescriptor:iosurface:plane:`'s debug line ([the module docs](self)): the
/// selector; the descriptor's pixel format, raw and by Metal's name; the plane asked for; the
/// descriptor's width by height, its usage (the runtime's flags and the raw bits beside them) and
/// its storage mode; then the surface's OSType (its four characters where they print, otherwise
/// in hex), its plane count, and each plane's width by height and bytes per row, one plane after
/// another, or the surface's own for a surface of no planes.
pub struct OnSurface<'a> {
    /// The descriptor's fields.
    pub fields: &'a TextureFields,
    /// The surface.
    pub surface: &'a OsSurface,
    /// The plane asked for.
    pub plane: u64,
}

impl fmt::Display for OnSurface<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use reims_vgpu_core::platform::os::surface::Surface as _;
        let OnSurface {
            fields,
            surface,
            plane,
        } = self;
        let d = &fields.desc;
        let format = (fields.ycbcr_format.or(fields.no_texture_format))
            .or(fields.lacked_format)
            .unwrap_or(u64::from(d.pixel_format.as_raw()));
        let name = u32::try_from(format)
            .ok()
            .and_then(|raw| metal2vulkan::metal::PixelFormat::try_from(raw).ok())
            .map_or("an unnamed format", metal2vulkan::metal::PixelFormat::name);
        let usage = u64::from(d.usage.as_raw()) | fields.private_usage;
        let storage = (d.resource_options.raw() >> 4) & 0xf;
        let ostype = surface.pixel_format().to_be_bytes();
        let count = surface.plane_count();
        write!(
            f,
            "newTextureWithDescriptor:iosurface:plane: format {format} {name}, plane {plane}, \
             {}x{}, usage {usage:#x}, storage {storage}; surface ",
            d.width, d.height
        )?;
        if ostype.iter().all(|c| c.is_ascii_graphic() || *c == b' ') {
            write!(f, "'{}'", String::from_utf8_lossy(&ostype))?;
        } else {
            write!(f, "{:#010x}", surface.pixel_format())?;
        }
        write!(f, ", {count} planes")?;
        if count == 0 {
            return write!(
                f,
                ": {}x{}, {} bytes per row",
                surface.width(),
                surface.height(),
                surface.bytes_per_row()
            );
        }
        for i in 0..count {
            let separator = if i == 0 { ": " } else { "; " };
            write!(f, "{separator}plane {i} ")?;
            match crate::os::plane_shape(surface, i) {
                Some((width, height, _)) => write!(f, "{width}x{height}")?,
                None => f.write_str("unsized")?,
            }
            match crate::os::plane(surface, i) {
                Some(p) => write!(f, ", {} bytes per row", p.bytes_per_row)?,
                None => f.write_str(", no rows")?,
            }
        }
        Ok(())
    }
}

/// A runtime device, as core holds it.
#[derive(Debug)]
pub struct DeviceObject {
    /// The runtime's device.
    pub raw: rt::Device,
    /// The registry ID this platform answers for it ([`registry_id`]).
    pub registry_id: u64,
    /// Its capabilities, once read.
    caps: OnceLock<Arc<DeviceCaps>>,
}

/// The registry ID this platform answers for `raw`: for the runtime's system default device,
/// Metal's for its system default device, asked at run time on macOS
/// ([`crate::os::default_registry_id`]), and the runtime's own for any other device, or where
/// there is no Metal default device.
pub fn registry_id(raw: &rt::Device) -> u64 {
    if crate::platform::default_device_registry_id() == Some(raw.registry_id())
        && let Some(id) = crate::os::default_registry_id()
    {
        return id;
    }
    raw.registry_id()
}

/// `id<MTLDevice>`.
pub type VulkanDevice = Obj<DeviceObject>;

/// The `NSError` Metal hands back for bytes that are no metallib (testing.md section 6.18).
fn invalid_library_file() -> NsErrorText {
    NsErrorText {
        domain: "MTLLibraryErrorDomain".to_owned(),
        code: 1,
        description: "Error Domain=MTLLibraryErrorDomain Code=1 \"Invalid library file\" \
                      UserInfo={NSLocalizedDescription=Invalid library file}"
            .to_owned(),
    }
}

/// The runtime's refusal `e` of the pipeline creator `sel`: the `NSError` where Metal answers one
/// ([`super::states::pipeline_error`]), else a death at the send, whose line names `descriptor`'s
/// formats ([`super::states::compute_formats`]). An unspecialized function constant is a death,
/// as Metal aborts for one.
fn compile_error(
    sel: &'static str,
    e: rt::Error,
    descriptor: Option<&ComputePipelineDescriptor>,
) -> NsErrorText {
    super::states::pipeline_error(&e).unwrap_or_else(|| {
        // Decided as a getter's refusal: none of them returns.
        let formats = descriptor.map(super::states::compute_formats);
        get_describing(sel, formats.as_deref(), || Err(e))
    })
}

/// The `NSError` a pipeline state made from a nil descriptor answers, which an empty serialized
/// descriptor deserializes to, from each of the three descriptor creators: Metal's code, 2, and
/// its text, "pipeline descriptor is nil", in the runtime's domain ([`RUNTIME_DOMAIN`]), as
/// Metal's domain is its driver's name (`AGXMetalG15X_M1` on the M3 Max) and only core's log line
/// reads it (`guest_descriptor_refusals` compares Metal's on the host).
pub(crate) fn nil_pipeline_descriptor() -> NsErrorText {
    NsErrorText {
        domain: RUNTIME_DOMAIN.to_owned(),
        code: 2,
        description: "pipeline descriptor is nil".to_owned(),
    }
}

/// The devices made in the process, by registry ID, each the one object every handle to it is.
/// Each is kept for the process, as Metal's device is: a device the host releases and opens again
/// is the same runtime device, whose spaces of IDs (textures', sampler states', pipeline states'
/// and indirect command buffers') live on, as Metal's do for the process. A runtime device whose
/// last handle dropped would tear down, and one opened again would number from a fresh pool where
/// Metal's and [`crate::gpu::icb`]'s go on. A device the runtime lost stays lost, kept, where one
/// made again would not be; a call the runtime refuses as lost dies at the send anyway
/// ([`crate::answer`]).
static DEVICES: Mutex<Vec<(u64, VulkanDevice)>> = Mutex::new(Vec::new());

/// The buffers over each surface at its range's address ([`metal_space::surface_buffer`]), by
/// the surface's key: a buffer made over the surface while another holds the address shares the
/// range of a live one (`newBufferWithIOSurface:`). Each holds its surface, so a key names one
/// surface while a buffer under it lives.
static SURFACE_BUFFERS: Mutex<BTreeMap<u64, Vec<Weak<BufferObject>>>> = Mutex::new(BTreeMap::new());

/// The handle of `raw`, which `sel` made: the object of the same device made before, or a new
/// one, kept. The process's first device's sampler ID pool is the one Metal's sampler mempool
/// follows ([`metal_space::count_sampler_ids`]). A new device's class joins the process's
/// placement of degradations ([`crate::answer::native::device_made`]).
pub fn adopt(sel: &'static str, raw: rt::Device) -> VulkanDevice {
    crate::platform::answered_at(sel, raw.properties().metal_level);
    let id = registry_id(&raw);
    let mut devices = DEVICES.lock();
    if let Some((_, kept)) = devices.iter().find(|(at, _)| *at == id) {
        return kept.clone_ref();
    }
    metal_space::count_sampler_ids(&raw);
    crate::answer::native::device_made(raw.properties());
    let made = Obj::new(DeviceObject {
        raw,
        registry_id: id,
        caps: OnceLock::new(),
    });
    devices.push((id, made.clone_ref()));
    made
}

/// Tells every device made in the process that the pages under the `length` host bytes at
/// `bytes` were replaced (`rt::Device::host_range_remapped`, [`crate::remap`]): a no-copy buffer
/// made over any of them after this imports the pages there then. Not a Metal call, so it never
/// stops an ended task's thread. Callable from any thread, a no-copy buffer's deallocator among
/// them: the runtime's lock under this one is a leaf. A null address or a length of 0 tells
/// nothing.
pub(crate) fn host_range_remapped(bytes: HostPtr, length: u64) {
    let Some(bytes) = NonNull::new(bytes.as_ptr::<u8>()) else {
        return;
    };
    if length == 0 {
        return;
    }
    for (_, device) in DEVICES.lock().iter() {
        device.raw.host_range_remapped(bytes, length);
    }
}

/// A deallocator of core's the runtime may run, once, or drop unrun when it refuses the buffer:
/// whichever comes first takes it from the slot.
type Slot<D> = Arc<Mutex<Option<D>>>;

/// The runtime's deallocator over `slot`: it takes core's and runs it, marked ours, with the
/// buffer's pointer and length.
fn runtime_deallocator<D: Send + 'static>(
    slot: &Slot<D>,
    run: impl FnOnce(D, NonNull<u8>, u64) + Send + 'static,
) -> rt::Deallocator {
    let slot = Arc::clone(slot);
    Box::new(move |pointer, length| {
        // A callout on the runtime's thread: marked ours, so a panic here aborts.
        let _env = BlockEnv::foreign();
        if let Some(d) = slot.lock().take() {
            run(d, pointer, length);
        }
    })
}

/// Core's deallocator of a buffer the runtime did not make: never run, and its captures kept for
/// good, as the Metal platform keeps them (what they hold stays mapped).
fn keep_unrun<D, T>(made: &RaiseResult<Option<T>>, slot: Option<Slot<D>>) {
    if !matches!(made, Ok(Some(_)))
        && let Some(d) = slot.and_then(|slot| slot.lock().take())
    {
        keep_forever(d, Kept::AddressSpace);
    }
}

/// What the runtime is given for a no-copy buffer over the `length` bytes at `start`, which
/// core's `HostRange` vouches for, with `run`, the runtime's form of core's deallocator: the first
/// byte it imports (the bytes, or a writable twin of them where it cannot import them as mapped,
/// [`crate::os::import_view`]), the deallocator, which runs at Metal's moment, given the bytes'
/// own address, and the keep-alive, which the runtime drops once it freed its import: it holds
/// the twin, and an unmap hold on the bytes' pages, so that core's unmap of its mapping of them
/// through the platform's VM, which its deallocator may make, waits for the free
/// ([`crate::remap::UnmapHold`]); and whether the import is a writable twin of read-only pages,
/// whose GPU writes reach them where Metal drops them ([`Backing::writes_kept`]).
fn no_copy_release(
    start: NonNull<u8>,
    length: u64,
    run: Option<rt::Deallocator>,
) -> (NonNull<u8>, Option<rt::Deallocator>, rt::KeepAlive, bool) {
    let view = crate::os::import_view(start, length);
    let imported = view.at();
    let kept = view.writes_kept();
    let (run, twin) = view.release(start, run);
    let hold = crate::remap::UnmapHold::new(start, length);
    (imported, run, backing::keep_alive((twin, hold)), kept)
}

/// What Metal answers where the runtime refuses a buffer or heap made at `placing`'s address:
/// `nil`, which answers nil for the runtime's refusal of the address, for a guest pin and for one
/// Metal's space refused ([`Placing::nil_at_conflict`]); for a range the space placed, nil only
/// for memory the device cannot allocate, and a death for the runtime's refusal of the address,
/// where Metal makes the resource ([`metal_space`]).
fn placed_nil(placing: &Placing, nil: Nil) -> Nil {
    match placing.nil_at_conflict() {
        true => nil,
        false => Nil::Memory,
    }
}

/// `descriptor` with `token`, the range token of its range of Metal's space
/// ([`metal_space::Hold`]), where it has one.
fn with_token(
    descriptor: rt::BufferDescriptor<'_>,
    token: Option<rt::RangeToken>,
) -> rt::BufferDescriptor<'_> {
    match token {
        Some(token) => descriptor.with_range_token(token),
        None => descriptor,
    }
}

/// The slot of a buffer `sel` makes on `raw` that Metal suballocates, where `placing` holds one
/// ([`Placing::pool_slot`]): the runtime's reservation of its pool and the slot's offset in it,
/// which the buffer's descriptor takes (`BufferDescriptor::with_pool_slot`). A refusal of the
/// reservation dies at `sel`, naming it, with the space unlocked.
fn pool_slot(
    sel: &'static str,
    raw: &rt::Device,
    placing: &mut Placing,
) -> RaiseResult<Option<(Arc<rt::AddressPool>, u64)>> {
    match placing.pool_slot(raw) {
        None => Ok(None),
        Some(Ok(slot)) => Ok(Some(slot)),
        Some(Err(e)) => decide(sel, Policy::Dies, e),
    }
}

/// `descriptor` in `pool`'s slot, the one Metal suballocates it in ([`pool_slot`]), where it has
/// one.
fn in_slot<'a>(
    descriptor: rt::BufferDescriptor<'a>,
    pool: &'a Option<(Arc<rt::AddressPool>, u64)>,
) -> rt::BufferDescriptor<'a> {
    match pool {
        Some((pool, offset)) => descriptor.with_pool_slot(pool, *offset),
        None => descriptor,
    }
}

/// What Metal's space holds a buffer of its own memory of `options` as: private or not
/// ([`metal_space::buffer_space`]).
fn own_bytes(options: rt::ResourceOptions) -> metal_space::Bytes {
    metal_space::Bytes::Own {
        class: metal_space::PoolClass::of(options.storage_mode().ok()),
    }
}

/// The first byte of `bytes`, which the runtime's no-copy buffers take non-null; a NULL one dies
/// at `sel`.
fn pointer(sel: &'static str, bytes: HostPtr) -> NonNull<u8> {
    NonNull::new(bytes.as_ptr::<u8>()).unwrap_or_else(|| unrepresentable(sel, "a NULL pointer"))
}

/// A buffer descriptor's `resourceOptions` before any setter: `Shared`.
const BUFFER_DESCRIPTOR_OPTIONS: u64 = 0;

/// A heap descriptor's `resourceOptions` before any setter: `Private` (measured).
const HEAP_DESCRIPTOR_OPTIONS: u64 = 0x20;

/// `MTLResourceOptions` bit 24, which core sets on a read-only detached buffer or heap, and which
/// the runtime takes and drops ([`crate::gpu::backing`] keeps it).
const READ_ONLY_OPTION: u64 = 0x100_0000;

/// The resource options a descriptor holds after core's storage properties, in order, from
/// `initial`: a storage mode replaces bits 4 to 7 and a hazard tracking mode bits 8 and 9, as
/// their setters do (measured on a heap descriptor: storage 2 then tracking 2 holds 0x220). A
/// buffer descriptor's value sends no hazard tracking mode.
fn descriptor_options(initial: u64, options: &[OptionSet]) -> u64 {
    options.iter().fold(initial, |held, option| match *option {
        OptionSet::ResourceOptions(options) => options,
        OptionSet::ResourceOptionsOr(bits) => held | bits,
        OptionSet::StorageMode(mode) => (held & !0xf0) | ((mode & 0xf) << 4),
        OptionSet::HazardTrackingMode(mode) => (held & !0x300) | ((mode & 0x3) << 8),
    })
}

impl VulkanDevice {
    /// The runtime's answer to `heapTextureSizeAndAlignWithDescriptor:` for `desc`, asked inside
    /// `sel`; `None` where it refuses it. What [`metal_space::own_heap_size`] and
    /// [`metal_space::metal_size_and_align`] make Metal's answer of.
    pub(crate) fn size_and_align_at(
        &self,
        sel: &'static str,
        desc: &rt::TextureDescriptor,
    ) -> Option<SizeAndAlign> {
        let answer = plain_at(sel, || self.raw.heap_texture_size_and_align(desc)).ok()?;
        Some(SizeAndAlign {
            size: answer.size,
            align: answer.align,
        })
    }

    /// The runtime's model of Metal's answer to `heapTextureSizeAndAlignWithDescriptor:` for
    /// `desc` (`Device::metal_heap_texture_size_and_align`), asked inside `sel`: the
    /// same on every device at the device's Metal level (the host's release on macOS,
    /// [`crate::os::METAL_LEVEL`] elsewhere), whatever its image needs, with no image made and
    /// nothing announced. `None` where the model has no answer (a format or a sample count not
    /// measured), and where the runtime refuses the descriptor. What
    /// [`metal_space::own_heap_size`] takes first.
    pub(crate) fn metal_answer_at(
        &self,
        sel: &'static str,
        desc: &rt::TextureDescriptor,
    ) -> Option<SizeAndAlign> {
        let answer = plain_at(sel, || self.raw.metal_heap_texture_size_and_align(desc));
        let answer = answer.ok()??;
        Some(SizeAndAlign {
            size: answer.size,
            align: answer.align,
        })
    }

    /// The handle of a texture this device made, on `surface` when it was made on one, answering
    /// the usage bits `private_usage` beyond the runtime's ([`TextureObject::private_usage`]),
    /// holding `space` of Metal's address space.
    pub(crate) fn texture_of(
        &self,
        raw: rt::Texture,
        surface: Option<Arc<OsSurface>>,
        private_usage: u64,
        space: Arc<metal_space::Hold>,
    ) -> VulkanTexture {
        Obj::new(TextureObject {
            raw,
            device: self.clone_ref(),
            surface,
            plane: None,
            under: None,
            private_usage,
            space,
        })
    }

    /// The handle of a texture this device made over a buffer's bytes or placed in a heap, which
    /// holds that buffer or heap while it lives ([`Under`]), answering the usage bits
    /// `private_usage` beyond the runtime's ([`TextureObject::private_usage`]). It takes none of
    /// Metal's space but its descriptor ([`metal_space::texture`]).
    pub(crate) fn texture_in(
        &self,
        raw: rt::Texture,
        under: Under,
        private_usage: u64,
    ) -> VulkanTexture {
        let kind = match under {
            Under::Buffer(..) => metal_space::TextureKind::OverBuffer,
            Under::Heap(..) => metal_space::TextureKind::InHeap,
        };
        let space = Arc::new(metal_space::texture(kind));
        Obj::new(TextureObject {
            raw,
            device: self.clone_ref(),
            surface: None,
            plane: None,
            under: Some(under),
            private_usage,
            space,
        })
    }

    /// The handle of a buffer this device made, over `surface` when it was made on one, over the
    /// guest's pages where `over_guest_pages` says so or the surface was made over them
    /// ([`BufferObject::over_guest_pages`]).
    pub(crate) fn buffer(
        &self,
        raw: rt::Buffer,
        surface: Option<Arc<OsSurface>>,
        over_guest_pages: bool,
    ) -> VulkanBuffer {
        let id = raw.object_id();
        let on_guest_pages = surface.as_deref().is_some_and(crate::os::over_guest_pages);
        Obj::new(BufferObject {
            raw,
            device: self.clone_ref(),
            surface,
            backing: Backing::new(false, id),
            heap: None,
            kept_whole: std::sync::atomic::AtomicU64::new(0),
            over_guest_pages: over_guest_pages || on_guest_pages,
        })
    }

    /// The handle of a buffer placed in `heap`, whose detach detaches the heap and whose replace
    /// answers NO ([`crate::gpu::backing`]).
    pub(crate) fn placed_buffer(&self, raw: rt::Buffer, heap: &VulkanHeap) -> VulkanBuffer {
        Obj::new(BufferObject {
            raw,
            device: self.clone_ref(),
            surface: None,
            backing: Backing::fixed(),
            heap: Some(heap.clone_ref()),
            kept_whole: std::sync::atomic::AtomicU64::new(0),
            over_guest_pages: false,
        })
    }

    /// A Shared no-copy buffer over all of `surface`'s memory (`IOSurfaceGetBaseAddress`, its
    /// allocation size), whose deallocator holds `surface` until the runtime releases the buffer,
    /// at Metal's moment (given a keep-alive), and whose range token is
    /// `token`, which the runtime drops as it frees the buffer's range. Over the surface's memory
    /// itself, nothing of this platform's keeps it mapped: the surface's last release may unmap
    /// it before the runtime frees its import (`crate::remap`'s docs, "What is not ours to
    /// unmap"); over an alias, the keep-alive holds the alias until then.
    ///
    /// At an offset into the surface's allocation (`kIOSurfaceOffset`,
    /// [`crate::os::surface_offset`]) the base address lies that far into it, and the buffer's last
    /// bytes, as many, lie past the allocation's end. Metal's buffer on such a surface
    /// (`newBufferWithIOSurface:`) answers the allocation size too, its first byte the surface's at
    /// the base address, and reads 0 past the allocation's end (measured on the M3 Max at 0x40,
    /// 0x1000 and 0x4000, by a kernel's copy). So the buffer is over an alias of the allocation's
    /// pages followed by zero-filled pages of the platform's own ([`backing::gather`]), from the
    /// offset into it, where the pages past the allocation would be other memory or none; what a
    /// write past the allocation's end does on Metal is not measured, and here it reaches those
    /// pages alone.
    pub(crate) fn buffer_over_surface(
        &self,
        sel: &'static str,
        surface: &Arc<OsSurface>,
        token: rt::RangeToken,
    ) -> RaiseResult<Option<rt::Buffer>> {
        let any = rt::AddressPlacement::Any;
        self.buffer_over_surface_at(sel, surface, (any, Some(token)), None, Nil::BufferLength)
    }

    /// [`buffer_over_surface`](Self::buffer_over_surface), at `placement` in the Metal address
    /// space with its range token where it has one ([`metal_space::surface_buffer`]), sharing
    /// the range of `shares`, a buffer over the surface, where given, `nil` saying which of the
    /// runtime's refusals are nil: `Outside` for one of the platform's own, which takes no Metal
    /// address ([`crate::gpu::own_buffer`]). The alias of the allocation, where the buffer lies
    /// over one, is one for the surface, which every buffer over it lies over while one lives
    /// ([`crate::os::surface_alias`]), so that each lies over the same first byte. Where the
    /// GPU's writes through the surface never reach its memory ([`crate::os::gpu_read_only`]),
    /// the alias is read-only: the GPU reads the same pages, and its writes are dropped as
    /// through a read-only alias of the guest's pages ([`crate::gpu::backing`]'s docs).
    pub(crate) fn buffer_over_surface_at(
        &self,
        sel: &'static str,
        surface: &Arc<OsSurface>,
        (placement, token): (rt::AddressPlacement, Option<rt::RangeToken>),
        shares: Option<&rt::Buffer>,
        nil: Nil,
    ) -> RaiseResult<Option<rt::Buffer>> {
        use reims_vgpu_core::platform::os::surface::Surface as _;
        let held = Arc::clone(surface);
        let run: rt::Deallocator = Box::new(move |_, _| {
            let _env = BlockEnv::foreign();
            drop(held);
        });
        let (base, length) = (pointer(sel, surface.base_address()), surface.alloc_size());
        let offset = crate::os::surface_offset(surface);
        let read_only = crate::os::gpu_read_only(surface);
        let (base, keep) = if offset == 0 && !read_only {
            log::debug!(
                "{sel}: a buffer of {length:#x} bytes over a surface in {}, imported at its base \
                 address",
                crate::os::memory_kind(surface)
            );
            (base, backing::keep_alive(()))
        } else {
            let first = u64::try_from(base.as_ptr().addr())
                .unwrap_or(u64::MAX)
                .saturating_sub(offset);
            let allocation = [ResourceAddressRange {
                address: first,
                length,
            }];
            let mut fresh = false;
            let made = || {
                fresh = true;
                backing::gather(sel, &allocation, length.saturating_add(offset), read_only)
            };
            let alias = crate::os::surface_alias(surface, made);
            log::debug!(
                "{sel}: a buffer of {length:#x} bytes over a surface in {}, at offset \
                 {offset:#x}{}, over {} alias of its allocation",
                crate::os::memory_kind(surface),
                if read_only {
                    ", read-only for the GPU"
                } else {
                    ""
                },
                match (&alias, fresh) {
                    (None, _) => "no",
                    (Some(_), true) => "a new",
                    (Some(_), false) => "the kept",
                }
            );
            let Some(alias) = alias else {
                return Ok(None);
            };
            let (at, keep) = alias.into_runtime();
            // The alias holds `offset` bytes more than the buffer, past its end.
            let from = at
                .as_ptr()
                .wrapping_add(usize::try_from(offset).unwrap_or(usize::MAX));
            (pointer(sel, HostPtr::from_ptr(from)), keep)
        };
        // SAFETY: the surface's memory stays mapped while the surface lives, which the
        // deallocator holds until the runtime lets go of the buffer; an alias of its allocation,
        // which the keep-alive holds, until the runtime freed its import.
        let memory = unsafe { crate::os::host_memory(base) };
        let contents = rt::BufferContents::NoCopy {
            memory,
            deallocator: Some(run),
            keep_alive: Some(keep),
        };
        let descriptor =
            rt::BufferDescriptor::new(length, rt::ResourceOptions::default(), contents)
                .with_placement(placement);
        let descriptor = match shares {
            Some(other) => descriptor.with_range_of(other),
            None => descriptor,
        };
        create(sel, Policy::Dies, nil, || {
            self.raw
                .new_buffer_with_descriptor(with_token(descriptor, token))
        })
    }

    /// A live buffer of this device over the surface `key` at its range's address
    /// ([`SURFACE_BUFFERS`]), forgetting those gone.
    fn live_surface_buffer(&self, key: u64) -> Option<VulkanBuffer> {
        let mut surfaces = SURFACE_BUFFERS.lock();
        let buffers = surfaces.get_mut(&key)?;
        buffers.retain(|buffer| buffer.strong_count() > 0);
        let live = (buffers.iter())
            .filter_map(Obj::upgrade)
            .find(|buffer| buffer.device.id() == self.id());
        if buffers.is_empty() {
            surfaces.remove(&key);
        }
        live
    }

    /// `buffer`, over the surface `key` at its range's address, for the buffers made over it
    /// after it to share ([`SURFACE_BUFFERS`]).
    fn keep_surface_buffer(key: u64, buffer: &VulkanBuffer) {
        let mut surfaces = SURFACE_BUFFERS.lock();
        surfaces.retain(|_, buffers| {
            buffers.retain(|buffer| buffer.strong_count() > 0);
            !buffers.is_empty()
        });
        surfaces.entry(key).or_default().push(buffer.downgrade());
    }
}

impl GpuDevice<VulkanPlatform> for VulkanDevice {
    /// Nothing to allow: the runtime makes a library of a metallib whatever platform it was
    /// built for (`newLibraryWithData:error:` checks no platform).
    fn allow_foreign_libraries(&self) {
        plain(|| ());
    }

    /// The runtime's answer, Metal's but for a depth or stencil texture with
    /// `allowGPUOptimizedContents` off, which MoltenVK keeps GPU-optimized: the runtime answers
    /// 66048 bytes for a 128 by 128 Depth32Float texture and 98816 for a Depth32Float_Stencil8
    /// one, where Metal answers 65536 and 81920 (measured on the M3 Max; with it on or unset,
    /// the sizes are Metal's; [`crate::gpu::caps`]' list). The query hears the runtime announce
    /// it (`OptimizedDepthStencilTexture`, [`crate::answer::HEARD`]) and answers Metal's: the
    /// runtime's model of Metal's answer ([`VulkanDevice::metal_answer_at`]), the size
    /// `newTextureWithDescriptor:` gives such a texture's range of Metal's space, and where the
    /// model has none, Metal's made of the runtime's answers for colour formats
    /// ([`metal_space::metal_size_and_align`]). It dies at the selector where neither answers. An
    /// `X32_Stencil8` texture, which the model answers as a `Stencil8` one of its shape and the
    /// colour rule does not, takes the model's, the Metal platform's answer at each shape
    /// measured (`resource`'s `x32_stencil8_size_queries_are_metals`); it died here before. One of
    /// `ShaderAtomic` usage, at which Metal asserts, the runtime refuses first (`Invalid`), which
    /// dies at the selector too. A heap's placement of such a texture still
    /// dies where the runtime's size decides it, as its image needs that size in the heap
    /// (`crate::gpu::resource::Placements`).
    /// It dies at the selector, too, naming the kind and the texture, for a texture whose layout
    /// in Metal's heaps the runtime does not model, a PVRTC format, whose size and alignment the
    /// runtime answers as its device's image's, larger or smaller than Metal's, and announces so
    /// (`unmeasured-heap-texture-size`, [`crate::answer::HEARD`];
    /// [`metal_space::unmeasured_size`]): the guest reads the answer, and no suite case or guest
    /// run asks it of such a texture. The runtime refuses 8 samples first on every device, as
    /// Metal stops them, so the query dies at that refusal. The runtime models a mipmapped texture
    /// Metal keeps losslessly compressed (a seven-level 64x64 `R32Uint` render target answers
    /// Metal's 0x5800 bytes) and a `Depth32Float_Stencil8` one of one sample. Where the device's
    /// image takes more than the model's answer, the runtime's answer
    /// is the device's (`heap-texture-larger-than-metal`, [`crate::answer`]), as a heap places
    /// the texture by it.
    /// Metal's 0 bytes, at Metal's alignment, for a format it makes no texture of from a
    /// descriptor alone, which the runtime answers as Metal does
    /// ([`crate::gpu::no_texture::descriptor`]).
    fn heap_texture_size_and_align(
        &self,
        desc: &VulkanTextureDescriptor,
    ) -> RaiseResult<SizeAndAlign> {
        let sel = "heapTextureSizeAndAlignWithDescriptor:";
        let fields = desc.fields();
        let desc = fields.descriptor(sel, Maker::SizeQuery);
        let raw = fields.raw_pixel_format();
        let (answer, heard) = hearing(|| {
            send(sel, Policy::Dies, || {
                (self.raw.heap_texture_size_and_align(&desc))
                    .map_err(|e| format_refused(sel, "a texture of", raw, e))
            })
        });
        if heard.contains(&rt::Degradation::OptimizedDepthStencilTexture) {
            let answer = |desc: &rt::TextureDescriptor| self.size_and_align_at(sel, desc);
            let metal = (self.metal_answer_at(sel, &desc))
                .or_else(|| metal_space::metal_size_and_align(&desc, answer));
            match metal {
                Some(metal) => return Ok(metal),
                None => crate::gpu::resource::optimized_size(sel, "the size and alignment of"),
            }
        }
        if heard.contains(&rt::Degradation::UnmeasuredHeapTextureSize) {
            metal_space::unmeasured_size(sel, "the size and alignment of", &desc);
        }
        let answer = answer?;
        Ok(SizeAndAlign {
            size: answer.size,
            align: answer.align,
        })
    }

    /// Any pointer and any length of at least 1 byte; a length of 0, or past the device's largest
    /// buffer, is nil, as Metal answers. So is memory with a page not mapped, or mapped without
    /// read access, which Metal makes no buffer over (measured: PROT_NONE pages, an unmapped
    /// page, and a range of a mapped and an unmapped page are nil; read-only pages make a
    /// buffer): the runtime imports the pages as it makes the buffer, as Metal wires them, and
    /// refuses those it cannot import, which [`OsVm::readable`] tells from its other
    /// refusals, which die. The deallocator runs once, when the runtime releases the buffer; one
    /// the runtime refuses never runs, and its captures are kept. At the address Metal's space
    /// gives it, keeping the pointer's offset into its page ([`metal_space`]), so made through
    /// the runtime's descriptor, which takes an address.
    fn new_buffer_over_host_memory(
        &self,
        _on: OnFault,
        bytes: HostRange<'_>,
        options: u64,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<Option<VulkanBuffer>> {
        let sel = "newBufferWithBytesNoCopy:length:options:deallocator:";
        let options = crate::gpu::resource_options(sel, options);
        let slot = deallocator.map(|d| Arc::new(Mutex::new(Some(d))));
        let run = slot
            .as_ref()
            .map(|slot| runtime_deallocator(slot, |d: NoCopyDeallocator, _, _| d()));
        let start = pointer(sel, bytes.ptr());
        let offset = u64::try_from(start.as_ptr().addr()).unwrap_or(0);
        let (placement, mut placing) =
            metal_space::placing(bytes.len(), metal_space::Bytes::NoCopy(offset), None);
        let token = placing.token();
        // The bytes, or a writable twin of them where the runtime cannot import them as mapped,
        // and what keeps them mapped until the runtime freed its import.
        let (imported, run, keep, kept) = no_copy_release(start, bytes.len(), run);
        let made = create(
            sel,
            Policy::Dies,
            placed_nil(&placing, Nil::BufferLength),
            || {
                // SAFETY: `HostRange::new`'s contract keeps the bytes mapped until the deallocator
                // runs, or for the buffer's life with none, and an unmap of core's through the
                // platform's VM waits for the keep-alive; a twin of them stays until it goes.
                let memory = unsafe { crate::os::host_memory(imported) };
                let contents = rt::BufferContents::NoCopy {
                    memory,
                    deallocator: run,
                    keep_alive: Some(keep),
                };
                let descriptor = rt::BufferDescriptor::new(bytes.len(), options, contents)
                    .with_placement(placement);
                match self
                    .raw
                    .new_buffer_with_descriptor(with_token(descriptor, token))
                {
                    Ok(raw) => Ok(Some(raw)),
                    // Pages Metal makes no buffer over: nil.
                    Err(_) if !OsVm::new().readable(bytes.ptr(), bytes.len()) => Ok(None),
                    Err(e) => Err(e),
                }
            },
        )
        .map(Option::flatten);
        keep_unrun(&made, slot);
        // Core makes a task's no-copy buffers over its own mappings of the guest's pages.
        let over_guest_pages = crate::remap::shows_guest_pages(bytes.ptr(), bytes.len());
        made.map(|b| {
            b.map(|raw| {
                let buffer = self.buffer(raw, None, over_guest_pages);
                buffer.backing.imported(kept);
                buffer
            })
        })
    }

    /// A no-copy buffer over all of the surface's memory, holding one reference of the surface
    /// while the runtime keeps the buffer, at the surface's address in Metal's space
    /// ([`metal_space::surface_buffer`]). The runtime takes no nil surface: a nil one dies. A
    /// buffer over the surface while another holds that address shares the range of a live one,
    /// as Metal answers the address for both; one made while none lives, after the last's
    /// release and before the runtime frees its range, dies: Metal makes it there, and the
    /// runtime shares a range only with a live buffer.
    fn new_buffer_on_surface(
        &self,
        iosurface: Option<&OsSurface>,
    ) -> RaiseResult<Option<VulkanBuffer>> {
        use reims_vgpu_core::platform::os::surface::Surface as _;
        let sel = "newBufferWithIOSurface:";
        let surface = iosurface.unwrap_or_else(|| unrepresentable(sel, "a nil IOSurface"));
        let surface = Arc::new(surface.clone_ref());
        let (key, alloc_size) = (surface.id().0, surface.alloc_size());
        let (place, space) = metal_space::surface_buffer(key, alloc_size);
        let address = space.address().unwrap_or_default();
        // At the surface's address, Metal makes the buffer: the runtime's refusal of the address
        // dies, as a placed range's does ([`placed_nil`]), and so does its refusal to share.
        let (placement, nil, live) = match place {
            SurfacePlace::At(placement @ rt::AddressPlacement::At(_)) => {
                (placement, Nil::Memory, None)
            }
            SurfacePlace::At(placement) => (placement, Nil::BufferLength, None),
            SurfacePlace::Shared => {
                let Some(live) = self.live_surface_buffer(key) else {
                    unrepresentable(
                        sel,
                        &format!(
                            "a buffer over IOSurface {key:#x} while a released one over it holds \
                             its address until the command buffers that used it complete, which \
                             Metal makes at that address too and the runtime shares with a live \
                             buffer alone"
                        ),
                    );
                };
                let at = rt::AddressPlacement::At(rt::GpuAddress(address));
                (at, Nil::Memory, Some(live))
            }
        };
        let token: rt::RangeToken = Box::new(space);
        let shares = live.as_ref().map(|live| &live.raw);
        let made =
            self.buffer_over_surface_at(sel, &surface, (placement, Some(token)), shares, nil)?;
        Ok(made.map(|raw| {
            let buffer = self.buffer(raw, Some(surface), false);
            if let rt::AddressPlacement::At(_) = placement {
                VulkanDevice::keep_surface_buffer(key, &buffer);
            }
            buffer
        }))
    }

    /// Nil for a length of 0 or past the device's largest buffer, and for memory the device
    /// cannot allocate, as Metal answers. At the address Metal's space gives it
    /// ([`metal_space`]), so made through the runtime's descriptor, which takes an address.
    fn new_buffer_of_length(
        &self,
        _on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<VulkanBuffer>> {
        let sel = "newBufferWithLength:options:";
        let options = crate::gpu::resource_options(sel, options);
        let (placement, mut placing) = metal_space::placing(length, own_bytes(options), None);
        let pool = pool_slot(sel, &self.raw, &mut placing)?;
        let descriptor = rt::BufferDescriptor::new(length, options, rt::BufferContents::Zero)
            .with_placement(placement);
        let descriptor = with_token(in_slot(descriptor, &pool), placing.token());
        let made = create(
            sel,
            Policy::Dies,
            placed_nil(&placing, Nil::BufferLength),
            || self.raw.new_buffer_with_descriptor(descriptor),
        );
        metal_space::let_go(pool);
        made.map(|b| b.map(|raw| self.buffer(raw, None, false)))
    }

    /// A queue of at most 64 uncompleted command buffers, as Metal's. A first use of Metal's
    /// device ([`metal_space::FirstUse::Queue`]).
    fn new_command_queue(&self) -> Option<VulkanCommandQueue> {
        let raw = get("newCommandQueue", || self.raw.new_command_queue());
        metal_space::first_use(metal_space::FirstUse::Queue);
        Some(Obj::new(QueueObject::new(raw, self.clone_ref())))
    }

    /// Decoded as Metal's deserializer decodes it ([`serialized::compute`]): nil for an empty blob;
    /// a blob Metal aborts on dies here, and the resolver's raise is the caller's.
    fn deserialize_compute_pipeline_descriptor(
        &self,
        hold: Hold<'_>,
        data: &[u8],
        context: FunctionResolver<'_, VulkanPlatform>,
    ) -> RaiseResult<Option<VulkanComputePipelineDescriptor>> {
        let decoded = timing::creation(
            "newComputePipelineDescriptorWithSerializedData:deserializationContext:",
            "",
            || serialized::compute(hold, data, context),
            || format!("{} bytes", data.len()),
            |d| match d {
                Ok(Some(d)) => format!("kernel {}", timing::stage(&d.desc.kernel)),
                Ok(None) => "nil".to_owned(),
                Err(_) => "raised".to_owned(),
            },
        )?;
        Ok(decoded.map(|d| Obj::new(PipelineDescriptorObject::new(d))))
    }

    /// The descriptor's state, placed at the resource index core set and labelled with its blob's
    /// label, under Metal's raw options; a nil descriptor is Metal's `NSError` for one
    /// ([`nil_pipeline_descriptor`]). The runtime's
    /// refusals are decided as the function form's: the compile's of a function or descriptor it
    /// finds wrong are the `NSError`, and the rest die (an unspecialized function constant, as
    /// Metal aborts for one; a resource index of 16384 or more, which ends Metal's process). One
    /// that supports indirect command buffers dies where the compile layer refuses its kernel's
    /// reflection, which decides whether Metal refuses it
    /// ([`super::states::kernel_direct_arguments_decided`]). Every state takes a new identifier
    /// ([`ComputePipelineObject::new`]).
    fn new_compute_pipeline_state(
        &self,
        descriptor: Option<&VulkanComputePipelineDescriptor>,
        options: u64,
    ) -> RaiseResult<Result<Option<VulkanComputePipeline>, NsErrorText>> {
        let sel = "newComputePipelineStateWithDescriptor:options:reflection:error:";
        let Some(descriptor) = descriptor else {
            return Ok(Err(nil_pipeline_descriptor()));
        };
        let desc = plain(|| descriptor.desc.lock().clone());
        let mut pipeline_options = rt::PipelineOptions::new(rt::PipelineOption::from_raw(options));
        pipeline_options.placement = descriptor.placement();
        pipeline_options.label = descriptor.label.clone();
        timing::creation(
            sel,
            timing::DISPATCH_LATER,
            || match plain_at(sel, || {
                self.raw
                    .new_compute_pipeline_state(&desc, &pipeline_options)
            }) {
                Ok(state) => {
                    if desc.support_indirect_command_buffers {
                        super::states::kernel_direct_arguments_decided(sel, &desc);
                    }
                    Ok(Ok(Some(Obj::new(ComputePipelineObject::of(
                        state,
                        desc.clone(),
                    )))))
                }
                Err(e) => Ok(Err(compile_error(sel, e, Some(&desc)))),
            },
            || format!("kernel {}", timing::stage(&desc.kernel)),
            |made| timing::raised_answer(made, |s| timing::state(s.unique_identifier)),
        )
    }

    /// Every refusal dies but the compile's of a function or descriptor it finds wrong, which is
    /// the `NSError`: an unspecialized function constant dies, as Metal aborts for one (testing.md
    /// section 6.18), and so does a nil function, which the runtime has no form for.
    fn new_compute_pipeline_state_for_function(
        &self,
        compute_function: Option<&VulkanFunction>,
    ) -> Result<Option<VulkanComputePipeline>, NsErrorText> {
        let sel = "newComputePipelineStateWithFunction:error:";
        let function = compute_function.unwrap_or_else(|| unrepresentable(sel, "a nil function"));
        let options = rt::PipelineOptions::default();
        timing::creation(
            sel,
            timing::DISPATCH_LATER,
            || match plain_at(sel, || {
                self.raw
                    .new_compute_pipeline_state_with_function(function, &options)
            }) {
                Ok(state) => {
                    let source = StageSource::Function(function.function_descriptor());
                    let desc = ComputePipelineDescriptor::new(source);
                    Ok(Some(Obj::new(ComputePipelineObject::of(state, desc))))
                }
                Err(e) => Err(compile_error(sel, e, None)),
            },
            || format!("kernel {:?}", function.name()),
            |made| timing::answer(made, |s| timing::state(s.unique_identifier)),
        )
    }

    /// The one creator whose result is kept: the library's own shaders, built into a metallib
    /// with this crate, are loaded once per device, and a later call answers the same library
    /// ([`super::shaders`]).
    fn new_shader_library(&self) -> Result<Option<VulkanLibrary>, NsErrorText> {
        super::shaders::library(self)
    }

    fn new_event(&self, _on: OnFault) -> RaiseResult<Option<VulkanEvent>> {
        send("newEvent", Policy::Dies, || self.raw.new_event()).map(|e| Some(Obj::new(e)))
    }

    /// The runtime makes a fence for every call.
    fn new_fence(&self) -> RaiseResult<Option<VulkanFence>> {
        Ok(Some(Obj::new(plain(|| self.raw.new_fence()))))
    }

    /// A metallib only, as Metal and the runtime take: other bytes, raw AIR and a bitcode wrapper
    /// among them, are Metal's `NSError` (testing.md section 6.18).
    fn new_library(&self, data: &[u8]) -> RaiseResult<Result<Option<VulkanLibrary>, NsErrorText>> {
        let sel = "newLibraryWithData:error:";
        let made = timing::creation(
            sel,
            "",
            || {
                send(sel, Policy::Dies, || {
                    match self.raw.new_library_with_data(data) {
                        Ok(raw) => Ok(Ok(raw)),
                        Err(rt::Error::Compile(metal2vulkan::error::Error::InvalidInput(_))) => {
                            Ok(Err(invalid_library_file()))
                        }
                        Err(e) => Err(e),
                    }
                })
            },
            || format!("{} bytes", data.len()),
            |made| match made {
                Ok(Ok(raw)) => format!("made, {} functions", raw.function_names().len()),
                Ok(Err(e)) => format!("refused: {}", e.description),
                Err(_) => "raised".to_owned(),
            },
        )?;
        Ok(made.map(|raw| {
            Some(Obj::new(LibraryObject {
                raw,
                device: self.clone_ref(),
            }))
        }))
    }

    fn new_rate_map(
        &self,
        descriptor: &VulkanRateMapDescriptor,
    ) -> RaiseResult<Option<VulkanRateMap>> {
        super::rate_map::new_map(self, descriptor)
    }

    /// Decoded as the compute form is ([`serialized::render`]).
    fn deserialize_render_pipeline_descriptor(
        &self,
        hold: Hold<'_>,
        data: &[u8],
        context: FunctionResolver<'_, VulkanPlatform>,
    ) -> RaiseResult<Option<VulkanRenderPipelineDescriptor>> {
        let decoded = timing::creation(
            "newRenderPipelineDescriptorWithSerializedData:deserializationContext:",
            "",
            || serialized::render(hold, data, context),
            || format!("{} bytes", data.len()),
            |d| match d {
                Ok(Some(d)) => timing::render_stages(&d.desc),
                Ok(None) => "nil".to_owned(),
                Err(_) => "raised".to_owned(),
            },
        )?;
        Ok(decoded.map(|d| Obj::new(PipelineDescriptorObject::new(d))))
    }

    /// The descriptor's state at its resource index ([`super::states`]).
    fn new_render_pipeline_state_from_descriptor(
        &self,
        descriptor: Option<&VulkanRenderPipelineDescriptor>,
        options: u64,
    ) -> RaiseResult<Result<Option<VulkanRenderPipeline>, NsErrorText>> {
        super::states::new_render_pipeline_state_from_descriptor(self, descriptor, options)
    }

    /// The descriptor's tile state ([`super::states`]).
    fn new_tile_render_pipeline_state(
        &self,
        descriptor: Option<&VulkanTileRenderPipelineDescriptor>,
        options: u64,
    ) -> RaiseResult<Result<Option<VulkanRenderPipeline>, NsErrorText>> {
        super::states::new_tile_render_pipeline_state(self, descriptor, options)
    }

    fn new_shared_event(&self) -> Option<VulkanSharedEvent> {
        Some(Obj::new(get("newSharedEvent", || {
            self.raw.new_shared_event()
        })))
    }

    /// The event of a handle of this process (the runtime's handles cross no process). The
    /// runtime has no nil handle: a nil one dies here.
    fn open_shared_event(
        &self,
        shared_event_handle: Option<&VulkanSharedEventHandle>,
    ) -> Option<VulkanSharedEvent> {
        let sel = "newSharedEventWithHandle:";
        let handle = shared_event_handle.unwrap_or_else(|| unrepresentable(sel, "a nil handle"));
        Some(Obj::new(get(sel, || {
            self.raw.new_shared_event_with_handle(handle)
        })))
    }

    /// A texture with in-process handles; nil for storage other than `Private`, as Metal answers,
    /// a size of 0 included, which it aborts at in `Private` ([`crate::gpu::desc`]); one the
    /// runtime makes at such a storage dies, as it would hold no range of Metal's space. Nil for
    /// a format Metal makes no texture of from a descriptor alone
    /// ([`crate::gpu::desc::TextureFields::formatless`]), as Metal answers before its validation,
    /// whatever the descriptor holds (measured: [`crate::gpu::no_texture`]'s docs, and
    /// [`crate::gpu::ycbcr`]'s), where the runtime validates first. One of protection options
    /// other than none is made as without them, relayed, where Metal answers nil
    /// ([`crate::gpu::desc::TextureFields::made_unprotected`]).
    fn new_shared_texture(
        &self,
        descriptor: &VulkanTextureDescriptor,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newSharedTextureWithDescriptor:";
        let fields = descriptor.fields();
        if fields.formatless().is_some() {
            return Ok(None);
        }
        // Metal answers nil for a value the runtime keeps raw at a storage other than `Private`,
        // as for every format, and at `Private` for 630 to 650, which it describes no format
        // for; it makes a texture of a format it describes, and asserts at 255 and 262 (measured
        // at macOS 26.7: `formats_the_runtime_lacks_die_where_textures_are_made`; at another
        // Metal level, one of them at `Private` dies where it is made, as nothing measured it).
        // It answers nil for `MTLPixelFormatInvalid` at either storage (measured:
        // `invalid_formats_are_metals`).
        const PRIVATE: u64 = 2;
        let storage = (fields.desc.resource_options.raw() >> 4) & 0xf;
        if let Some(raw) = fields.lacked_format
            && (storage != PRIVATE
                || (matches!(raw, 630..=650)
                    && crate::platform::answering_level(sel, || {
                        format!("a shared texture of pixel format {raw} at Private storage")
                    }) == rt::MetalLevel::MacOS26_7))
        {
            return Ok(None);
        }
        if fields.desc.pixel_format == metal2vulkan::metal::PixelFormat::Invalid
            && fields.lacked_format.is_none()
        {
            return Ok(None);
        }
        let desc = fields.descriptor(sel, Maker::Shared);
        // Its range of Metal's space, placed before the runtime makes it, at `Private` storage,
        // where it would be made: its allocation is sized by Apple's layout from Metal's
        // description of its format, so a format the device's Metal level describes and macOS
        // 26.7, where the layout was measured, does not (630 to 645 at macOS 27.0.1) dies here
        // before the runtime makes anything ([`crate::gpu::format_info::block`]). At another
        // storage the runtime answers nil, as Metal does, and nothing is placed; a texture the
        // runtime made there would hold no range of Metal's space where Metal makes none, so it
        // dies. The runtime's range token holds it until the runtime frees the texture, and its
        // object shares it ([`metal_space::Hold`]).
        let space = match storage == PRIVATE {
            true => metal_space::texture(metal_space::shared_backing(sel, &desc)),
            false => metal_space::Hold::NONE,
        };
        let space = Arc::new(space);
        let token: rt::RangeToken = Box::new(Arc::clone(&space));
        let raw = fields.raw_pixel_format();
        let made = create(sel, Policy::Dies, Nil::StorageMode, || {
            (self.raw.new_shared_texture_with_range_token(&desc, token))
                .map_err(|e| format_refused(sel, "a texture of", raw, e))
        })?;
        if made.is_some() && storage != PRIVATE {
            unrepresentable(
                sel,
                &format!(
                    "a shareable texture of storage mode {storage}, which the runtime makes where \
                     Metal answers nil, and which takes no range of Metal's space"
                ),
            );
        }
        if made.is_some() {
            fields.made_unprotected(sel);
        }
        Ok(made.map(|raw| self.texture_of(raw, None, 0, space)))
    }

    /// The texture of a handle of this process, its resource ID at `resource_index` (0: one the
    /// runtime picks), answering its usage as the handle's texture does. It takes no range of
    /// Metal's space and shares the handle's texture's (`vaprobe surfaces`), which the shared
    /// texture's range token holds while it lives, and while a command buffer that holds it
    /// runs, and its object shares ([`metal_space::Hold`]; `vaprobe sharedrelease`: with the
    /// shared texture and its handle released, Metal kept the range while the opened texture
    /// lived, and freed it once that was released too). The runtime takes no nil handle: a nil
    /// one dies.
    fn open_shared_texture(
        &self,
        handle: Option<&VulkanSharedTextureHandle>,
        resource_index: u64,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newSharedTextureWithHandle:withResourceIndex:";
        let handle = handle.unwrap_or_else(|| unrepresentable(sel, "a nil handle"));
        let placement = rt::IdPlacement::At(rt::ResourceId(resource_index));
        let raw = send(sel, Policy::Dies, || {
            self.raw
                .new_shared_texture_with_handle(&handle.raw, placement)
        })?;
        let usage = handle.private_usage;
        Ok(Some(self.texture_of(
            raw,
            None,
            usage,
            Arc::new(metal_space::Hold::sharing(Arc::clone(&handle.space))),
        )))
    }

    /// Nil for memory the device cannot allocate, and for a descriptor Metal makes no texture
    /// of ([`crate::gpu::desc::metal_makes_none`], and a format Metal makes no texture of from a
    /// descriptor alone, which the runtime answers as Metal does,
    /// [`crate::gpu::no_texture::descriptor`]), as Metal answers. It holds what Metal's space
    /// gives a texture of memory of its own ([`metal_space::texture_space`]): a range of
    /// Metal's heap size for it ([`metal_space::own_heap_size`]: the runtime's model of Metal's
    /// answer, [`VulkanDevice::metal_answer_at`], on every device, where the device's own answer
    /// may be larger), a slot for one Metal suballocates (of a heap size under a page), none for
    /// a memoryless one, placed before the runtime makes it, which the runtime's range token
    /// holds until it frees the texture, and its object shares ([`metal_space::Hold`]). Where
    /// the model has no answer, a depth or stencil texture without `allowGPUOptimizedContents`
    /// whose size the runtime answers larger than Metal's (`optimized-depth-stencil-texture`)
    /// and Metal's is not measured for dies here, at every shape; so does one whose
    /// layout in Metal's heaps the runtime does not model (`unmeasured-heap-texture-size`: a
    /// PVRTC format), at every shape ([`metal_space::unmeasured_size`]). One of protection
    /// options other than none is made as without them, relayed, where Metal answers nil
    /// ([`crate::gpu::desc::TextureFields::made_unprotected`]).
    fn new_texture(
        &self,
        _on: OnFault,
        descriptor: &VulkanTextureDescriptor,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureWithDescriptor:";
        let fields = descriptor.fields();
        let desc = fields.descriptor(sel, Maker::Other);
        if crate::gpu::desc::metal_makes_none(&desc) {
            return Ok(None);
        }
        let raw = fields.raw_pixel_format();
        let kind = match desc.resource_options.storage_mode() {
            Ok(rt::StorageMode::Memoryless) => TextureKind::Memoryless,
            _ => {
                let metal = self.metal_answer_at(sel, &desc);
                let answer = |desc: &rt::TextureDescriptor| self.size_and_align_at(sel, desc);
                let (heap_size, heard) =
                    hearing(|| metal_space::own_heap_size(&desc, metal, answer));
                if heard.contains(&rt::Degradation::OptimizedDepthStencilTexture) {
                    crate::gpu::resource::optimized_size(sel, "the range of Metal's space of");
                }
                if heard.contains(&rt::Degradation::UnmeasuredHeapTextureSize) {
                    metal_space::unmeasured_size(sel, "the range of Metal's space of", &desc);
                }
                metal_space::own_memory(sel, &desc, heap_size)
            }
        };
        let space = Arc::new(metal_space::texture(kind));
        let token: rt::RangeToken = Box::new(Arc::clone(&space));
        let made = create(sel, Policy::Dies, Nil::Memory, || {
            (self.raw.new_texture_with_range_token(&desc, token))
                .map_err(|e| format_refused(sel, "a texture of", raw, e))
        })?;
        if made.is_some() {
            fields.made_unprotected(sel);
        }
        Ok(made.map(|raw| self.texture_of(raw, None, 0, space)))
    }

    /// A texture over a no-copy buffer of all of the surface's memory, of its plane at the plane's
    /// offset with rows the plane's bytes per row apart, under Metal's IOSurface rules
    /// (`Buffer::new_texture_with_iosurface_plane`), holding one reference of the surface while
    /// the runtime keeps the texture. Nil for a format Metal makes no texture of
    /// ([`crate::gpu::no_texture`]) once the runtime's validation of it passes, which dies where
    /// Metal's aborts ([`crate::gpu::no_texture::descriptor`]), for a plane past a planar
    /// surface's last, and for what the runtime answers nil for over a plane (`Error::Nil`), as
    /// Metal answers; a nil surface dies, and so does a plane past 0 of a surface without planes,
    /// where Metal asserts. A
    /// texture of a two-plane YCbCr format the runtime reads (`YCBCR8_420_2P`, 505 and 543 among
    /// them) is the runtime's, over the surface's facts it reads ([`crate::os::ycbcr_facts`],
    /// [`crate::gpu::ycbcr::runtime_makes`]): each surface, plane, size or matrix it refuses dies
    /// here, naming the refusal. So is one of a two-plane XR10 format (550, 551), over the color
    /// words of plane 0 and the alpha bytes of plane 1 ([`crate::os::plane_facts`]), which Metal
    /// reads alike ('b3a8'); what the runtime refuses of it (a single-plane surface, plane 1, a
    /// rotation) dies here too, naming the refusal. It takes the descriptor's rotation, which the
    /// runtime's texture holds ([`crate::gpu::desc`]).
    fn new_texture_on_surface(
        &self,
        descriptor: &VulkanTextureDescriptor,
        iosurface: Option<&OsSurface>,
        plane: u64,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureWithDescriptor:iosurface:plane:";
        let surface = iosurface.unwrap_or_else(|| unrepresentable(sel, "a nil IOSurface"));
        let fields = descriptor.fields();
        log::debug!(
            "{}",
            OnSurface {
                fields: &fields,
                surface,
                plane
            }
        );
        // A format of `no_texture` is nil, once Metal's validation passes, before the plane and
        // the surface are looked at: the runtime's heap size query validates its descriptor as
        // the runtime's creators do (over a surface's plane too), and answers 0 bytes for it.
        if fields.no_texture_format.is_some()
            && let Some(pf) = fields.formatless()
        {
            let desc = crate::gpu::no_texture::descriptor(sel, &fields, pf);
            send(sel, Policy::Dies, || {
                self.raw.heap_texture_size_and_align(&desc)
            })?;
            return Ok(None);
        }
        let mut desc = match fields.ycbcr_format {
            Some(format) => fields.on_surface_ycbcr(sel, format),
            None => fields.descriptor(sel, Maker::Surface),
        };
        let usage = match crate::os::gpu_read_only(surface) {
            true => crate::gpu::resource::read_only_usage(&mut desc),
            false => 0,
        };
        // Metal asserts at a plane past 0 of a surface without planes, whatever the format
        // ("IOSurface is non-planar, but plane 1 specified"), and answers nil at a plane past a
        // planar surface's last (measured: `textures_over_planes_a_surface_lacks_are_metals`).
        if plane != 0 && reims_vgpu_core::platform::os::surface::Surface::plane_count(surface) == 0
        {
            metal_aborts(sel, &format!("plane {plane} of a surface without planes"));
        }
        let Some(mut at) = crate::os::plane(surface, plane) else {
            return Ok(None);
        };
        let facts = if crate::gpu::ycbcr::runtime_makes(desc.pixel_format) {
            crate::os::ycbcr_facts(surface, plane)
        } else if two_plane_xr10(desc.pixel_format) {
            crate::os::plane_facts(surface, plane)
        } else {
            None
        };
        if let Some(facts) = facts {
            at = at.with_ycbcr(facts);
        }
        let plane = at;
        let surface = Arc::new(surface.clone_ref());
        // The texture's hold on its surface's range, which the range token of the buffer it is
        // over holds too: the range goes once the runtime frees that buffer, after the last
        // command buffer that used the texture completed, as Metal frees it ([`metal_space::Hold`]).
        let space = Arc::new(metal_space::surface(
            surface.id().0,
            reims_vgpu_core::platform::os::surface::Surface::alloc_size(&*surface),
        ));
        let token: rt::RangeToken = Box::new(Arc::clone(&space));
        let Some(buffer) = self.buffer_over_surface(sel, &surface, token)? else {
            return Ok(None);
        };
        let format = fields.raw_pixel_format();
        let Some(raw) = create(sel, Policy::Dies, Nil::Never, || {
            (buffer.new_texture_with_iosurface_plane(&desc, plane))
                .map_err(|e| format_refused(sel, "a texture of", format, e))
        })?
        else {
            return Ok(None);
        };
        let over_guest_pages = crate::os::over_guest_pages(&surface);
        let texture = Obj::new(TextureObject {
            raw,
            device: self.clone_ref(),
            space,
            surface: Some(surface),
            plane: Some(plane),
            under: None,
            private_usage: usage,
        });
        // A texture on a surface over the guest's pages meets a texture a heap over them keeps
        // apart through them.
        if over_guest_pages {
            let memory = crate::gpu::writes::texture_memory(&texture).ranges;
            crate::gpu::over_pages::over_surface(sel, memory, texture.life());
        }
        Ok(Some(texture))
    }

    /// Decoded as the compute form is ([`serialized::tile`]), from `data` alone: Metal's data
    /// spans it, and the rest of core's buffer is never read.
    fn deserialize_tile_render_pipeline_descriptor(
        &self,
        hold: Hold<'_>,
        data: &[u8],
        _buffer_length: u64,
        context: FunctionResolver<'_, VulkanPlatform>,
    ) -> RaiseResult<Option<VulkanTileRenderPipelineDescriptor>> {
        let decoded = timing::creation(
            "newTileRenderPipelineDescriptorWithSerializedData:deserializationContext:",
            "",
            || serialized::tile(hold, data, context),
            || format!("{} bytes", data.len()),
            |d| match d {
                Ok(Some(d)) => format!("tile {}", timing::stage(&d.desc.tile)),
                Ok(None) => "nil".to_owned(),
                Err(_) => "raised".to_owned(),
            },
        )?;
        Ok(decoded.map(|d| Obj::new(PipelineDescriptorObject::new(d))))
    }

    /// Keeps `indices` from plain allocation of IDs of the type for the device's life: textures
    /// (type 0) and sampler states (1) in the runtime's spaces, which number them as Metal does,
    /// indirect command buffers (5) in the numbers this platform gives them and, below 2^23, in the
    /// runtime's, which park the thread for good where Metal spins and die where Metal goes on to
    /// hand out IDs twice ([`crate::gpu::icb`]). The runtime takes only a space's first call with
    /// an ID, before the space handed out one, and ignores any other, as Metal does; a call holding
    /// an ID of 2^32 or more it drops whole, leaving the reservation to a later call, as Metal
    /// does. An ID the runtime refuses dies: a texture or sampler state ID of 2^28 or more, which
    /// ended Metal's process, or a texture ID past what the device's texture arrays hold, which
    /// Metal takes; so does any other type, which core never reserves.
    fn reserve_resource_indices(&self, resource_type: u64, indices: &[u64]) -> RaiseResult<()> {
        let sel = "reserveResourceIndicesForResourceType:indices:indexCount:";
        let kind = match resource_type {
            0 => rt::ResourceIdKind::Texture,
            1 => rt::ResourceIdKind::SamplerState,
            5 => return crate::gpu::icb::reserve(self, sel, indices),
            other => unrepresentable(sel, &format!("resource type {other}")),
        };
        let ids: Vec<rt::ResourceId> = indices.iter().map(|&id| rt::ResourceId(id)).collect();
        send(sel, Policy::Dies, || {
            self.raw.reserve_resource_ids(kind, &ids)
        })
    }

    /// Metal's description of the format at the device's Metal level, the runtime's
    /// ([`crate::gpu::format_info::description`]).
    fn pixel_format_info(&self, format: u64) -> PixelFormatInfo {
        let sel = "MTLPixelFormatGetInfoForDevice()";
        plain(|| crate::gpu::format_info::description(sel, &self.raw, format))
    }

    /// Metal's layout of the level at the device's Metal level, the runtime's
    /// ([`crate::gpu::format_info::level_layout`]).
    fn texture_level_info(&self, q: &TextureLevelQuery) -> TextureLevelInfo {
        let sel = "MTLGetTextureLevelInfoForDeviceWithOptions()";
        plain(|| crate::gpu::format_info::level_layout(sel, &self.raw, q))
    }

    /// Read on the first call for this device object, then shared: the heap-buffer query's length
    /// is the host's page size, as on the Metal platform.
    fn caps(&self) -> Arc<DeviceCaps> {
        Arc::clone(
            self.caps
                .get_or_init(|| caps::read(&self.raw, self.registry_id, OsVm::new().page_size())),
        )
    }

    /// The runtime's buffer descriptor of the value: zero bytes, a copy of the contents, or,
    /// with `no_copy`, the contents themselves with the deallocator, at the pinned address when
    /// there is one. Nil as `newBufferWithLength:options:` is, for a pinned address the space
    /// cannot honour, and for no-copy contents of length 0, whose deallocator never runs, as
    /// Metal answers (measured: NULL contents of length 0, as core's detached buffer of length 0
    /// is). What the runtime's descriptor has no form for (no-copy without contents, a
    /// deallocator of a buffer that is not no-copy, copied contents of another length) dies.
    fn new_buffer(&self, desc: BufferDesc<'_>) -> RaiseResult<Option<VulkanBuffer>> {
        let sel = "newBufferWithDescriptor:";
        let held = descriptor_options(BUFFER_DESCRIPTOR_OPTIONS, &desc.options);
        let options = crate::gpu::resource_options(sel, held);
        let slot = desc.deallocator.map(|d| Arc::new(Mutex::new(Some(d))));
        if desc.no_copy && desc.length == 0 {
            let made = Ok(None);
            keep_unrun(&made, slot);
            return made;
        }
        let bytes = match (desc.contents, desc.no_copy) {
            (Some(bytes), true) => {
                metal_space::Bytes::NoCopy(u64::try_from(bytes.ptr().addr()).unwrap_or(0))
            }
            (None, true) => metal_space::Bytes::NoCopy(0),
            (_, false) => own_bytes(options),
        };
        let (placement, mut placing) =
            metal_space::placing(desc.length, bytes, desc.pinned_gpu_address);
        // A no-copy buffer over one of core's mappings of the guest's pages is over them.
        let over_guest_pages = match (desc.contents, desc.no_copy) {
            (Some(bytes), true) => crate::remap::shows_guest_pages(bytes.ptr(), bytes.len()),
            _ => false,
        };
        // The copied contents, read once from the caller's memory before the runtime sees them:
        // its descriptor takes only a slice, and guest memory may change while it is read.
        let copied: Vec<u8>;
        // A no-copy buffer's memory, whose deallocator's run a detach may hold
        // ([`backing::Held`]).
        let mut own = None;
        // Whether the runtime imports a no-copy buffer's memory through a writable twin of
        // read-only pages.
        let mut kept = false;
        let contents = match (desc.contents, desc.no_copy, &slot) {
            (None, false, None) => rt::BufferContents::Zero,
            (Some(bytes), false, None) if bytes.len() == desc.length => {
                // A NULL pointer dies whatever the length, as the no-copy creators' does.
                pointer(sel, bytes.ptr());
                copied = crate::gpu::resource::host_bytes(sel, bytes);
                rt::BufferContents::Copy(&copied)
            }
            (Some(bytes), true, slot) => {
                let start = pointer(sel, bytes.ptr());
                // The bytes, or a writable twin of them where the runtime cannot import them as
                // mapped, and what keeps them mapped until the runtime freed its import; the
                // deallocator is given the bytes' own address, at Metal's moment.
                let held = Arc::new(backing::Held::default());
                let run = slot.as_ref().map(|slot| {
                    let held = Arc::clone(&held);
                    runtime_deallocator(slot, move |d: BufferDeallocator, p, len| {
                        let at = p.as_ptr().addr();
                        held.run(Box::new(move || d(HostPtr::from_addr(at), len)));
                    })
                });
                own = Some(backing::Own::new(start, held));
                let (imported, run, keep, twinned) = no_copy_release(start, bytes.len(), run);
                kept = twinned;
                rt::BufferContents::NoCopy {
                    // SAFETY: `HostRange::new`'s contract keeps the bytes mapped until the
                    // deallocator runs, or for the buffer's life with none, and an unmap of
                    // core's through the platform's VM waits for the keep-alive; a twin of them
                    // stays until it goes.
                    memory: unsafe { crate::os::host_memory(imported) },
                    deallocator: run,
                    keep_alive: Some(keep),
                }
            }
            _ => unrepresentable(sel, "that descriptor's contents"),
        };
        let pool = pool_slot(sel, &self.raw, &mut placing)?;
        let descriptor =
            rt::BufferDescriptor::new(desc.length, options, contents).with_placement(placement);
        let descriptor = with_token(in_slot(descriptor, &pool), placing.token());
        let made = create(
            sel,
            Policy::Dies,
            placed_nil(&placing, Nil::BufferLength),
            || self.raw.new_buffer_with_descriptor(descriptor),
        );
        metal_space::let_go(pool);
        keep_unrun(&made, slot);
        made.map(|b| {
            b.map(|raw| {
                let backing = Backing::new(held & READ_ONLY_OPTION != 0, raw.object_id());
                if let Some(own) = own {
                    backing.made_over_own(own);
                }
                backing.imported(kept);
                Obj::new(BufferObject {
                    raw,
                    device: self.clone_ref(),
                    surface: None,
                    backing,
                    heap: None,
                    kept_whole: std::sync::atomic::AtomicU64::new(0),
                    over_guest_pages,
                })
            })
        })
    }

    /// A heap of the value, at the pinned address when there is one. The descriptor's options
    /// are those Metal's descriptor holds after the value's setters ([`descriptor_options`]),
    /// and the heap's those of its modes, an untracked mode where the value names the default,
    /// as Metal's are (measured). Nil for a size of 0, for memory the device cannot allocate and
    /// for a pinned address the space cannot honour, as Metal answers; a `Managed` or
    /// `Memoryless` heap dies, as Metal asserts at it ("Heap Descriptor Validation"). A type the
    /// runtime lacks dies at `setType:`.
    ///
    /// Over address ranges (`setAddressRanges:`), the heap's memory is an alias of the ranges'
    /// pages ([`crate::gpu::backing`]), read-only for options with bit 24. Nil, as Metal answers
    /// (measured), for a size of 0 before the ranges are read, and for ranges Metal makes no heap
    /// over: none, a NULL range, an unmapped one or one with a page mapped without access.
    /// Storage other than `Shared` dies, as the runtime makes no heap of another storage over
    /// host memory.
    fn new_heap(
        &self,
        desc: &HeapDesc<'_, VulkanPlatform>,
    ) -> RaiseResult<HeapMade<VulkanPlatform>> {
        let sel = "newHeapWithDescriptor:";
        let heap_type = u64::try_from(desc.heap_type)
            .ok()
            .and_then(|raw| rt::HeapType::from_raw(raw).ok())
            .unwrap_or_else(|| {
                unrepresentable("setType:", &format!("heap type {}", desc.heap_type))
            });
        let held = descriptor_options(HEAP_DESCRIPTOR_OPTIONS, &desc.options);
        let options = crate::gpu::resource_options(sel, held);
        let read_only = held & READ_ONLY_OPTION != 0;
        // A sparse heap's ranges are its range token's, which the runtime drops once the heap,
        // its textures and the command buffers that hold any of them are gone; the runtime holds
        // none of the space for it. One the space refuses dies: the runtime would make it, and
        // Metal's answer is not measured ([`metal_space::placing_sparse`]).
        let (placement, placing, token) = match heap_type {
            rt::HeapType::Sparse => {
                let (placement, mut placing) =
                    metal_space::placing_sparse(desc.size, desc.pinned_gpu_address);
                if placing.refused() {
                    let at = match desc.pinned_gpu_address {
                        Some(pin) => format!("at the pin {pin:#x}"),
                        None => "with no room for its range".to_owned(),
                    };
                    unrepresentable(
                        sel,
                        &format!(
                            "a sparse heap of {:#x} bytes {at}, which the model of Metal's space \
                             refuses and nothing measured",
                            desc.size
                        ),
                    );
                }
                let token = placing.token();
                (placement, placing, token)
            }
            _ => {
                let (placement, mut placing) = metal_space::placing(
                    desc.size,
                    metal_space::Bytes::Heap,
                    desc.pinned_gpu_address,
                );
                let token = placing.token();
                (placement, placing, token)
            }
        };
        let nil = placed_nil(&placing, Nil::Heap);
        let descriptor =
            rt::HeapDescriptor::new(heap_type, desc.size, options).with_placement(placement);
        let storage = |made: Result<rt::Heap, rt::Error>| match made {
            Err(rt::Error::Invalid(rt::Invalid {
                problem: rt::Problem::StorageMode,
                ..
            })) => metal_aborts(sel, &format!("a heap of resource options {held:#x}")),
            made => made,
        };
        // The alias a heap over address ranges is made over.
        let mut made_over = None;
        let heap = match desc.address_ranges {
            None => create(sel, Policy::Dies, nil, || {
                storage(match token {
                    Some(token) => self.raw.new_heap_with_range_token(&descriptor, token),
                    None => self.raw.new_heap(&descriptor),
                })
            })?,
            Some(_) if desc.size == 0 => None,
            Some(ranges) => {
                made_over = desc
                    .size
                    .checked_next_multiple_of(HEAP_GRANULE)
                    .and_then(|size| backing::gather(sel, &ranges.ranges, size, read_only))
                    .map(Arc::new);
                match made_over
                    .as_ref()
                    .map(|alias| Arc::clone(alias).into_runtime())
                {
                    None => None,
                    Some((first, keep)) => create(sel, Policy::Dies, nil, || {
                        // SAFETY: the alias covers the heap's size, the descriptor's rounded up
                        // to the runtime's granule, and starts on a page, the runtime's import
                        // unit on this host. It is as a replace's alias is otherwise
                        // (`VulkanBuffer::replace_backing`).
                        storage(unsafe {
                            match token {
                                Some(token) => {
                                    self.raw.new_heap_with_bytes_no_copy_and_range_token(
                                        crate::os::host_memory(first),
                                        &descriptor,
                                        None,
                                        Some(keep),
                                        token,
                                    )
                                }
                                None => self.raw.new_heap_with_bytes_no_copy(
                                    crate::os::host_memory(first),
                                    &descriptor,
                                    None,
                                    Some(keep),
                                ),
                            }
                        })
                    })?,
                }
            }
        };
        Ok(HeapMade {
            heap: heap.map(|raw| {
                // Only a heap over address ranges detaches ([`crate::gpu::backing`]).
                let backing = match &desc.address_ranges {
                    Some(ranges) => {
                        let backing = Backing::heap(read_only, raw.object_id());
                        backing.made_over(&ranges.ranges, made_over);
                        backing
                    }
                    None => Backing::fixed(),
                };
                Obj::new(HeapObject {
                    raw,
                    device: self.clone_ref(),
                    descriptor_size: desc.size,
                    backing,
                    placed: Mutex::default(),
                    space: placing.made(),
                })
            }),
            descriptor_resource_options: held,
        })
    }

    /// One object for equal requests while it lives, placed at the value's resource index
    /// ([`super::states`]).
    fn new_sampler_state(&self, desc: &SamplerDesc) -> RaiseResult<Option<VulkanSamplerState>> {
        super::states::new_sampler_state(self, desc)
    }

    /// A state of any descriptor ([`super::states`]).
    fn new_depth_stencil_state(
        &self,
        desc: &DepthStencilDesc,
    ) -> RaiseResult<Option<VulkanDepthStencilState>> {
        super::states::new_depth_stencil_state(self, desc)
    }

    fn new_indirect_command_buffer(
        &self,
        desc: &IcbDesc,
        max_count: u64,
        options: u64,
    ) -> RaiseResult<Option<VulkanIcb>> {
        crate::gpu::icb::new_indirect_command_buffer(self, desc, max_count, options)
    }

    /// The display's present pipeline ([`super::states`]).
    fn new_render_pipeline_state(
        &self,
        desc: &RenderPipelineDesc<'_, VulkanPlatform>,
    ) -> Result<Option<VulkanRenderPipeline>, NsErrorText> {
        super::states::new_render_pipeline_state(self, desc)
    }
}

/// Whether `pf` is a two-plane XR10 format: `RGB10A8_2P_XR10` (550) or its sRGB twin (551),
/// which Metal makes over a surface's two planes, the color words in plane 0 and the alpha bytes
/// in plane 1.
pub(crate) fn two_plane_xr10(pf: metal2vulkan::metal::PixelFormat) -> bool {
    use metal2vulkan::metal::PixelFormat;
    matches!(
        pf,
        PixelFormat::RGB10A82PXR10 | PixelFormat::RGB10A82PXR10SRGB
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::LazyLock;

    use reims_vgpu_core::platform::Handle;
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;

    /// The system default device, one handle to the runtime device the platform keeps for the
    /// process ([`DEVICES`]).
    static DEVICE: LazyLock<VulkanDevice> = LazyLock::new(|| {
        VulkanPlatform::system_default_device().expect("a Vulkan device (MoltenVK)")
    });

    /// The system default device (MoltenVK on the suite's host).
    pub(crate) fn dev() -> VulkanDevice {
        DEVICE.clone_ref()
    }

    /// A sparse heap takes its range where the model of Metal's space places it: unpinned or at a
    /// pin it honours it is made; at a pin it refuses (one off a 32-KiB granule) it dies naming
    /// the pin, as the runtime would make it and Metal's answer there is not measured
    #[test]
    fn sparse_heaps_die_at_pins_the_space_refuses() {
        use reims_vgpu_core::platform::values::{HeapDesc, OptionSet};
        let d = dev();
        let sparse = |pin: Option<u64>| HeapDesc {
            size: 0x40000,
            heap_type: 2,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: pin,
        };
        let made = d.new_heap(&sparse(None)).unwrap();
        assert!(made.heap.is_some(), "unpinned");
        let far = metal_space::END / 2;
        let made = d.new_heap(&sparse(Some(far))).unwrap();
        assert!(
            made.heap.is_some(),
            "pinned far from every range, on a granule"
        );
        // A granule's second page, far from the first heap's range.
        let refused = far + 0x10_0000 + metal_space::PAGE;
        let site = catch_death(|| {
            let _ = d.new_heap(&sparse(Some(refused)));
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "newHeapWithDescriptor:")
        );
        let line = crate::answer::LAST_LINE.lock().clone().unwrap();
        assert!(line.contains(&format!("{refused:#x}")), "{line}");
    }

    /// A no-copy buffer over a mapping of core's, released while a command buffer that reads it
    /// runs: its deallocator runs as the command buffer completes, Metal's moment, and core's
    /// unmap of the mapping inside it, through the platform's VM, answers success and leaves the
    /// pages mapped, as the device still imports them; the runtime frees the import after the
    /// deallocator and then drops the keep-alive, whose hold unmaps them. Nothing waits on another
    /// thread: the command buffer completes, the deallocator returns, and the unmap lands. The
    /// command buffer reads the buffer's bytes, and no deallocator is delayed. The release lands
    /// while the kernel runs, its end mark unset, or the test fails: a kernel waiting for a word
    /// the CPU writes after the release cannot hold the command buffer instead, as Metal's GPU saw
    /// no CPU write to a word it polled (`tests/new_images.rs`)
    #[test]
    fn an_unmap_in_a_deallocator_waits_for_the_import() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::mpsc;
        use std::time::Duration;

        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
        use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
        use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder};
        use reims_vgpu_core::platform::resource::{Buffer, Library};
        use reims_vgpu_core::platform::types::{CommandBufferStatus, MtlSize, NsRange};

        /// The kernel that keeps the GPU busy (`src/gpu/spin.metal`): about 60 us a round on the
        /// M3 Max, so about 60 ms, long enough to release the buffer while it runs and short of
        /// the half second Metal fails under other GPU work (`tests/new_images.rs`'s kernel).
        const SPIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spin.metallib"));
        const ROUNDS: u32 = 1_000;
        const LEN: u64 = 0x8000;
        const BYTE: u8 = 0xa5;
        let wait = Duration::from_secs(30);
        // Whether `done` came true within `wait`, polled each millisecond.
        let within = |done: &dyn Fn() -> bool| {
            (0..wait.as_millis()).any(|_| {
                done() || {
                    std::thread::sleep(Duration::from_millis(1));
                    false
                }
            }) || done()
        };
        let d = dev();
        let vm = crate::remap::VulkanVm::new();
        let va = vm.allocate_anywhere(LEN).unwrap();
        let len = usize::try_from(LEN).unwrap();
        // SAFETY: pages just mapped, readable and writable.
        unsafe { std::ptr::write_bytes(va.as_ptr::<u8>(), BYTE, len) };
        // What the deallocator saw after core's unmap: its answer, whether the first page was
        // still mapped, and whether the unmap waited.
        let (ran, seen) = mpsc::channel();
        let deallocator: NoCopyDeallocator = Box::new(move || {
            // Core's mapping's release (`Teardown::Vm`).
            let freed = crate::remap::VulkanVm::new().deallocate(va, LEN);
            let waits = crate::remap::unmap_waits(va, LEN);
            let _ = ran.send((freed.is_success(), crate::os::mapped(va), waits));
        });
        // SAFETY: the pages stay mapped until the deallocator runs, which unmaps them.
        let bytes = unsafe { HostRange::new(va, LEN) };
        let buffer = d
            .new_buffer_over_host_memory(OnFault::Die, bytes, 0, Some(deallocator))
            .unwrap()
            .unwrap();
        let library = d.new_library(SPIN).unwrap().unwrap().unwrap();
        let function = library.new_function(OnFault::Die, "spin").unwrap().unwrap();
        let state = d
            .new_compute_pipeline_state_for_function(Some(&function))
            .unwrap()
            .unwrap();
        let words = d
            .new_buffer_of_length(OnFault::Die, 16, 0)
            .unwrap()
            .unwrap();
        let word = |i: usize| {
            // SAFETY: one of the buffer's four words, which the kernel reads and writes.
            unsafe { AtomicU32::from_ptr(words.contents().as_ptr::<u32>().add(i)) }
        };
        word(0).store(ROUNDS, Ordering::SeqCst);
        let out = d
            .new_buffer_of_length(OnFault::Die, LEN, 0)
            .unwrap()
            .unwrap();
        let cb = (d.new_command_queue().unwrap())
            .command_buffer(OnFault::Die)
            .unwrap()
            .unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&state),
            },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&words)],
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let b = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&buffer),
                source_offset: 0,
                destination_buffer: Some(&out),
                destination_offset: 0,
                size: LEN,
            },
            BlitCmd::EndEncoding,
        ] {
            b.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_scheduled().unwrap();
        let running = word(1).load(Ordering::SeqCst) == 0;
        drop(buffer);
        assert!(running, "released while the kernel ran");
        // The command buffer holds the buffer: no deallocator while the kernel runs.
        let early = seen.try_recv().ok();
        if early.is_some() {
            assert_ne!(
                word(1).load(Ordering::SeqCst),
                0,
                "ran before the kernel ended"
            );
        }
        let ended = [CommandBufferStatus::COMPLETED, CommandBufferStatus::ERROR];
        assert!(
            within(&|| ended.contains(&cb.status())),
            "the command buffer did not complete"
        );
        assert_eq!(
            cb.status(),
            CommandBufferStatus::COMPLETED,
            "{:?}",
            cb.error()
        );
        let (freed, mapped, waits) = early
            .or_else(|| seen.recv_timeout(wait).ok())
            .expect("the deallocator ran");
        assert!(freed, "the unmap answered success");
        assert!(
            mapped && waits,
            "the device still imported the pages: the unmap waited"
        );
        assert!(
            within(&|| !crate::remap::unmap_waits(va, LEN)),
            "the keep-alive was not dropped"
        );
        // SAFETY: a read of the first page where it is mapped; another test's thread may have
        // mapped it again since, without these bytes.
        let gone = !crate::os::mapped(va) || unsafe { va.as_ptr::<u8>().read_volatile() } != BYTE;
        assert!(gone, "unmapped once the import was freed");
        // SAFETY: a Shared buffer the GPU's writes completed.
        let read = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), len) };
        assert!(
            read.iter().all(|&b| b == BYTE),
            "the command buffer read the bytes"
        );
        assert_eq!(d.raw.work_counts().host_deallocators_delayed, 0);
    }

    /// Every handle to one runtime device is one object
    #[test]
    fn a_device_is_one_object() {
        let d = dev();
        let again = adopt("MTLCopyAllDevices()", d.raw.clone());
        assert_eq!(again.id(), d.id());
        assert!(!d.raw.name().is_empty());
    }

    /// The debug line of a texture over a surface names the descriptor's format, raw and by
    /// name, the plane, the size, the usage (the runtime's flags and the raw bits beside them)
    /// and the storage mode, then the surface's OSType, its plane count and each plane's size
    /// and rows, or the surface's own for a surface of no planes
    #[test]
    fn a_texture_over_a_surface_has_a_debug_line_of_its_shape() {
        use reims_vgpu_core::platform::values::TextureDesc;
        let fields = |pixel_format, (width, height), usage, resource_options| {
            TextureFields::of(&TextureDesc {
                texture_type: 2,
                pixel_format,
                width,
                height,
                usage,
                resource_options: Some(resource_options),
                ..TextureDesc::default()
            })
        };
        let x420 = crate::gpu::ycbcr::measure::Shape::of(b"x420").surface((40, 24));
        let line = OnSurface {
            fields: &fields(505, (40, 24), 0x9, 0x20),
            surface: &x420,
            plane: 1,
        };
        assert_eq!(
            line.to_string(),
            "newTextureWithDescriptor:iosurface:plane: format 505 MTLPixelFormatYCBCR10_420_2P, \
             plane 1, 40x24, usage 0x9, storage 2; surface 'x420', 2 planes: plane 0 40x24, 128 \
             bytes per row; plane 1 20x12, 128 bytes per row"
        );
        let bgra = crate::os::tests::bgra_surface(16, 8);
        let line = OnSurface {
            fields: &fields(80, (16, 8), 0x1, 0),
            surface: &bgra,
            plane: 0,
        };
        assert_eq!(
            line.to_string(),
            "newTextureWithDescriptor:iosurface:plane: format 80 MTLPixelFormatBGRA8Unorm, plane \
             0, 16x8, usage 0x1, storage 0; surface 'BGRA', 0 planes: 16x8, 64 bytes per row"
        );
    }

    /// A multisample texture of the padded 10-bit formats (576 to 579), which the runtime stops
    /// as Metal's validation does (`Invalid`, `TextureProblem::TypeForFormat`), dies at its
    /// creator, where Metal aborts in the descriptor's validation: measured on this host (macOS
    /// 26.7), plain and under `MTL_DEBUG_LAYER`, 16 by 16 at 4 samples, `Private`, render target
    /// usage, a 2D multisample texture and an array of 2 ("-[MTLTextureDescriptorInternal
    /// validateWithDevice:]:1416: failed assertion `Texture Descriptor Validation
    /// MTLTextureDescriptor pixelFormat (MTLPixelFormatR10Unorm_X6) does not support multisample
    /// textures.'"). The creator's decision of the runtime's stop, without a device
    #[test]
    fn a_stopped_multisample_padded_texture_dies_at_its_creator() {
        let sel = "newTextureWithDescriptor:";
        for raw in 576..=579 {
            let stop = rt::Error::Invalid(rt::Invalid::new(
                "new_texture",
                rt::Problem::Texture(rt::TextureProblem::TypeForFormat),
            ));
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| {
                create(sel, Policy::Dies, Nil::Memory, || {
                    Err::<(), _>(format_refused(sel, "a texture of", raw, stop))
                })
            })
            .unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{raw}");
            assert!(line.contains("TypeForFormat"), "{raw}: {line}");
        }
    }

    /// The same at the device: a 2D multisample texture of 576 to 579, and an array of 2, at 4
    /// samples, die at `newTextureWithDescriptor:`, named by the runtime's stop, where Metal
    /// aborts ([`a_stopped_multisample_padded_texture_dies_at_its_creator`])
    #[test]
    fn multisample_padded_textures_die_where_metal_aborts() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::values::TextureDesc;
        let p = crate::platform::tests::platform();
        let sel = "newTextureWithDescriptor:";
        for raw in 576..=579 {
            for (texture_type, array_length) in [(4, 1), (8, 2)] {
                let desc = p.texture_descriptor(&TextureDesc {
                    texture_type,
                    pixel_format: raw,
                    width: 16,
                    height: 16,
                    array_length,
                    sample_count: 4,
                    usage: 4,
                    resource_options: Some(2 << 4),
                    ..TextureDesc::default()
                });
                *crate::answer::LAST_LINE.lock() = None;
                let site =
                    catch_death(|| drop(dev().new_texture(OnFault::Return, &desc))).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                let case = format!("{raw} type {texture_type}");
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{case}");
                assert!(line.contains("TypeForFormat"), "{case}: {line}");
            }
        }
    }

    /// Every compute pipeline creation takes a new nonzero identifier, even of one function twice,
    /// and every reference to a state answers its own
    #[test]
    fn each_compute_pipeline_creation_takes_a_new_identifier() {
        use reims_vgpu_core::platform::resource::ComputePipeline as _;
        use reims_vgpu_core::platform::resource::Library as _;
        let d = dev();
        let library = d
            .new_library(include_bytes!(
                "../../../../tests/suite/fixtures/shaders/kernel.metallib"
            ))
            .unwrap()
            .unwrap()
            .unwrap();
        let name = &library.function_names()[0];
        let f = library
            .new_function(OnFault::Return, name)
            .unwrap()
            .unwrap();
        let a = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let b = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        assert_ne!(a.unique_identifier(), 0);
        assert_ne!(a.unique_identifier(), b.unique_identifier());
        assert_eq!(a.clone_ref().unique_identifier(), a.unique_identifier());
    }

    /// A serialized compute descriptor makes the state its function makes, with the descriptor's
    /// limits; a forced resource index places an ICB state there; each state takes a new
    /// identifier; a nil descriptor is Metal's error
    #[test]
    fn compute_pipelines_from_serialized_descriptors() {
        use crate::gpu::serialized::tests::{KERNEL, Resolve, fixture_function, hex};
        use reims_vgpu_core::platform::descriptor::ComputePipelineDescriptor as _;
        use reims_vgpu_core::platform::resource::ComputePipeline as _;
        let d = dev();
        let f = fixture_function(KERNEL);
        let mut resolve = Resolve::answering(Some(f.clone_ref()));
        let mut descriptor = |blob: &str| {
            d.deserialize_compute_pipeline_descriptor(Hold::NONE, &hex(blob), &mut |h, r| {
                resolve.call(h, r)
            })
            .unwrap()
            .unwrap()
        };
        let plain_blob = descriptor("01 00 04 2a000000");
        let limited = descriptor("02 08 04 40000000 00 04 2a000000");
        let icb = descriptor("02 07 04 01000000 00 04 2a000000");
        assert_eq!(resolve.refs, vec![0x2a; 3]);
        let state = |desc| {
            d.new_compute_pipeline_state(Some(desc), 0)
                .unwrap()
                .unwrap()
                .unwrap()
        };
        let of_function = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let a = state(&plain_blob);
        assert_eq!(
            (
                a.thread_execution_width(),
                a.max_total_threads_per_threadgroup()
            ),
            (
                of_function.thread_execution_width(),
                of_function.max_total_threads_per_threadgroup()
            )
        );
        assert_eq!(state(&limited).max_total_threads_per_threadgroup(), 64);
        icb.set_resource_index(12, true);
        let placed = state(&icb);
        assert!(placed.support_indirect_command_buffers());
        assert_eq!(placed.gpu_resource_id(), 12);
        let b = state(&plain_blob);
        assert_ne!(a.unique_identifier(), b.unique_identifier());
        assert_ne!(a.unique_identifier(), 0);
        let e = d.new_compute_pipeline_state(None, 0).unwrap().unwrap_err();
        assert!(e.description.contains("pipeline descriptor is nil"), "{e}");
        assert!(
            d.deserialize_compute_pipeline_descriptor(Hold::NONE, &[], &mut |_, _| Ok(None))
                .unwrap()
                .is_none()
        );
    }

    /// A blob's label is its state's, a diagnostic that changes nothing else: the state takes
    /// its identifier and its resource ID as an unlabelled one does, and a compile's refusal is
    /// the same error
    #[test]
    fn a_blobs_label_is_its_states_and_changes_nothing_else() {
        use crate::gpu::serialized::tests::{KERNEL, Resolve, fixture_function, hex};
        use reims_vgpu_core::platform::descriptor::ComputePipelineDescriptor as _;
        use reims_vgpu_core::platform::resource::ComputePipeline as _;
        let d = dev();
        let descriptor = |f: &VulkanFunction, blob: &str| {
            let mut resolve = Resolve::answering(Some(f.clone_ref()));
            d.deserialize_compute_pipeline_descriptor(Hold::NONE, &hex(blob), &mut |h, r| {
                resolve.call(h, r)
            })
            .unwrap()
            .unwrap()
        };
        let f = fixture_function(KERNEL);
        let state = |blob| {
            d.new_compute_pipeline_state(Some(&descriptor(&f, blob)), 0)
                .unwrap()
                .unwrap()
                .unwrap()
        };
        let (plain, labelled) = (
            "01 00 04 2a000000",
            "02 02 04 0d000000 00 04 2a000000 616200",
        );
        let [a, b, c] = [plain, labelled, plain].map(state);
        assert_eq!(
            (a.raw.label(), b.raw.label(), c.raw.label()),
            (None, Some("ab"), None)
        );
        assert_eq!(
            b.unique_identifier() - a.unique_identifier(),
            c.unique_identifier() - b.unique_identifier()
        );
        let icb = descriptor(&f, "03 07 04 01000000 02 04 13000000 00 04 2a000000 616200");
        icb.set_resource_index(12, true);
        let placed = d
            .new_compute_pipeline_state(Some(&icb), 0)
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            (placed.gpu_resource_id(), placed.raw.label()),
            (12, Some("ab"))
        );
        let vert = fixture_function(include_bytes!(
            "../../../../tests/suite/fixtures/shaders/vert.metallib"
        ));
        let refused = |blob| {
            d.new_compute_pipeline_state(Some(&descriptor(&vert, blob)), 0)
                .unwrap()
                .unwrap_err()
        };
        assert_eq!(
            refused("01 00 04 05000000"),
            refused("02 02 04 0d000000 00 04 05000000 616200")
        );
    }

    /// The suite's stage-input blob makes its kernel's state
    #[test]
    fn a_stage_input_descriptor_makes_its_state() {
        use crate::gpu::serialized::tests::{Resolve, STAGEIN, fixture_function, stage_in_blob};
        let d = dev();
        let mut resolve = Resolve::answering(Some(fixture_function(STAGEIN)));
        let desc = d
            .deserialize_compute_pipeline_descriptor(Hold::NONE, &stage_in_blob(3), &mut |h, r| {
                resolve.call(h, r)
            })
            .unwrap()
            .unwrap();
        assert!(desc.desc.lock().stage_input.is_some());
        let made = d.new_compute_pipeline_state(Some(&desc), 0).unwrap();
        assert!(matches!(made, Ok(Some(_))), "{:?}", made.err());
    }

    /// A vertex function as the compute function is Metal's error, as on Metal
    #[test]
    fn a_vertex_function_as_the_compute_function_is_an_error() {
        use crate::gpu::serialized::tests::{Resolve, fixture_function, hex};
        let d = dev();
        let vert = fixture_function(include_bytes!(
            "../../../../tests/suite/fixtures/shaders/vert.metallib"
        ));
        let mut resolve = Resolve::answering(Some(vert));
        let desc = d
            .deserialize_compute_pipeline_descriptor(
                Hold::NONE,
                &hex("01 00 04 05000000"),
                &mut |h, r| resolve.call(h, r),
            )
            .unwrap()
            .unwrap();
        assert!(
            d.new_compute_pipeline_state(Some(&desc), 0)
                .unwrap()
                .is_err()
        );
    }

    /// A fixture function, `fixtures/shaders/<name>.metallib`'s one.
    fn shader(name: &str) -> VulkanFunction {
        let path = format!(
            "{}/../../tests/suite/fixtures/shaders/{name}.metallib",
            env!("CARGO_MANIFEST_DIR")
        );
        crate::gpu::serialized::tests::fixture_function(&std::fs::read(path).unwrap())
    }

    /// A resolver of the suite's sentinel refs: vertex 0x21, fragment 0x22, tile 0x31.
    fn by_ref(
        functions: &[(u64, &VulkanFunction)],
    ) -> impl FnMut(Hold<'_>, u64) -> RaiseResult<Option<VulkanFunction>> + use<> {
        let functions: Vec<_> = functions.iter().map(|(r, f)| (*r, f.clone_ref())).collect();
        move |_, r| {
            Ok(functions
                .iter()
                .find(|(at, _)| *at == r)
                .map(|(_, f)| f.clone_ref()))
        }
    }

    /// The suite's render blobs make their states, placed at the resource index core sets, each
    /// with a new identifier and Metal's imageblock length; a kernel as the vertex function is
    /// Metal's error, and an empty blob is nil, whose state is Metal's error for a nil descriptor
    #[test]
    fn render_pipelines_from_serialized_descriptors() {
        use crate::gpu::serialized::tests::pipeline_blob;
        use reims_vgpu_core::platform::descriptor::RenderPipelineDescriptor as _;
        use reims_vgpu_core::platform::resource::RenderPipeline as _;
        let d = dev();
        let (vert, frag, kernel) = (shader("vert"), shader("frag"), shader("kernel"));
        let mut resolve = by_ref(&[(0x21, &vert), (0x22, &frag), (0x23, &kernel)]);
        let mut descriptor = |blob: &[u8]| {
            d.deserialize_render_pipeline_descriptor(Hold::NONE, blob, &mut resolve)
                .unwrap()
        };
        let state = |desc: &VulkanRenderPipelineDescriptor| {
            d.new_render_pipeline_state_from_descriptor(Some(desc), 0)
                .unwrap()
        };
        for name in ["basic", "blend", "msaa4", "two", "nocolor", "nocolor4"] {
            let desc = descriptor(&pipeline_blob(name, 0x21, 0x22, 0)).unwrap();
            let made = state(&desc);
            assert!(matches!(made, Ok(Some(_))), "{name}: {:?}", made.err());
        }
        let basic = descriptor(&pipeline_blob("basic", 0x21, 0x22, 0)).unwrap();
        let a = state(&basic).unwrap().unwrap();
        let b = state(&basic).unwrap().unwrap();
        assert!(a.unique_identifier() != 0 && a.unique_identifier() != b.unique_identifier());
        // Metal's answer for this pipeline (`info_replies`' 458), with no raise.
        let eight = reims_vgpu_core::platform::types::MtlSize::new(8, 8, 1);
        assert_eq!(a.imageblock_memory_length(eight), Ok(256));
        let mut icb = pipeline_blob("basic", 0x21, 0x22, 0);
        icb[0] += 1;
        icb.splice(1..1, [0x19, 4, 1, 0, 0, 0]);
        // The colour attachments' offset moves with the entry added before it
        icb[9] += 6;
        let icb = descriptor(&icb).unwrap();
        icb.set_resource_index(11, true);
        assert_eq!(state(&icb).unwrap().unwrap().gpu_resource_id(), 11);
        let kernel_as_vertex = descriptor(&pipeline_blob("basic", 0x23, 0x22, 0)).unwrap();
        assert!(state(&kernel_as_vertex).is_err());
        assert!(descriptor(&[]).is_none());
        let e = d
            .new_render_pipeline_state_from_descriptor(None, 0)
            .unwrap()
            .unwrap_err();
        assert_eq!(e, nil_pipeline_descriptor());
    }

    /// The suite's tile blob makes a tile function's state, and an empty blob is nil
    #[test]
    fn tile_pipelines_from_serialized_descriptors() {
        use crate::gpu::serialized::tests::pipeline_blob;
        let d = dev();
        // A tile function that never touches its imageblock
        let tile = shader("mutdec_tgwrite_blue");
        let mut resolve = by_ref(&[(0x31, &tile)]);
        let desc = d
            .deserialize_tile_render_pipeline_descriptor(
                Hold::NONE,
                &pipeline_blob("tile", 0, 0, 0x31),
                0x100,
                &mut resolve,
            )
            .unwrap()
            .unwrap();
        assert!(desc.desc.lock().threadgroup_size_matches_tile_size);
        let made = d.new_tile_render_pipeline_state(Some(&desc), 0).unwrap();
        assert!(matches!(made, Ok(Some(_))), "{:?}", made.err());
        assert!(
            d.deserialize_tile_render_pipeline_descriptor(Hold::NONE, &[], 0, &mut resolve)
                .unwrap()
                .is_none()
        );
    }

    /// A tile function whose half4 imageblock lies over RGBA8Unorm makes its state, as Metal makes
    /// it
    #[test]
    fn a_tile_pipeline_of_a_half_imageblock_over_rgba8unorm() {
        use crate::gpu::serialized::tests::pipeline_blob;
        let d = dev();
        assert!(d.caps().flags.supports_tile_shaders);
        let tile = shader("tile");
        let mut resolve = by_ref(&[(0x31, &tile)]);
        let desc = d
            .deserialize_tile_render_pipeline_descriptor(
                Hold::NONE,
                &pipeline_blob("tile", 0, 0, 0x31),
                0x100,
                &mut resolve,
            )
            .unwrap()
            .unwrap();
        let made = d.new_tile_render_pipeline_state(Some(&desc), 0).unwrap();
        assert!(matches!(made, Ok(Some(_))), "{:?}", made.err());
    }

    /// The creators of queues, events and fences each make a new object of the device; a shared
    /// event's handle opens the same event, and a nil handle dies
    #[test]
    fn creators_make_objects_of_the_device() {
        use reims_vgpu_core::platform::event::SharedEvent as _;
        let d = dev();
        d.allow_foreign_libraries();
        let (a, b) = (
            d.new_command_queue().unwrap(),
            d.new_command_queue().unwrap(),
        );
        assert_ne!(a.id(), b.id());
        assert_eq!(a.device.id(), d.id());
        let e = d.new_event(OnFault::Return).unwrap().unwrap();
        assert_ne!(e.id(), d.new_event(OnFault::Die).unwrap().unwrap().id());
        assert!(d.new_fence().unwrap().is_some());
        let shared = d.new_shared_event().unwrap();
        shared.set_signaled_value(7).unwrap();
        let handle = shared.new_shared_event_handle().unwrap();
        let opened = d.open_shared_event(Some(&handle)).unwrap();
        assert_eq!(opened.signaled_value(), 7);
        let site = catch_death(|| d.open_shared_event(None)).unwrap_err();
        assert_eq!(site.file, "newSharedEventWithHandle:");
    }

    #[cfg(target_os = "macos")]
    pub(crate) mod metal_tests;
}
