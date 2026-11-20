//! Task 0's IOSurface-backed planes ([`BackingResource`]): a display's frame source, made from a
//! backing record and paged plane by plane between the guest pages and one texture per plane
//! over a new IOSurface. It answers no getter: getting its texture or resource is unrecognized.

use std::sync::Arc;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::guest::GuestPtr;
use crate::memory::DeathKind;
use crate::memory::map::GuestPages;
use crate::nil_send;
use crate::num::{ToU64, ToUsize};
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::descriptor::TextureDescriptor;
use crate::platform::device::GpuDevice;
use crate::platform::resource::Buffer as _;
use crate::platform::os::surface::{BackingPlaneSpec, BackingSurfaceSpec};
use crate::platform::types::{MtlOrigin, MtlSize};
use crate::platform::values::TextureDesc;
use crate::platform::{Handle, Platform};
use crate::raise::{Kept, keep_forever};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::resource::{Resource, ResourceClass, ResourceKind, TaskRef};
use crate::resources::texture::{RaiseKind, raise};
use crate::rt::{Held, Hold};
use crate::sync::Mutex;

/// `MTLPixelFormatInvalid`.
const PIXEL_FORMAT_INVALID: u64 = 0;
/// `MTLPixelFormatR8Unorm`.
const PIXEL_FORMAT_R8_UNORM: u64 = 10;
/// `MTLPixelFormatRG8Unorm`.
const PIXEL_FORMAT_RG8_UNORM: u64 = 30;
/// `MTLPixelFormatRGBA8Unorm`.
const PIXEL_FORMAT_RGBA8_UNORM: u64 = 70;
/// `MTLPixelFormatRGBA16Uint`.
const PIXEL_FORMAT_RGBA16_UINT: u64 = 113;
/// `MTLPixelFormatRGBA32Uint`.
const PIXEL_FORMAT_RGBA32_UINT: u64 = 123;
/// `MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget`.
const PLANE_TEXTURE_USAGE: u64 = 7;
/// The row length's 24 bits.
const BYTES_PER_ROW_MASK: u32 = 0x00ff_ffff;

/// A backing record's fixed part.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct BackingDescriptor {
    /// The guest pages' length.
    pub length: u64,
    /// The guest pages' first virtual page.
    pub page: u32,
    /// The IOSurface's pixel format.
    pub pixel_format: u32,
    /// The plane count.
    pub plane_count: u8,
    /// Padding.
    pub unknown11: [u8; 7],
}

const _: () = assert!(size_of::<BackingDescriptor>() == 24);

/// One plane of a backing record. `bits` holds the row length (24 bits) and the bytes per
/// element (8 bits).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct BackingPlane {
    /// The plane's offset in the guest pages.
    pub offset: u32,
    /// Width in elements.
    pub width: u32,
    /// Height in rows.
    pub height: u32,
    /// The row length and the bytes per element.
    pub bits: u32,
}

const _: () = assert!(size_of::<BackingPlane>() == 16);

impl BackingPlane {
    /// The row length (bits 0-23).
    pub fn bytes_per_row(&self) -> u32 {
        self.bits & BYTES_PER_ROW_MASK
    }

    /// The bytes per element (bits 24-31).
    pub fn bytes_per_element(&self) -> u32 {
        self.bits.wrapping_shr(24)
    }

    /// A copy's bytes per image: the row length times the height, wrapping in 32 bits, then
    /// widened.
    fn bytes_per_image(&self) -> u64 {
        // 32-bit wrapping product.
        u64::from(self.bytes_per_row().wrapping_mul(self.height))
    }

    /// The copy's size: width by height by 1.
    fn size(&self) -> MtlSize {
        MtlSize {
            width: u64::from(self.width),
            height: u64::from(self.height),
            depth: 1,
        }
    }
}

/// Task 0's IOSurface-backed planes.
pub struct BackingResource<P: Platform> {
    /// The task, unretained.
    task: TaskRef<P>,
    /// The IOSurface (never checked for nil).
    io_surface: Option<P::Surface>,
    /// The validated planes.
    planes: Vec<BackingPlane>,
    /// The guest pages: dropped by a discard, bound again by the next paging. Never made where
    /// the paging is [`Self::staged`].
    transfer_buffer: Mutex<Option<P::Buffer>>,
    /// Whether a paging copies the guest pages through a fresh buffer of its own instead of a
    /// buffer over them: where the platform's alias of scattered guest pages costs more than
    /// copying them (`HostVm::aliases_guest_in_place`; a window's surface is a thousand pages).
    staged: bool,
    /// One texture per plane over the IOSurface.
    textures: Vec<P::Texture>,
    /// The guest pages' length.
    length: u64,
    /// The guest pages' first virtual page.
    page: u32,
}

impl<P: Platform> BackingResource<P> {
    /// A backing of `planes` (the record's planes): the IOSurface, the guest pages and a texture
    /// per plane, made in a landing frame ([`Held::raising`]): what binding the guest pages holds
    /// (the page-table read's device handle) is released at its end, or goes on with its raise.
    /// Never `None`: a refusal raises.
    ///
    /// `platform` makes the IOSurface and the texture descriptor.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        platform: &P,
        task: TaskRef<P>,
        descriptor: &BackingDescriptor,
        planes: &[BackingPlane],
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        Held::raising(|held| {
            let hold = Hold::of(held);
            let plane_count = planes.len();
            if plane_count == 0 || plane_count > 3 {
                return raise(RaiseKind::BackingPlaneCount);
            }
            let mut own = vec![BackingPlane::default(); plane_count];
            let page = descriptor.page;
            let length = descriptor.length;
            if !validate(planes, &mut own, descriptor.length) {
                return raise(RaiseKind::BackingPlaneInvalid);
            }
            // The fields box as `int`.
            let spec = BackingSurfaceSpec {
                pixel_format: descriptor.pixel_format,
                planes: own
                    .iter()
                    .map(|p| BackingPlaneSpec {
                        width: p.width,
                        height: p.height,
                        bytes_per_element: p.bytes_per_element().cast_signed(),
                        bytes_per_row: p.bytes_per_row().cast_signed(),
                        base: p.offset,
                        offset: p.offset,
                    })
                    .collect(),
                width: own.first().map_or(0, |p| p.width),
                height: own.first().map_or(0, |p| p.height),
            };
            let io_surface = platform.surfaces().create_backing(&spec);
            let staged = platform.vm().aliases_guest_in_place();
            // A torn-down task answers nothing: no buffer.
            let host = task.upgrade();
            let transfer_buffer = match &host {
                Some(_) if staged => None,
                Some(t) => match t.new_buffer_for_virtual_page(hold, page, length) {
                    Ok(b) => Some(b),
                    Err(e) => {
                        // The task's raise releases the half-made backing's IOSurface, which is
                        // ours alone: no Metal object, no table and no lookup reaches it.
                        drop(io_surface);
                        return Err(e);
                    }
                },
                None => {
                    drop(io_surface);
                    return crate::resources::raise(crate::resources::RaiseKind::NoGuestBuffer);
                }
            };
            let mut textures: Vec<P::Texture> = Vec::with_capacity(plane_count);
            // One descriptor for every plane: made from the first plane's value, then each later
            // plane's size, usage and pixel format set on it.
            let mut texture_descriptor: Option<P::TextureDescriptor> = None;
            for (i, plane) in own.iter().enumerate() {
                let (width, height) = (u64::from(plane.width), u64::from(plane.height));
                let td = match &texture_descriptor {
                    Some(td) => {
                        td.set_width(width);
                        td.set_height(height);
                        td.set_usage(PLANE_TEXTURE_USAGE);
                        td.set_pixel_format(pixel_format(plane));
                        td
                    }
                    None => texture_descriptor.insert(platform.texture_descriptor(&TextureDesc {
                        width,
                        height,
                        usage: PLANE_TEXTURE_USAGE,
                        pixel_format: pixel_format(plane),
                        ..TextureDesc::default()
                    })),
                };
                // The task's device, asked for every plane (`host` holds the task).
                let made = match &host {
                    Some(t) => nil_send!(
                        t.mtl_device(),
                        new_texture_on_surface(td, io_surface.as_ref(), i.to_u64())
                    ),
                    None => Ok(None),
                };
                let texture = match made {
                    Ok(t) => t,
                    Err(e) => {
                        // A Metal raise keeps the half-made backing for good: its IOSurface, the
                        // guest pages' buffer (its GPU VA) and the textures made so far (their
                        // resource indices).
                        keep_forever(
                            (io_surface, transfer_buffer, textures),
                            Kept::MetalNumbering,
                        );
                        return Err(e);
                    }
                };
                let Some(texture) = texture else {
                    // The half-made backing is released: its IOSurface and guest pages go now;
                    // the textures made so far (each holding the IOSurface) go on with the raise,
                    // and are released, most recent first, where it ends.
                    drop(io_surface);
                    drop(transfer_buffer);
                    for t in textures {
                        held.keep(t);
                    }
                    return raise(RaiseKind::BackingTextureRefused);
                };
                textures.push(texture);
            }
            drop(texture_descriptor);
            Ok(Some(Resource::new(ResourceKind::Backing(
                BackingResource {
                    task,
                    io_surface,
                    planes: own,
                    transfer_buffer: Mutex::new(transfer_buffer),
                    staged,
                    textures,
                    length,
                    page,
                },
            ))))
        })
    }

    /// The guest pages, bound again if a discard dropped them; retained for the caller's copies.
    /// A torn-down task answers nothing, which raises.
    ///
    /// The lock is a leaf held only to read or store the field: the task and the encoder are
    /// asked outside it.
    fn ensure_physical(&self, hold: Hold<'_>) -> RaiseResult<P::Buffer> {
        if let Some(b) = self.transfer_buffer.lock().as_ref() {
            return Ok(b.clone_ref());
        }
        let made = match self.task.upgrade() {
            Some(t) => Some(t.new_buffer_for_virtual_page(hold, self.page, self.length)?),
            None => None,
        };
        let Some(made) = made else {
            return crate::resources::raise(crate::resources::RaiseKind::NoGuestBuffer);
        };
        let copy = made.clone_ref();
        // A plain store: a buffer a racing caller stored meanwhile is overwritten unreleased,
        // its GPU VA kept taken.
        if let Some(raced) = self.transfer_buffer.lock().replace(made) {
            keep_forever(raced, Kept::MetalNumbering);
        }
        Ok(copy)
    }

    /// The guest pages and a fresh shared buffer of their length, for a [`Self::staged`]
    /// paging; `None` where the paging is not staged, the task answers no pages (a buffer over
    /// them is used then) or Metal makes no buffer. A torn-down task raises, as for the buffer
    /// over the pages.
    fn staging(&self, hold: Hold<'_>) -> RaiseResult<Option<(GuestPages<P>, P::Buffer)>> {
        if !self.staged {
            return Ok(None);
        }
        let Some(task) = self.task.upgrade() else {
            return crate::resources::raise(crate::resources::RaiseKind::NoGuestBuffer);
        };
        let Some(pages) = task.guest_pages_for_virtual_page(hold, self.page, self.length)? else {
            return Ok(None);
        };
        let buffer = nil_send!(
            task.mtl_device(),
            new_buffer_of_length(OnFault::Return, self.length, 0)
        )?;
        Ok(buffer.map(|b| (pages, b)))
    }

    /// The IOSurface.
    pub fn io_surface(&self) -> Option<P::Surface> {
        self.io_surface.as_ref().map(Handle::clone_ref)
    }

    /// The plane textures (the first is what a display presents).
    pub fn textures(&self) -> Vec<P::Texture> {
        self.textures.iter().map(Handle::clone_ref).collect()
    }
}

/// A plane's texture pixel format, by its bytes per element; an unknown size logs and answers
/// the invalid format.
fn pixel_format(plane: &BackingPlane) -> u64 {
    match plane.bytes_per_element() {
        16 => PIXEL_FORMAT_RGBA32_UINT,
        8 => PIXEL_FORMAT_RGBA16_UINT,
        4 => PIXEL_FORMAT_RGBA8_UNORM,
        2 => PIXEL_FORMAT_RG8_UNORM,
        1 => PIXEL_FORMAT_R8_UNORM,
        other => {
            log::warn!("backing plane of {other} bytes per element has no pixel format");
            PIXEL_FORMAT_INVALID
        }
    }
}

/// The plane rows a synchronize's copies wrote into `staging`, copied into the guest `pages`
/// (only the rows' pixels, as the copies into a buffer over the pages wrote only those). Pages
/// that are read-only, or a part with no host address, are logged and left as they are.
fn write_back_rows<P: Platform>(
    pages: &GuestPages<P>,
    staging: &P::Buffer,
    planes: &[BackingPlane],
    length: u64,
) {
    if !pages.writable() {
        log::warn!("a synchronized backing's guest pages are read-only: they keep their bytes");
        return;
    }
    let mut bytes = vec![0u8; length.to_usize()];
    GuestPtr::new(staging.contents()).copy_to(0, &mut bytes, DeathKind::MapReadFault);
    for plane in planes {
        let row = u64::from(plane.width).wrapping_mul(u64::from(plane.bytes_per_element()));
        for y in 0..u64::from(plane.height) {
            let at = u64::from(plane.offset)
                .wrapping_add(y.wrapping_mul(u64::from(plane.bytes_per_row())));
            let Some(end) = at.checked_add(row).filter(|&end| end <= length) else {
                break;
            };
            if !pages.write(at, &bytes[at.to_usize()..end.to_usize()]) {
                log::warn!("a synchronized backing's guest page has no host address");
                return;
            }
        }
    }
}

/// Copies each plane into `dest_planes` and checks it: its rows hold its width, and it ends
/// within `length` (exactly at the end passes). A refusal leaves the planes up to the refused
/// one copied, and has logged one line.
fn validate(src_planes: &[BackingPlane], dest_planes: &mut [BackingPlane], length: u64) -> bool {
    for (dest, src) in dest_planes.iter_mut().zip(src_planes) {
        *dest = *src;
        let plane = *dest;
        // 64 bits: no product overflows.
        if u64::from(plane.width).wrapping_mul(u64::from(plane.bytes_per_element()))
            > u64::from(plane.bytes_per_row())
        {
            log::warn!(
                "backing plane refused: {} pixels of {} bytes do not fit its {} bytes per row",
                plane.width,
                plane.bytes_per_element(),
                plane.bytes_per_row()
            );
            return false;
        }
        // 64 bits (at most 2^32 + 2^56): no overflow.
        let end = u64::from(plane.offset)
            .wrapping_add(u64::from(plane.height).wrapping_mul(u64::from(plane.bytes_per_row())));
        if end > length {
            log::warn!(
                "backing plane refused: it ends at {end:#x}, past the backing's {length:#x}"
            );
            return false;
        }
    }
    true
}

impl<P: Platform> ResourceClass<P> for BackingResource<P> {
    fn initial_guest_valid(&self) -> bool {
        // A backing starts guest-valid.
        true
    }

    /// Guest pages to the plane textures. No devolatilize, no protection check. A nil encoder
    /// makes the copies sends to nil, and host-valid is still set.
    fn prepare_in_encoder(
        &self,
        hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let state = &this.state;
        if state.host_valid() {
            return Ok(());
        }
        if !state.guest_valid() {
            log::warn!("preparing a backing whose guest copy is not valid");
        }
        // Staged: the guest pages are copied into a buffer of the paging's own now, which the
        // command buffer holds until its copies ran.
        let staged = match encoder {
            Some(_) => self.staging(hold)?,
            None => None,
        };
        let transfer_buffer = match staged {
            Some((pages, staging)) => {
                let mut bytes = vec![0u8; self.length.to_usize()];
                if !pages.read(0, &mut bytes) {
                    log::warn!("backing pages without a host address: its prepare copies zeros");
                }
                GuestPtr::new(staging.contents()).copy_from(0, &bytes, DeathKind::MapWriteNull);
                staging
            }
            None => self.ensure_physical(hold)?,
        };
        let buffer = &transfer_buffer;
        for (plane, texture) in self.planes.iter().zip(&self.textures) {
            // The form without options.
            nil_send!(
                encoder,
                encode(
                    OnFault::Return,
                    BlitCmd::CopyBufferToTexture {
                        source_buffer: buffer,
                        source_offset: u64::from(plane.offset),
                        source_bytes_per_row: u64::from(plane.bytes_per_row()),
                        source_bytes_per_image: plane.bytes_per_image(),
                        source_size: plane.size(),
                        destination_texture: texture,
                        destination_slice: 0,
                        destination_level: 0,
                        destination_origin: MtlOrigin::default()
                    }
                )
            )?;
        }
        state.set_host_valid(true);
        Ok(())
    }

    /// The plane textures back to the guest pages. A nil encoder makes the copies sends to nil,
    /// and guest-valid is still set.
    fn synchronize_in_encoder(
        &self,
        hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let state = &this.state;
        // Guest-valid from the start and after a synchronize, so a second 53 writes nothing.
        if state.guest_valid() {
            return Ok(());
        }
        if !state.host_valid() {
            log::warn!("synchronizing a backing whose host copy is not valid");
        }
        // Staged: the copies land in a buffer of the paging's own, whose plane rows are copied
        // into the guest pages once the command buffer completed, before the guest hears of it.
        let staged = match encoder {
            Some(e) => match self.staging(hold)? {
                Some((pages, staging)) => {
                    let rows = staging.clone_ref();
                    let planes = self.planes.clone();
                    let length = self.length;
                    if e.at_completion(Box::new(move || {
                        write_back_rows(&pages, &rows, &planes, length);
                    })) {
                        Some(staging)
                    } else {
                        None
                    }
                }
                None => None,
            },
            None => None,
        };
        let transfer_buffer = match staged {
            Some(staging) => staging,
            None => self.ensure_physical(hold)?,
        };
        let buffer = &transfer_buffer;
        for (plane, texture) in self.planes.iter().zip(&self.textures) {
            nil_send!(
                encoder,
                encode(
                    OnFault::Return,
                    BlitCmd::CopyTextureToBuffer {
                        source_texture: texture,
                        source_slice: 0,
                        source_level: 0,
                        source_origin: MtlOrigin::default(),
                        source_size: plane.size(),
                        destination_buffer: buffer,
                        destination_offset: u64::from(plane.offset),
                        destination_bytes_per_row: u64::from(plane.bytes_per_row()),
                        destination_bytes_per_image: plane.bytes_per_image()
                    }
                )
            )?;
        }
        state.set_guest_valid(true);
        Ok(())
    }

    /// Drops the guest pages only (no purgeable state, no validity change); the next paging
    /// binds them again.
    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        // 63 on a backing only drops the guest pages.
        let old = self.transfer_buffer.lock().take();
        drop(old);
        Ok(())
    }
}

impl<P: Platform> Drop for BackingResource<P> {
    /// In order: the IOSurface, the guest pages, the textures, then the planes.
    fn drop(&mut self) {
        drop(self.io_surface.take());
        drop(self.transfer_buffer.get_mut().take());
        drop(std::mem::take(&mut self.textures));
        drop(std::mem::take(&mut self.planes));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::platform::os::surface::Surface;
    use crate::platform::resource::Texture;
    use crate::resources::host::ResourceHost;
    use crate::resources::resource::test_support::logged;
    use crate::resources::texture::test_support::raised;
    use crate::testkit::calllog::Arg;
    use crate::testkit::resource_host::FakeResourceHost;
    use crate::testkit::{FakePlatform, Fk, Step, kind};

    type P = FakePlatform;

    struct Env {
        p: Arc<P>,
        rec: Arc<FakeResourceHost>,
        host: Arc<dyn ResourceHost<P>>,
    }

    impl Env {
        fn new() -> Env {
            let p = Arc::new(FakePlatform::new());
            let rec = FakeResourceHost::new(p.clone());
            let host: Arc<dyn ResourceHost<P>> = rec.clone();
            Env { p, rec, host }
        }

        fn task(&self) -> TaskRef<P> {
            Arc::downgrade(&self.host)
        }

        fn encoder(&self) -> Fk<kind::BlitEncoder> {
            self.p.world().new_obj(crate::testkit::objects::Model::None)
        }
    }

    fn plane(offset: u32, width: u32, height: u32, bpr: u32, bpe: u32) -> BackingPlane {
        BackingPlane {
            offset,
            width,
            height,
            bits: (bpe << 24) | (bpr & BYTES_PER_ROW_MASK),
        }
    }

    fn desc(length: u64, page: u32, planes: u8) -> BackingDescriptor {
        BackingDescriptor {
            length,
            page,
            pixel_format: 0x4247_5241,
            plane_count: planes,
            ..BackingDescriptor::default()
        }
    }

    /// `BackingResource::new` with the process's platform, inside its scope.
    fn new_in(
        env: &Env,
        task: TaskRef<P>,
        d: &BackingDescriptor,
        planes: &[BackingPlane],
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        crate::resources::resource::with_platform(&env.p, || {
            BackingResource::new(&*env.p, task, d, planes)
        })
    }

    fn backing(r: &Resource<P>) -> &BackingResource<P> {
        match &r.kind {
            ResourceKind::Backing(b) => b,
            _ => panic!("not a backing"),
        }
    }

    /// A two-plane backing.
    fn make(env: &Env) -> Arc<Resource<P>> {
        new_in(
            env,
            env.task(),
            &desc(0x1000, 3, 2),
            &[plane(0, 4, 4, 16, 4), plane(64, 2, 2, 8, 2)],
        )
        .unwrap()
        .unwrap()
    }

    /// The two bit-fields of the last word
    #[test]
    fn plane_bit_fields() {
        let p = plane(0, 1, 1, 0x00ab_cdef, 0x12);
        assert_eq!(p.bytes_per_row(), 0x00ab_cdef);
        assert_eq!(p.bytes_per_element(), 0x12);
        assert_eq!(p.as_bytes()[12..16], [0xef, 0xcd, 0xab, 0x12]);
    }

    /// Every case, and the logged refusal
    #[test]
    fn pixel_format_by_element_size() {
        let f = |bpe| pixel_format(&plane(0, 1, 1, 1, bpe));
        let (formats, lines) = logged(|| [f(16), f(8), f(4), f(2), f(1)]);
        assert_eq!(formats, [123, 113, 70, 30, 10]);
        assert!(lines.is_empty());
        let (formats, lines) = logged(|| [f(3), f(0), f(255)]);
        assert_eq!(formats, [0, 0, 0]);
        assert_eq!(
            lines,
            vec![
                "backing plane of 3 bytes per element has no pixel format",
                "backing plane of 0 bytes per element has no pixel format",
                "backing plane of 255 bytes per element has no pixel format"
            ]
        );
    }

    /// Pass, exact fit, stride, overrun, copy before check
    #[test]
    fn validate_planes() {
        let v = |src: &[BackingPlane], dst: &mut [BackingPlane], len| validate(src, dst, len);
        // Exact fit: offset + height * bytesPerRow == length passes.
        let src = [plane(0, 4, 4, 16, 4), plane(64, 2, 2, 8, 2)];
        let mut dst = [BackingPlane::default(); 2];
        assert!(v(&src, &mut dst, 80));
        assert_eq!(dst, src);
        // One byte short: the second plane overruns; the first two were copied.
        let mut dst = [BackingPlane::default(); 2];
        let (valid, lines) = logged(|| v(&src, &mut dst, 79));
        assert!(!valid);
        assert_eq!(dst, src);
        assert_eq!(
            lines,
            vec!["backing plane refused: it ends at 0x50, past the backing's 0x4f"]
        );
        // Stride: width * bytesPerElement > bytesPerRow; the later plane is not copied.
        let bad = [plane(0, 5, 1, 16, 4), plane(64, 2, 2, 8, 2)];
        let mut dst = [BackingPlane::default(); 2];
        let (valid, lines) = logged(|| v(&bad, &mut dst, 1 << 40));
        assert!(!valid);
        assert_eq!(dst[0], bad[0]);
        assert_eq!(dst[1], BackingPlane::default());
        assert_eq!(
            lines,
            vec!["backing plane refused: 5 pixels of 4 bytes do not fit its 16 bytes per row"]
        );
        // Width * bytesPerElement == bytesPerRow passes; the check is 64-bit.
        let wide = [plane(u32::MAX, u32::MAX, 0xffff_ffff, 0xff_ffff, 0)];
        let mut dst = [BackingPlane::default(); 1];
        assert!(v(&wide, &mut dst, u64::MAX));
        let mut dst = [BackingPlane::default(); 1];
        let end = u64::from(u32::MAX) + u64::from(u32::MAX) * 0xff_ffff;
        assert!(v(&wide, &mut dst, end));
        assert!(!v(&wide, &mut dst, end - 1));
        // No planes: YES.
        assert!(v(&[], &mut [], 0));
    }

    /// The calls in order, the state and the surface
    #[test]
    fn new_makes_surface_buffer_and_textures() {
        let env = Env::new();
        let log = env.p.call_log();
        let mark = log.mark();
        let r = make(&env);
        let b = backing(&r);
        assert!(!r.state.host_valid());
        assert!(r.state.guest_valid());
        assert!(!r.state.is_volatile());
        let want = [
            "initWithProperties:",
            "new_buffer_for_virtual_page",
            "setPixelFormat:",
            "mtl_device",
            "newTextureWithDescriptor:iosurface:plane:",
            "setPixelFormat:",
            "mtl_device",
            "newTextureWithDescriptor:iosurface:plane:",
        ];
        let got: Vec<_> = log
            .since(mark)
            .iter()
            .map(|c| c.sel)
            .filter(|s| want.contains(s))
            .collect();
        assert_eq!(got, want);
        let spec = BackingSurfaceSpec {
            pixel_format: 0x4247_5241,
            planes: vec![
                BackingPlaneSpec {
                    width: 4,
                    height: 4,
                    bytes_per_element: 4,
                    bytes_per_row: 16,
                    base: 0,
                    offset: 0,
                },
                BackingPlaneSpec {
                    width: 2,
                    height: 2,
                    bytes_per_element: 2,
                    bytes_per_row: 8,
                    base: 64,
                    offset: 64,
                },
            ],
            width: 4,
            height: 4,
        };
        assert_eq!(
            log.calls_to("initWithProperties:")[0].args,
            vec![Arg::val(&spec)]
        );
        assert_eq!(
            log.calls_to("new_buffer_for_virtual_page")[0].args,
            vec![Arg::U32(3), Arg::U64(0x1000)]
        );
        let fmts: Vec<_> = log
            .calls_to("setPixelFormat:")
            .iter()
            .map(|c| c.args[0].clone())
            .collect();
        assert_eq!(fmts, vec![Arg::U64(70), Arg::U64(30)]);
        let tex = log.calls_to("newTextureWithDescriptor:iosurface:plane:");
        let s = b.io_surface().unwrap();
        // One descriptor for both planes.
        assert_eq!(tex[0].args[0], tex[1].args[0]);
        assert_eq!(tex[0].args[1], Arg::Obj(Some(s.id())));
        assert_eq!(tex[1].args[2], Arg::U64(1));
        assert_eq!(b.textures().len(), 2);
        let made: Vec<_> = b
            .textures()
            .iter()
            .map(|t| (t.width(), t.height(), t.usage()))
            .collect();
        assert_eq!(
            made,
            [(4, 4, PLANE_TEXTURE_USAGE), (2, 2, PLANE_TEXTURE_USAGE)]
        );
        // The surface: our handle, the one just taken, and one per plane texture.
        assert_eq!(Surface::retain_count(&s), 4);
    }

    /// Plane counts 0 and 4 raise before anything is made
    #[test]
    fn new_refuses_plane_counts() {
        let env = Env::new();
        for n in [0usize, 4] {
            let planes = vec![plane(0, 1, 1, 4, 4); n];
            let e = new_in(&env, env.task(), &desc(64, 0, 0), &planes)
                .err()
                .unwrap();
            assert_eq!(raised(&e), Some(RaiseKind::BackingPlaneCount));
        }
        assert!(env.p.call_log().calls_to("initWithProperties:").is_empty());
        // Three planes pass the count check.
        let planes = [plane(0, 1, 1, 4, 4); 3];
        assert!(new_in(&env, env.task(), &desc(64, 0, 3), &planes).is_ok());
    }

    /// A refused plane raises before the surface
    #[test]
    fn new_refuses_invalid_planes() {
        let env = Env::new();
        let e = new_in(&env, env.task(), &desc(16, 0, 1), &[plane(0, 4, 4, 16, 4)])
            .err()
            .unwrap();
        assert_eq!(raised(&e), Some(RaiseKind::BackingPlaneInvalid));
        assert!(env.p.call_log().calls_to("initWithProperties:").is_empty());
    }

    /// A raise names where it is raised
    #[test]
    fn new_raise_names_where_it_is_raised() {
        let env = Env::new();
        let r = new_in(&env, env.task(), &desc(16, 0, 0), &[]);
        let d = r.map(|_| ()).unwrap_err().site();
        assert_eq!(d.kind, crate::raise::death::DeathKind::UnhandledRaise);
        assert!(d.file.ends_with("backing.rs"), "{d:?}");
    }

    /// The task's raise releases the IOSurface
    #[test]
    fn new_buffer_raise_releases_surface() {
        let env = Env::new();
        env.p.script().push(
            "new_buffer_for_virtual_page",
            Step::raise("NSGenericException", "Failed to get guest pointer"),
        );
        let e = new_in(
            &env,
            env.task(),
            &desc(0x1000, 0, 1),
            &[plane(0, 4, 4, 16, 4)],
        )
        .err()
        .unwrap();
        assert_eq!(
            e.detail().unwrap_or_default(),
            "Failed to get guest pointer"
        );
        // The surface is released, and no texture was made.
        assert!(env.p.world().live("IOSurface").is_empty());
        assert!(
            env.p
                .call_log()
                .calls_to("newTextureWithDescriptor:iosurface:plane:")
                .is_empty()
        );
    }

    /// A torn-down task gives no buffer: the creation raises, the surface released
    #[test]
    fn new_without_task_refuses() {
        let env = Env::new();
        let gone: TaskRef<P> = {
            let other: Arc<dyn ResourceHost<P>> = env.rec.sibling();
            Arc::downgrade(&other)
        };
        let e = new_in(&env, gone, &desc(0x1000, 0, 1), &[plane(0, 4, 4, 16, 4)])
            .err()
            .unwrap();
        assert_eq!(
            crate::resources::raised_kind(&e),
            Some(crate::resources::RaiseKind::NoGuestBuffer)
        );
        assert!(env.p.world().live("IOSurface").is_empty());
    }

    /// A nil plane texture: released, earlier textures going on with the raise to the holder
    /// where it ends
    #[test]
    fn new_nil_texture_refuses() {
        let env = Env::new();
        env.p
            .script()
            .after("newTextureWithDescriptor:iosurface:plane:", 1, Step::Nil);
        let outer = Held::new();
        let e = new_in(
            &env,
            env.task(),
            &desc(0x1000, 0, 2),
            &[plane(0, 4, 4, 16, 4), plane(64, 2, 2, 8, 2)],
        )
        .err()
        .unwrap();
        assert_eq!(raised(&e), Some(RaiseKind::BackingTextureRefused));
        // Plane 0's texture goes on with the raise, lands in the holder of the hold that
        // ends it, and holds the surface.
        Hold::of(&outer).land(e);
        assert_eq!(env.p.world().live("MTLTexture").len(), 1);
        assert_eq!(env.p.world().live("IOSurface").len(), 1);
        assert!(env.p.world().live("MTLBuffer").is_empty());
        drop(outer);
        assert!(env.p.world().live("MTLTexture").is_empty());
        assert!(env.p.world().live("IOSurface").is_empty());
    }

    /// A nil IOSurface is not checked: the texture send gets nil
    #[test]
    fn new_nil_surface_unchecked() {
        let env = Env::new();
        env.p.script().push("initWithProperties:", Step::Nil);
        let e = new_in(
            &env,
            env.task(),
            &desc(0x1000, 0, 1),
            &[plane(0, 4, 4, 16, 4)],
        )
        .err()
        .unwrap();
        let tex = env
            .p
            .call_log()
            .calls_to("newTextureWithDescriptor:iosurface:plane:");
        assert_eq!(tex[0].args[1], Arg::Obj(None));
        assert_eq!(raised(&e), Some(RaiseKind::BackingTextureRefused));
    }

    /// A Metal raise leaves the half-made object leaked
    #[test]
    fn new_metal_raise_leaks() {
        let env = Env::new();
        env.p.script().push(
            "newTextureWithDescriptor:iosurface:plane:",
            Step::raise("NSInvalidArgumentException", "bad"),
        );
        let e = new_in(
            &env,
            env.task(),
            &desc(0x1000, 0, 1),
            &[plane(0, 4, 4, 16, 4)],
        )
        .err()
        .unwrap();
        assert_eq!(e.detail().unwrap_or_default(), "bad");
        assert_eq!(env.p.world().live("IOSurface").len(), 1);
        assert_eq!(env.p.world().live("MTLBuffer").len(), 1);
    }

    /// One copy per plane with its arguments, then host-valid; a second prepare does nothing
    #[test]
    fn prepare_copies_each_plane() {
        let env = Env::new();
        let r = make(&env);
        let b = backing(&r);
        let enc = env.encoder();
        let log = env.p.call_log();
        let mark = log.mark();
        r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
        assert!(r.state.host_valid());
        let copies: Vec<_> = log
            .since(mark)
            .into_iter()
            .filter(|c| c.sel.starts_with("copyFromBuffer:"))
            .collect();
        assert_eq!(copies.len(), 2);
        let tb = b.transfer_buffer.lock().as_ref().unwrap().id();
        let tex = b.textures();
        assert_eq!(
            copies[1].args,
            vec![
                Arg::Obj(Some(tb)),
                Arg::U64(64),
                Arg::U64(8),
                Arg::U64(16),
                Arg::val(&MtlSize {
                    width: 2,
                    height: 2,
                    depth: 1
                }),
                Arg::Obj(Some(tex[1].id())),
                Arg::U64(0),
                Arg::U64(0),
                Arg::val(&MtlOrigin::default()),
            ]
        );
        let mark = log.mark();
        r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
        assert!(log.since(mark).is_empty());
    }

    /// Guest-invalid still prepares; a nil encoder still sets host-valid
    #[test]
    fn prepare_guest_invalid_and_nil_encoder() {
        let env = Env::new();
        let r = make(&env);
        r.state.set_guest_valid(false);
        r.prepare_in_encoder(Hold::NONE, None).unwrap();
        assert!(r.state.host_valid());
        assert!(!r.state.guest_valid());
        // Synchronize from an invalid host still copies.
        r.state.set_host_valid(false);
        r.synchronize_in_encoder(Hold::NONE, None).unwrap();
        assert!(r.state.guest_valid());
    }

    /// `bytesPerImage` wraps in 32 bits
    #[test]
    fn bytes_per_image_wraps_in_32_bits() {
        let p = plane(0, 1, 0x100, 0x100_0000 - 1, 1);
        assert_eq!(
            p.bytes_per_image(),
            u64::from((0xff_ffffu32).wrapping_mul(0x100))
        );
        let big = plane(0, 1, 0x1000, 0x10_0000, 1);
        // 2^20 * 2^12 = 2^32 wraps to 0.
        assert_eq!(big.bytes_per_image(), 0);
    }

    /// Discard drops the buffer; the next prepare makes one
    #[test]
    fn discard_then_prepare_rebinds() {
        let env = Env::new();
        let r = make(&env);
        let b = backing(&r);
        let first = b.transfer_buffer.lock().as_ref().unwrap().id();
        let log = env.p.call_log();
        let mark = log.mark();
        r.discard().unwrap();
        assert!(b.transfer_buffer.lock().is_none());
        assert!(r.state.guest_valid());
        assert!(!r.state.host_valid());
        r.prepare_in_encoder(Hold::NONE, Some(&env.encoder()))
            .unwrap();
        let again = b.transfer_buffer.lock().as_ref().unwrap().id();
        assert_ne!(first, again);
        assert_eq!(
            log.since(mark)
                .iter()
                .filter(|c| c.sel == "new_buffer_for_virtual_page")
                .map(|c| c.args.clone())
                .collect::<Vec<_>>(),
            vec![vec![Arg::U32(3), Arg::U64(0x1000)]]
        );
    }

    /// The task's raise propagates; a gone task gives no buffer, which raises
    #[test]
    fn ensure_physical_failures() {
        let env = Env::new();
        let r = make(&env);
        r.discard().unwrap();
        env.p.script().push(
            "new_buffer_for_virtual_page",
            Step::raise("NSGenericException", "Failed to create buffer"),
        );
        let hold = Hold::NONE;
        let e = r.prepare_in_encoder(hold, None).err().unwrap();
        assert_eq!(
            e.origin(),
            crate::raise::Origin::Boundary("new_buffer_for_virtual_page")
        );
        assert!(!r.state.host_valid());
        // A backing whose task is gone.
        let lone = BackingResource::<P> {
            task: {
                let other: Arc<dyn ResourceHost<P>> = env.rec.sibling();
                Arc::downgrade(&other)
            },
            io_surface: None,
            planes: vec![plane(0, 1, 1, 4, 4)],
            transfer_buffer: Mutex::new(None),
            textures: Vec::new(),
            length: 4,
            page: 0,
            staged: false,
        };
        let e = lone.ensure_physical(hold).err().unwrap();
        assert_eq!(
            crate::resources::raised_kind(&e),
            Some(crate::resources::RaiseKind::NoGuestBuffer)
        );
    }

    /// Skipped while guest-valid; else one copy per plane, then guest-valid
    #[test]
    fn synchronize_copies_back() {
        let env = Env::new();
        let r = make(&env);
        let b = backing(&r);
        let enc = env.encoder();
        let log = env.p.call_log();
        let mark = log.mark();
        r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
        assert!(log.since(mark).is_empty());
        r.state.set_guest_valid(false);
        let mark = log.mark();
        r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
        assert!(r.state.guest_valid());
        let copies: Vec<_> = log
            .since(mark)
            .into_iter()
            .filter(|c| c.sel.starts_with("copyFromTexture:"))
            .collect();
        assert_eq!(copies.len(), 2);
        let tb = b.transfer_buffer.lock().as_ref().unwrap().id();
        assert_eq!(
            copies[0].args,
            vec![
                Arg::Obj(Some(b.textures()[0].id())),
                Arg::U64(0),
                Arg::U64(0),
                Arg::val(&MtlOrigin::default()),
                Arg::val(&MtlSize {
                    width: 4,
                    height: 4,
                    depth: 1
                }),
                Arg::Obj(Some(tb)),
                Arg::U64(0),
                Arg::U64(16),
                Arg::U64(64),
            ]
        );
        // Host-invalid still copies (it only logs).
        r.state.set_guest_valid(false);
        r.state.set_host_valid(false);
        r.synchronize_in_encoder(Hold::NONE, None).unwrap();
        assert!(r.state.guest_valid());
    }

    /// A Metal raise from a copy propagates with guest still invalid
    #[test]
    fn synchronize_copy_raise() {
        let env = Env::new();
        let r = make(&env);
        r.state.set_guest_valid(false);
        env.p.script().push(
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:",
            Step::raise("NSInvalidArgumentException", "x"),
        );
        let hold = Hold::NONE;
        assert!(
            r.synchronize_in_encoder(hold, Some(&env.encoder()))
                .is_err()
        );
        assert!(!r.state.guest_valid());
    }

    /// The last drop releases the surface, buffer and textures
    #[test]
    fn the_last_release_releases_everything() {
        let env = Env::new();
        let r = make(&env);
        assert_eq!(env.p.world().live("MTLTexture").len(), 2);
        drop(r);
        assert!(env.p.world().live("MTLTexture").is_empty());
        assert!(env.p.world().live("IOSurface").is_empty());
        assert!(env.p.world().live("MTLBuffer").is_empty());
    }

    /// What a landing frame holds goes at its end on success, and on a raise goes on with the
    /// raise to the holder where it ends
    #[test]
    fn a_landing_frame_holds_to_its_end_or_where_its_raise_ends() {
        let env = Env::new();
        let outer = Held::new();
        let t: Fk<kind::Texture> = env.p.world().new_obj(crate::testkit::objects::Model::None);
        let id = t.id();
        let r: RaiseResult<()> = Held::raising(|held| {
            held.keep(t);
            Ok(())
        });
        assert!(r.is_ok());
        assert!(env.p.world().find(id).is_none());
        let t: Fk<kind::Texture> = env.p.world().new_obj(crate::testkit::objects::Model::None);
        let id = t.id();
        {
            let r: RaiseResult<()> = Held::raising(|held| {
                held.keep(t);
                Err(crate::resources::texture::test_support::a_raise())
            });
            Hold::of(&outer).land(r.unwrap_err());
            assert!(env.p.world().find(id).is_some());
            drop(outer);
            assert!(env.p.world().find(id).is_none());
        }
    }

    /// The texture descriptor goes with a nil plane texture's raise
    #[test]
    fn a_refused_texture_lets_the_descriptor_go() {
        let env = Env::new();
        env.p
            .script()
            .after("newTextureWithDescriptor:iosurface:plane:", 1, Step::Nil);
        let hold = Hold::NONE;
        let e = new_in(
            &env,
            env.task(),
            &desc(0x1000, 0, 2),
            &[plane(0, 4, 4, 16, 4), plane(64, 2, 2, 8, 2)],
        )
        .err()
        .unwrap();
        assert_eq!(raised(&e), Some(RaiseKind::BackingTextureRefused));
        assert!(env.p.world().live("MTLTextureDescriptor").is_empty());
        hold.land(e);
    }

    /// A Metal raise lets the descriptor go too, and so does a normal return
    #[test]
    fn a_metal_raise_lets_the_descriptor_go() {
        let env = Env::new();
        env.p.script().push(
            "newTextureWithDescriptor:iosurface:plane:",
            Step::raise("NSInvalidArgumentException", "bad"),
        );
        let _ = new_in(
            &env,
            env.task(),
            &desc(0x1000, 0, 1),
            &[plane(0, 4, 4, 16, 4)],
        )
        .err()
        .unwrap();
        assert!(env.p.world().live("MTLTextureDescriptor").is_empty());
        let _r = make(&env);
        assert!(env.p.world().live("MTLTextureDescriptor").is_empty());
    }

    /// The transfer buffer's lock is not held across the task's buffer creation or the copies
    #[test]
    fn the_transfer_buffer_lock_is_not_held_across_calls() {
        let env = Env::new();
        let r = make(&env);
        r.discard().unwrap();
        let enc = env.encoder();
        r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
        // The field is free after the call, and the buffer made again is the one stored.
        let b = backing(&r);
        assert!(b.transfer_buffer.try_lock().is_some_and(|g| g.is_some()));
        assert_eq!(env.p.world().live("MTLBuffer").len(), 1);
    }
}
