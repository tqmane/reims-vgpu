//! The answer layer: where the runtime's typed errors become core's outcomes, as the metal
//! crate's message layer (`reims_vgpu_metal::msg`) decides a raise inside Metal.
//!
//! Every runtime call core asks for runs here, named by the Metal selector it stands for, which a
//! raise and a death name:
//! - [`send`]: the call's [`Policy`], its label, decides an [`Error::Invalid`]. By default
//!   (`Dies`, every encoder command and creator) it dies at the send naming the runtime's error,
//!   as every other refusal does: what Metal's validation stops, Metal without it goes on, nils
//!   or asserts at. Only where the evidence shows that Metal raises is it answered as the Metal
//!   platform answers a raise at that send: `Caught` (and `On(Return)`) as
//!   `Err(Raise::platform(sel))`, the caller's to catch, `Fatal` (and `On(Die)`) as a death at
//!   the send.
//! - [`get`]: a call whose every refusal dies at the call (a getter, a property setter, a send
//!   no path reaches under a handler); [`plain`], a call the runtime cannot refuse.
//! - [`create`]: a creator, where Metal answers nil: every refusal the runtime types
//!   [`Error::Nil`], what plain Metal answers nil for, and those of [`Nil`]'s set, which the
//!   runtime types otherwise where plain Metal was measured to answer nil, are `None`; the rest
//!   are as [`send`] decides them.
//! - [`error_text`]: a runtime error where Metal hands back an `NSError`.
//!
//! **What the runtime refuses where Metal would go on** (`NotImplemented`, `Unsupported`,
//! `AppleOnly`, a Vulkan failure, a lost device, a wait it will not make) is never a raise the
//! guest could catch: it is a death at the send, naming the selector and the runtime's error
//! ([`refused`]). **A method not built yet** dies the same way ([`unbuilt`]), and so does a raw
//! value the runtime's types cannot hold ([`unrepresentable`]).
//!
//! **A value of Metal's this platform does not follow**, which no runtime notice announces, dies
//! where the guest would read it, named by its kind ([`Unfollowed`], [`unfollowed`]); what made
//! the object it is read of went on. No such value is relayed.
//!
//! **A call Metal never returns from** parks the calling thread for good ([`hang`]), as the
//! contract keeps a hang a hang.
//!
//! **A write the runtime refuses where Metal's GPU writes or drops it** dies at the command: a
//! dispatch's or a draw's write through a texture over a surface of the 4:4:4 two-plane YCbCr
//! formats whose writes the runtime does not store as Metal does, the `_sRGB` twins 523 and 527
//! and the packed 510 and 530 ([`crate::gpu::ycbcr::refused_write`]). The runtime stores the
//! writes of 503, 507, 542 and 545 into the surface's planes as Metal's GPU does, and refuses
//! where the command is encoded the writes of those it does not store: through a texture over
//! plane 1, which dies there naming the refusal (`writes_of_444_textures_over_plane_1_die`), and
//! through a slot that also reads the texture, or by a command that writes the planes through the
//! texture and another way, which the runtime refuses alike.
//!
//! **A task that has ended sends nothing**, as on the Metal platform: before every call, a thread
//! running the work of an isolated task that died or was invalidated stops
//! (`stop_if_task_ended`).
//!
//! **A degradation of a kind [`DIES`] lists is relayed, and dies in strict mode** ([`notice`],
//! the runtime's message sink), but those of [`DIES_BY_DEFAULT`], which die in either mode: the
//! runtime announces each kind of degradation once per device, but these kinds at every
//! occurrence (`DeviceOptions::announce_each`, [`crate::platform`]).
//! By default the process goes on: the first occurrence of each kind writes the runtime's line,
//! and every occurrence is counted ([`relayed_counts`]), as a native device's relay is
//! (docs/accuracy-ledger.md: a difference from Metal's answer is logged, never a death of its
//! own). **Strict mode** ([`strict`], `REIMS_VGPU_STRICT=1`, read once per process) restores the
//! deaths, for tests and audits: the call that announced it dies once it returns, at its selector,
//! naming the notice, and a notice announced on a thread outside every call here (the runtime's
//! scheduler thread, at a host step) dies where it is announced. Where a kind's docs below say it
//! dies, they say what strict mode does. This crate's tests run strict, as each asserts the
//! deaths ([`relaxed`] runs a call under the default).
//!
//! **A difference this platform finds itself** ([`Inexact`], [`inexact`]): a value it computes
//! otherwise than Metal, which no runtime notice announces as a death (a `half` colour output's
//! blend, a resolve's rounding, a NaN or bfloat clear, a bfloat attachment's codes, a
//! `textureWriteRoundingMode` the runtime does not take), or an answer it gives otherwise (a
//! texture of protection options made unprotected), is relayed alike, its first occurrence
//! writing a line naming the kind, and dies in strict mode. A death where Metal itself ends the
//! process, or where the runtime cannot perform the call at all, is a death in either mode.
//!
//! **On a native device** (off macOS, a driver other than MoltenVK, [`native`]), the kinds of
//! [`native::NATIVE_RELAYED`] are relayed in strict mode too, each with its ground, the first
//! occurrence's line written and every occurrence counted; every other kind is placed as above,
//! as it is on MoltenVK. The placement is the process's: native only while every device it
//! adopted is.
//!
//! **In census mode** (`REIMS_VGPU_CENSUS_DIR`, a diagnostic for inventory runs on Linux alone,
//! [`census`]), a degradation of [`DIES`] is recorded with the selector of the call it was
//! announced inside, and relayed where it would die, so one sweep finds every kind. The mode
//! says on stderr, once per process as it turns on, that the run is not faithful to Metal. On
//! macOS the variable is ignored: every such degradation dies there.
//!
//! **Every death here ends the process**, one inside an isolated task's own call included: each
//! is a platform death (`DeathId::Platform`), which core's isolation leaves to the process, as it
//! leaves a crash inside Metal (`reims_vgpu_core::tasks::isolation::life`), and every task's work
//! runs on the one runtime device and its scheduler thread. The runtime catches a panic in its
//! sink, which only the test capture's death is: under the capture, a notice kept before it died
//! dies again at the thread's next call here.
//!
//! **A degradation of a kind [`HEARD`] lists is heard** by the call here it is announced inside
//! ([`hearing`]), at every occurrence too: the call dies where an answer it gives would differ
//! from Metal's, and otherwise goes on, the notice relayed (the first time alone, for
//! [`RELAYED_ONCE`]'s kinds).
//!
//! **Every other kind is relayed**: it goes to stderr as the runtime writes it without a sink
//! ([`relay`]), as every other message of the runtime's own does (its pipeline-creation log among
//! them; a `Verbose` one, which the runtime gives a sink alone, goes to the log at debug), each
//! line naming its kind. Each is relayed for one of these reasons, named at the kind
//! in the placement test (`every_degradation_kind_is_placed_once`):
//! - Metal's answer at a cost Metal does not pay, in time or memory (the performance kinds);
//! - measured equal to Metal's answer on this host;
//! - a difference the device itself forces, a stated fact of its hardware (as close as the
//!   hardware allows, the rest logged, no interpreter);
//! - an accepted class, within its stated bound: 8-bit linear filtering
//!   within one code of Metal's (`device-unorm8-linear-filter`), fast-math rewrites within what
//!   the function's AIR contract and reassociation flags allow (`fast-math-rewrites`), maths
//!   functions whose bound Metal does not state, each device's deviation measured
//!   (`device-transcendentals`), and a `half` colour output's rounding within its measured bound
//!   (`half-output-precision`, which past it is relayed as [`Inexact::HalfOutputBlend`], and dies
//!   in strict mode, [`HEARD`]); and a native device's own maths and sampling, each bound
//!   recorded ([`native::NATIVE_RELAYED`]);
//! - Metal's own answer there is not one value: undefined, memory-dependent or racing;
//! - no value the guest or the host sees;
//! - the difference dies where it shows, at a later call;
//! - pending a decision: `dont-care-load-tiles`, `extended-range-bits-kept` and
//!   `unreached-tiles-rewritten`, and `heap-texture-lossless-layout` ([`HEARD`]).
//!
//! That a guest's boot or a suite case reaches a kind is no reason: a kind none of these covers
//! dies ([`DIES`]) until the runtime makes it exact or it is measured equal. The other heap
//! layout, `heap-texture-block-compressed-layout`, dies.

use std::cell::{Cell, RefCell};

use metal2vulkan_runtime as rt;
use metal2vulkan_runtime::Error;
use reims_vgpu_core::platform::types::NsErrorText;
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult};
use reims_vgpu_core::tasks::isolation::life::stop_if_task_ended;

pub mod census;
pub mod native;

/// What a refusal the runtime types `Invalid` does at a call: the call's label, as the metal
/// crate's `msg::Policy` labels the same send.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Policy {
    /// A call where Metal raises at nothing: the refusal dies at the send, naming the runtime's
    /// error ([`refused`]). The default for an encoder command and a creator.
    Dies,
    /// A call Metal raises at, which no path reaches without a handler: the refusal is `Err`,
    /// the caller's to catch.
    Caught,
    /// A call whose refusal dies whatever handler is live.
    Fatal,
    /// A call in a platform method reached both ways, whose caller says by the [`OnFault`] which
    /// way this call is.
    On(OnFault),
}

impl Policy {
    /// The way this call is reached: `Return` under a handler, `Die` with none.
    fn on(self) -> OnFault {
        match self {
            Policy::Caught => OnFault::Return,
            Policy::Fatal | Policy::Dies => OnFault::Die,
            Policy::On(on) => on,
        }
    }
}

/// Which refusals of a creator Metal answers with nil, beside every [`Error::Nil`] ([`create`]):
/// those the runtime types otherwise, where plain Metal was measured to answer nil, which are
/// the runtime's memory refusals. A buffer's length, a heap's size and a buffer's place in a
/// heap, which plain Metal answers nil for, are [`Error::Nil`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nil {
    /// None: every refusal is decided as [`send`] decides it.
    Never,
    /// A buffer's length or memory: a length of 0 or past the device's largest buffer, no memory
    /// for it, or a pinned address the space cannot honour (measured on the Metal platform:
    /// `newBufferWithLength:options:` of `u64::MAX`, and a no-copy buffer of length 0 or 2^46,
    /// answer nil).
    BufferLength,
    /// What a library lacks: `newFunctionWithName:` of a name the library does not hold.
    MissingFunction,
    /// Memory the device cannot allocate.
    Memory,
    /// No memory for a shared texture: one of storage other than `Private` is an [`Error::Nil`].
    StorageMode,
    /// A heap's size or memory: a size of 0, no memory for it, or a pinned address the space
    /// cannot honour (measured on the Metal platform: a heap of size 0 is nil).
    Heap,
    /// Where a resource is placed in a heap: an offset off the resource's alignment, bytes past
    /// the heap's end, or no memory for it (measured on the Metal platform: a buffer at an offset
    /// off 256 bytes, and a buffer or texture past the heap's end, are nil).
    Placed,
}

impl Nil {
    /// Whether Metal answers nil where the runtime refuses with `e`.
    fn covers(self, e: &Error) -> bool {
        match self {
            Nil::Never => false,
            Nil::BufferLength => matches!(e, Error::OutOfMemory(_) | Error::PlacementConflict(_)),
            Nil::MissingFunction => matches!(
                e,
                Error::Compile(metal2vulkan::error::Error::InvalidDescriptor(_))
            ),
            Nil::Memory => matches!(e, Error::OutOfMemory(_)),
            Nil::StorageMode => matches!(e, Error::OutOfMemory(_)),
            Nil::Heap => matches!(e, Error::OutOfMemory(_) | Error::PlacementConflict(_)),
            Nil::Placed => matches!(e, Error::OutOfMemory(_)),
        }
    }
}

/// Runs `call`, the runtime's form of `sel`: `policy` decides a refusal Metal's validation stops,
/// and any other refusal dies at the send ([`decide`]).
pub fn send<T>(
    sel: &'static str,
    policy: Policy,
    call: impl FnOnce() -> Result<T, Error>,
) -> RaiseResult<T> {
    stop_if_task_ended();
    inside(Some(sel), call).or_else(|e| decide(sel, policy, e))
}

/// [`send`], the line of a death or a raise also carrying what `describe` says of the call
/// ([`decide_describing`]), asked only where the runtime refuses it.
pub fn send_describing<T>(
    sel: &'static str,
    policy: Policy,
    describe: impl FnOnce() -> String,
    call: impl FnOnce() -> Result<T, Error>,
) -> RaiseResult<T> {
    stop_if_task_ended();
    inside(Some(sel), call).or_else(|e| decide_describing(sel, policy, e, Some(&describe())))
}

/// Runs `call`, the runtime's form of `sel`, whose every refusal dies at the call.
pub fn get<T>(sel: &'static str, call: impl FnOnce() -> Result<T, Error>) -> T {
    get_describing(sel, None, call)
}

/// [`get`], the line of a death also carrying `descriptor` ([`decide_describing`]).
pub fn get_describing<T>(
    sel: &'static str,
    descriptor: Option<&str>,
    call: impl FnOnce() -> Result<T, Error>,
) -> T {
    stop_if_task_ended();
    inside(Some(sel), call).unwrap_or_else(|e| {
        match decide_describing::<T>(sel, Policy::Fatal, e, descriptor) {
            Ok(v) => v,
            Err(r) => r.fatal(),
        }
    })
}

/// Runs `call`, a runtime call that cannot refuse and stands for no selector: a degradation of
/// [`DIES`] announced inside it dies at the kind's code. A call that stands for a selector is
/// [`plain_at`]'s.
pub fn plain<T>(call: impl FnOnce() -> T) -> T {
    stop_if_task_ended();
    inside(None, call)
}

/// Runs `call`, the runtime's form of `sel`, whose result the caller decides: a degradation of
/// [`DIES`] announced inside it dies at `sel`, as one inside [`send`]'s call does.
pub fn plain_at<T>(sel: &'static str, call: impl FnOnce() -> T) -> T {
    stop_if_task_ended();
    inside(Some(sel), call)
}

/// [`plain_at`] without the check of the task's end before `call`: for a call no thread may stop
/// before, as one holding an encoder it took ([`crate::gpu::encode_blit::EncoderObject::end`]).
pub fn unstopped_at<T>(sel: &'static str, call: impl FnOnce() -> T) -> T {
    inside(Some(sel), call)
}

/// Runs `call`, a creator standing for `sel`: `None` for an [`Error::Nil`] and for the refusals
/// `nil` names, as Metal answers nil; the rest as [`send`] decides them.
pub fn create<T>(
    sel: &'static str,
    policy: Policy,
    nil: Nil,
    call: impl FnOnce() -> Result<T, Error>,
) -> RaiseResult<Option<T>> {
    stop_if_task_ended();
    match inside(Some(sel), call) {
        Ok(v) => Ok(Some(v)),
        Err(Error::Nil(_)) => Ok(None),
        Err(e) if nil.covers(&e) => Ok(None),
        Err(e) => decide(sel, policy, e),
    }
}

/// The degradations whose cost reaches what the guest sees, which die ([`notice`]), each
/// announced at every occurrence:
/// - `ReleasedCommandState`: an execution of an indirect command naming a state the program
///   released, whose ID no live state of its kind holds, which runs nothing where Metal's
///   execution is undefined. Announced by the execution alone: at `commit` for records the CPU
///   wrote, a pass whose pipelines were still being made then included, which `commit` waits for
///   before it records the pass; and for records the GPU may write or a range a buffer holds, at
///   the host step, which runs on the runtime's scheduler thread alone, outside every call here:
///   it dies there and ends the process, never at `commit` (a pass one of whose pipelines failed
///   to be made goes to the host step too, which fails the command buffer);
/// - `TwoPlaneTextureCopy` and `TwoPlaneBufferBlit`: a blit of the two-plane XR10 formats whose
///   answer is not Metal's measured one ([`crate::gpu::encode_blit`]);
/// - `YCbCrFilterNotExact`: a pipeline's filtered sample of a texture slot bound to a two-plane
///   YCbCr texture (any of the 24 formats of [`crate::gpu::ycbcr`]'s family) that does not take
///   Metal's exact filter of the format (a swizzle that moves its channels on MoltenVK, whose own
///   sampler composes it, where another device filters exactly and swizzles the result; a
///   constexpr sampler with a mip filter; a sample the emulation does not
///   reach), which reads off Metal's grid (of 1/2048 for a converted format, of 1/8192 for an
///   `_sRGB` twin), naming the slot and the format: the runtime's generic sample filter notes
///   (`HalfSampleFilter`, `FloatSampleFilter`) cover no slot bound to a YCbCr texture, and a
///   runtime sampler's mip filter over its one level takes the exact filter, announcing nothing.
///   Announced at each dispatch, tile dispatch and draw that binds the pipeline's variant, on the
///   encoding thread, inside its call, which dies at its selector: not where the variant is made,
///   which may be the runtime's pipeline-making thread, off every call here. An indirect command's
///   is announced where its execution records it: at `commit`, or at the host step on the
///   scheduler's thread alone, which dies there and ends the process, as `ReleasedCommandState`
///   does. A variant made where the command does not know its formats (an execution set's,
///   `VK_EXT_device_generated_commands`, which MoltenVK lacks) binds no texture slot, so its
///   samples are none of these: a texture it reaches through handles is refused
///   (`MappedTextureSampling`, [`refusal`]);
/// - `RotatedBlendRounding`: a pipeline's blend into an `ARGB8Unorm` attachment in none of the
///   forms measured to blend as Metal's, or in one of them under a fragment output that is not
///   32-bit float, which may round a code apart from Metal's (the runtime's
///   `PrecisionNote::RotatedBlendRounding`). Announced at each draw that binds the variant, on the
///   encoding thread, inside its call, which dies at its selector, as `YCbCrFilterNotExact` is.
///   It dies on every device, a native one included: Apple's blend arithmetic for those forms is
///   not measured, and the exact form is a measured one or the blend in the function;
/// - `EncodedUseRebound`: a command buffer's command named a buffer or heap over the guest's
///   pages that was rebound (a replace, `crate::gpu::backing`) after the command was encoded and
///   before the command buffer's first submission, in an encoder the rebind could not encode
///   again (a predicated command buffer's, a render pass recorded later or
///   whose end captured its draws' tile reach, a call refused again; every other compute, blit
///   and render encoder it encodes again over the new pages, as Metal runs it), and neither the
///   submission check nor the failure the command buffer was committed with (`commit_failed`)
///   failed the submission holding it: the runtime runs nothing of that submission and fails the
///   command buffer, where Metal runs the command over the pages the object lies in as it submits.
///   A command buffer an earlier submission failed already (a use detached behind an event wait,
///   Metal's code 8) keeps that code. A submission the check fails keeps that code and announces
///   nothing. Announced once per device, whatever `announce_each` says, as the submission is made,
///   on the thread that makes it: the runtime's scheduler thread, outside every call here, or a
///   thread inside a later command buffer's `commit`, in which the runtime submits every batch
///   ready then. It dies there and ends the process. The failure it leaves dies where it is first
///   read too (`crate::gpu::command`), so a later one, which the device does not announce again,
///   dies;
/// - `ZeroLayerMapAfterRelease`: a draw in a pass over a rasterization rate map of no layers made
///   after the device released a map of layers, which writes nothing where Metal draws rows of
///   the pass. Announced at the draw, inside its call, which dies at its selector. No draw over a
///   map of no layers reaches the runtime here ([`crate::gpu::rate_map`]);
/// - `HeapTextureBlockCompressedLayout`: a texture made in a heap over the guest's pages of a
///   block-compressed format whose placement the device's Metal level does not measure (PVRTC
///   at every level; BC, ETC2, EAC and ASTC are measured at this host's macOS 26.7 and at
///   27.0.1), whose blocks the runtime lays out as its own where Metal tiles them. Its reads,
///   writes and copies are Metal's, but a buffer or texture over its bytes meets other bytes than
///   Metal's would. Announced once per texture, inside the call that made it (the heap's
///   `newTextureWithDescriptor:offset:`), which dies at its selector;
/// - `BiasedDepthPastUnitRange`: a draw of triangles in depth clip mode under a depth bias, on a
///   device that enabled `VK_EXT_depth_range_unrestricted` but cannot clamp to `[0, 1]` while it
///   clips, which stores a depth the bias takes past `[0, 1]` unclamped, where Metal stores 0
///   or 1. Announced at the draw, inside its call, which dies at its selector. MoltenVK does not
///   enable the extension, so no draw here reaches it;
/// - the kinds whose answer this host's driver gives as Metal's, which another device's may not,
///   so that no device here announces them, each dying where a device announces it, inside the
///   guest's call that met it (a draw, a pass, an upload, a resource's or the device's creation;
///   a note at each draw or dispatch that binds the pipeline's variant), at its selector:
///   - `UnormDepthBias` and `UnormDepthRounding`, a draw biasing or testing a `Depth16Unorm`
///     depth, which MoltenVK stores as Metal does (the runtime's driver facts
///     `UnormDepthBiasIsMetals` and `Depth16UnormStoresMetals`);
///   - `RestrictedDepthUpload` and the note `RestrictedDepthRange`, a `Depth32Float` value outside
///     `[0, 1]` that a device stores clamped, which MoltenVK keeps
///     (`DepthImageKeepsValuesOutsideUnitRange`);
///   - `UnmeasuredResolve`, a resolve past 4 samples, of a texture neither this device nor Metal's
///     here makes;
///   - `MoltenVkArgumentBuffersUnread` and the notes `BeyondMeasuredDeviation` and
///     `FunctionStorageAssumed`: a driver release, a MoltenVK configuration or a device's private
///     memory the runtime has not measured, where this host's are measured. A device whose
///     driver, GPU and release the runtime has not measured takes no fact measured elsewhere, only
///     the exact path, and says so once as it is made in an informational message
///     (`metal2vulkan-unmeasured-driver`), which is relayed ([`relay`]), no degradation;
///   - the notes `HalfSampleFilter`, `SrgbTexelDecode`, `FloatSampleFilter`,
///     `DeviceGatherFootprint`, `CubeFaceReciprocal`, `DeviceDepthComparison` and
///     `DeviceTexelDenormals`: a sample, gather or comparison a device makes otherwise than
///     Metal's texture unit, which MoltenVK's is (the runtime requires
///     `VkProperty::MetalSampler` there instead). A native device relays `HalfSampleFilter`,
///     `SrgbTexelDecode` and `FloatSampleFilter`, its sampling, as the class of a native
///     device's own maths and sampling ([`native::NATIVE_RELAYED`]);
///   - the notes `AnisotropicFilter` and `DeviceAnisotropicFilter`: a sample through a sampler
///     of `maxAnisotropy` above 1 on a device without Metal's own sampler, which takes Metal's
///     anisotropic filter emulated by the device's samples at explicit levels, a near-equivalent
///     whose level and tap spacing are a fitted model's (a 2D
///     texture or 2D array sampled without comparison through a constexpr sampler, or through a
///     runtime sampler slot under a bound sampler that states anisotropy), or, for a shape the
///     emulation does not reach (a cube, a 3D texture, a comparison, a sample whose residency
///     the function reads, a texture of another class, a swizzled storage slot), the device's
///     own anisotropic filter, up to 62 codes off Metal's on an RTX 3090's 2D samples and not
///     measured on those shapes. MoltenVK samples through Metal's own sampler, where the runtime
///     emulates nothing and notes neither. Both die on every device, a native one included:
///     the emulation's level and its tap spacing past two taps are a fitted model, not Metal's
///     rule, and each tap is the device's own filter;
///   - the notes `DeviceInterpolation`, `QuadsSplitByClipping`, `VertexSimdgroupGrouping` and
///     `FragmentSimdgroupGrouping`: NVIDIA's interpolation, llvmpipe's clipping, ANV's vertex
///     subgroups, and the fragment subgroups of a device whose are not Metal's simdgroups;
///   - `NonStrictResidency` and the note `SparseResidency`: a sparse texture's unmapped texels as
///     the device reads them, where MoltenVK binds no sparse memory;
///   - the note `DeviceRayIntersection`: a ray query, which MoltenVK's profile lacks;
///   - the note `UnorderedFramebufferFetch`: a framebuffer fetch without rasterization-order
///     access, whose pipeline the runtime refuses where it makes it, whatever its policy;
/// - the note `ViewBaseTexelOffset`: a sample or gather of an argument buffer's texture with a
///   constant texel offset on an axis that may repeat or mirror, which the runtime places by
///   Metal's rule on the bound view's own base-level size, where Metal reads its parent's: on a
///   view whose first level lies above its parent's base, an axis of size 2 or 4 under a parent
///   whose base-level size is not a power of two keeps the offset Metal drops, and reads one
///   texel or footprint away. A texture slot's access reads the parent's and announces
///   nothing. MoltenVK's own sampler places the offset as Metal's, so there only the forms the
///   runtime emulates on every device announce it: a bicubic filter, and a sample or gather of a
///   texture a heap holds by its raw texels (an extended-range or bfloat format); no guest run on
///   the Mac has announced it. Announced at each draw or dispatch that binds the
///   pipeline's variant, whatever texture it binds, inside its call, which dies at its selector.
///   It dies on MoltenVK: the hardware allows the offset placed by the parent's base level, as a
///   texture slot's is, which the runtime has not built for an argument buffer's texture. A
///   native device relays it, its sampling, as the class of a native device's own maths and
///   sampling ([`native::NATIVE_RELAYED`]);
/// - the depth a draw stores where the runtime's depth state answers otherwise than Metal's,
///   each announced at the draw, inside its call, which dies at its selector (an indirect
///   command's where its execution records it, as `ReleasedCommandState`'s):
///   - `PointGuardBand`: a point whose centre lies past its viewport, within Metal's guard band
///     of 1.5 times the viewport's half-size, where no Vulkan viewport the device's limits take
///     holds the band, which the device culls where Metal draws it;
///   - `NanDepth`: a NaN depth the fragment function writes, which stores the depth clamp
///     range's low end where Metal stores 0, or runs no stencil operation where Metal runs the
///     depth-failure one;
///   - `ClippedDepthPieces` and `SampleDepthClamp`: a primitive past Metal's near or far plane
///     or its guard band, or a biased one in depth clamp mode, whose depths the device clamps at
///     its vertices or its samples, where Metal clamps the pieces it clips the primitive into,
///     or each vertex's depth before the interpolation;
///   - `DepthBiasPastOne`: a depth a bias moves past 1 in depth clamp mode, which stores 1
///     where Metal clamps it to the range's larger depth;
/// - `InPlaceVertexReads` and `FragmentsBeforeGeometry`: a draw's vertex, object or mesh stage
///   that reads in place bytes an earlier fragment stage of its pass wrote, or writes bytes one
///   read or wrote, which the device may run after those fragments, and does across a split of
///   the pass, where Metal's vertex, object and mesh functions ran before the pass's fragments
///   in every run measured. Announced at the draw, inside its call, which dies at its selector;
/// - `MipmapsWithinSpread`: `generateMipmapsForTexture:` of a texture whose levels the
///   device's linear blit makes, within 1 unit in the last place (2 for 32-bit float) of an
///   Apple M3 Max's or an Apple M4's on the inputs checked, at neither's exactly. Announced at the
///   call, which dies at its selector;
/// - `HeapDepthStencilApart` and `HeapTextureHidesAliasedWrites`: a heap's texture whose texels
///   do not alias the heap's bytes as Metal's do: a depth or stencil image a driver lays apart
///   from its memory (MoltenVK, the runtime's `DepthStencilImagesLieApart`), where the runtime
///   keeps no shadow of it in the heap's bytes, or a colour image a driver may compress (ANV's
///   `CompressedImagesHideAliasedWrites`, which no device here has). A write through a buffer
///   or texture over its bytes changes none of its texels, or not all, and its writes change
///   none of those resources' bytes. Announced inside the heap's creator of the texture, which
///   dies at its selector;
/// - a resource a function may reach through its ID or address without the command naming it,
///   which the read would find other than Metal's, each announced where the reach becomes
///   possible, not where a function reads it, inside the call that makes it possible, which
///   dies at its selector:
///   - `ResourceIdTakeover`: a sampler state made while a committed command buffer lists the
///     argument samplers, whose ID a function may read as an earlier holder's or the neutral
///     sampler's;
///   - `UnreadableHandleTexture`: a PVRTC or 4:2:2 texture a command that reads texture
///     handles reaches, which reads the neutral texture where Metal decodes it;
///   - `UnnamedTextureUnviewable` and `UnnamedTexturesPastCapacity`: a texture a heap array
///     cannot view, or one past a pipeline's heap arrays, which a function reading its ID
///     unnamed reads as the neutral texture;
///   - `HeapSetCopied`: a heap set a command buffer binds on a device that cannot write a set a
///     pending command buffer binds, whose functions read each ID's texture as it stood at the
///     commit;
///   - `DetachedStorageKeepsWrites`: a storage texture over detached bytes a committed command
///     buffer's heap set reaches, whose unnamed storage reads read what an earlier such write
///     left, where Metal drops those writes and reads zero. Announced at `commit`;
/// - what a pass or a copy leaves in its texture otherwise than Metal, each announced inside the
///   call that encodes it, which dies at its selector:
///   - `MixedDepthStencilResolve`: a pass that stores and resolves one texture's depth beside
///     its stencil resolved alone, which Metal's validation stops and plain Metal answers with a
///     layout pattern of the resolved depth, where the runtime resolves both and keeps both
///     aspects' samples. Announced at the pass's creation;
///   - `UncoveredColorKept`: a draw of a state that lacks a colour attachment of the pass, whose
///     texels it keeps where Metal's compiler may fold the output to a constant that stores zero.
///     Announced at the draw;
/// - `CappedAnisotropy`: a sampler state of a `maxAnisotropy` past 16, which Metal samples with
///   32, where the device filters with its largest (16), so a footprint more than 16 times as
///   long as it is wide reads a coarser level. Announced at the state's creation, which dies at
///   its selector;
/// - `CombinedArchiveStages`: a render pipeline state the runtime makes from binary archives
///   where Metal may refuse it with an archive miss (its two adds' fragment functions reading
///   different vertex outputs). Announced at the state's creation, which dies at its selector;
/// - `ImplementationDefinedReconvergence`: a stage that reads simdgroup state after divergent
///   control flow on a device that does not reconverge maximally, whose simdgroup operations
///   there can see other lanes active than Metal's. Announced at each draw or dispatch that binds
///   the pipeline's variant, inside its call, which dies at its selector. The runtime notes every
///   stage holding a simdgroup or quad operation, a derivative, an implicit-level sample or a
///   barrier on a driver without `VK_KHR_shader_maximal_reconvergence`, which decides which lanes
///   run together after divergence: other sets of lanes than Metal's can be active together, so
///   a ballot or a sum can cover another set of lanes, and a `simd_is_first` election can pick
///   several lanes. A native device relays it, as it does the precision notes below, on the
///   rule for native devices stated there ([`native::NATIVE_RELAYED`]). A fragment stage with
///   simdgroup operations still dies there, of `FragmentSimdgroupGrouping`, which the runtime
///   records on every driver but MoltenVK;
/// - the precision notes of a pipeline's stage whose arithmetic, filtering or ordering can give
///   other bits than Metal's, as the runtime's documentation of each says, which no measurement
///   holds equal here: `F16FmaDoubleRounding` and `F16IntConvDoubleRounding` (a widened half fused
///   multiply-add or half-integer conversion rounding twice), `GradFromBias` (a biased sample
///   emulated through gradients), `ReductionOrder` (a float reduction's association),
///   `DeviceSignedZeroInfNan` (float operations run without `SignedZeroInfNanPreserve`),
///   `TessellationFactorScale` and `TessellationFactorInstance` (a scaled factor's rounding, a
///   per-instance factor's first instance), `Fp8Value` (an 8-bit float converted to an integer),
///   `MultisamplePosition` (a value Metal reads at the pixel's centre, evaluated elsewhere),
///   `CubeGradientFootprint` (a cube sample's level), `QuadsCountedBeforeDiscard` (a simdgroup
///   operation before a discard counting a quad Metal's may not) and `BlendedWriteRounding` (a
///   blended half-float write under `textureWriteRoundingMode` 2). Announced at each draw or
///   dispatch that binds the pipeline's variant, inside its call, which dies at its selector, as
///   `RotatedBlendRounding` is. MoltenVK's placement is as measured on the Mac GPUs: every one
///   dies there. A native device relays those whose difference is the device's own rounding,
///   evaluation position or reduction order, a fact of the device ([`native::NATIVE_RELAYED`],
///   each with its ground): `GradFromBias`, `ReductionOrder`, `DeviceSignedZeroInfNan` and
///   `MultisamplePosition`, and, as the class of the device's own maths and sampling,
///   `F16FmaDoubleRounding`, `F16IntConvDoubleRounding` and `CubeGradientFootprint`, on the rule
///   for native devices: as close to Metal's answer as the hardware allows, a difference that
///   cannot match logged as a degradation, with no fallback to an interpreter for correctness.
///   Every other one dies there too, as no fact of
///   the device decides it: `TessellationFactorInstance` and `Fp8Value` are other values, not
///   other bits of the same arithmetic (the factors of a draw whose base instance is not 0, read
///   at an offset counted from it; an 8-bit float's integer from the format's value, where Metal
///   on an Apple M3 answers its carrier byte's); `QuadsCountedBeforeDiscard` differs where
///   Apple's compiler moves the operation past the discard, which that compiler decides shape by
///   shape; the hardware allows `BlendedWriteRounding` an exact form the runtime has not built
///   (the blend in the function under the pixel interlock); and Metal's rule for
///   `TessellationFactorScale` is not read (whether its fractional partitions take the scaled
///   factor in `float` or rounded up to `half`). On MoltenVK, `CubeGradientFootprint` (Apple's
///   combination of a cube sample's two gradient components and its minification threshold,
///   whose rounding is not identified, and an implicit-level cube sample's level) dies too;
/// - the note `LinkedSamplerNumbering`: the constexpr sampler IDs of a stage that links visible
///   functions, where the runtime does not number them as Metal does. It notes a stage that
///   links two or more with constexpr samplers, whose order Metal itself does not decide (two
///   machines numbered one pipeline in different orders), a dynamic library's, a binary
///   function's or a function-pointer group's, which nothing measured, and a function a table
///   can hold whose samplers the runtime cannot read, which Metal numbers and the runtime does
///   not, so every sampler state made while the state lives may answer another `gpuResourceID`.
///   One linked function alone, of `functions` or `privateFunctions`, called or not, is
///   numbered as Metal numbers it and noted for nothing. One note
///   covers them all, so it dies. No guest form here links a function: no serialized key
///   carries one, and a dynamic library's import is refused. Announced at each draw or dispatch
///   that binds the pipeline's variant, inside its call, which dies at its selector;
/// - the note `PixelSamplerScaling`: a stage that samples through a sampler state bound at a slot
///   whose coordinates are pixels, or through a constexpr pixel sampler, where none of the
///   runtime's exact paths reaches and the device filters a coordinate the runtime scaled. On
///   MoltenVK a nearest or linear 2D sample through either takes an exact path and notes nothing,
///   Metal's texels in each of the runtime's measured configurations on an M4, so what it notes
///   is what nothing measured. No recorded VM boot relayed `pixel-sampler-scaling`, and the suite's
///   one pixel sampler state filters nearest, which notes nothing. Announced at each draw or
///   dispatch that binds the pipeline's variant (a slot's where the state bound has pixel
///   coordinates), inside its call, which dies at its selector;
/// - the note `ArgumentPixelSamplerScaling`: a stage that samples through an argument buffer's
///   sampler, which may hand the device a coordinate the runtime scaled from pixels, where the
///   state the buffer names has pixel coordinates and a linear filter or a comparison. On
///   MoltenVK the runtime notes only a comparison, as every other sample through such a sampler
///   reads Metal's texels there (60 configurations measured on an M4), and Metal's answer for a
///   comparison through a pixel sampler is not measured. No recorded VM boot relayed
///   `pixel-sampler-scaling`, and the suite's one pixel sampler state filters nearest and compares
///   nothing. Announced at each draw, dispatch, tile dispatch or execution of indirect commands
///   of the pipeline while the device holds a live sampler state of pixel coordinates with a
///   linear filter or a comparison, inside its call, which dies at its selector;
/// - the note `DiscardDropsSimdgroupStores`: on MoltenVK on an Apple GPU, a fragment function
///   that samples, and after a discard computes a floating-point value and writes storage,
///   where Apple's compiler drops the texture writes of every live thread of a SIMD group holding
///   a discarded thread, whole 8 by 4 blocks of a storage-form fragment's colour writes. The
///   runtime waits with such a discard until after the writes, which announces nothing; it
///   keeps the discard and announces the note where a SIMD-group operation follows
///   it, so the function's storage writes may be lost. A store, atomic or image write that
///   follows a discard it keeps otherwise runs only where the thread was not discarded,
///   announcing nothing. No VM boot or suite run is known to make such a function. Announced at
///   each draw or dispatch that binds the pipeline's variant (the runtime announces a note of a
///   kind in `announce_each` there, not where the variant is made), inside its call, which dies at
///   its selector;
/// - `FreshPatternZeroed`: a new `Private` texture Metal keeps losslessly compressed, whose
///   fixed pattern the device cannot write exactly (a NaN depth a copy may store clamped, a
///   multisampled `Depth32Float` one's NaN, a NaN colour of a multisampled image of a format
///   the device makes no integer scratch image of, or an image of other bytes than Metal's
///   texel), which reads zero where Metal reads the pattern. The runtime copies a multisampled
///   NaN pattern from its integer scratch through a view of the image's format, so MoltenVK's
///   copies hold it (`rgbx`'s `new_textures_read_as_metals`). No VM boot or suite run is known
///   to read such a texture before writing it, so nothing grounds a relay. Announced where a
///   command first reads a texel of that content that no clear, resolve or blit wrote (a pass that
///   loads it, a blit that copies from it, a shader that reads it), inside its call, which dies at
///   its selector. The runtime counts what writes the texels by box and aspect: a blit that writes
///   a whole level, blits of parts that together cover one (an upload in pieces), a resolve into
///   the texture, separate depth and stencil attachments and a stand-in's own texture, and a pass's
///   load or a blit reads only its region; a copy of one aspect of a `Depth32Float_Stencil8`
///   texture writes that aspect alone. So a guest that copies a buffer into a new multisampled
///   `Depth32Float` target and reads it with no clearing pass between reads Metal's texels
///   (`encode_blit`'s `multisampled_depth_and_stencil_buffer_copies_are_the_metal_platforms`). A
///   pass's clear counts as a write, as the pass ends, only where it reaches the texture: an
///   attachment the pass renders itself, or one it renders through a stand-in or a separate
///   depth or stencil scratch that it stores. A new separate depth or stencil texture, or a
///   stand-in's own, that a pass cleared through its scratch and stored `DontCare` keeps its
///   fresh pattern, as Metal's keeps its texels there, and a later read of it announces the kind
///   and dies;
/// - the note `ExtendedRangeFloatBlendSource`: a fragment function's `float` colour output
///   blended into an extended-range attachment where Metal may round the blend's source otherwise
///   than the runtime's model: under fast math a lane of the output that is not a load, or a
///   module that names no math mode (a note of its own, beside the `half` output's
///   `ExtendedRangeBlendSource`). A computed `float4` under fast math misses Metal's by
///   a code of a channel under 17 of the 66 blended configurations `encode_render`'s sweeps draw
///   on this host's M3 Max, and no VM boot or suite run is known to make such a state, so nothing
///   grounds a relay. Announced at each draw that binds the state's variant, inside its call,
///   which dies at its selector; the state is made (`encode_render`'s
///   `float4_extended_range_blends_die_naming_their_source`);
/// - the note `ExtendedRangeBlendSource`, the same for a `half` colour output: Metal's compiler
///   folds the blend into the function, so its rounding follows how the function computed the
///   output, where the runtime rounds each product to `half`. A module
///   compiled without fast math, and an output each of whose lanes is a load under fast math,
///   announce nothing and blend as Metal's; a computed `half4` under fast math, or a module that
///   names no math mode, may store a channel a code apart from Metal's. The function can take
///   Metal's rounding once the runtime reads how Apple's compiler folds the blend, so nothing in
///   the hardware forces the difference, and the blend is not in the function's own arithmetic,
///   whose fast-math rewrites `FastMathRewrites` covers. It dies on every device. Announced at
///   each draw that binds the state's variant, inside its call, which dies at its selector;
/// - `MipmapsAsAppleM4`: `generateMipmapsForTexture:` of a chain whose levels the runtime takes
///   from an Apple M4's filter where the M3 Max's passes are not known (a compressed chain with a
///   pass from a level more than 128 times wider than tall or taller than wide, an
///   `A1BGR5Unorm` 2D chain, an `ABGR4Unorm` one of three levels or more, a 3D chain of a format
///   summed as half floats), with a few measured values of the 4-bit
///   and 1-bit formats and of 3D half-float sums one unit off. The M3 Max's passes can be read
///   for those chains as for the others, so nothing in the hardware forces the difference, and it
///   dies on every device;
/// - `MipmapTieRounding`: `generateMipmapsForTexture:` of a chain whose levels take a rounding
///   the runtime's chain does not fit, at a sum a hair from a rounding boundary: every chain of
///   a format with an 8-bit sRGB colour field, of an extended-range sRGB format, and a 16-bit float
///   chain with a level read from the level above's kept values, at most one code of the colour or
///   one unit in the last place of the half (measured: 1 of 5,331,270 sRGB colour values and 8 of
///   538,088 half values, the M3 Max's and an M4's alike). The tie rules can be fitted, as the
///   runtime fitted the 10-bit and extended-range linear formats', so nothing in the hardware
///   forces the difference; the kind holds `half` and 10-bit sRGB chains beside the 8-bit ones, so
///   it is not 8-bit filtering alone, and it dies on every device.
///
///   Each is announced at the call, for every chain of its shapes, not at a value met, inside
///   the call, which dies at its selector;
/// - `PrivateResourceOption`: a resource made with `MTLResourceOptions` bit 24, a private option
///   Metal takes, which the runtime makes as without it: on Metal a no-copy buffer made with it
///   keeps the GPU's writes from the CPU (the runtime's probe `nc_bit24`), where the runtime's
///   reach the CPU. A device can hold the GPU's writes in memory of its own, apart from the
///   guest's pages, so nothing in the hardware forces the difference, and it dies on every
///   device. Announced where the resource is made, inside the creator, which dies at its
///   selector;
/// - `StaleDetachedReach`: a command buffer that bound a heap set, whose commands may reach a
///   texture through its ID without naming it, submitted while the texture's bytes were
///   detached after a submitted command buffer made them resident: a read that does not name
///   it reads zeros, where Metal's reads the pages the guest gave back. Those pages hold what
///   they held until the guest writes them again, so Metal's answer is one value, and no fact of
///   the hardware forces the difference, so it dies on every device. Announced once per device,
///   at the first such submission, inside its call, which dies at its selector;
/// - `CrossClassTextureNeutral`: a draw or dispatch binding, at a texture slot declared of
///   another numeric class (`float` or `half`, `int`, `uint`) than its texture's format, one the
///   runtime has no exact answer for (a storage slot, a texture buffer, a multisampled, sparse,
///   depth, extended-range or YCbCr texture, or a read of a `short`, `ushort` or `ulong`
///   result), whose slot reads its neutral texture, zeros, where Metal reads the texture in its
///   format's own class and hands back its words as the declared type. A sampled colour texture
///   read with `float`, `half` or 32-bit integer results, at a slot or held by an argument buffer,
///   the runtime answers as Metal does and announces nothing (`rgbx`'s
///   `cross_class_samples_are_metals`, `mistyped_reads_are_metals`). Only a guest
///   shader that mistypes a texture reaches it, and no boot or suite case is known to make one,
///   so nothing grounds a relay. Announced at the draw or dispatch that binds it, inside its
///   call, which dies at its selector;
/// - `IndirectNilBufferOffset`: an execution of indirect commands the runtime replays on the
///   host, of a command that holds a nil buffer set at an offset whose low 32 bits are not 0,
///   which Metal's function sees as the buffer's address, where the replay binds no buffer, a
///   null address (an execution the GPU replays passes the bits, as Metal does). Only the address
///   differs, but a function may read through it or keep it, and no boot or suite case is known to
///   set such a buffer, so nothing grounds a relay. Announced where the runtime replays the
///   execution, for records the GPU may write or a range a buffer holds: at the host step, on its
///   scheduler thread alone, outside every call here, which dies there and ends the process, as
///   `ReleasedCommandState` does;
/// - `CompressedTexelDecode`: a texture of a BC1 to BC5 format (the sRGB forms too) on a device
///   whose driver is not measured to decode their blocks as Metal does, which keeps no decoded
///   twin of Metal's texels: one over the guest's pages in a heap, laid out apart from a heap's
///   bytes, memoryless or sparse, over a surface, a view of another decode than its texture's
///   twin or a shadowed view, or one whose twin's format the device cannot sample, filter and
///   copy into. Its reads, samples and gathers answer the device's decode,
///   where Metal interpolates with 6-bit weights (NVIDIA 615.71.09: up to 3/255 off for BC1 to
///   BC3, 7.3/255 for BC4 and BC5). MoltenVK samples the blocks through Metal itself, and the
///   runtime holds its decode as Metal's on every Apple GPU, so no device on Metal announces it.
///   A texture with a twin reads Metal's texels and announces `decoded-block-twin`, a
///   performance kind, relayed, and `cpu-block-decode`, another,
///   where a CPU copy into it (`replaceRegion:`) while a command buffer that may read it is
///   submitted and not seen complete is decoded on the CPU into the twin before the call
///   returns, so that the command buffer samples the copied texels, as Metal's does
///   (`WorkCounts::cpu_block_decodes` counts them). The hardware allows an exact answer, a decode
///   of the raw blocks in the shader by Metal's rules, which the runtime has not built; a native
///   device relays the device's decode, its sampling, as the class of a native device's own
///   maths and sampling ([`native::NATIVE_RELAYED`]). Announced where the texture or view is made,
///   inside its creator's call, which dies at its selector elsewhere;
/// - `HeapTextureLargerThanMetal`: a texture sized or placed in a heap whose image on the device
///   takes more bytes or a coarser alignment than the runtime's model of Metal's answer, not one
///   `OptimizedDepthStencilTexture` covers (on NVIDIA 615.71.09, a 4x4
///   `Depth32Float` texture takes 1024 bytes, where Metal answers 128). The runtime answers the
///   device's size and alignment, so the guest's heap size query differs from Metal's; the range
///   of Metal's space a texture of memory of its own takes is of the runtime's model of Metal's
///   answer ([`crate::gpu::metal_space::own_heap_size`]), which asks no image, so it is Metal's.
///   A native device relays it, on the hardware's ground: the device's memory granularity decides
///   it, and the guest sizes and places its heaps by the same answers, so a heap it sizes by them
///   is larger, and the buffers and heaps made after it can lie at other gpuAddresses than on Metal
///   ([`native::NATIVE_RELAYED`]). Announced
///   inside the call that sizes or places the texture (`heapTextureSizeAndAlignWithDescriptor:`,
///   a heap's `newTextureWithDescriptor:offset:`, and `newTextureWithDescriptor:` of a texture
///   the model has no answer for), which dies at its selector;
/// - `HostDeallocatorDelayed`: a no-copy buffer or heap made with a deallocator and no keep-alive,
///   released while the device's import of its pages is still held (a command buffer submitted
///   before a texture over them was released may still read that texture through its ID), whose
///   deallocator the runtime runs once that command buffer completed, where Metal's runs at the
///   release. Core's deallocator is what the guest and the host see of the
///   release (core's mapping's release, an IOSurface's), so its moment is Metal's. Never
///   announced here: every no-copy buffer, heap and rebind this platform makes gives the runtime
///   a keep-alive ([`crate::gpu::backing::keep_alive`]), with which it runs the deallocator at the
///   release. Announced once per device, on the thread that lets go of the buffer last, which may
///   be one of the runtime's as a command buffer completes, outside every call here: it dies
///   where it is announced;
/// - three kinds no device on Metal announces, each dying where a device announces it, at its
///   selector, as no fact of the hardware forces it:
///   - the note `DevicePointSize`: a vertex function's `[[point_size]]`, or a mesh function's
///     `set_point_size`, on a device whose `pointSizeRange` ends below Metal's largest point,
///     511 pixels (255 on ANV, 256 on lavapipe), which draws a point between its bound and 511
///     pixels smaller than Metal's. The device's limit is on its point primitives alone: drawing
///     such a point as two triangles is the exact form, which the runtime has not built.
///     Announced at each draw that binds the pipeline's variant, inside its call;
///   - the note `ClippedCoverageSplit`: a multisampled pipeline's fragment function that reads
///     `[[sample_mask]]` on a device whose rasterizer is not Apple's, which hands a pixel's
///     fragment the whole primitive's coverage where Apple's GPU runs it once per piece its
///     clipper splits a triangle past the viewport into, each reading that piece's coverage.
///     Announced at each draw that binds the pipeline's variant, inside
///     its call;
///   - `MetalLevelNotMacOS`: a device made off macOS with no Metal level named, which answers as
///     the runtime's newest measured release, where Metal's answers differ by release. This
///     platform names one on every OS but macOS ([`crate::os::METAL_LEVEL`]: on Linux macOS 26.7),
///     so the runtime announces it for no device here. Announced as the device is made, inside
///     the creator's call;
/// - the note `DeviceNanEncoding`: a NaN whose bits the device decides, where Metal answers its
///   own: the canonical `0x7fc00000` for its float arithmetic and functions, a 16-bit float
///   texel's own NaN for its read, sample, gather or pixel fetch and the canonical one for a
///   32-bit texel's, and the half's own bits for a `half` written to a 16-bit float texel.
///   MoltenVK answers the canonical NaN for a 16-bit texel's read and a 32-bit texel's own. A
///   written `half` keeps its bits through a 16-bit float texel's unsigned integer view, but
///   where a heap's handle reaches a texture that view cannot hold (a
///   swizzled one, or a wrapped image made without `MUTABLE_FORMAT`) and in a packed small float
///   texel (`RG11B10Float`, `RGB9E5Float`), where the device stores its own conversion of the
///   NaN; another device may answer another sign or payload for its arithmetic (llvmpipe:
///   `0xffc00000`). The hardware allows Metal's bits, the function selecting them itself, as the
///   runtime does for its own functions and a 32-bit texel's write, so it dies on every
///   device. Announced at each draw or dispatch that binds the pipeline's variant (where it binds
///   a texture whose format may hold a NaN), inside its call, which dies at its selector;
/// - the note `DeviceArithmetic`: a `float` division, remainder, square root, fused multiply-add
///   or dot product at Vulkan's precision bounds (a division within 2.5 ulp, an `Fma` that may run
///   unfused) where Metal answers the correctly rounded value, and a fast division's reciprocal,
///   which Metal rounds within about 0.64 ulp, as the device's or as `RN(1 / y)`, an ulp apart
///   near rounding midpoints. The device's integer and `float`
///   operations can compute Metal's values in the function, as the runtime already does for
///   `sqrt` and `fmod`, so nothing in the hardware forces the difference, and it dies on MoltenVK;
///   a native device relays it, its maths, as the class of a native device's own maths and
///   sampling ([`native::NATIVE_RELAYED`]). Not announced where the device answers as Metal's own
///   function (MoltenVK, under fast math). Announced at each draw or dispatch that binds the
///   pipeline's variant, inside its call, which dies at its selector;
/// - the note `DeviceDenormals`: an `f32` denormal the device keeps or flushes by its own float
///   controls, where Metal reads a denormal operand of its float arithmetic, comparisons and
///   functions as the zero of its sign and flushes a denormal result to one: at a remainder, a
///   conversion to or from `double`, `dot`'s fused sum and a function the translation does not
///   compute itself. An exponent-field select gives Metal's reading on every device, as the runtime
///   already does for `+ - *`, the comparisons, the fused multiply-adds, `ldexp`, `sqrt`, `floor`
///   and `ceil`, so nothing in the hardware forces the difference, and it dies on MoltenVK; a
///   native device relays it, its maths, as the class of a native device's own maths and sampling
///   ([`native::NATIVE_RELAYED`]). Not announced where the device answers as Metal's own function
///   (MoltenVK, under fast math). Announced at each draw or dispatch that binds the pipeline's
///   variant, inside its call, which dies at its selector;
/// - the note `HalfTexelConversion`: a read, gather or sample with a `half` result of a texture
///   bound in a format the translation does not convert as Metal does (an sRGB format otherwise
///   than its 8-bit slot, a packed or depth format and the rest) or of an argument buffer's
///   element, which keeps the runtime's rounding of the device's `f32` texel to nearest even,
///   where Metal rounds a 32-bit float texel toward zero and an 8- or 16-bit normalized one onto
///   its format's grid first. The function can convert by Metal's
///   rule, as the runtime already does for the formats it measured, so nothing in the hardware
///   forces the difference, and it dies on MoltenVK; a native device relays it, its sampling, as
///   the class of a native device's own maths and sampling ([`native::NATIVE_RELAYED`]). Announced
///   at each draw or dispatch that binds the pipeline's variant, inside its call, which dies at its
///   selector;
/// - the note `BicubicFilterArithmetic`: Metal's bicubic filter, emulated by fetches, whose sums
///   differ from Metal's where the runtime has not pinned Metal's rule (a 2D normalized sum near
///   a rounding tie, one grid step apart; an argument buffer's sRGB element; the formats no case
///   reads; a denormal tap and the sign of a zero sum; a coordinate past `2^23` texels). The
///   fetches can compute Metal's sums once the rule is read, so
///   nothing in the hardware forces the difference, and it dies on MoltenVK; a native device
///   relays it, its sampling, as the class of a native device's own maths and sampling
///   ([`native::NATIVE_RELAYED`]). Announced at each draw or dispatch that binds the pipeline's
///   variant, wherever the function samples through a bicubic filter, whatever the format, inside
///   its call, which dies at its selector;
/// - the notes `TessellationWinding` and `TessellationFactorEdgeCases`: a tessellation
///   pipeline's output winding, declared as MoltenVK maps Metal's under Vulkan's upper-left
///   domain origin, where Metal's mapping is not read, and the other one reverses the facing of
///   every generated triangle; and a subnormal edge factor that does not cull and a NaN inside
///   factor at the range's lower end, which Metal answers so under the `integer` partition and is
///   not read under the others. Metal's answers can be
///   read and taken, so nothing in the hardware forces the difference, and each dies on every
///   device. Announced at each draw that binds the pipeline's variant, inside its call, which
///   dies at its selector;
/// - the note `ComputedSampleDepth`: a fragment function that runs its draw's depth test itself
///   at more than one sample, on a device whose `FragCoord.z` is the pixel centre's there
///   (MoltenVK), computes each sample's depth from its vertex function's window depth, which
///   the runtime measured on the M3 Max and the M4 exact in 10 of 17 depth cases and otherwise up
///   to 2 ulp from the depth Metal stores, so a depth test between surfaces
///   that close may decide otherwise. The runtime defers an exact path, and no hardware fact
///   forces the difference, so it dies. Announced at each draw that binds the pipeline's
///   variant, inside its call, which dies at its selector;
/// - the note `HelperBranchDropsSimdgroupStores`: on MoltenVK on an Apple GPU, a fragment
///   function where only helper invocations may run an operation SPIRV-Cross writes as an
///   `optnone` call before a storage write, which the runtime cannot keep them out of exactly,
///   where Apple's compiler may drop the texture writes of the live threads of such a SIMD group
///   (a branch every invocation of the draw takes alike leaves no helper alone and announces
///   nothing). The runtime can keep the helpers away from that call by
///   another form of the function, and defers it, so nothing in the hardware forces the
///   difference, and it dies. Announced at each draw that binds the pipeline's variant, inside
///   its call, which dies at its selector;
/// - `ExtendedRangeBlendsStoredCodes`: a draw that blends into, or reads, an extended-range
///   colour attachment an earlier draw of its pass may have written, or whose `Clear` left a
///   colour between two codes, and an execution of render commands in a pass with such an
///   attachment, where the runtime holds no `half` shadow of the colour: Metal's tile memory
///   holds the colour as a `half` through the pass and rounds it to its codes once, at the store,
///   where the runtime stores the codes at each draw, a value a code or two off that can grow
///   over draws (measured on an M4 and the M3 Max). The runtime holds the
///   shadow, exactly, through a pass of one sample and one layer without a rate map on a device
///   that fetches in the pixel interlock, announcing nothing; where it does
///   not yet (another pass, an execution of render commands or a tile dispatch before the draw,
///   an earlier render pass instance of the pass, a pipeline with no shadowed form) nothing in
///   the hardware forces the difference, and it dies on every device. Announced once per
///   device, at the first such draw or execution, inside its call, which dies at its selector;
/// - `RotatedLoadTiles`: a pass that loads a texture of rotation 4 and draws, whose 32 by 32
///   tiles no draw reaches the runtime lands mirrored as the others, where Metal keeps them in
///   place ([`crate::gpu::desc`]). That difference is one value on each side: Metal's M3 Max
///   left a block of a reached tile in place in two of 88 passes, a different block each time,
///   which is not the tiles the kind names. The runtime can load the unreached tiles in place, so
///   nothing in the hardware forces the difference, and it dies on every device. Announced once
///   per device, at the first such pass, inside its call, which dies at its selector;
/// - the notes of the runtime's native defaults on a GPU that is not Apple's, whose exact forms
///   are the device policy's opt-ins (`rt::DevicePolicy`), which this platform takes only where
///   `REIMS_VGPU_EXACT` names them ([`crate::platform`]'s `exact_policy`), each recording its
///   bound: the device's maths (`NativeTranscendentals`, `NativeDivision`,
///   `NativeFma`, `NativeNanEncoding`, `NativeMinMax`, `NativeDot` and `NativeMix`), its rounding
///   of a `half` result (`NativeHalfConversion` and `NativeHalfQuantize`) and its sampling
///   (`DeviceSampling` and `DeviceTexelConversion`). On a device that runs on Metal (MoltenVK) the
///   runtime answers as Metal's own functions, sampler and conversions, so none is announced
///   there; a native device relays each as the class of a native device's own maths and
///   sampling ([`native::NATIVE_RELAYED`]). Announced at each draw or dispatch that binds the
///   pipeline's variant, inside its call, which dies at its selector.
///
/// Every kind not listed here or in [`HEARD`] is relayed, those the guest sees included, each for
/// one of the reasons the module's docs give: logged once per device and not dying. Each relayed
/// kind is reviewed, its reason named in
/// the test `every_degradation_kind_is_placed_once`, which a kind the runtime adds fails until it
/// is placed; a kind whose answer the runtime states is Metal's (`rt::Exactness::Exact`: a
/// performance or validity kind) or one of Metal's where Metal's own is not one value
/// (`rt::Exactness::OneOfMetals`) is placed by that statement.
pub const DIES: [rt::Degradation; 109] = [
    rt::Degradation::ReleasedCommandState,
    rt::Degradation::TwoPlaneTextureCopy,
    rt::Degradation::TwoPlaneBufferBlit,
    rt::Degradation::YCbCrFilterNotExact,
    rt::Degradation::RotatedBlendRounding,
    rt::Degradation::EncodedUseRebound,
    rt::Degradation::ZeroLayerMapAfterRelease,
    rt::Degradation::HeapTextureBlockCompressedLayout,
    rt::Degradation::BiasedDepthPastUnitRange,
    rt::Degradation::UnormDepthBias,
    rt::Degradation::UnormDepthRounding,
    rt::Degradation::RestrictedDepthUpload,
    rt::Degradation::RestrictedDepthRange,
    rt::Degradation::UnmeasuredResolve,
    rt::Degradation::MoltenVkArgumentBuffersUnread,
    rt::Degradation::BeyondMeasuredDeviation,
    rt::Degradation::FunctionStorageAssumed,
    rt::Degradation::HalfSampleFilter,
    rt::Degradation::SrgbTexelDecode,
    rt::Degradation::FloatSampleFilter,
    rt::Degradation::DeviceGatherFootprint,
    rt::Degradation::CubeFaceReciprocal,
    rt::Degradation::DeviceDepthComparison,
    rt::Degradation::DeviceTexelDenormals,
    rt::Degradation::AnisotropicFilter,
    rt::Degradation::DeviceAnisotropicFilter,
    rt::Degradation::ViewBaseTexelOffset,
    rt::Degradation::DeviceInterpolation,
    rt::Degradation::QuadsSplitByClipping,
    rt::Degradation::VertexSimdgroupGrouping,
    rt::Degradation::FragmentSimdgroupGrouping,
    rt::Degradation::NonStrictResidency,
    rt::Degradation::SparseResidency,
    rt::Degradation::DeviceRayIntersection,
    rt::Degradation::UnorderedFramebufferFetch,
    rt::Degradation::PointGuardBand,
    rt::Degradation::NanDepth,
    rt::Degradation::ClippedDepthPieces,
    rt::Degradation::SampleDepthClamp,
    rt::Degradation::DepthBiasPastOne,
    rt::Degradation::InPlaceVertexReads,
    rt::Degradation::FragmentsBeforeGeometry,
    rt::Degradation::MipmapsWithinSpread,
    rt::Degradation::HeapDepthStencilApart,
    rt::Degradation::HeapTextureHidesAliasedWrites,
    rt::Degradation::ResourceIdTakeover,
    rt::Degradation::UnreadableHandleTexture,
    rt::Degradation::UnnamedTextureUnviewable,
    rt::Degradation::UnnamedTexturesPastCapacity,
    rt::Degradation::HeapSetCopied,
    rt::Degradation::DetachedStorageKeepsWrites,
    rt::Degradation::MixedDepthStencilResolve,
    rt::Degradation::UncoveredColorKept,
    rt::Degradation::CappedAnisotropy,
    rt::Degradation::CombinedArchiveStages,
    rt::Degradation::ImplementationDefinedReconvergence,
    rt::Degradation::F16FmaDoubleRounding,
    rt::Degradation::F16IntConvDoubleRounding,
    rt::Degradation::GradFromBias,
    rt::Degradation::ReductionOrder,
    rt::Degradation::DeviceSignedZeroInfNan,
    rt::Degradation::TessellationFactorScale,
    rt::Degradation::TessellationFactorInstance,
    rt::Degradation::Fp8Value,
    rt::Degradation::MultisamplePosition,
    rt::Degradation::CubeGradientFootprint,
    rt::Degradation::QuadsCountedBeforeDiscard,
    rt::Degradation::BlendedWriteRounding,
    rt::Degradation::LinkedSamplerNumbering,
    rt::Degradation::PixelSamplerScaling,
    rt::Degradation::ArgumentPixelSamplerScaling,
    rt::Degradation::DiscardDropsSimdgroupStores,
    rt::Degradation::FreshPatternZeroed,
    rt::Degradation::ExtendedRangeFloatBlendSource,
    rt::Degradation::CrossClassTextureNeutral,
    rt::Degradation::IndirectNilBufferOffset,
    rt::Degradation::CompressedTexelDecode,
    rt::Degradation::HeapTextureLargerThanMetal,
    rt::Degradation::HostDeallocatorDelayed,
    rt::Degradation::DevicePointSize,
    rt::Degradation::ClippedCoverageSplit,
    rt::Degradation::MetalLevelNotMacOS,
    rt::Degradation::DeviceNanEncoding,
    rt::Degradation::DeviceArithmetic,
    rt::Degradation::DeviceDenormals,
    rt::Degradation::HalfTexelConversion,
    rt::Degradation::BicubicFilterArithmetic,
    rt::Degradation::TessellationWinding,
    rt::Degradation::TessellationFactorEdgeCases,
    rt::Degradation::ComputedSampleDepth,
    rt::Degradation::HelperBranchDropsSimdgroupStores,
    rt::Degradation::ExtendedRangeBlendsStoredCodes,
    rt::Degradation::ExtendedRangeBlendSource,
    rt::Degradation::MipmapsAsAppleM4,
    rt::Degradation::MipmapTieRounding,
    rt::Degradation::PrivateResourceOption,
    rt::Degradation::StaleDetachedReach,
    rt::Degradation::RotatedLoadTiles,
    rt::Degradation::NativeTranscendentals,
    rt::Degradation::NativeDivision,
    rt::Degradation::NativeFma,
    rt::Degradation::NativeNanEncoding,
    rt::Degradation::NativeMinMax,
    rt::Degradation::NativeDot,
    rt::Degradation::NativeMix,
    rt::Degradation::NativeHalfConversion,
    rt::Degradation::NativeHalfQuantize,
    rt::Degradation::DeviceSampling,
    rt::Degradation::DeviceTexelConversion,
];

/// The degradations a creator here learns of, announced at every occurrence too, which do not
/// die ([`hearing`]):
/// - `HeapTwoPlaneTexture`, a heap's two-plane XR10 texture whose texels lie in memory of their
///   own, apart from the heap's bytes (a mipmapped one, an array, any but a 2D texture of one
///   level, slice and sample, or one past the device's limits), announced on the caller's thread
///   inside the placement's call, which dies where a buffer or texture of the heap that lives
///   shares a byte with it: the heap's cover, the buffer over all of it core makes with each
///   heap, included, so in every heap core makes ([`crate::gpu::resource::Placements`]);
/// - `MetalLevelBelowHost`, a device made on a macOS host whose release the runtime has not
///   measured, above a release it has, which answers as the newest measured release below the
///   host's (`rt::DeviceProperties::metal_level`). Metal's answers differ by release only at a
///   few formats: which values a texture descriptor's setter takes, which formats Metal
///   describes, and what follows from that (a view's nil, a shared texture's nil, a texture of
///   such a format and its heap size, a state's attachment of one, a format's description). The
///   device is made and the notice relayed; each answer that depends on the level dies where it
///   is asked, naming it (`crate::platform::answering_level`), as the host's Metal may answer
///   otherwise. Announced once per device as the runtime makes it, on the thread that asks for
///   it, inside the device creator here (`MTLCreateSystemDefaultDevice()`,
///   `MTLCopyAllDevices()`), which keeps it. A host below every measured release, or whose
///   release does not read, still dies at the creator (`Problem::HostReleaseUnmeasured`). The
///   measured hosts' releases are measured, so the runtime announces nothing there;
/// - `DeviceRoundedResolve`, a pass's colour resolve the device averages and rounds: one of a
///   format without the runtime's own resolve rules, where MoltenVK's average of those with
///   rules is measured to be Metal's. Announced as the pass ends, inside `endEncoding`, which
///   hears it and dies at its selector unless every attachment the device may round is of a
///   format and a sample count measured to resolve as Metal's here (`R16Float`, `RG16Float` and
///   `RGBA16Float` at 2 and 4 samples, `crate::gpu::encode_render`'s `device_rounded`). Relayed
///   the first time alone ([`RELAYED_ONCE`]);
/// - `OptimizedDepthStencilTexture`, a depth or stencil texture without
///   `allowGPUOptimizedContents`, whose layout MoltenVK keeps GPU-optimized: the runtime sizes it
///   as Metal's texture with the property on, larger than Metal's with it off (16512 bytes for a 64
///   by 64 `Depth32Float` texture, where Metal answers 16384). Announced where such a texture is
///   sized or made. Its reads and writes are Metal's, so its creation relays it, a device's
///   texture's, whose range or slot of Metal's space is of Metal's size, the runtime's model of
///   Metal's answer, which asks no image and hears nothing, else made of the runtime's answers for
///   colour formats ([`crate::gpu::metal_space::own_heap_size`]), and a heap's placement's that
///   fits it. Where the model has none, a device's texture of a descriptor Metal's size is not
///   measured for (another storage than `Private` or `Shared`; one of `ShaderAtomic` usage the
///   runtime refuses first, as Metal asserts) dies at `newTextureWithDescriptor:`, at every shape;
///   an `X32_Stencil8` texture takes the model's size, a `Stencil8` one's of its shape, and dies
///   where the model has none. `heapTextureSizeAndAlignWithDescriptor:` hears it and answers
///   Metal's, the runtime's model of it first, as the device's texture takes it, an `X32_Stencil8`
///   one's among them, else made of the runtime's answers for colour formats
///   ([`crate::gpu::metal_space::metal_size_and_align`]), and dies at its selector where neither
///   answers ([`crate::gpu::device`]). A placement that depends on the runtime's size dies: a
///   heap's placement that hears it dies where it answers nil inside the heap and Metal's smaller
///   size fits there or is not measured (where Metal's does not fit either, the nil is Metal's),
///   where it is made but Metal answers nil or a size the runtime's falls short of, and in a heap
///   over no address ranges at every shape (`crate::gpu::resource::HeapObject`'s placement).
///   Elsewhere the heap keeps the texture apart: a placement over a byte of it dies while it lives,
///   and so does such a texture placed over a byte of a buffer or texture that lives, or while its
///   heap's cover lives, the buffer over all of the heap's bytes core makes with each heap, whose
///   bytes the guest reads with no call to die at (`crate::gpu::resource::Placements`). A heap's
///   `usedSize`, `currentAllocatedSize` and `maxAvailableSize`, which would answer it too, are no
///   answer core asks for. Relayed the first time alone ([`RELAYED_ONCE`]);
/// - `RasterizationRateMap`, a rasterization rate map whose rates vary, which the runtime compiles
///   to other than the identity: a pass over it draws by regions whose lines are its tiles' exact
///   lines, where Apple's are its fixed-point ones. Its parameter data and coordinate maps are
///   Metal's, and so are a pass's loads, clears and stores over it. Announced as the runtime
///   compiles the map, inside `newRasterizationRateMapWithDescriptor:` or `resetUsingDescriptor:`,
///   which hear it ([`crate::gpu::rate_map`]): a draw or an execution of indirect commands that is
///   not empty, and a tile dispatch, in a pass over such a map die at their selector
///   ([`crate::gpu::encode_render`]). Relayed the first time alone ([`RELAYED_ONCE`]);
/// - `HeapTextureUntiledLayout`, a texture in a heap over the guest's pages that the runtime does
///   not lay out in Metal's tiles: a multisample texture, a two-aspect depth and stencil format, a
///   PVRTC format, an image that holds other bytes than Metal's elements, or one past the device's
///   limits; and in any heap a multisample texture Metal keeps uncompressed whose image the runtime
///   makes in memory of its own, where the heap's memory cannot hold it (a `Shared` heap's depth or
///   stencil texture without `allowGPUOptimizedContents` on MoltenVK). Its reads, writes
///   and copies are Metal's, but its bytes in the heap are the runtime's layout, where Metal's are
///   its tiles, so a buffer or texture over them would meet other bytes than on Metal. Announced at
///   each such texture, inside `newTextureWithDescriptor:offset:`, which hears it. The heap keeps
///   it apart, as an optimized one: a buffer or texture placed over a byte of it while it lives
///   dies at its placement, and so does such a texture placed over a byte of a buffer or texture
///   that lives, or while its heap's cover lives, the buffer over all of the heap's bytes core
///   makes with each heap, so in every heap core makes: the cover's bytes and the guest's pages
///   under them would hold the runtime's layout where Metal's hold its tiles, which the guest reads
///   with no call to die at (`crate::gpu::resource::Placements`). Relayed the first time alone
///   ([`RELAYED_ONCE`]), where nothing shares its bytes;
/// - `HeapTextureLosslessLayout`, a texture in a heap over the guest's pages that Metal keeps
///   losslessly compressed, whose bytes on Metal are no layout of its texels: a single-sample one
///   laid out as the runtime's own, which guests make by the tens of thousands a boot, and a 2- or
///   4-sample one, its image in memory of its own, the heap's bytes keeping what they held, as
///   in any heap whose memory cannot hold its image (a `Shared` heap's depth or stencil texture
///   on MoltenVK). Its reads, writes, resolves and copies are Metal's:
///   only another resource over its bytes meets other bytes. Announced at each such texture, inside
///   `newTextureWithDescriptor:offset:`, which hears it. A single-sample one is made and the notice
///   relayed, the first time alone ([`RELAYED_ONCE`]), pending a decision. A multisample one
///   dies there, named: a buffer or texture over its bytes, which
///   on Metal sees and changes them, would neither see nor change its image;
/// - `UnmeasuredHeapTextureSize`, a texture whose layout in Metal's heaps the runtime does not
///   model, whose size and alignment it answers as its device's image's, larger or smaller than
///   Metal's: at the pinned runtime a PVRTC format. The runtime names 8 samples too, but no
///   texture of 8 samples reaches it: the runtime takes Apple's sample counts alone (1, 2 and 4)
///   on every device, a native one included, so its descriptor check refuses 8 at every creator
///   and the size query first (`SampleCount`), which dies there, as Metal stops them on the M3
///   Max and the M4 (the runtime's measurements). Announced where such a
///   texture is sized or placed,
///   inside `heapTextureSizeAndAlignWithDescriptor:`, a heap's `newTextureWithDescriptor:offset:`
///   and `newTextureWithDescriptor:`, whose texture of memory of its own takes a range of Metal's
///   space of the runtime's size ([`crate::gpu::metal_space::own_heap_size`]). Each hears it and
///   dies at its selector, naming the kind and the texture
///   ([`crate::gpu::metal_space::unmeasured_size`]), every PVRTC texture of memory of its own
///   among them, at every shape, but for a heap's nil at or past its end, Metal's whatever the
///   size. Relayed the first time alone ([`RELAYED_ONCE`]) where nothing takes the size;
/// - `HalfOutputPrecision`, a fragment function's `half` colour output, of a module compiled
///   without fast math or naming no math mode (one compiled with fast math notes
///   `HalfOutputFastMathBlend` in its place, relayed in the accepted class of fast-math
///   rewrites), which the runtime rounds to `half` before the blend and the
///   format's conversion, where Metal may carry it at `float` precision; a module without fast
///   math notes it only where an attachment the function writes blends, and
///   not for a `half4` output whose 16-bit float blend the function computes as Metal's under the
///   opt-in `HalfBlendPolicy::Exact`. Announced at each draw that binds such a state,
///   inside its call, which hears it
///   and relays it as [`Inexact::HalfOutputBlend`] (in strict mode dies at its selector) where an
///   attachment the function writes stores a value past the
///   bound the rounding is relayed within: a 16-bit float
///   attachment blended otherwise than by `One`, `Zero` and `Add` or by `Min` or `Max`, a 32-bit
///   float one blended by a factor that reads the source beside one neither `Zero` nor `One`, and
///   an 8-bit `_sRGB` one blended by `DestinationColor` and `OneMinusDestinationColor`
///   (`crate::gpu::encode_render`'s `half_apart`, each measured against Metal on the M3 Max).
///   Elsewhere the draw relays it: within one code of a unorm, snorm or packed float format, a
///   16-bit float format's `half` unblended or so blended, and a 32-bit float format's value, as
///   measured. An execution's commands announce it at the commit, which hears nothing and relays
///   it. Relayed the first time alone ([`RELAYED_ONCE`]).
pub const HEARD: [rt::Degradation; 9] = [
    rt::Degradation::HeapTwoPlaneTexture,
    rt::Degradation::MetalLevelBelowHost,
    rt::Degradation::DeviceRoundedResolve,
    rt::Degradation::OptimizedDepthStencilTexture,
    rt::Degradation::RasterizationRateMap,
    rt::Degradation::HeapTextureUntiledLayout,
    rt::Degradation::HeapTextureLosslessLayout,
    rt::Degradation::UnmeasuredHeapTextureSize,
    rt::Degradation::HalfOutputPrecision,
];

/// The kinds of [`HEARD`] the runtime would announce once per device but for `announce_each`,
/// which asks for each occurrence only so a call here can hear it: relayed the first time alone,
/// as without a sink, so a guest's every pass does not log one ([`notice`]).
const RELAYED_ONCE: [rt::Degradation; 7] = [
    rt::Degradation::DeviceRoundedResolve,
    rt::Degradation::OptimizedDepthStencilTexture,
    rt::Degradation::RasterizationRateMap,
    rt::Degradation::HeapTextureUntiledLayout,
    rt::Degradation::HeapTextureLosslessLayout,
    rt::Degradation::UnmeasuredHeapTextureSize,
    rt::Degradation::HalfOutputPrecision,
];

/// The kinds of [`RELAYED_ONCE`] this process relayed.
static RELAYED_KINDS: reims_vgpu_core::sync::Mutex<Vec<rt::Degradation>> =
    reims_vgpu_core::sync::Mutex::new(Vec::new());

thread_local! {
    /// How many calls here this thread is inside.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// A notice of [`DIES`] the runtime announced on this thread inside a call here, which the
    /// call dies at once it returns.
    static PENDING: RefCell<Option<rt::DegradeNotice>> = const { RefCell::new(None) };
    /// The kinds of [`HEARD`] the runtime announced on this thread inside the innermost
    /// [`hearing`], where one runs.
    static HEARING: RefCell<Option<Vec<rt::Degradation>>> = const { RefCell::new(None) };
}

/// Whether this process dies at a difference from Metal's answer it could go on past: a
/// degradation of [`DIES`] or one of [`Inexact`] (the module's docs). `REIMS_VGPU_STRICT` set to
/// `1`, read once per process, turns it on; by default the process relays them and goes on. A
/// test of this crate runs strict, but where [`relaxed`] runs it on its thread.
pub fn strict() -> bool {
    #[cfg(test)]
    if let Some(strict) = TEST_STRICT.with(Cell::get) {
        return strict;
    }
    static STRICT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        cfg!(test) || std::env::var_os("REIMS_VGPU_STRICT").is_some_and(|v| v == "1")
    });
    *STRICT
}

#[cfg(test)]
thread_local! {
    /// The strictness a test runs its thread's calls under ([`relaxed`]), in place of the
    /// process's.
    static TEST_STRICT: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Runs `call` with the calls and notices on this thread under the default, not strict: a test's
/// of what the process does without `REIMS_VGPU_STRICT`.
#[cfg(test)]
pub(crate) fn relaxed<T>(call: impl FnOnce() -> T) -> T {
    /// Puts back the outer strictness on drop, an unwinding call's included.
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_STRICT.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(TEST_STRICT.with(|s| s.replace(Some(false))));
    call()
}

/// How many times this process relayed each kind of [`DIES`], in its order, outside strict mode.
static RELAYED_DIES: [std::sync::atomic::AtomicU64; DIES.len()] =
    [const { std::sync::atomic::AtomicU64::new(0) }; DIES.len()];

/// Counts an occurrence of `kind`, of [`DIES`], relayed: whether it is the process's first, whose
/// line is written.
fn count_relayed(kind: rt::Degradation) -> bool {
    use std::sync::atomic::Ordering;
    DIES.iter()
        .position(|&k| k == kind)
        .and_then(|at| RELAYED_DIES.get(at))
        .is_some_and(|c| c.fetch_add(1, Ordering::Relaxed) == 0)
}

/// Each kind of [`DIES`] this process relayed outside strict mode, with how many times.
pub fn relayed_counts() -> Vec<(rt::Degradation, u64)> {
    use std::sync::atomic::Ordering;
    (DIES.iter().zip(&RELAYED_DIES))
        .map(|(&k, c)| (k, c.load(Ordering::Relaxed)))
        .filter(|&(_, n)| n > 0)
        .collect()
}

/// Runs `call` on this thread, and the kinds of [`HEARD`] the runtime announced on this thread
/// inside it, in order, each as many times as announced.
pub fn hearing<T>(call: impl FnOnce() -> T) -> (T, Vec<rt::Degradation>) {
    /// Puts back the outer hearing on drop, an unwinding call's included.
    struct Restore(Option<Option<Vec<rt::Degradation>>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            if let Some(outer) = self.0.take() {
                HEARING.with(|h| *h.borrow_mut() = outer);
            }
        }
    }
    let mut restore = Restore(Some(HEARING.with(|h| h.replace(Some(Vec::new())))));
    let out = call();
    let heard = HEARING.with(|h| h.replace(restore.0.take().flatten()));
    (out, heard.unwrap_or_default())
}

/// Runs `call`, a runtime call standing for `sel` (none for [`plain`]'s), and dies at `sel`, or
/// at the notice's code, where the runtime announced a degradation of [`DIES`] inside it, or on
/// this thread before it ([`notice`]). A call that unwinds, a death inside it, lets its notice
/// go: the death that unwinds it is the call's, and the thread's next call here must not die of
/// a notice of this one.
fn inside<T>(sel: Option<&'static str>, call: impl FnOnce() -> T) -> T {
    /// Leaves the call on drop, an unwinding one included, whose notice it drops.
    struct Leave {
        /// Whether the call returned.
        returned: bool,
    }
    impl Drop for Leave {
        fn drop(&mut self) {
            DEPTH.with(|d| d.set(d.get() - 1));
            if !self.returned {
                PENDING.with(|p| drop(p.take()));
            }
        }
    }
    DEPTH.with(|d| d.set(d.get() + 1));
    let out = {
        let mut leave = Leave { returned: false };
        let out = call();
        leave.returned = true;
        out
    };
    if let Some(notice) = PENDING.with(RefCell::take) {
        let call = sel.unwrap_or(notice.degradation.code());
        if census::on() {
            census::selector(notice.degradation, call);
        } else {
            degraded(call, &notice);
        }
    }
    out
}

/// The kinds of [`DIES`] that die by default too, not only in strict mode, as the guest or the
/// host would otherwise go on past a break of what it relies on, which no log line repairs:
/// - `ReleasedCommandState`: the command runs nothing, where Metal's execution is undefined;
/// - `ResourceIdTakeover`: a function reads the guest's sampler ID as another sampler's, an
///   earlier holder's or the neutral one;
/// - `PrivateResourceOption`: the GPU's writes reach the guest's pages, which Metal's never do
///   for such a resource;
/// - `HostDeallocatorDelayed`: the deallocator, which the host sees as the release, runs after
///   the release, out of the host's release order.
pub const DIES_BY_DEFAULT: [rt::Degradation; 4] = [
    rt::Degradation::ReleasedCommandState,
    rt::Degradation::ResourceIdTakeover,
    rt::Degradation::PrivateResourceOption,
    rt::Degradation::HostDeallocatorDelayed,
];

/// What [`notice`] does with a degradation of [`DIES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fate {
    /// Strict mode ([`strict`]), or a kind of [`DIES_BY_DEFAULT`]: it dies at the call it was
    /// announced inside, or where it was announced outside every call.
    Dies,
    /// A native device relays it ([`native::NATIVE_RELAYED`]): counted, the first relayed.
    RelayedNatively,
    /// Census mode records it ([`census`]), with the call it was announced inside, and relays it.
    Counted,
    /// The default: counted, the first relayed ([`relayed_counts`]), and the process goes on.
    Relayed,
}

/// The fate of `kind`, a kind of [`DIES`], under `placement`, with census mode and strict mode
/// each on or off.
fn fate(kind: rt::Degradation, placement: native::Placement, census: bool, strict: bool) -> Fate {
    if placement == native::Placement::Native && native::relayed(kind) {
        Fate::RelayedNatively
    } else if census {
        Fate::Counted
    } else if strict || DIES_BY_DEFAULT.contains(&kind) {
        Fate::Dies
    } else {
        Fate::Relayed
    }
}

/// The runtime's message sink (`DeviceOptions::messages`, [`crate::platform`]): a degradation
/// of [`DIES`] is relayed and counted, or in strict mode kept for the call here it was announced
/// inside to die at, or, announced outside every call here, dies now, ending the process (the
/// module's docs), but where a native device relays it or census mode records it ([`Fate`]); one
/// of [`HEARD`] is kept for the
/// [`hearing`] it was announced inside; every other message, and every one of [`HEARD`] too,
/// goes to [`relay`], to stderr (a `Verbose` one of the runtime's own to the log at debug), but
/// a kind of [`RELAYED_ONCE`] after its first.
pub fn notice(message: &rt::DebugMessage) {
    let Some(notice) = &message.degradation else {
        relay(message);
        return;
    };
    if !DIES.contains(&notice.degradation) {
        if HEARD.contains(&notice.degradation) {
            HEARING.with(|h| {
                if let Some(heard) = h.borrow_mut().as_mut() {
                    heard.push(notice.degradation);
                }
            });
        }
        #[cfg(test)]
        RELAYED.lock().push(notice.degradation);
        if RELAYED_ONCE.contains(&notice.degradation) {
            let mut relayed = RELAYED_KINDS.lock();
            if relayed.contains(&notice.degradation) {
                return;
            }
            relayed.push(notice.degradation);
        }
        relay(message);
        return;
    }
    let kind = notice.degradation;
    match fate(kind, native::placement(), census::on(), strict()) {
        Fate::Dies => {}
        Fate::Relayed => {
            if count_relayed(kind) {
                relay(message);
            }
            return;
        }
        Fate::RelayedNatively => {
            census::record(kind, &notice.object, true);
            if native::count(kind) {
                relay(message);
            }
            return;
        }
        Fate::Counted => {
            if census::record(kind, &notice.object, false) {
                relay(message);
            }
            if DEPTH.with(Cell::get) == 0 {
                census::selector(kind, census::OUTSIDE);
            } else {
                PENDING.with(|p| *p.borrow_mut() = Some(notice.clone()));
            }
            return;
        }
    }
    // Kept first: the death below ends the process, but the test capture's unwinds into the
    // runtime, which catches it, and the thread then dies at its next call here.
    PENDING.with(|p| *p.borrow_mut() = Some(notice.clone()));
    if DEPTH.with(Cell::get) == 0 {
        degraded(notice.degradation.code(), notice);
    }
}

/// The degradations not of [`DIES`] the runtime announced in this process, in order, which
/// [`notice`] relays and a test reads.
#[cfg(test)]
pub(crate) static RELAYED: reims_vgpu_core::sync::Mutex<Vec<rt::Degradation>> =
    reims_vgpu_core::sync::Mutex::new(Vec::new());

/// What the id names of the runtime's own messages start with: its degradations'
/// (`metal2vulkan-degrade-`), its pipeline-creation log's (`metal2vulkan-pipeline-creation`) and
/// its other diagnostics' (`metal2vulkan-late-reservation`, `metal2vulkan-host-import-freed`, the
/// shader cache's `metal2vulkan-shader-cache-evicted` and
/// `metal2vulkan-shader-cache-write-failed`, each said once per device, and
/// `metal2vulkan-unmeasured-driver`, said as a device whose driver, GPU and release the runtime
/// has not measured is made).
const RUNTIME_MESSAGES: &str = "metal2vulkan-";

/// Writes `message`, which the runtime gave the sink, on stderr ([`relay_to`]).
fn relay(message: &rt::DebugMessage) {
    relay_to(
        &mut std::io::stderr(),
        message.severity,
        &message.id_name,
        &message.text,
    );
}

/// Writes `text`, a message of `severity` and id name `id_name` the runtime gave the sink, on `w`
/// as the runtime writes it on stderr without a sink, one line, unchanged, where it is one of the
/// runtime's own ([`RUNTIME_MESSAGES`]). One of its own of severity `Verbose`, which the runtime
/// gives a sink alone, goes to the log at debug instead, as the library's other diagnostics do:
/// the runtime's line for each host import freed on a device that keeps MoltenVK's residency set
/// (`metal2vulkan-host-import-freed`; this host's MoltenVK keeps it). A driver's or a layer's
/// (`VK_EXT_debug_utils`), which the runtime reports to a sink alone, is dropped, as without one.
fn relay_to(w: &mut impl std::io::Write, severity: rt::MessageSeverity, id_name: &str, text: &str) {
    if !id_name.starts_with(RUNTIME_MESSAGES) {
        return;
    }
    if severity == rt::MessageSeverity::Verbose {
        log::debug!("{text}");
        return;
    }
    let mut line = text.to_owned();
    line.push('\n');
    // One write, so the line never interleaves with another thread's.
    let _ = w.write_all(line.as_bytes());
}

/// The death at `call` of a degradation of [`DIES`], its line naming the notice.
fn degraded(call: &'static str, notice: &rt::DegradeNotice) -> ! {
    dies(
        call,
        &format!(
            "the Vulkan runtime degraded it ({}): {}: {}; {}",
            notice.degradation.code(),
            notice.object,
            notice.reason,
            notice.cost
        ),
    )
}

/// The runtime's refusal `e` of `sel`, decided: what Metal's validation stops (`Invalid`) by the
/// call's label, anything else, and anything at a call labelled `Dies`, a death here
/// ([`refused`]).
pub fn decide<T>(sel: &'static str, policy: Policy, e: Error) -> RaiseResult<T> {
    decide_describing(sel, policy, e, None)
}

/// [`decide`], the line of a death or a raise also carrying `descriptor`
/// ([`refusal_describing`]): what the refused call's descriptor holds that the runtime's error
/// does not say, a pipeline creator's formats (`gpu::states::formats`).
pub fn decide_describing<T>(
    sel: &'static str,
    policy: Policy,
    e: Error,
    descriptor: Option<&str>,
) -> RaiseResult<T> {
    let line = refusal_describing(&e, descriptor);
    match e {
        Error::Invalid(_) if policy != Policy::Dies => {
            policy.on().throw(Raise::platform(sel).with_detail(line))
        }
        _ => dies(sel, &line),
    }
}

/// The domain of every `NSError` this platform answers: Metal's is its driver's name, which
/// differs between GPU families, and only core's log line reads it
/// ([`crate::gpu::states`]' module docs).
pub const RUNTIME_DOMAIN: &str = "metal2vulkan-runtime";

/// The runtime's error as the `NSError` Metal hands back for a creator with an `error:`
/// out-parameter: the runtime's domain ([`RUNTIME_DOMAIN`]), no code of Metal's (0), and its
/// text.
pub fn error_text(e: &Error) -> NsErrorText {
    NsErrorText {
        domain: RUNTIME_DOMAIN.to_owned(),
        code: 0,
        description: e.to_string(),
    }
}

/// The death of a send the runtime refuses where Metal goes on: named by the selector, the line
/// carrying the runtime's error ([`refusal`]).
pub fn refused(sel: &'static str, e: &Error) -> ! {
    dies(sel, &refusal(e))
}

/// The line of the last death [`dies`] made in this process, which a test reads: the test
/// capture takes a death before its line is written.
#[cfg(test)]
pub(crate) static LAST_LINE: reims_vgpu_core::sync::Mutex<Option<String>> =
    reims_vgpu_core::sync::Mutex::new(None);

/// The death at `sel` of [`refused`], [`metal_aborts`] and [`unrepresentable`], its line
/// `line`.
fn dies(sel: &'static str, line: &str) -> ! {
    #[cfg(test)]
    {
        *LAST_LINE.lock() = Some(line.to_owned());
    }
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: sel,
        },
        line,
    )
}

/// The line of [`refused`]: the runtime's error, and where it is a command sampling a texture whose
/// address modes are mapped (one of the two-plane YCbCr family the runtime makes over a surface,
/// [`crate::gpu::ycbcr::runtime_makes`], whose modes it maps as Metal addresses the format:
/// `YCBCR8_420_2P`'s repeat and mirror modes read as clamp-to-edge), which of the paths the runtime
/// cannot map it is: a sampler slot shared with a texture of another map, sampling through
/// handles (an argument buffer's or a heap's), or a constexpr sampler whose modes the runtime
/// cannot map, which it maps where its modes are stated and the map sends each to one a constexpr
/// sampler addresses, and refuses otherwise; and where it is a refusal of SIMD-group operations
/// in a stage, the stage and the class ([`simd_stage`]); and where it is a refusal of host memory
/// after the import's dma-buf source made none, why it made none ([`host_memory_refusal`]).
pub fn refusal(e: &Error) -> String {
    use metal2vulkan_runtime::{MappedSampling, RuntimeGap};
    let path = match e {
        Error::NotImplemented(RuntimeGap::MappedTextureSampling { through, .. }) => match through {
            MappedSampling::ConstexprSampler => {
                Some("through a constexpr sampler whose address modes the map changes")
            }
            MappedSampling::SharedSamplerSlot => Some(
                "through a sampler slot that samples, in the same stage, a texture the map \
                 addresses otherwise",
            ),
            MappedSampling::Handles => Some(
                "by a function that samples through handles (an argument buffer's or a heap's) \
                 able to reach it",
            ),
            _ => Some("where the runtime cannot map its sampler"),
        },
        _ => None,
    };
    let named = path
        .map(|path| {
            format!(
                "a texture whose address modes are mapped to Metal's format's (a member of \
                 gpu::ycbcr's family) is sampled {path}"
            )
        })
        .or_else(|| simd_stage(e))
        .or_else(|| host_memory_refusal(e));
    match named {
        Some(named) => format!("the Vulkan runtime refused it: {e}; {named}"),
        None => format!("the Vulkan runtime refused it: {e}"),
    }
}

/// What a refusal of host memory names, for [`refusal`]: why the import's dma-buf source on this
/// thread made no dma-buf of its pages (`crate::os::host_memory`), such as udmabuf's limits, which
/// the device would have imported where it refuses the pages as host memory. `None` for any other
/// refusal, and where the source made one or was not asked.
fn host_memory_refusal(e: &Error) -> Option<String> {
    use metal2vulkan::error::Feature;
    match e {
        Error::Unsupported(u) if u.feature == Feature::HostMemory => crate::os::dma_buf_refusal(),
        _ => None,
    }
}

/// What a refusal of SIMD-group or quad-group operations names, for [`refusal`]: the stage the
/// runtime does not take them in on this device, and the class. Device info's key 33 answers
/// the SIMD reduction and shuffle-and-fill bits no where the runtime does not take every function
/// they promise in every stage ([`crate::gpu::caps`]), but a guest may still use the class in
/// such a stage, which Metal runs: it reaches this refusal where its pipeline state is made, and
/// dies there. The line keys on the refusal, never on the stage: where the runtime takes the
/// class in a stage, no refusal comes. `None` for any other refusal.
fn simd_stage(e: &Error) -> Option<String> {
    use metal2vulkan::device::VkProperty;
    use metal2vulkan::error::{Capability, Feature, Needs};
    use metal2vulkan::pipeline::Stage;
    let Error::Unsupported(u) = e else {
        return None;
    };
    let class = match u.feature {
        Feature::SubgroupOperations => "SIMD-group operations",
        Feature::QuadOperations => "quad-group operations",
        _ => return None,
    };
    let Needs::AnyOf(alternatives) = &u.needs else {
        return None;
    };
    let mut stage = None;
    let mut ops = Vec::new();
    for capability in alternatives.iter().flatten() {
        match capability {
            Capability::Property(VkProperty::SubgroupStage(s)) => stage = Some(*s),
            // Quad operations outside the fragment and kernel stages are a vertex function's.
            Capability::Property(VkProperty::SubgroupQuadOperationsInAllStages) => {
                stage = Some(Stage::Vertex)
            }
            Capability::Property(VkProperty::SubgroupOperation(op)) => ops.push(format!("{op:?}")),
            _ => {}
        }
    }
    let whose = match stage {
        Some(Stage::Vertex) => "a vertex function's",
        Some(Stage::Fragment) => "a fragment function's",
        Some(Stage::Kernel) => "a kernel's",
        Some(Stage::Object) => "an object function's",
        Some(Stage::Mesh) => "a mesh function's",
        Some(_) | None => "a function's",
    };
    let of = if ops.is_empty() {
        String::new()
    } else {
        format!(" (the classes {})", ops.join(", "))
    };
    Some(format!(
        "{whose} {class}{of}, which the runtime does not take in its stage on this device, \
         where Metal runs them (device info's key 33 answers no SIMD bits for a class some \
         stage lacks: gpu::caps)"
    ))
}

/// [`refusal`]'s line, then `descriptor`, what the refused call's descriptor holds, after a
/// semicolon.
pub fn refusal_describing(e: &Error, descriptor: Option<&str>) -> String {
    match descriptor {
        Some(descriptor) => format!("{}; {descriptor}", refusal(e)),
        None => refusal(e),
    }
}

/// The death of a method this platform has not built yet: its line names the selector.
pub fn unbuilt(sel: &'static str) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: sel,
        },
        "not built on the Vulkan platform",
    )
}

/// The death of a send Metal itself aborts at (an assertion inside Metal, whatever handler is
/// live), where the runtime would refuse or go on: `what` says what Metal stops.
pub fn metal_aborts(sel: &'static str, what: &str) -> ! {
    dies(sel, &format!("Metal aborts at {what}"))
}

/// The death of a send at which Metal reads `what` at a NULL or near-NULL address, in the zero
/// region every arm64 macOS executable maps (`__PAGEZERO`, below core's `ZERO_REGION_END`), where
/// the read ends the process with a segmentation fault.
pub fn metal_reads_near_null(sel: &'static str, what: &str) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::NullDeref,
            call: sel,
        },
        &format!("Metal reads {what} in the zero region (a segmentation fault)"),
    )
}

/// The death of a send whose raw value the runtime's types cannot hold: `what` names it (a pixel
/// format no `MTLPixelFormat` is, an option bit the runtime does not define). Metal itself aborts
/// for some of these (a pixel format it refuses); for the others the runtime has no form.
pub fn unrepresentable(sel: &'static str, what: &str) -> ! {
    dies(sel, &format!("the Vulkan runtime cannot hold {what}"))
}

/// A value of Metal's this platform does not follow, which no runtime notice announces: it dies
/// where the guest would read it ([`unfollowed`]), its line naming the kind's code. Each kind is
/// placed once: listed once in [`Unfollowed::ALL`], under a code of its own that is none of the
/// runtime's kinds' (`every_unfollowed_kind_is_placed_once`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unfollowed {
    /// The `uniqueIdentifier` of an indirect command buffer made after an encoder of any kind was
    /// made in the process (info command 460): Metal's address of the buffer then depends on
    /// what each encoder took of its range by its kind, its work and whether it was committed
    /// ([`crate::gpu::icb`]).
    IcbIdentifierAfterEncoder,
}

impl Unfollowed {
    /// Every kind, once.
    pub const ALL: [Self; 1] = [Self::IcbIdentifierAfterEncoder];

    /// The kind's code, which its death's line names.
    pub const fn code(self) -> &'static str {
        match self {
            Self::IcbIdentifierAfterEncoder => "icb-identifier-after-encoder",
        }
    }
}

/// The death at `sel` of a read of a value of kind `kind` this platform does not follow, `what`
/// saying what the guest would read.
pub fn unfollowed(sel: &'static str, kind: Unfollowed, what: &str) -> ! {
    dies(
        sel,
        &format!(
            "this platform does not follow Metal's answer here ({}): {what}",
            kind.code()
        ),
    )
}

/// A value this platform computes otherwise than Metal, which no runtime notice of [`DIES`]
/// announces: relayed by default, a death in strict mode ([`inexact`]). Each kind is placed
/// once, under a code of its own (`every_inexact_kind_is_placed_once`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inexact {
    /// A `half` colour output stored past the bound its rounding was measured within, as blended
    /// or converted at its attachment (`HalfOutputPrecision`, `crate::gpu::encode_render`'s
    /// `half_apart`).
    HalfOutputBlend,
    /// A colour resolve the device averages and rounds, of a format and sample count not
    /// measured to resolve as Metal's (`DeviceRoundedResolve`).
    RoundedResolve,
    /// An attachment cleared to a NaN, or a bfloat one to a value, the runtime stores otherwise
    /// than Metal.
    Clear,
    /// A bfloat attachment loaded and stored by a pass that draws, whose codes the runtime moves
    /// through `half` otherwise than Metal's tile store.
    BfloatLoaded,
    /// A write under `textureWriteRoundingMode` 2 that the runtime rounds by its default.
    WriteRounding,
    /// A heap texture's size, alignment or bytes in the heap the runtime answers or lays out as
    /// its device's image, where Metal's are not measured or are its own: a texture is made, and
    /// its size answered, by the runtime's.
    HeapLayout,
    /// An answer that depends on Metal's release, given at a release the rule was not measured at
    /// (a host's release the runtime has not measured, a format's description or view at a
    /// release other than macOS 26.7).
    MetalLevel,
    /// A draw at the last vertex provoking, under a state that reads a flat input, barycentrics
    /// or a layer or viewport index, which has no exact rotation: drawn as it stands, its first
    /// vertex provoking.
    ProvokingVertex,
    /// Lines at a line width other than 1: drawn 1 pixel wide.
    LineWidth,
    /// A draw or tile dispatch in a pass of a rasterization rate map whose rates vary, or a tile
    /// dispatch in one of no layers, which the runtime runs by its own regions and tiles.
    RateMap,
    /// A view of a two-plane YCbCr texture as another type, which the runtime reads as the
    /// texture where Metal reads the planes' codes unconverted.
    YcbcrView,
    /// A colour fill packed in a pixel format other than the texture's, whose bytes the runtime
    /// repeats where Metal's follow no rule measured.
    FillFormat,
    /// A late resolve texture beside a resolve the pass dropped, a shape Metal is not measured
    /// at: the dropped resolve stays dropped, and a depth or stencil resolve goes on.
    LateResolve,
    /// A texture of protection options other than none, made by `newTextureWithDescriptor:` or
    /// `newSharedTextureWithDescriptor:` as without them, as Metal's other creators make one
    /// (it answers protection options 0), where Metal's two answer nil to a process without the
    /// protected-content entitlement, QEMU among them ([`crate::gpu::desc`]). Nothing here holds
    /// protected memory, so its contents are not kept from the host.
    ProtectedContent,
}

impl Inexact {
    /// Every kind, once.
    pub const ALL: [Self; 14] = [
        Self::HalfOutputBlend,
        Self::RoundedResolve,
        Self::Clear,
        Self::BfloatLoaded,
        Self::WriteRounding,
        Self::HeapLayout,
        Self::MetalLevel,
        Self::ProvokingVertex,
        Self::LineWidth,
        Self::RateMap,
        Self::YcbcrView,
        Self::FillFormat,
        Self::LateResolve,
        Self::ProtectedContent,
    ];

    /// The kind's code, which its line names.
    pub const fn code(self) -> &'static str {
        match self {
            Self::HalfOutputBlend => "inexact-half-output-blend",
            Self::RoundedResolve => "inexact-rounded-resolve",
            Self::Clear => "inexact-clear",
            Self::BfloatLoaded => "inexact-bfloat-loaded",
            Self::WriteRounding => "inexact-write-rounding",
            Self::HeapLayout => "inexact-heap-layout",
            Self::MetalLevel => "inexact-metal-level",
            Self::ProvokingVertex => "inexact-provoking-vertex",
            Self::LineWidth => "inexact-line-width",
            Self::RateMap => "inexact-rate-map",
            Self::YcbcrView => "inexact-ycbcr-view",
            Self::FillFormat => "inexact-fill-format",
            Self::LateResolve => "inexact-late-resolve",
            Self::ProtectedContent => "inexact-protected-content",
        }
    }
}

/// How many times this process relayed each kind of [`Inexact`], in [`Inexact::ALL`]'s order.
static RELAYED_INEXACT: [std::sync::atomic::AtomicU64; Inexact::ALL.len()] =
    [const { std::sync::atomic::AtomicU64::new(0) }; Inexact::ALL.len()];

/// A value of kind `kind` at `sel` that the runtime computes otherwise than Metal, `what` saying
/// which: in strict mode the death of [`unrepresentable`]; by default relayed, the process's
/// first of the kind writing a line on stderr, and counted ([`inexact_counts`]).
pub fn inexact(sel: &'static str, kind: Inexact, what: &str) {
    use std::sync::atomic::Ordering;
    if strict() {
        unrepresentable(sel, what);
    }
    let at = Inexact::ALL.iter().position(|&k| k == kind);
    let first = (at.and_then(|at| RELAYED_INEXACT.get(at)))
        .is_some_and(|c| c.fetch_add(1, Ordering::Relaxed) == 0);
    if first {
        let line = format!(
            "reims-vgpu: {sel}: the Vulkan runtime cannot hold {what}; relayed ({}), the first of \
             its kind: REIMS_VGPU_STRICT=1 makes it a death\n",
            kind.code()
        );
        let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
    }
}

/// Each kind of [`Inexact`] this process relayed, with how many times.
pub fn inexact_counts() -> Vec<(Inexact, u64)> {
    use std::sync::atomic::Ordering;
    (Inexact::ALL.iter().zip(&RELAYED_INEXACT))
        .map(|(&k, c)| (k, c.load(Ordering::Relaxed)))
        .filter(|&(_, n)| n > 0)
        .collect()
}

/// Parks this thread for good, where Metal's call never returns: a wait Metal never returns
/// from ([`crate::gpu::command`]), and a reservation Metal spins in
/// ([`crate::gpu::icb`]).
pub fn hang() -> ! {
    loop {
        std::thread::park();
    }
}

/// Runs `call`, a test's, which must die of a degradation of `kind` ([`DIES`]): the site it
/// died at, its line naming the kind's code. A call that returns, or dies of another, fails the
/// test.
#[cfg(test)]
pub(crate) fn dies_of<T>(
    kind: rt::Degradation,
    call: impl FnOnce() -> T,
) -> reims_vgpu_core::raise::death::DeathSite {
    *LAST_LINE.lock() = None;
    let Err(site) = reims_vgpu_core::testkit::capture::catch_death(call) else {
        panic!("it did not die, where {} dies", kind.code());
    };
    let line = LAST_LINE.lock().take().unwrap_or_default();
    assert!(
        line.contains(&format!("({})", kind.code())),
        "it died of another than {}: {line}",
        kind.code()
    );
    site
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan_runtime::{Invalid, Problem, RuntimeGap};
    use reims_vgpu_core::raise::{Origin, RaiseId};
    use reims_vgpu_core::testkit::capture::catch_death;

    fn invalid() -> Error {
        Error::Invalid(Invalid::new("fill_buffer", Problem::StorageMode))
    }

    /// The runtime's own messages that are no degradation, its pipeline-creation log, its late
    /// reservation's warning and its shader cache's warnings of an entry evicted and one not
    /// written, reach stderr's writer as the runtime writes them without a sink, a line each,
    /// unchanged, and so does a degradation it announces; its own `Verbose` line of a host import
    /// freed goes to the log alone, and a layer's and a driver's are dropped
    #[test]
    fn the_runtimes_own_messages_are_relayed_unchanged() {
        use rt::MessageSeverity as S;
        let creation = "metal2vulkan: pipeline creation: state 3 \"blit\" pipeline 2: 4.1 ms";
        let late = "metal2vulkan: reserve() must be called before any allocation takes place";
        let degraded =
            "metal2vulkan: texture 7: its rows are copied; slower [metal2vulkan-degrade-x]";
        let freed = "metal2vulkan: a host import of 16384 bytes freed with 2 submissions in flight";
        let evicted = "metal2vulkan: the shader cache directory /c reached its cap of 26843545600";
        let unwritten = "metal2vulkan: an entry of the shader cache directory /c was not written";
        let mut out = Vec::new();
        for (severity, id_name, text) in [
            (S::Info, "metal2vulkan-pipeline-creation", creation),
            (
                S::Error,
                "VUID-RuntimeSpirv-None-06343",
                "Validation Error: [ VUID-RuntimeSpirv-None-06343 ]",
            ),
            (S::Warning, "metal2vulkan-late-reservation", late),
            (S::Verbose, "metal2vulkan-host-import-freed", freed),
            (S::Warning, "metal2vulkan-shader-cache-evicted", evicted),
            (
                S::Warning,
                "metal2vulkan-shader-cache-write-failed",
                unwritten,
            ),
            (S::Warning, "", "MoltenVK: a message of the driver's"),
            (
                S::Verbose,
                "",
                "MoltenVK: a verbose message of the driver's",
            ),
            (
                S::Warning,
                "metal2vulkan-degrade-shadowed-texture",
                degraded,
            ),
        ] {
            relay_to(&mut out, severity, id_name, text);
        }
        assert_eq!(
            String::from_utf8(out).unwrap(),
            format!("{creation}\n{late}\n{evicted}\n{unwritten}\n{degraded}\n")
        );
    }

    /// Every kind of degradation the runtime has is placed here once: it dies ([`DIES`]), a
    /// creator hears it ([`HEARD`]), or it is relayed, in one of the reviewed lists below. A kind
    /// the runtime adds fails this until it is placed, so none is relayed unreviewed
    #[test]
    fn every_degradation_kind_is_placed_once() {
        use rt::Degradation as D;
        // Metal's answer, at a cost Metal does not pay in time or memory (a performance kind), or
        // through a use of the device its driver does not advertise but runs as Metal does (a
        // validity kind, `UnadvertisedVertexSimdgroups` alone: a vertex function's simdgroup
        // functions): relayed, as the runtime states (`rt::Exactness::Exact`). Two performance
        // kinds are Metal's answer while a pool of the command buffer holds the work
        // (`rt::Exactness::ExactWithinPool`): `IndirectIndexBoundsPass` and
        // `EmulatedMeshDraws`, whose overflow fails the command buffer, which dies where it is
        // read (`crate::gpu::command`'s `pool_overflow` and `mesh_records_overflow`): the
        // difference dies where it shows.
        let exact = D::ALL.iter().copied().filter(|k| {
            matches!(
                k.exactness(),
                rt::Exactness::Exact | rt::Exactness::ExactWithinPool
            )
        });
        // One of Metal's answers where Metal's own is not one value, as the runtime states
        // (`rt::Exactness::OneOfMetals`), unless it dies here: relayed. Among them:
        // - `DontCareStore`: the texels a `DontCare` store leaves are the device's undefined ones
        //   here, where Apple's GPU keeps the earlier texels unless its tiler overflows, which no
        //   call sees coming. On MoltenVK 1.4.1 on the M3 Max, as on
        //   the M4, the runtime does not announce it (its `DontCareStoreIsMetals`):
        //   the device's stores there kept every kept case of its lane.
        // - `ZeroedIndexFetch`: an indexed draw from past its index buffer draws index 0 here,
        //   where Metal reads whatever memory follows the buffer.
        // - `RepeatedExecutionStart`: executions of indirect commands from a start an earlier one
        //   ran with another length each run their own range here, where Metal's run one with the
        //   other's length, differently from run to run (the runtime's probes on the M3 Max and
        //   the M4).
        // - `NarrowSurfaceRows`: a GPU read of a row past the first of a surface plane's texture
        //   under 16 bytes a row reads what the GPU's writes put there, where Metal's GPU reads
        //   memory-dependent bytes (the runtime's `linear_plan`: they differ between the M3 Max and
        //   the M4). The suite runs it twice (`textures_backing_records`, which reads no row past
        //   the first).
        // - `OverlappingTextureCopyAsMemmove`: a blit copy between overlapping regions of one
        //   single-sampled texture's texels, which the runtime gives `memmove`'s answer through a
        //   scratch buffer. That is Metal's for regions of up to a few hundred texels (measured
        //   on the M3 Max and an M4); past that Metal's copy
        //   races, some texels reading what it already wrote, differently from run to run.
        //   Announced once per device. A multisampled texture's overlapping copy is still refused,
        //   which dies at the copy.
        // - `PlainStateUniqueIdentifier`: the `uniqueIdentifier` of a pipeline state made without
        //   indirect command buffer support, which Metal answers as an address that differs from
        //   run to run. No value the guest or the host sees: this platform
        //   answers its own numbering ([`crate::gpu::unique_id`]), and never calls the runtime's
        //   identifier getters, whose answers announce it.
        // - `AbsentArgumentRead`: a pipeline state whose function may read an argument its
        //   function constants removed, announced as it is made. The runtime takes zero for such a
        //   read and the command buffer completes, where Metal's compiler drops or folds the read
        //   and its debug layer reads zero.
        // The runtime states one more, `MipmapsWithinSpread`, which dies here ([`DIES`]): the
        // levels lie within the spread of an M3 Max's and an M4's, at neither's exactly, so they
        // are not one of Metal's answers.
        let one_of_metals = D::ALL
            .iter()
            .copied()
            .filter(|k| k.exactness() == rt::Exactness::OneOfMetals && !DIES.contains(k));
        // Among the exact kinds, `MisplacedNotice` (`misplaced-notice`): in a
        // build without debug assertions, a notice the runtime announced where its rule announces
        // none (inside a pipeline pool's piece, or a kind of `announce_each` outside the
        // program's call and outside a host step) reaches the sink unchanged from the runtime's
        // notice thread, after the call that met it, and this kind follows it there, naming the
        // kind and the place (`rt::DegradeNotice::misplaced`). Metal's answers are unchanged, as
        // the runtime states. Relayed: the notice it follows is placed by its own kind, outside
        // every call here, where a kind of `DIES` dies as it is announced and a kind of `HEARD`
        // reaches no hearing. A build with debug assertions, every test here, ends the process at
        // such a notice instead, with the runtime's line.
        // A recorded near-equivalent of Metal's answer (`rt::Exactness::NearEquivalent`), or work
        // the runtime fails instead (`FailsInstead`): relayed and logged, each for one of the
        // module's reasons, named below.
        // - `UncalibratedGpuTimes`, `UntimedCommandBuffer` and `GpuTimesFromHost`: no value the
        //   guest or the host sees. They are a command buffer's GPU times, which core's traits
        //   answer neither; only debug lines read them, and `gpu_times_from_host`
        //   (`crate::platform`'s `device_options`) leaves the first two unreachable. VM boots
        //   relay `GpuTimesFromHost`, which the runtime also takes on MoltenVK 1.4.1 whatever the
        //   option (its `TimestampsWaitForEveryQueue`).
        // - `UnmeasuredUniqueIdentifier`: no value the guest or the host sees. The runtime announces
        //   it where a call answers an identifier it has not measured to be Metal's number: a
        //   state's with indirect command buffer support, a buffer's made after an
        //   encoder, and an indirect command's state identifier, which its read-back of the
        //   commands answers (`IndirectCommandBuffer::commands`). This platform reads that
        //   read-back for every getter of a command ([`crate::gpu::icb`]), so the runtime announces
        //   the kind there, but answers its own numbers: a state's from [`crate::gpu::unique_id`],
        //   a command's state identifier by the state core set or the states that held the ID, and
        //   a buffer's from Metal's range, dying after an encoder. It never answers the
        //   runtime's identifiers.
        // - `IndirectValuePastRecord`: no value the guest or the host sees. The runtime announces
        //   it at a command's buffer set at a slot past 31, or threadgroup length at an index past
        //   30, which no function binds and its records do not hold, so its read-back answers a
        //   fresh command's value there. This platform keeps such a value beside the
        //   command and answers Metal's getters from it, as Metal keeps it for its getters alone
        //   (`icb`'s `slots_past_the_record_read_back_as_on_metal` and
        //   `lengths_past_the_record_read_back_as_on_metal`, measured on this host).
        // - `UnlistedArgumentSampler`: the difference dies where it shows. It is announced before
        //   any command shows it (a sampler state no set-1 array can hold, as it is made or
        //   listed), where most such work never shows it; the command buffer the runtime fails for
        //   it dies as it fails (`crate::gpu::command`'s `unlisted_argument_sampler`), where
        //   Metal's function reads the sampler through its ID.
        // Among the others:
        // - `UnreachedTilesRewritten`, `DontCareLoadTiles` and `ExtendedRangeBitsKept`: the bytes
        //   a pass leaves where its draws write nothing, announced at a pass Metal runs. Metal's
        //   `DontCare` load zeroes every tile in a pass
        //   of one layer with a draw call that clears and stores an attachment, none in a pass
        //   whose draws draw nothing and that clears and stores none, and else the tiles a
        //   draw's primitive reaches within that draw's viewport and scissor rectangle, keeping
        //   the rest (an M4). The runtime takes the first two exactly, and in the third keeps
        //   the tiles outside its draws' viewports and scissors, clearing or rewriting the tiles
        //   inside them that no primitive touches. In a pass whose load
        //   rewrites an extended-range or padded colour's unused bits, of one sample, one layer and
        //   32 by 32 tiles, with no rotated frame, whose draws are direct triangle lists or
        //   strips the runtime can record again, it finds the tiles their primitives reach before
        //   the pass from the positions its vertex functions compute, and rewrites those alone,
        //   announcing nothing, once each draw's capture form is made; the runtime makes that form
        //   on its own threads, so a pipeline's first passes still take the viewports and
        //   scissors, and announce. Its tiles are the pass's, as Metal's grid
        //   is: its `tileWidth` and `tileHeight`, or the default for its
        //   samples. A pass at a sample count with no measured default takes every tile, and so
        //   does a load pass's rewrite under a tile size other than 32 by 32. A rotated frame, a
        //   colour held in sample planes, and a pass through a rate map, with an execution of
        //   render commands or with a tile dispatch, still take every tile. A tile it clears
        //   takes the texel a `Clear` of (0, 0, 0, 0) writes, an extended-range format's own
        //   zero, which reads 0, as Metal's. Pending a decision: the runtime's in-place load of the
        //   tiles no draw reaches is the exact form. The runtime announces
        //   `UnreachedTilesRewritten` and `ExtendedRangeBitsKept` for a bfloat attachment too,
        //   whose codes it moves through `half` or keeps
        //   otherwise than Metal, so a draw in a pass that loads and stores one dies first
        //   (`encode_render`'s `bfloat_loaded`).
        // - `DrawnDepthCopy`: a copy into a multisampled depth texture, or a pass's copies of a
        //   separate depth texture through its scratch image, which a fragment-depth draw writes
        //   where Metal copies the bytes: no copy of the device keeps a multisampled depth's
        //   samples (Vulkan copies a buffer into a single-sampled image alone, and a depth aspect
        //   to or from another format only under maintenance8, which MoltenVK 1.4.1 and NVIDIA's
        //   GA10x lack or get wrong), so a draw writes them through the fragment's depth. Measured
        //   equal: the runtime measured each of its four cases exact at 1 and 4 samples over a
        //   4-sample `Depth32Float_Stencil8` on the M4, the M3 Max's 4-sample row equal to the
        //   M4's. The runtime announces it at a copy from a buffer into a
        //   multisampled 32-bit float depth texture too, whose samples the same draw writes where
        //   the device's maintenance8 copies are missing or wrong (MoltenVK 1.4.1, NVIDIA's GA10x):
        //   every value Metal's but the edges, which Metal keeps bit for bit, where MoltenVK stores
        //   a value with the sign bit set (-0, a negative value, -inf, a negative NaN) as +0 and
        //   one past 1 or +inf as 1, a positive NaN writing nothing, and NVIDIA flushes denormals
        //   and stores NaN as -inf (the notice names which): the device forces
        //   those edges, as a fragment's depth reaches the attachment through the device's depth
        //   unit, which clamps and flushes them, and no other write of a multisampled depth exists
        //   there. Relayed on the rule for native devices: as close as the hardware allows, the
        //   rest logged.
        // - `UnnamedShadowedTexture`: a shadowed texture (its texels a buffer's, a surface's
        //   among them) a heap set holds, which a function reads through its ID as the texels an
        //   encoder that named it last copied in. Announced at the first command that binds a
        //   heap set holding its view while neither naming it nor making it resident, once per
        //   device, not at a read; a set no command binds announces nothing. The read Metal
        //   defines is by ID with the texture resident, and is exact: an
        //   encoder that names it (binds it, or makes it resident by `useResource` or by `useHeap`,
        //   a texture over a heap's buffer included) copies its texels in, and so does each later
        //   compute or render encoder of a command buffer that made it resident, as Metal's
        //   residency lasts the command buffer (the runtime's lanes against the M4's answers).
        //   Measured here: by ID through an argument buffer under `useResource`
        //   (`ycbcr::sampling`'s `the_guests_id_is_metals`, a 505 texture's reads and IDs as
        //   Metal's). What stays stale is outside Metal's residency rule: a read in an encoder
        //   before the first that makes the texture resident in its command buffer, which Metal
        //   orders against nothing either, and a read in a command buffer that never makes it
        //   resident, which on the M4 read the older texels, or zeros for a heap's buffer's
        //   texture. Metal's own answer there is not one value, though the runtime states
        //   the kind a near-equivalent.
        // - `DepthDenormals`: a write that may store a denormal in a 32-bit float depth (a
        //   pass's clear to one, a blit into a `Depth32Float` or `Depth32Float_Stencil8`
        //   texture's depth whatever bytes it copies, a `replace_region` whose bytes hold one) on
        //   a device whose driver may store it as a zero, or whose depth test may compare it
        //   otherwise than Metal's, which keeps it and compares a positive one above zero and a
        //   negative one not below zero. Announced once per device, where
        //   the denormal may be stored, not where a depth test or a read meets one. Grounds:
        //   - this host's answer, measured equal to Metal's: MoltenVK keeps a denormal in a depth
        //     image on the M3 Max (the runtime's `DepthImageKeepsDenormals`),
        //     and its depth test compares one as Metal's does there, positive and negative,
        //     written by a blit or a pass's clear, under every compare function but `Never` and
        //     `Always` (`serialized::draws`'
        //     `denormal_depths_store_and_compare_as_metals`, which announces the kind); the
        //     runtime measured that depth test on an Apple M4 alone (`DepthTestKeepsDenormals`),
        //     so it announces the kind on this host too;
        //   - on another device, a fact of the device's depth unit, which the runtime measured:
        //     NVIDIA's depth test compares a stored positive denormal as zero, and RADV and ANV
        //     store one a GPU write gives as a zero. No exact form is the device's but a depth
        //     test run in the shader, an interpreter's, which the rule for native devices does
        //     not ask for.
        const PRECISION: [D; 12] = [
            D::UncalibratedGpuTimes,
            D::UntimedCommandBuffer,
            D::GpuTimesFromHost,
            D::DontCareLoadTiles,
            D::UnlistedArgumentSampler,
            D::ExtendedRangeBitsKept,
            D::UnreachedTilesRewritten,
            D::DrawnDepthCopy,
            D::UnnamedShadowedTexture,
            D::UnmeasuredUniqueIdentifier,
            D::IndirectValuePastRecord,
            D::DepthDenormals,
        ];
        // A pipeline stage's precision notes, each arithmetic, filtering or ordering of the device
        // that its documentation says can differ from Metal's: relayed, but those [`DIES`] and
        // [`HEARD`] list, each for one of the module's reasons:
        // - `FastMathRewrites`: an accepted class, fast-math rewrites
        //   within what the function's AIR contract and reassociation flags allow, which Metal's
        //   compiler may make too.
        // - `DeviceTranscendentals`: an accepted class, maths functions
        //   whose bound Metal does not state, each device's deviation measured and recorded, and
        //   fixed where far off. Later runtimes do not record it where MoltenVK's float functions
        //   are Metal's own.
        // - `HalfOutputFastMathBlend`: the accepted class of fast-math rewrites, extended to
        //   blends. A fast-math module's `half` colour output to an attachment, whose blend
        //   and conversion Metal's compiler folds into the function and regroups per pipeline in
        //   no form the AIR and the blend state decide: the device's blend of the output's `half`
        //   stores a value at most three `half` units in the last place at the largest operand's
        //   magnitude apart from Metal's into a 16-bit float attachment, and two `float` ones into
        //   a 32-bit float one (MoltenVK on the M3 Max against plain Metal). A
        //   module without fast math records `HalfOutputPrecision` instead ([`HEARD`]).
        // And `DeviceColorConversion`: a colour channel the device converts, which on MoltenVK,
        // whose device stores Metal's codes, is only a half-float channel under
        // `textureWriteRoundingMode` 2: the alpha that alpha to coverage reads at colour 0 of
        // `RGBA16Float` (a write mask with alpha dies first, at the state's creation,
        // `crate::gpu::serialized::rounded_writes`). Its conversion is measured as Metal's, byte
        // for byte over sweeps through unorm values, past 1 by float steps and past the largest
        // half, by `encode_render`'s
        // `mode_2_half_float_attachments_no_other_test_compares_round_as_metals`, at `R16Float`,
        // `RG16Float` and `RGBA16Float`. A `bfloat` colour output no longer reaches the note: the
        // runtime refuses one as invalid input, as Apple's compiler rejects the return
        // type, which the pipeline creators answer as an `NSError` of code 0
        // (`crate::gpu::states::pipeline_error`); what Metal does with one is not measured.
        const NOTES: [D; 4] = [
            D::DeviceTranscendentals,
            D::FastMathRewrites,
            D::HalfOutputFastMathBlend,
            D::DeviceColorConversion,
        ];
        // A difference of an accepted class, within the class's stated bound: relayed,
        // each the class's alone. `DeviceUnorm8LinearFilter`, the class of 8-bit linear filtering
        // within one code of Metal's: an `R8Unorm`, `RG8Unorm`,
        // `RGBA8Unorm` or `BGRA8Unorm` 2D or 2D array texture's linear sample, without comparison,
        // offset or pixel coordinates, that the device filters at the coordinate of Metal's
        // footprint step, on a device the runtime measured within the bound (at most
        // 0.849 of a code on NVIDIA 615.71.09, 0.031 on RADV and ANV, just under 1 on lavapipe,
        // against an Apple M4). Every other filtered sample keeps Metal's filter, emulated
        // exactly, or announces its own kind.
        const ACCEPTED: [D; 1] = [D::DeviceUnorm8LinearFilter];
        // No value the guest or the host sees: `PrivatePropertiesUnknown`, Apple's private device
        // properties the runtime has no answer of the device's for, announced where
        // `rt::Device::private_properties` is first read. The runtime gives none there
        // (`rt::PrivateAnswer::value` is `None`), never a value of its own, and this platform reads
        // no private property of the runtime's: it answers each of core's from the device's facts
        // or a stand-in its table names ([`crate::gpu::caps`]).
        const UNSEEN: [D; 1] = [D::PrivatePropertiesUnknown];
        let relayed: Vec<D> = exact
            .chain(one_of_metals)
            .chain(PRECISION)
            .chain(NOTES)
            .chain(ACCEPTED)
            .chain(UNSEEN)
            .collect();
        for kind in D::ALL {
            let places = [
                DIES.contains(kind),
                HEARD.contains(kind),
                relayed.contains(kind),
            ];
            let placed = places.into_iter().filter(|&p| p).count();
            assert_eq!(placed, 1, "{kind:?} ({}): {places:?}", kind.code());
        }
        assert_eq!(DIES.len() + HEARD.len() + relayed.len(), D::ALL.len());
        // On a native device, the kinds of `native::NATIVE_RELAYED` are relayed where they die
        // on MoltenVK: each a kind of DIES, listed once, with its ground at the entry. They are
        // the device's interpolation, level arithmetic and denormal reads, the heap size the
        // device's memory granularity decides, which the guest reads, and, on the rule for native
        // devices, the precision notes whose difference is the device's rounding, evaluation
        // position or order; then, as the class of a native device's own maths and sampling, the
        // device's maths
        // and sampling, each with its bound or the runtime's record of it, the notes of the
        // runtime's native defaults off Apple's GPUs last.
        let natively: Vec<D> = native::NATIVE_RELAYED.iter().map(|r| r.kind).collect();
        assert_eq!(
            natively,
            [
                D::DeviceInterpolation,
                D::GradFromBias,
                D::DeviceTexelDenormals,
                D::HeapTextureLargerThanMetal,
                D::ReductionOrder,
                D::DeviceSignedZeroInfNan,
                D::MultisamplePosition,
                D::ImplementationDefinedReconvergence,
                D::F16FmaDoubleRounding,
                D::F16IntConvDoubleRounding,
                D::DeviceArithmetic,
                D::DeviceDenormals,
                D::FloatSampleFilter,
                D::HalfSampleFilter,
                D::CubeGradientFootprint,
                D::CompressedTexelDecode,
                D::ViewBaseTexelOffset,
                D::SrgbTexelDecode,
                D::BicubicFilterArithmetic,
                D::HalfTexelConversion,
                D::NativeTranscendentals,
                D::NativeDivision,
                D::NativeFma,
                D::NativeNanEncoding,
                D::NativeMinMax,
                D::NativeDot,
                D::NativeMix,
                D::NativeHalfConversion,
                D::NativeHalfQuantize,
                D::DeviceSampling,
                D::DeviceTexelConversion,
            ]
        );
        // These stay deaths on native devices too, as neither a fact of the device nor the
        // class of the device's own maths and sampling covers them: two precision notes
        // are other values, not other bits, the third is where Apple's compiler places an
        // operation, Metal's rule for tessellation factors is not read, blending and storage are
        // neither maths nor sampling, and the anisotropic emulation is a fitted model.
        for kind in [
            D::TessellationFactorInstance,
            D::Fp8Value,
            D::QuadsCountedBeforeDiscard,
            D::TessellationFactorScale,
            D::BlendedWriteRounding,
            D::RotatedBlendRounding,
            D::AnisotropicFilter,
            D::DeviceNanEncoding,
            D::TessellationWinding,
            D::TessellationFactorEdgeCases,
        ] {
            assert!(DIES.contains(&kind) && !native::relayed(kind), "{kind:?}");
        }
        for (at, r) in native::NATIVE_RELAYED.iter().enumerate() {
            let code = r.kind.code();
            assert!(
                DIES.contains(&r.kind),
                "{code} relayed natively, but not of DIES"
            );
            assert!(
                !native::NATIVE_RELAYED[..at]
                    .iter()
                    .any(|o| o.kind == r.kind),
                "{code} relayed natively twice"
            );
            assert!(
                !r.ground.is_empty(),
                "{code} relayed natively without a ground"
            );
        }
    }

    /// In strict mode a kind of DIES dies on a device that runs on Metal, MoltenVK's, as before;
    /// on a native device the kinds of `native::NATIVE_RELAYED` are relayed and every other one
    /// dies; census mode counts and relays what would die, and leaves a native relay a native
    /// relay. By default every kind is relayed, counted, on every device, and the process goes
    /// on, but those of `DIES_BY_DEFAULT`, which die
    #[test]
    fn a_kind_of_dies_is_placed_by_the_device_class() {
        use native::Placement::{Native, OnMetal};
        for kind in DIES {
            let natively = native::relayed(kind);
            assert_eq!(fate(kind, OnMetal, false, true), Fate::Dies, "{kind:?}");
            assert_eq!(fate(kind, OnMetal, true, true), Fate::Counted, "{kind:?}");
            let by_default = if DIES_BY_DEFAULT.contains(&kind) {
                Fate::Dies
            } else {
                Fate::Relayed
            };
            assert_eq!(fate(kind, OnMetal, false, false), by_default, "{kind:?}");
            assert_eq!(fate(kind, OnMetal, true, false), Fate::Counted, "{kind:?}");
            let (native, census, relayed) = if natively {
                (
                    Fate::RelayedNatively,
                    Fate::RelayedNatively,
                    Fate::RelayedNatively,
                )
            } else {
                (Fate::Dies, Fate::Counted, by_default)
            };
            assert_eq!(fate(kind, Native, false, true), native, "{kind:?}");
            assert_eq!(fate(kind, Native, true, true), census, "{kind:?}");
            assert_eq!(fate(kind, Native, false, false), relayed, "{kind:?}");
        }
        assert_eq!(
            fate(rt::Degradation::DeviceInterpolation, Native, false, true),
            Fate::RelayedNatively
        );
        // A precision note the device's evaluation position decides: relayed on a native device,
        // a death on MoltenVK in strict mode, as before; so is the device's sampling, as the class
        // of the device's own maths and sampling. One whose Metal rule is not read, and not the
        // device's maths or sampling, dies on both in strict mode, and is relayed on both by
        // default.
        let position = rt::Degradation::MultisamplePosition;
        assert_eq!(fate(position, Native, false, true), Fate::RelayedNatively);
        assert_eq!(fate(position, OnMetal, false, true), Fate::Dies);
        let cube = rt::Degradation::CubeGradientFootprint;
        assert_eq!(fate(cube, Native, false, true), Fate::RelayedNatively);
        assert_eq!(fate(cube, OnMetal, false, true), Fate::Dies);
        let scale = rt::Degradation::TessellationFactorScale;
        assert_eq!(fate(scale, Native, false, true), Fate::Dies);
        assert_eq!(fate(scale, OnMetal, false, true), Fate::Dies);
        assert_eq!(fate(scale, Native, false, false), Fate::Relayed);
        assert_eq!(fate(scale, OnMetal, false, false), Fate::Relayed);
        // A kind whose difference breaks what the guest or host relies on dies by default too,
        // on every device; each is one of DIES, and none a native relay.
        for kind in DIES_BY_DEFAULT {
            assert!(DIES.contains(&kind) && !native::relayed(kind), "{kind:?}");
            assert_eq!(fate(kind, OnMetal, false, false), Fate::Dies, "{kind:?}");
            assert_eq!(fate(kind, Native, false, false), Fate::Dies, "{kind:?}");
        }
    }

    /// A test of this crate runs strict; [`relaxed`] runs its call under the default, on its
    /// thread alone, and puts strict mode back after, an unwinding call's included
    #[test]
    fn tests_run_strict_but_where_relaxed() {
        assert!(strict());
        assert!(!relaxed(strict));
        assert!(strict());
        let unwound = catch_death(|| relaxed(|| dies("sel:", "a death inside")));
        assert!(unwound.is_err() && strict());
        let elsewhere = relaxed(|| std::thread::scope(|s| s.spawn(strict).join().unwrap()));
        assert!(elsewhere);
    }

    /// Every kind this platform relays as computed otherwise than Metal is placed once: listed
    /// once, under a code of its own, none a runtime kind's or an unfollowed kind's
    #[test]
    fn every_inexact_kind_is_placed_once() {
        let codes: Vec<&str> = Inexact::ALL.iter().map(|k| k.code()).collect();
        for (at, code) in codes.iter().enumerate() {
            assert!(!codes[..at].contains(code), "{code} placed twice");
            assert!(
                rt::Degradation::ALL.iter().all(|d| d.code() != *code)
                    && Unfollowed::ALL.iter().all(|u| u.code() != *code),
                "{code} is another kind's"
            );
        }
        for kind in Inexact::ALL {
            match kind {
                Inexact::HalfOutputBlend
                | Inexact::RoundedResolve
                | Inexact::Clear
                | Inexact::BfloatLoaded
                | Inexact::WriteRounding
                | Inexact::HeapLayout
                | Inexact::MetalLevel
                | Inexact::ProvokingVertex
                | Inexact::LineWidth
                | Inexact::RateMap
                | Inexact::YcbcrView
                | Inexact::FillFormat
                | Inexact::LateResolve
                | Inexact::ProtectedContent => {}
            }
        }
    }

    /// By default a value computed otherwise than Metal is relayed and counted, and the call
    /// returns; in strict mode it dies at its selector, as before, its line saying what
    #[test]
    fn an_inexact_value_is_relayed_but_in_strict_mode() {
        let counted = || {
            (inexact_counts().into_iter())
                .find(|&(k, _)| k == Inexact::Clear)
                .map_or(0, |(_, n)| n)
        };
        let before = counted();
        relaxed(|| inexact("sel:", Inexact::Clear, "a value"));
        relaxed(|| inexact("sel:", Inexact::Clear, "a value"));
        assert_eq!(counted(), before + 2);
        *LAST_LINE.lock() = None;
        let site = reims_vgpu_core::testkit::capture::catch_death(|| {
            inexact("sel:", Inexact::Clear, "a value");
        })
        .unwrap_err();
        assert_eq!(site.file, "sel:");
        let line = LAST_LINE.lock().take().unwrap_or_default();
        assert!(line.contains("cannot hold a value"), "{line}");
        assert_eq!(counted(), before + 2);
    }

    /// Every kind of value this platform does not follow is placed once: listed once, under a
    /// code of its own, none a runtime kind's
    #[test]
    fn every_unfollowed_kind_is_placed_once() {
        let codes: Vec<&str> = Unfollowed::ALL.iter().map(|k| k.code()).collect();
        for (at, code) in codes.iter().enumerate() {
            assert!(!codes[..at].contains(code), "{code} placed twice");
            assert!(
                rt::Degradation::ALL.iter().all(|d| d.code() != *code),
                "{code} is a runtime kind's"
            );
        }
        for kind in Unfollowed::ALL {
            match kind {
                Unfollowed::IcbIdentifierAfterEncoder => {}
            }
        }
    }

    /// A call here that the runtime announced a degradation of [`DIES`] inside, and that then
    /// unwinds, a death of its own inside it, dies of that death alone, and lets the notice go:
    /// the thread's next call here returns. A copy within one texture of a two-plane XR10 format
    /// announces `TwoPlaneTextureCopy` (`gpu::encode_blit`'s
    /// `two_plane_xr10_blits_metal_was_not_measured_at_die`)
    #[test]
    fn a_call_that_unwinds_lets_its_notice_go() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::values::TextureDesc;
        use reims_vgpu_core::raise::OnFault;
        let d = crate::gpu::device::tests::dev();
        let desc = crate::platform::tests::platform().texture_descriptor(&TextureDesc {
            pixel_format: 550,
            width: 16,
            height: 4,
            usage: 0x3,
            ..TextureDesc::default()
        });
        let t = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
        let queue = d.new_command_queue().unwrap();
        let cb = queue.raw.command_buffer().unwrap();
        let mut e = cb.blit_command_encoder().unwrap();
        let site = catch_death(|| {
            inside(Some("the call"), || {
                let texels = t.copied("the call");
                let copied = e.copy_from_texture_to_texture(
                    texels,
                    0,
                    0,
                    rt::Origin::new(0, 0, 0),
                    rt::Size::new(4, 1, 1),
                    texels,
                    0,
                    0,
                    rt::Origin::new(8, 0, 0),
                );
                assert!(PENDING.with(|p| p.borrow().is_some()), "{copied:?}");
                dies("its own death", "a death inside the call")
            })
        })
        .unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "its own death"));
        assert!(PENDING.with(|p| p.borrow().is_none()));
        assert_eq!(DEPTH.with(Cell::get), 0);
        assert_eq!(plain(|| 7), 7);
        drop((e, cb, t));
    }

    /// By default a degradation of [`DIES`] announced inside a call is relayed: the call
    /// returns, nothing is kept for it to die at, and the kind is counted at each occurrence.
    /// The copy of `a_call_that_unwinds_lets_its_notice_go` announces `TwoPlaneTextureCopy`
    #[test]
    fn a_kind_of_dies_is_relayed_and_counted_by_default() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::values::TextureDesc;
        use reims_vgpu_core::raise::OnFault;
        let kind = rt::Degradation::TwoPlaneTextureCopy;
        let counted = || {
            (relayed_counts().into_iter())
                .find(|&(k, _)| k == kind)
                .map_or(0, |(_, n)| n)
        };
        let d = crate::gpu::device::tests::dev();
        let desc = crate::platform::tests::platform().texture_descriptor(&TextureDesc {
            pixel_format: 550,
            width: 16,
            height: 4,
            usage: 0x3,
            ..TextureDesc::default()
        });
        let t = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
        let queue = d.new_command_queue().unwrap();
        let cb = queue.raw.command_buffer().unwrap();
        let mut e = cb.blit_command_encoder().unwrap();
        let before = counted();
        for at in [8, 12] {
            let copied = relaxed(|| {
                inside(Some("the call"), || {
                    let texels = t.copied("the call");
                    e.copy_from_texture_to_texture(
                        texels,
                        0,
                        0,
                        rt::Origin::new(0, 0, 0),
                        rt::Size::new(4, 1, 1),
                        texels,
                        0,
                        0,
                        rt::Origin::new(at, 0, 0),
                    )
                })
            });
            assert!(copied.is_ok(), "{copied:?}");
            assert!(PENDING.with(|p| p.borrow().is_none()));
        }
        assert_eq!(counted(), before + 2);
        drop((e, cb, t));
    }

    /// What Metal's validation stops is the caller's raise under a handler and a death with none,
    /// as the send's label says
    #[test]
    fn an_invalid_refusal_is_decided_by_the_label() {
        let r = send("fillBuffer:range:value:", Policy::Caught, || {
            Err::<(), _>(invalid())
        })
        .unwrap_err();
        assert_eq!(r.id(), RaiseId::Platform);
        assert_eq!(r.origin(), Origin::Platform("fillBuffer:range:value:"));
        assert!(
            r.detail().unwrap_or_default().contains("StorageMode"),
            "{r}"
        );
        let r = send(
            "fillBuffer:range:value:",
            Policy::On(OnFault::Return),
            || Err::<(), _>(invalid()),
        );
        assert!(r.is_err());
        for policy in [Policy::Fatal, Policy::On(OnFault::Die)] {
            let site = catch_death(|| {
                send(
                    "fillBuffer:range:value:",
                    policy,
                    || Err::<(), _>(invalid()),
                )
            })
            .unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::UncaughtObjc, "fillBuffer:range:value:")
            );
        }
        assert_eq!(send("length", Policy::Caught, || Ok(3)), Ok(3));
    }

    /// A runtime gap is a death naming the selector whatever the label: never a raise the guest
    /// could catch
    #[test]
    fn a_runtime_gap_dies_at_the_send() {
        let site = catch_death(|| {
            send("generateMipmapsForTexture:", Policy::Caught, || {
                Err::<(), _>(Error::NotImplemented(RuntimeGap::Textures))
            })
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file, site.line),
            (DeathKind::Abort, "generateMipmapsForTexture:", 0)
        );
        let site = catch_death(|| get("length", || Err::<u64, _>(Error::DeviceLost))).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "length"));
        let site = catch_death(|| get("length", || Err::<u64, _>(invalid()))).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::UncaughtObjc, "length"));
    }

    /// A command the runtime refuses for sampling a texture of an address-mode map dies naming
    /// which path it samples it by
    #[test]
    fn a_mapped_texture_sampling_refusal_names_its_path() {
        use metal2vulkan::metal::SamplerAddressMode;
        use metal2vulkan_runtime::{AddressModeMap, MappedSampling};
        let map = AddressModeMap::IDENTITY
            .with(SamplerAddressMode::Repeat, SamplerAddressMode::ClampToEdge);
        for (through, named) in [
            (
                MappedSampling::ConstexprSampler,
                "through a constexpr sampler",
            ),
            (
                MappedSampling::SharedSamplerSlot,
                "through a sampler slot that samples",
            ),
            (MappedSampling::Handles, "samples through handles"),
        ] {
            let e = Error::NotImplemented(RuntimeGap::MappedTextureSampling { map, through });
            let text = refusal(&e);
            assert!(text.contains(named), "{text}");
            assert!(text.contains(&format!("{through:?}")), "{text}");
        }
        let gap = refusal(&Error::NotImplemented(RuntimeGap::Textures));
        assert!(!gap.contains("address modes"), "{gap}");
    }

    /// A refusal of SIMD-group or quad-group operations names the stage and the class the
    /// refusal names, and no other refusal names one
    #[test]
    fn a_simd_stage_refusal_names_its_stage_and_class() {
        use metal2vulkan::device::{SubgroupOp, VkProperty};
        use metal2vulkan::error::{Capability, Feature, Unsupported};
        use metal2vulkan::pipeline::Stage;
        let property = |p| Capability::Property(p);
        for (unsupported, named) in [
            (
                Unsupported::needs(
                    Feature::SubgroupOperations,
                    property(VkProperty::SubgroupStage(Stage::Vertex)),
                ),
                "a vertex function's SIMD-group operations, which",
            ),
            (
                Unsupported::needs(
                    Feature::QuadOperations,
                    property(VkProperty::SubgroupQuadOperationsInAllStages),
                ),
                "a vertex function's quad-group operations, which",
            ),
            (
                Unsupported::needs_any_of(
                    Feature::SubgroupOperations,
                    vec![vec![
                        property(VkProperty::SubgroupOperation(SubgroupOp::Arithmetic)),
                        property(VkProperty::SubgroupOperation(SubgroupOp::ShuffleRelative)),
                    ]],
                ),
                "a function's SIMD-group operations (the classes Arithmetic, ShuffleRelative)",
            ),
        ] {
            let e = Error::Unsupported(unsupported);
            let text = refusal(&e);
            assert!(text.contains(named), "{text}");
            assert!(text.contains("key 33"), "{text}");
            // A pipeline's death line carries the clause, then its descriptor's formats.
            let formats = "the descriptor's formats: no colour attachments";
            let line = refusal_describing(&e, Some(formats));
            assert_eq!(line, format!("{text}; {formats}"));
        }
        for other in [
            Error::Unsupported(Unsupported::not_implemented(Feature::SubgroupOperations)),
            Error::Unsupported(Unsupported::needs(
                Feature::SimdgroupMatrix,
                property(VkProperty::SubgroupStage(Stage::Vertex)),
            )),
            Error::NotImplemented(RuntimeGap::Textures),
        ] {
            let text = refusal(&other);
            assert!(!text.contains("key 33"), "{text}");
        }
    }

    /// A creator answers nil for the refusals Metal answers nil for, every `Error::Nil` and
    /// those its set names, and decides the rest; an `Error::Nil` at a send that makes nothing
    /// dies naming the selector
    #[test]
    fn a_creator_answers_nil_where_metal_does() {
        let nil_of = || {
            Err::<(), _>(Error::Nil(metal2vulkan_runtime::Nil::new(
                "new_texture",
                Problem::StorageMode,
            )))
        };
        for policy in [Policy::Dies, Policy::Caught] {
            assert_eq!(
                create("newTextureWithDescriptor:", policy, Nil::Never, nil_of),
                Ok(None)
            );
        }
        let site =
            catch_death(|| send("fillBuffer:range:value:", Policy::Caught, nil_of)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "fillBuffer:range:value:")
        );
        let nil = create(
            "newBufferWithLength:options:",
            Policy::Caught,
            Nil::BufferLength,
            || Err::<(), _>(Error::OutOfMemory(metal2vulkan_runtime::MemoryKind::Device)),
        );
        assert_eq!(nil, Ok(None));
        let r = create(
            "newBufferWithLength:options:",
            Policy::Caught,
            Nil::Never,
            || Err::<(), _>(invalid()),
        );
        assert!(r.is_err());
        assert_eq!(
            create("newFence", Policy::Caught, Nil::Never, || Ok(1)),
            Ok(Some(1))
        );
    }

    /// An unbuilt method, and a value the runtime cannot hold, die naming the selector
    #[test]
    fn unbuilt_and_unrepresentable_die_at_the_selector() {
        let site = catch_death(|| unbuilt("newHeapWithDescriptor:")).unwrap_err();
        assert_eq!(
            (site.kind, site.file, site.line),
            (DeathKind::Abort, "newHeapWithDescriptor:", 0)
        );
        let site =
            catch_death(|| unrepresentable("setPixelFormat:", "pixel format 1")).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "setPixelFormat:")
        );
    }

    /// A runtime error as an NSError carries the runtime's text
    #[test]
    fn error_text_is_the_runtimes() {
        let t = error_text(&Error::DeviceLost);
        assert_eq!(t.description, "the Vulkan device was lost");
        assert_eq!(t.to_string(), t.description);
    }
}
