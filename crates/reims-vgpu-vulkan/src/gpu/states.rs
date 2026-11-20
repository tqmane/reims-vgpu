//! The render family's state objects, made from core's values and the translator's descriptors:
//! sampler states, depth-stencil states and render pipeline states (`GpuDevice`'s creators of
//! them call here).
//!
//! - **A value's raw properties** become the runtime's types in the value's order, each named by
//!   its setter: a raw value no Metal enumerator is dies at that setter
//!   ([`crate::gpu::metal_value`]), as the runtime's types cannot hold it.
//! - **A sampler state** of equal requests is one object while it lives, as Metal's is on this
//!   host ([`crate::gpu::resource::sampler_state`]). The value's resource index, which core sets
//!   without the force flag, places the state at that ID: Metal on this host places a sampler at
//!   an index set so (`render_objects_placement_sampler`: index 7 gives `gpuResourceID` 7), and the
//!   runtime's placement is that ID (index 0 is plain allocation, as Metal's is). Plain
//!   allocation hands out the IDs Metal's does, as the runtime numbers them: Metal's
//!   device keeps a sampler ID of its own for good from its first command queue, indirect
//!   command buffer or texture of its own memory (sampler states made after a texture take 2 and
//!   on, or 3 and on with 2 reserved); and a pipeline state, as it is made, takes one for each of
//!   its functions' constexpr samplers, texture reads and texture writes, and the device's own
//!   for a render state with a fragment function made before that, unless a live holder holds an
//!   equal one. The state holds them, as do every function object of its functions and every
//!   live library of their bytes, and each goes back with its last holder. The constexpr samplers
//!   of the visible functions a stage links come after the stage's own, as Metal numbers them.
//!   Where it does not number them as Metal does, the runtime announces
//!   `linked-sampler-numbering`, which dies ([`crate::answer::DIES`]): a stage that links two or
//!   more with constexpr samplers, whose order Metal itself does not decide, and a function a
//!   table can hold whose samplers the runtime cannot read, among them. One linked function
//!   alone, of `functions` or `privateFunctions`, called or not, is numbered as Metal numbers it.
//!   No guest form here links a function. This platform's libraries, functions and states hold the
//!   runtime's exactly as long as the guest's handles to them live, so its sampler IDs are Metal's
//!   (`tests::sampler_ids_follow_pipeline_holders_as_metals`).
//! - **A pipeline's refusals** that Metal answers with an `NSError` are one here
//!   ([`pipeline_error`]): what the compile finds wrong with the functions or the descriptor, a
//!   vertex function writing a point size under a line or triangle topology, a function Metal's
//!   compiler itself refuses on this device, an import no dynamic library supplies, and a state
//!   that supports indirect command buffers whose functions use a direct texture or sampler
//!   argument. The rest die at the selector: an ID past Metal's limit, which ends Metal's process,
//!   a gap of the runtime (an unspecialized function constant among them, which Metal aborts for;
//!   and a function's SIMD-group operations in a stage the runtime does not take them in, a vertex
//!   function's on MoltenVK, which Metal makes and for which device info's key 33 answers no,
//!   [`crate::gpu::caps`]), and its other refusals, as Metal raises at none. Such a death's line
//!   also names the descriptor's formats (`formats`: its colour attachments' with their write
//!   masks, its depth and stencil, its vertex attributes'), which the runtime's error does not
//!   carry. A binary archive miss cannot happen: core's descriptors list no archive.
//!
//!   Measured on the M3 Max (macOS 26.7, one case per process, plain and under `MTL_DEBUG_LAYER`):
//!   the compute creators (both selectors), the render creator and the tile creator answer nil and
//!   an `NSError` of code 3 for a render state setter on an indirect render command
//!   (`set_cull_mode`, `set_depth_stencil_state`) in a kernel, vertex, fragment or tile function,
//!   and for a fragment function's read of a program-scope `thread_index_in_quadgroup` (a read of
//!   `thread_index_in_simdgroup` or `threads_per_simdgroup` there is made); and of code 2
//!   ("Undefined symbols:") for a function whose import the library at its install name lacks, or
//!   whose install name names no file. The functions that draw without a setter, and those whose
//!   library exports the import, are made. The runtime refuses a fragment function's read of a
//!   program-scope `thread_index_in_simdgroup` as a gap of its own (`Lowering(StageIo)`), which
//!   dies where Metal makes the pipeline. The compute and render creators answer nil and an
//!   `NSError` of code 3 for a descriptor with `supportIndirectCommandBuffers` whose kernel,
//!   vertex or fragment function uses a `[[texture(n)]]` or `[[sampler(n)]]` argument once Metal
//!   links the pipeline ("Fragment shader cannot be used with indirect command buffers", naming
//!   the fragment function alone where both use one): Metal makes the state where a vertex
//!   function's read reaches only outputs the fragment function does not read, or there is no
//!   fragment function, and where a fragment function's reaches only a colour no attachment
//!   writes (none, colour 0 `Invalid`, or a write mask of none or of none of its format's channels,
//!   measured). The runtime refuses the same states with Metal's error, deciding the use
//!   by liveness as Metal links, and the creators answer that `NSError`. A use only a rule the
//!   runtime keeps beyond Metal keeps is unproven (`Error::Unproven`): an endless loop, stack
//!   memory, a gated-off mesh output, or a colour whose attachment's write is unmeasured
//!   (`UnmeasuredColorWrite`: a mask of some of the format's channels but not all, a format of
//!   fewer than four channels even written in full, an unspecialized mask, or a dual-source second
//!   colour under blending that reads no second source). Metal's answer is unmeasured, and the
//!   creator dies, naming the stage and the rule. The compile layer decides the use from the
//!   arguments' roles alone, so an argument whose reflection Metal's answer has not been measured
//!   for decides nothing; where it refuses to decide (the decode's or the
//!   specialization's own refusal), the runtime refuses the state, unproven, and the creator dies
//!   there, naming the refusal. Once the runtime made such a state, the creator asks the compile
//!   layer as the runtime did, and dies naming a refusal there too
//!   ([`direct_arguments_decided`], [`kernel_direct_arguments_decided`]). The runtime
//!   refuses a render state with rasterization off whose vertex function returns a value with
//!   Metal's error too (`UnrasterizedVertexReturn`, code 3). With rasterization off the descriptor
//!   keeps the fragment function, which the runtime checks as Metal links it and leaves out of the
//!   Vulkan compile: it answers Metal's error (code 3) where a written colour attachment keeps the
//!   fragment function's read of a `stage_in` input, which no vertex function writes then, or, with
//!   `supportIndirectCommandBuffers`, its direct texture or sampler, and where the function writes
//!   depth with no depth attachment; it makes the state otherwise, as Metal does (measured:
//!   `unrasterized_states_are_metals`).
//!
//!   The `NSError`'s domain is the driver's (`AGXMetalG15X_M1` on the M3 Max; another GPU family's
//!   driver names another), and its text names the intrinsic or the symbol, which the runtime's
//!   refusal does not always carry. Core reads a pipeline creator's `NSError` only for its
//!   description, which goes into a log line and the creation's error string
//!   (`resources::manager::create::pipeline`): no domain, code or text reaches the guest or the
//!   host. So the `NSError` here carries the runtime's domain and text, with Metal's code; the
//!   refusal, in place of a death, is what the guest observes.
//! - **A sampler's descriptor** is a debug line under this module's target
//!   (`REIMS_VGPU_LOG=debug`, or `warn,reims_vgpu_vulkan::gpu::states=debug`), written before its
//!   first setter ([`Sampler`]): each property of the guest's record, so a death at one setter
//!   (`setMinFilter:` of a filter the runtime has no enumerator of, say) shows the whole
//!   descriptor. The record holds no LOD bias, which only a binding's setter takes, and no label.
//!   The line is a diagnostic: it changes nothing the guest or the host sees, and with the
//!   target's debug lines off nothing formats it.
//! - **A cubic sampler** (a min or mag filter of 2, `MTLSamplerMinMagFilterCubic`) is the
//!   runtime's `Cubic`. Without a mip filter the runtime samples through it as a bicubic sampler
//!   of its address modes and border colour, whatever its other filter, and announces
//!   `bicubic-filter-arithmetic` at the dispatch or draw, which dies at its selector, naming the
//!   kind ([`crate::answer::DIES`]). With a mip filter, under mirror clamp to the edge, with a
//!   minimum or maximum reduction, or with anisotropy, the runtime refuses the dispatch or draw
//!   that samples through it, which dies at its selector, naming the refusal, where Metal
//!   samples; and so under mirrored repeat where the function also gathers through it, or samples
//!   in pixel coordinates or a 1D texture. The state itself is made, as Metal makes it. Through
//!   an argument buffer, a stage other than a fragment one tests each sampler it reaches, and
//!   the runtime fails the command buffer of a function that reached a cubic one, which dies
//!   where its failure is first read ([`crate::gpu::command`]), where Metal samples; one that
//!   never reaches a cubic sampler runs as Metal's while a cubic state is alive. A fragment
//!   stage tests none: while any cubic state is alive on the device, whether or not an argument
//!   buffer holds it, a draw whose fragment stage samples through an argument buffer's sampler,
//!   of any filter, dies at its selector, naming the refusal, as the runtime lists every live
//!   sampler state as an argument sampler and refuses that stage while the list holds a cubic
//!   one. The runtime lists the live states again as a command buffer commits, and fails it,
//!   whether or not a function reaches the state it leaves out, where a command of any stage
//!   that samples through an argument buffer's sampler meets more states than its set-1 sampler
//!   arrays (at most 65536 elements each, fewer where the device's limits are) or its sampler
//!   tables hold, or such a draw meets a cubic state made after it was encoded: that dies where
//!   its failure is first read (`UnlistedArgumentSampler`, [`crate::gpu::command`]), where
//!   Metal's functions reach every live state by its ID. A command a host step records after
//!   the commit, which no commit lists, fails it the same way where it meets more states than
//!   its set-1 sampler arrays hold.
//! - **A texture slot sampled through an argument buffer's sampler**, whatever its filter, is a
//!   function the runtime refuses where its pipeline state is made (`ArgumentBufferTextures`),
//!   which dies there, where Metal makes and runs it. A texture an argument buffer holds, sampled
//!   through a sampler it holds, samples as Metal's does.
//! - **A sampler's raw border colour** is the colour Metal samples for it
//!   (`rt::sampler_border_color`): Metal takes any value and keeps its low two bits, 3 sampling
//!   as transparent black.
//! - **A sampler's or a depth-stencil state's refusal** dies at the selector: Metal makes a
//!   state of any descriptor these take (measured: a sampler of `maxAnisotropy` 0 or 17, of a
//!   minimum LOD clamp past the maximum, or of border colour 3, and a depth-stencil state of the
//!   default descriptor, each made, no exception raised), and so does the runtime of each of
//!   those.
//! - **Labels** are diagnostics: a pipeline state takes its descriptor's or its value's
//!   (`PipelineOptions::label`), which no answer to the guest reads and the runtime compiles
//!   nothing of.

use std::sync::Weak;

use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::PrimitiveTopologyClass;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
    TileRenderPipelineDescriptor,
};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::resource::RenderPipeline as _;
use reims_vgpu_core::platform::types::NsErrorText;
use reims_vgpu_core::platform::values::{
    DepthStencilDesc, RenderPipelineDesc, SamplerDesc, StencilDesc,
};
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::sync::Mutex;

use crate::answer::{
    Policy, RUNTIME_DOMAIN, decide, decide_describing, error_text, get_describing, plain, plain_at,
    unrepresentable,
};
use crate::gpu::desc::{VulkanRenderPipelineDescriptor, VulkanTileRenderPipelineDescriptor};
use crate::gpu::device::{VulkanDevice, nil_pipeline_descriptor};
use crate::gpu::metal_value;
use crate::gpu::resource::{
    RenderDescriptor, RenderPipelineObject, VulkanDepthStencilState, VulkanRenderPipeline,
    VulkanSamplerState, render_pipeline_of, sampler_state,
};
use crate::gpu::serialized;
use crate::gpu::timing;
use crate::platform::{Obj, VulkanPlatform};

/// The render pipeline states made with indirect command buffer support, held weakly: an
/// indirect command names one by the runtime's state, whose input primitive topology class only
/// this platform holds ([`points_sized`]).
static INDIRECT_STATES: Mutex<Vec<Weak<RenderPipelineObject>>> = Mutex::new(Vec::new());

/// Whether the runtime's state `raw`, which an indirect command names, sizes its points as Metal
/// does (`crate::gpu::encode_render`'s `sized`): its vertex function writes `[[point_size]]`, or
/// each live state of the program's that is `raw` is of the point input primitive topology class.
/// `None` where it writes none and no live state of the program's is `raw` (the program released
/// it).
pub(crate) fn points_sized(raw: &rt::RenderPipelineState) -> Option<bool> {
    if plain(|| raw.writes_point_size()) {
        return Some(true);
    }
    let states: Vec<_> = plain(|| INDIRECT_STATES.lock().clone());
    let classes: Vec<_> = states
        .iter()
        .filter_map(Obj::upgrade)
        .filter(|state| state.raw == *raw)
        .map(|state| state.input_primitive_topology)
        .collect();
    (!classes.is_empty()).then(|| classes.iter().all(|c| *c == PrimitiveTopologyClass::Point))
}

/// A `u64` property the runtime holds as `u32`, sent with `sel`.
fn small(sel: &'static str, what: &str, raw: u64) -> u32 {
    u32::try_from(raw).unwrap_or_else(|_| unrepresentable(sel, &format!("{what} {raw}")))
}

/// The resource ID a value's or a descriptor's resource index places a state at: Metal's index,
/// forced or not (the module docs).
fn placement(resource_index: Option<(u64, bool)>) -> rt::IdPlacement {
    resource_index.map_or(rt::IdPlacement::Any, |(index, _force)| {
        rt::IdPlacement::At(rt::ResourceId(index))
    })
}

/// A sampler descriptor's debug line ([the module docs](self)): the selector, then each property
/// of `SamplerDesc` raw, `unset` for one the value does not set: the min, mag and mip filters,
/// the address modes, the border colour, normalized coordinates, the LOD clamps and average, the
/// maximum anisotropy, the compare function, the reduction mode, `supportArgumentBuffers` and the
/// resource index.
pub struct Sampler<'a>(pub &'a SamplerDesc);

/// A property of a [`Sampler`] line: its value, or `unset`.
struct Set<T>(Option<T>);

impl<T: std::fmt::Display> std::fmt::Display for Set<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(v) => v.fmt(f),
            None => f.write_str("unset"),
        }
    }
}

impl std::fmt::Display for Sampler<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.0;
        write!(
            f,
            "newSamplerStateWithDescriptor: min filter {}, mag filter {}, mip filter {}, address \
             modes s {} t {} r {}, border colour {}, normalized coordinates {}, lod min {} max {} \
             average {}, max anisotropy {}, compare function {}, reduction mode {}, \
             supportArgumentBuffers {}, resource index {}",
            Set(d.min_filter),
            Set(d.mag_filter),
            Set(d.mip_filter),
            Set(d.s_address_mode),
            Set(d.t_address_mode),
            Set(d.r_address_mode),
            Set(d.border_color),
            Set(d.normalized_coordinates),
            Set(d.lod_min_clamp),
            Set(d.lod_max_clamp),
            Set(d.lod_average),
            Set(d.max_anisotropy),
            Set(d.compare_function),
            Set(d.reduction_mode),
            Set(d.support_argument_buffers),
            Set(d.resource_index),
        )
    }
}

/// The runtime's sampler descriptor and options of `d`, each property set in the value's order,
/// after its debug line ([`Sampler`]).
fn sampler_descriptor(d: &SamplerDesc) -> (SamplerDescriptor, rt::SamplerStateOptions) {
    log::debug!("{}", Sampler(d));
    let mut s = SamplerDescriptor::new();
    if let Some(v) = d.min_filter {
        s.min_filter = metal_value("setMinFilter:", "sampler filter", v);
    }
    if let Some(v) = d.mag_filter {
        s.mag_filter = metal_value("setMagFilter:", "sampler filter", v);
    }
    if let Some(v) = d.lod_average {
        s.lod_average = v;
    }
    if let Some(v) = d.compare_function {
        s.compare_function = metal_value("setCompareFunction:", "compare function", v);
    }
    if let Some(v) = d.s_address_mode {
        s.s_address_mode = metal_value("setSAddressMode:", "address mode", v);
    }
    if let Some(v) = d.t_address_mode {
        s.t_address_mode = metal_value("setTAddressMode:", "address mode", v);
    }
    if let Some(v) = d.r_address_mode {
        s.r_address_mode = metal_value("setRAddressMode:", "address mode", v);
    }
    if let Some(v) = d.mip_filter {
        s.mip_filter = metal_value("setMipFilter:", "mip filter", v);
    }
    if let Some(v) = d.max_anisotropy {
        s.max_anisotropy = small("setMaxAnisotropy:", "an anisotropy", v);
    }
    if let Some(v) = d.normalized_coordinates {
        s.normalized_coordinates = v;
    }
    let mut options = rt::SamplerStateOptions::default();
    if let Some(v) = d.support_argument_buffers {
        options.support_argument_buffers = v;
    }
    if let Some(v) = d.reduction_mode {
        s.reduction_mode = metal_value("setReductionMode:", "reduction mode", v);
    }
    if let Some(v) = d.lod_min_clamp {
        s.lod_min_clamp = v;
    }
    if let Some(v) = d.lod_max_clamp {
        s.lod_max_clamp = v;
    }
    if let Some(v) = d.border_color {
        s.border_color = rt::sampler_border_color(v);
    }
    options.placement = placement(d.resource_index.map(|index| (index, false)));
    (s, options)
}

/// `newSamplerStateWithDescriptor:`: a refusal dies at the selector, as Metal makes a state of
/// any descriptor, and its process ends at an ID past its limit.
pub(crate) fn new_sampler_state(
    device: &VulkanDevice,
    desc: &SamplerDesc,
) -> RaiseResult<Option<VulkanSamplerState>> {
    let sel = "newSamplerStateWithDescriptor:";
    let (descriptor, options) = sampler_descriptor(desc);
    match plain_at(sel, || {
        device
            .raw
            .new_sampler_state_with_options(&descriptor, options)
    }) {
        Ok(raw) => Ok(Some(sampler_state(raw))),
        Err(e) => decide(sel, Policy::Dies, e),
    }
}

/// A face's stencil of `s`, every property set in field order.
fn stencil(s: &StencilDesc) -> StencilDescriptor {
    let mut out = StencilDescriptor::default();
    out.stencil_compare_function = metal_value(
        "setStencilCompareFunction:",
        "compare function",
        s.stencil_compare_function,
    );
    out.stencil_failure_operation = metal_value(
        "setStencilFailureOperation:",
        "stencil operation",
        s.stencil_failure_operation,
    );
    out.depth_failure_operation = metal_value(
        "setDepthFailureOperation:",
        "stencil operation",
        s.depth_failure_operation,
    );
    out.depth_stencil_pass_operation = metal_value(
        "setDepthStencilPassOperation:",
        "stencil operation",
        s.depth_stencil_pass_operation,
    );
    out.read_mask = s.read_mask;
    out.write_mask = s.write_mask;
    out
}

/// `newDepthStencilStateWithDescriptor:`: Metal makes a state of any descriptor, and so does the
/// runtime; a refusal dies at the selector.
pub(crate) fn new_depth_stencil_state(
    device: &VulkanDevice,
    desc: &DepthStencilDesc,
) -> RaiseResult<Option<VulkanDepthStencilState>> {
    let sel = "newDepthStencilStateWithDescriptor:";
    let mut d = DepthStencilDescriptor::default();
    if let Some(v) = desc.depth_compare_function {
        d.depth_compare_function = metal_value("setDepthCompareFunction:", "compare function", v);
    }
    if let Some(v) = desc.depth_write_enabled {
        d.depth_write_enabled = v;
    }
    if let Some(front) = &desc.front {
        d.front_face_stencil = stencil(front);
    }
    if let Some(back) = &desc.back {
        d.back_face_stencil = stencil(back);
    }
    match plain_at(sel, || device.raw.new_depth_stencil_state(&d)) {
        Ok(raw) => Ok(Some(Obj::new(raw))),
        Err(e) => decide(sel, Policy::Dies, e),
    }
}

/// The `NSError` a pipeline creator answers for the runtime's refusal `e` of a pipeline, where
/// Metal answers one (the module docs): the runtime's domain and text, and Metal's code where it
/// was measured, else 0; `None` for every other refusal, which dies.
/// - Code 0: what the compile finds wrong with the functions or the descriptor, and a vertex
///   function writing a point size under a line or triangle topology.
/// - Code 3 ([`METAL_REFUSES`]): a function the runtime refuses as Metal itself refuses it on
///   this device, by a GPU family the device lacks (`Needs::MetalGpuFamily`: a render state set
///   on an indirect render command) or by the devices Metal was measured to refuse it on
///   (`Needs::MetalRefuses`: a fragment function's read of a program-scope
///   `thread_index_in_quadgroup`).
/// - Code 2 ([`UNDEFINED_SYMBOLS`]): a function whose import no linked dynamic library exports,
///   or whose install name loaded no library (`NeedsInput` of `DynamicSymbol` or
///   `DynamicLibrary`).
/// - Metal's error as the runtime states it, with its code ([`rt::Error::MetalError`]), which the
///   runtime answers only where plain Metal answers nil with that `NSError`: a state that
///   supports indirect command buffers whose function uses a direct texture or sampler argument
///   once Metal links the pipeline (`IndirectCommandsDirectArgument`), and a render state with
///   rasterization off whose vertex function returns a value (`UnrasterizedVertexReturn`), among
///   them; a problem the runtime adds there is the `NSError` too.
///
/// An answer the runtime cannot prove ([`rt::Error::Unproven`]) dies, naming it (`unproven`), as
/// does every other refusal.
pub(crate) fn pipeline_error(e: &rt::Error) -> Option<NsErrorText> {
    use metal2vulkan::error::{Error as Compile, Needs, RuntimeInput};
    let code = match e {
        rt::Error::Compile(Compile::InvalidInput(_) | Compile::InvalidDescriptor(_)) => 0,
        rt::Error::Invalid(i) if i.problem == rt::Problem::PointSizeTopology => 0,
        rt::Error::Unsupported(u)
            if matches!(u.needs, Needs::MetalGpuFamily(_) | Needs::MetalRefuses(_)) =>
        {
            METAL_REFUSES
        }
        rt::Error::Compile(Compile::NeedsInput(
            RuntimeInput::DynamicSymbol { .. } | RuntimeInput::DynamicLibrary { .. },
        )) => UNDEFINED_SYMBOLS,
        rt::Error::MetalError(m) => m.code,
        _ if !crate::answer::strict() => {
            log::error!(
                target: "reims_vgpu_vulkan::gpu::states",
                "pipeline creation rejected by runtime (non-fatal): {e}"
            );
            METAL_REFUSES
        }
        _ => return None,
    };
    Some(NsErrorText {
        code,
        ..error_text(e)
    })
}

/// The `NSError` code of Metal's pipeline creators for a function its compiler refuses on the
/// device ("Encountered unlowered function call to air.set_cull_mode_render_command", to
/// `air.get_thread_index_in_quadgroup.i32`), at the compute, render and tile creators (measured:
/// the module docs).
const METAL_REFUSES: i64 = 3;

/// The `NSError` code of Metal's pipeline creators for a function whose import no library it
/// links exports, the library at its install name loading or not ("Undefined symbols:"), at the
/// compute, render and tile creators (measured: the module docs).
const UNDEFINED_SYMBOLS: i64 = 2;

/// The colour attachment of a raw pixel format, sent with `setPixelFormat:`: none for
/// `MTLPixelFormatInvalid`.
fn color_attachment(raw: u64) -> Option<ColorAttachmentDescriptor> {
    (raw != 0)
        .then(|| ColorAttachmentDescriptor::new(crate::gpu::pixel_format("setPixelFormat:", raw)))
}

/// `newRenderPipelineStateWithDescriptor:error:` of the display's value: its two functions and
/// colour 0's format, blending off, no vertex descriptor, the resource index as the placement.
/// The compile's refusal of a format the functions cannot write (an integer target) is the
/// `NSError`, as Metal answers nil and an error there; no path reaches it under a handler, so
/// every other refusal dies at the send.
pub(crate) fn new_render_pipeline_state(
    device: &VulkanDevice,
    desc: &RenderPipelineDesc<'_, VulkanPlatform>,
) -> Result<Option<VulkanRenderPipeline>, NsErrorText> {
    let sel = "newRenderPipelineStateWithDescriptor:error:";
    let mut d =
        RenderPipelineDescriptor::new(StageSource::Function(desc.vertex.function_descriptor()));
    d.fragment = Some(StageSource::Function(desc.fragment.function_descriptor()));
    d.color_attachments[0] = color_attachment(desc.color0_pixel_format);
    let mut options = rt::PipelineOptions::default();
    options.placement = placement(desc.resource_index);
    options.label = desc.label.map(str::to_owned);
    timing::creation(
        sel,
        timing::RENDER_LATER,
        || match plain_at(sel, || device.raw.new_render_pipeline_state(&d, &options)) {
            Ok(state) => Ok(Some(render_pipeline_of(
                state,
                PrimitiveTopologyClass::Unspecified,
                Some(RenderDescriptor::Render(Box::new(d.clone()))),
            ))),
            Err(e) => match pipeline_error(&e) {
                Some(error) => Err(error),
                // Decided as a getter's refusal: none of them returns.
                None => get_describing(sel, Some(&render_formats(&d)), || Err(e)),
            },
        },
        || timing::render_stages(&d),
        |made| timing::answer(made, |s| timing::state(s.unique_identifier())),
    )
}

/// The runtime's options of Metal's raw `MTLPipelineOption`, the state placed at `placement` and
/// labelled `label`.
fn pipeline_options(
    options: u64,
    placement: rt::IdPlacement,
    label: Option<String>,
) -> rt::PipelineOptions {
    let mut out = rt::PipelineOptions::new(rt::PipelineOption::from_raw(options));
    out.placement = placement;
    out.label = label;
    out
}

/// A pipeline creator's answer to core: the state, of the descriptor's input primitive topology
/// `class`, the `NSError` ([`pipeline_error`]), or a death at the selector for the rest: an ID past
/// Metal's limit, a gap of the runtime (an unspecialized function constant among them, which
/// Metal aborts for), and its other refusals, as Metal raises at none. A state the runtime made
/// that supports indirect command buffers dies where whether Metal refuses it is not decided
/// ([`direct_arguments_decided`]).
fn made(
    sel: &'static str,
    (class, descriptor): (PrimitiveTopologyClass, RenderDescriptor),
    made: Result<rt::RenderPipelineState, rt::Error>,
) -> RaiseResult<Result<Option<VulkanRenderPipeline>, NsErrorText>> {
    match made {
        Ok(state) => {
            let indirect = matches!(&descriptor,
                RenderDescriptor::Render(d) if d.support_indirect_command_buffers);
            if let RenderDescriptor::Render(d) = &descriptor
                && indirect
            {
                direct_arguments_decided(sel, d);
            }
            let made = render_pipeline_of(state, class, Some(descriptor));
            if indirect {
                plain(|| {
                    let mut states = INDIRECT_STATES.lock();
                    states.retain(|state| state.strong_count() > 0);
                    states.push(made.downgrade());
                });
            }
            Ok(Ok(Some(made)))
        }
        Err(e) => match pipeline_error(&e) {
            Some(error) => Ok(Err(error)),
            None => match &e {
                rt::Error::Unproven(unproven) => {
                    self::unproven(sel, unproven, &e, &formats(&descriptor))
                }
                _ => decide_describing(sel, Policy::Dies, e, Some(&formats(&descriptor))),
            },
        },
    }
}

/// Dies at `sel`, the creation of a render state of `d`, which supports indirect command buffers
/// and the runtime made, where the compile layer refuses to say whether its functions use a
/// direct texture or sampler argument once linked (`stages_using_direct_arguments`, under the
/// standard budget, as the runtime asks it): it decides from the arguments' roles alone, and
/// refuses where it cannot decode or specialize a function, among others. Metal answers nil and
/// an `NSError` for a state whose functions use one (the module docs). The stages are asked with
/// the dynamic libraries their imports reach loaded, as the runtime asks them
/// ([`with_dependencies`]).
fn direct_arguments_decided(sel: &'static str, d: &RenderPipelineDescriptor) {
    let mut loaded = d.clone();
    loaded.vertex = with_dependencies(&d.vertex);
    loaded.fragment = d.fragment.as_ref().map(with_dependencies);
    let entries = metal2vulkan::pipeline::StageEntries::new();
    let budget = metal2vulkan::Budget::standard();
    if let Err(e) = plain_at(sel, || {
        loaded.stages_using_direct_arguments(&budget, &entries)
    }) {
        undecided_direct_arguments(sel, &e, &render_formats(d));
    }
}

/// Dies at `sel`, the creation of a compute state of `d`, which supports indirect command buffers
/// and the runtime made, where the compile layer refuses to say whether the kernel uses a direct
/// texture or sampler argument (`uses_direct_arguments`, as the runtime asks it), as
/// [`direct_arguments_decided`] dies for a render state's functions.
pub(crate) fn kernel_direct_arguments_decided(sel: &'static str, d: &ComputePipelineDescriptor) {
    let mut loaded = d.clone();
    loaded.kernel = with_dependencies(&d.kernel);
    let budget = metal2vulkan::Budget::standard();
    if let Err(e) = plain_at(sel, || loaded.uses_direct_arguments(&budget)) {
        undecided_direct_arguments(sel, &e, &compute_formats(d));
    }
}

/// The death at `sel` of a state that supports indirect command buffers whose functions' use of
/// a direct texture or sampler argument the compile layer refused to decide, with `e`; `formats`
/// are the descriptor's.
fn undecided_direct_arguments(sel: &'static str, e: &metal2vulkan::error::Error, formats: &str) {
    let what = format!(
        "whether a state that supports indirect command buffers has a function that uses a \
         direct texture or sampler argument, for which Metal answers nil and an NSError: the \
         compile layer refused to decide it ({e}), and the runtime made the state without \
         deciding; {formats}"
    );
    if !crate::answer::strict() {
        log::error!(
            target: "reims_vgpu_vulkan::gpu::states",
            "platform call {sel} skipped direct-argument check (non-fatal): {what}"
        );
        return;
    }
    unrepresentable(sel, &what);
}

/// `stage` with the dynamic libraries its imports reach loaded from the files their install
/// names name, as the runtime loads them as it makes a pipeline state (its
/// `stage_with_dependencies`): a name that is no absolute path, or a file that holds no dynamic
/// library, loads nothing.
fn with_dependencies(stage: &StageSource) -> StageSource {
    match stage {
        StageSource::Function(function) => {
            StageSource::Function(function.clone().with_loaded_dependencies(|name| {
                let path = std::path::Path::new(name);
                if !path.is_absolute() {
                    return None;
                }
                let bytes = std::fs::read(path).ok()?;
                let library = metal2vulkan::library::Library::from_metallib(&bytes).ok()?;
                metal2vulkan::library::DynamicLibrary::loaded(library).ok()
            }))
        }
        StageSource::Compiled(_) => stage.clone(),
    }
}

/// The death at `sel` of `unproven`, a call whose answer on Metal the runtime cannot prove (`e`,
/// [`rt::Error::Unproven`]): Metal may make what it asks for, or answer nil and an `NSError`, and
/// the runtime does neither. Its line names the call and the problem: for a state that supports
/// indirect command buffers, the stage of the function whose direct texture or sampler argument
/// only a rule the runtime keeps beyond Metal keeps, and the rule (`LinkedUseGap`); for any other
/// problem, the problem as the runtime states it. Then the runtime's error and the descriptor's
/// formats, `formats`.
fn unproven(sel: &'static str, unproven: &rt::Unproven, e: &rt::Error, formats: &str) -> ! {
    let what = match &unproven.problem {
        rt::Problem::IndirectCommandsDirectArgumentUnproven { stage, reason } => format!(
            "a state that supports indirect command buffers whose {} function uses a direct \
             texture or sampler argument only by a rule the runtime keeps beyond Metal \
             ({reason:?})",
            format!("{stage:?}").to_lowercase()
        ),
        problem => format!("{problem:?}"),
    };
    unrepresentable(
        sel,
        &format!(
            "{what}, whose answer on Metal (made, or nil and an NSError) is unmeasured, in {}: {}; \
             {formats}",
            unproven.call,
            crate::answer::refusal(e)
        ),
    )
}

/// What a refused pipeline's descriptor holds that the runtime's error does not say, for its
/// death's line ([`decide_describing`]): a render descriptor's formats ([`render_formats`]) or a
/// tile descriptor's ([`tile_formats`]).
pub(crate) fn formats(d: &RenderDescriptor) -> String {
    match d {
        RenderDescriptor::Render(d) => render_formats(d),
        RenderDescriptor::Tile(d) => tile_formats(d),
    }
}

/// A raw Metal value as a death's line names it: the number, then its enumerator.
fn named(raw: u32, name: &str) -> String {
    format!("{raw} ({name})")
}

/// The line of a descriptor's formats, `items`, or `none` for no items.
fn listed(items: Vec<String>, none: &str) -> String {
    let items = if items.is_empty() {
        none.to_owned()
    } else {
        items.join(", ")
    };
    format!("the descriptor's formats: {items}")
}

/// A render descriptor's formats, by index: each colour attachment's pixel format, its blend
/// state ([`blend_state`]) and write mask, the depth and stencil formats, and its vertex
/// descriptor's attribute formats.
pub(crate) fn render_formats(d: &RenderPipelineDescriptor) -> String {
    let format = |f: metal2vulkan::metal::PixelFormat| named(f.as_raw(), f.name());
    let mut items: Vec<String> = (d.color_attachments.iter().enumerate())
        .filter_map(|(i, c)| c.as_ref().map(|c| (i, c)))
        .map(|(i, c)| {
            let (format, mask) = (format(c.pixel_format), c.write_mask.as_raw());
            let blend = blend_state(c);
            format!("colour {i} {format} {blend} write mask {mask:#x}")
        })
        .collect();
    if items.is_empty() {
        items.push("no colour attachments".to_owned());
    }
    items.push(
        d.depth_format
            .map_or("no depth".to_owned(), |f| format!("depth {}", format(f))),
    );
    items.push(d.stencil_format.map_or("no stencil".to_owned(), |f| {
        format!("stencil {}", format(f))
    }));
    match &d.vertex_descriptor {
        None => items.push("no vertex descriptor".to_owned()),
        Some(v) if v.attributes.is_empty() => items.push("no vertex attributes".to_owned()),
        Some(v) => items.extend(v.attributes.iter().map(|(i, a)| {
            format!(
                "vertex attribute {i} {}",
                named(a.format.as_raw(), a.format.name())
            )
        })),
    }
    listed(items, "")
}

/// A colour attachment's blend state as a death's line names it: whether it blends, then its rgb
/// and alpha factors and operations, numeric and named, which it holds blending or not.
fn blend_state(c: &ColorAttachmentDescriptor) -> String {
    let factor = |f: metal2vulkan::metal::BlendFactor| named(f.as_raw(), f.name());
    let operation = |o: metal2vulkan::metal::BlendOperation| named(o.as_raw(), o.name());
    format!(
        "blending {} rgb source {} destination {} operation {} alpha source {} destination {} \
         operation {}",
        if c.blending_enabled { "on" } else { "off" },
        factor(c.source_rgb_blend_factor),
        factor(c.destination_rgb_blend_factor),
        operation(c.rgb_blend_operation),
        factor(c.source_alpha_blend_factor),
        factor(c.destination_alpha_blend_factor),
        operation(c.alpha_blend_operation),
    )
}

/// A tile descriptor's colour attachment formats, by index.
pub(crate) fn tile_formats(d: &TileRenderPipelineDescriptor) -> String {
    let colours = (d.color_attachment_formats.iter().enumerate())
        .filter_map(|(i, f)| f.map(|f| format!("colour {i} {}", named(f.as_raw(), f.name()))));
    listed(colours.collect(), "no colour attachments")
}

/// A compute descriptor's stage-in attribute formats, by index.
pub(crate) fn compute_formats(d: &ComputePipelineDescriptor) -> String {
    match &d.stage_input {
        None => listed(Vec::new(), "no stage input descriptor"),
        Some(s) => listed(
            (s.attributes.iter())
                .map(|(i, a)| {
                    format!(
                        "stage-in attribute {i} {}",
                        named(a.format.as_raw(), a.format.name())
                    )
                })
                .collect(),
            "no stage-in attributes",
        ),
    }
}

fn raw_format_or_die(sel: &'static str, raw: &serialized::RawFormat) -> Result<(), NsErrorText> {
    if crate::answer::strict() {
        raw.dies(sel);
    }
    let description = format!(
        "a state whose {} is pixel format {} (unsupported)",
        raw.attachment, raw.raw
    );
    log::error!(
        target: "reims_vgpu_vulkan::gpu::states",
        "platform call {sel} skipped with error (non-fatal): {description}"
    );
    Err(NsErrorText {
        domain: RUNTIME_DOMAIN.to_owned(),
        code: METAL_REFUSES,
        description,
    })
}

/// `newRenderPipelineStateWithDescriptor:options:reflection:error:` of a guest's descriptor, at
/// its resource index ([`crate::gpu::desc::PipelineDescriptorObject::placement`]); no reflection
/// is asked for, as core asks for none. A nil descriptor, an empty blob's, is Metal's `NSError`
/// for one ([`nil_pipeline_descriptor`]). A pixel format the runtime holds raw dies here
/// ([`serialized::RawFormat`]). A blob's `textureWriteRoundingMode` of
/// [`serialized::NEAREST_EVEN`] is decided once the runtime made the state, its compile's refusal
/// staying the `NSError`: it dies where Metal rounds a write the state may make otherwise than
/// the runtime ([`serialized::rounded_writes`]), but a texture write of a state without indirect
/// command buffer support, which its draws decide ([`serialized::rounded_texture_writes`]).
pub(crate) fn new_render_pipeline_state_from_descriptor(
    device: &VulkanDevice,
    descriptor: Option<&VulkanRenderPipelineDescriptor>,
    options: u64,
) -> RaiseResult<Result<Option<VulkanRenderPipeline>, NsErrorText>> {
    let sel = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
    let Some(descriptor) = descriptor else {
        return Ok(Err(nil_pipeline_descriptor()));
    };
    if let Some(raw) = &descriptor.raw_format
        && let Err(err) = raw_format_or_die(sel, raw)
    {
        return Ok(Err(err));
    }
    let (d, placement) = plain(|| (descriptor.desc.lock().clone(), descriptor.placement()));
    let options = pipeline_options(options, placement, descriptor.label.clone());
    timing::creation(
        sel,
        timing::RENDER_LATER,
        || {
            let state = plain_at(sel, || device.raw.new_render_pipeline_state(&d, &options))
                .map_err(|e| {
                    if crate::answer::strict() {
                        let colors = d.color_attachments.map(|c| c.map(|c| c.pixel_format));
                        let depth = (d.depth_format, d.stencil_format);
                        serialized::RawFormat::refused(
                            sel,
                            &colors,
                            depth,
                            || render_formats(&d),
                            e,
                        )
                    } else {
                        e
                    }
                });
            if descriptor.nearest_even
                && let Ok(raw) = &state
                && let Some(what) = plain(|| {
                    serialized::rounded_writes(
                        &d,
                        &raw.fragment_color_outputs(),
                        d.support_indirect_command_buffers && raw.fragment_writes_textures(),
                    )
                })
            {
                serialized::rounded_writes_inexact(sel, &what);
            }
            let class = d.input_primitive_topology;
            made(
                sel,
                (class, RenderDescriptor::Render(Box::new(d.clone()))),
                state,
            )
        },
        || timing::render_stages(&d),
        |made| timing::raised_answer(made, |s| timing::state(s.unique_identifier())),
    )
}

/// `newRenderPipelineStateWithTileDescriptor:options:reflection:error:` of a guest's descriptor;
/// no reflection is asked for. A tile state takes no resource ID, as Metal's answers 0: core sets
/// no resource index on a tile descriptor, so its placement is the runtime's. A nil descriptor,
/// an empty blob's, is Metal's `NSError` for one ([`nil_pipeline_descriptor`]). A pixel format
/// the runtime holds raw dies here ([`serialized::RawFormat`]).
pub(crate) fn new_tile_render_pipeline_state(
    device: &VulkanDevice,
    descriptor: Option<&VulkanTileRenderPipelineDescriptor>,
    options: u64,
) -> RaiseResult<Result<Option<VulkanRenderPipeline>, NsErrorText>> {
    let sel = "newRenderPipelineStateWithTileDescriptor:options:reflection:error:";
    let Some(descriptor) = descriptor else {
        return Ok(Err(nil_pipeline_descriptor()));
    };
    if let Some(raw) = &descriptor.raw_format
        && let Err(err) = raw_format_or_die(sel, raw)
    {
        return Ok(Err(err));
    }
    let (d, placement) = plain(|| (descriptor.desc.lock().clone(), descriptor.placement()));
    let options = pipeline_options(options, placement, descriptor.label.clone());
    timing::creation(
        sel,
        timing::DISPATCH_LATER,
        || {
            made(
                sel,
                (
                    PrimitiveTopologyClass::Unspecified,
                    RenderDescriptor::Tile(Box::new(d.clone())),
                ),
                plain_at(sel, || {
                    device.raw.new_tile_render_pipeline_state(&d, &options)
                })
                .map_err(|e| {
                    if crate::answer::strict() {
                        let colors = d.color_attachment_formats;
                        serialized::RawFormat::refused(
                            sel,
                            &colors,
                            (None, None),
                            || tile_formats(&d),
                            e,
                        )
                    } else {
                        e
                    }
                }),
            )
        },
        || format!("tile {}", timing::stage(&d.tile)),
        |made| timing::raised_answer(made, |s| timing::state(s.unique_identifier())),
    )
}

#[cfg(test)]
mod tests {
    use metal2vulkan::metal::PixelFormat;
    use metal2vulkan::pipeline::TileRenderPipelineDescriptor;
    use reims_vgpu_core::platform::Handle;
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::resource::{DepthStencilState, SamplerState};
    use reims_vgpu_core::platform::resource::{Library, RenderPipeline};
    use reims_vgpu_core::raise::OnFault;
    use reims_vgpu_core::sync::Mutex;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::desc::PipelineDescriptorObject;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::VulkanFunction;

    /// `MTLSamplerMinMagFilterLinear`.
    const LINEAR: u64 = 1;

    /// A sampler descriptor's debug line names every property of the record, raw, and `unset`
    /// for those the value does not set; a float prints as Rust prints it
    #[test]
    fn a_samplers_debug_line_is_its_whole_descriptor() {
        let d = SamplerDesc {
            min_filter: Some(2),
            mag_filter: Some(LINEAR),
            lod_average: Some(false),
            compare_function: Some(7),
            s_address_mode: Some(0),
            t_address_mode: Some(2),
            r_address_mode: Some(5),
            mip_filter: Some(0),
            max_anisotropy: Some(16),
            normalized_coordinates: Some(true),
            support_argument_buffers: Some(true),
            reduction_mode: Some(0),
            lod_min_clamp: Some(0.0),
            lod_max_clamp: Some(f32::MAX),
            border_color: Some(1),
            resource_index: None,
        };
        assert_eq!(
            Sampler(&d).to_string(),
            "newSamplerStateWithDescriptor: min filter 2, mag filter 1, mip filter 0, address \
             modes s 0 t 2 r 5, border colour 1, normalized coordinates true, lod min 0 max \
             340282350000000000000000000000000000000 average false, max anisotropy 16, compare \
             function 7, reduction mode 0, supportArgumentBuffers true, resource index unset"
        );
        assert_eq!(
            Sampler(&SamplerDesc::default()).to_string(),
            "newSamplerStateWithDescriptor: min filter unset, mag filter unset, mip filter unset, \
             address modes s unset t unset r unset, border colour unset, normalized coordinates \
             unset, lod min unset max unset average unset, max anisotropy unset, compare function \
             unset, reduction mode unset, supportArgumentBuffers unset, resource index unset"
        );
    }

    /// The display's vertex and plain fragment functions, from the platform's shader library.
    fn display_functions(d: &VulkanDevice) -> (VulkanFunction, VulkanFunction) {
        use reims_vgpu_core::display::pipeline::{FRAGMENT_FUNCTION, VERTEX_FUNCTION};
        let library = d.new_shader_library().unwrap().unwrap();
        let function = |name| {
            library
                .new_function(OnFault::Return, name)
                .unwrap()
                .unwrap()
        };
        (function(VERTEX_FUNCTION), function(FRAGMENT_FUNCTION))
    }

    /// The suite's vertex and fragment functions, which use no texture or sampler: a state that
    /// supports indirect command buffers takes them, where Metal and the runtime
    /// refuse one of the display's, whose fragment function samples its texture.
    fn indirect_functions() -> (VulkanFunction, VulkanFunction) {
        use crate::gpu::serialized::tests::fixture_function;
        (
            fixture_function(include_bytes!(
                "../../../../tests/suite/fixtures/shaders/vert.metallib"
            )),
            fixture_function(include_bytes!(
                "../../../../tests/suite/fixtures/shaders/frag.metallib"
            )),
        )
    }

    /// A guest's render pipeline descriptor, as the decoder makes one: the two functions, colour
    /// 0 of `format`, and the resource index core sets.
    fn guest_descriptor(
        (vertex, fragment): &(VulkanFunction, VulkanFunction),
        format: PixelFormat,
        resource_index: Option<(u64, bool)>,
    ) -> VulkanRenderPipelineDescriptor {
        let mut desc =
            RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
        desc.fragment = Some(StageSource::Function(fragment.function_descriptor()));
        desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
        Obj::new(PipelineDescriptorObject {
            desc: Mutex::new(desc),
            resource_index: Mutex::new(resource_index),
            ignored: Vec::new(),
            nearest_even: false,
            label: None,
            raw_format: None,
        })
    }

    /// A guest's descriptor makes a pipeline, one of indirect command buffer support at its
    /// resource index (of functions that use no direct texture or sampler), with a new identifier
    /// at every creation
    #[test]
    fn guest_descriptors_make_pipelines() {
        let d = dev();
        let functions = display_functions(&d);
        let indirect = indirect_functions();
        let placed = guest_descriptor(&indirect, PixelFormat::BGRA8Unorm, Some((9, true)));
        plain(|| placed.desc.lock().support_indirect_command_buffers = true);
        let a = d
            .new_render_pipeline_state_from_descriptor(Some(&placed), 0)
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(a.gpu_resource_id(), 9);
        let any = guest_descriptor(&functions, PixelFormat::BGRA8Unorm, None);
        let b = d
            .new_render_pipeline_state_from_descriptor(Some(&any), 0)
            .unwrap()
            .unwrap()
            .unwrap();
        // A state without indirect command buffer support has none, as on Metal.
        assert_eq!(b.gpu_resource_id(), 0);
        let c = d
            .new_render_pipeline_state_from_descriptor(Some(&any), 0)
            .unwrap()
            .unwrap()
            .unwrap();
        let ids = [
            a.unique_identifier(),
            b.unique_identifier(),
            c.unique_identifier(),
        ];
        assert!(ids.iter().all(|&id| id != 0));
        assert!(
            ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2],
            "{ids:?}"
        );
        assert_eq!(b.clone_ref().unique_identifier(), b.unique_identifier());
    }

    /// `desc` again, labelled `label`.
    fn labelled(
        desc: &VulkanRenderPipelineDescriptor,
        label: &str,
    ) -> VulkanRenderPipelineDescriptor {
        Obj::new(PipelineDescriptorObject {
            desc: Mutex::new(plain(|| desc.desc.lock().clone())),
            resource_index: Mutex::new(plain(|| *desc.resource_index.lock())),
            ignored: Vec::new(),
            nearest_even: false,
            label: Some(label.to_owned()),
            raw_format: None,
        })
    }

    /// A descriptor's label and a value's are their states', a diagnostic that changes nothing
    /// else: the state takes its resource ID, and a compile's refusal is the error, as an
    /// unlabelled one's
    #[test]
    fn render_labels_are_the_states() {
        use reims_vgpu_core::platform::values::RenderPipelineDesc;
        let d = dev();
        let functions = display_functions(&d);
        let indirect = indirect_functions();
        let placed = guest_descriptor(&indirect, PixelFormat::BGRA8Unorm, Some((9, true)));
        let placed = labelled(&placed, "guest");
        plain(|| placed.desc.lock().support_indirect_command_buffers = true);
        let a = d
            .new_render_pipeline_state_from_descriptor(Some(&placed), 0)
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!((a.gpu_resource_id(), a.raw.label()), (9, Some("guest")));
        let integer = guest_descriptor(&functions, PixelFormat::R32Uint, None);
        let refused = |desc: &VulkanRenderPipelineDescriptor| {
            d.new_render_pipeline_state_from_descriptor(Some(desc), 0)
                .unwrap()
                .unwrap_err()
        };
        assert_eq!(refused(&integer), refused(&labelled(&integer, "guest")));
        let (vertex, fragment) = &functions;
        let value = d
            .new_render_pipeline_state(&RenderPipelineDesc {
                label: Some("display"),
                vertex,
                fragment,
                color0_pixel_format: 80,
                resource_index: None,
            })
            .unwrap()
            .unwrap();
        assert_eq!(value.raw.label(), Some("display"));
    }

    /// A tile descriptor whose tile function is no kernel is refused, as Metal's compile refuses
    /// it
    #[test]
    fn a_tile_descriptor_of_a_vertex_function_is_refused() {
        let d = dev();
        let (vertex, _) = display_functions(&d);
        let mut desc =
            TileRenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
        desc.color_attachment_formats[0] = Some(PixelFormat::BGRA8Unorm);
        let tile = Obj::new(PipelineDescriptorObject {
            desc: Mutex::new(desc),
            resource_index: Mutex::new(None),
            ignored: Vec::new(),
            nearest_even: false,
            label: None,
            raw_format: None,
        });
        let made = d.new_tile_render_pipeline_state(Some(&tile), 0);
        assert!(!matches!(made, Ok(Ok(Some(_)))), "{made:?}");
    }

    /// Equal requests answer one object while it lives, as Metal's do; another request another
    #[test]
    fn equal_sampler_requests_are_one_object() {
        let d = dev();
        let linear = SamplerDesc {
            mag_filter: Some(LINEAR),
            ..SamplerDesc::default()
        };
        let a = d.new_sampler_state(&linear).unwrap().unwrap();
        let b = d.new_sampler_state(&linear).unwrap().unwrap();
        assert_eq!(a.id(), b.id());
        assert_eq!(a.gpu_resource_id(), b.gpu_resource_id());
        let nearest = d
            .new_sampler_state(&SamplerDesc::default())
            .unwrap()
            .unwrap();
        assert_ne!(nearest.id(), a.id());
        assert_ne!(nearest.gpu_resource_id(), a.gpu_resource_id());
    }

    /// A value's resource index places the sampler at that ID, and its resource index is its ID
    #[test]
    fn a_sampler_takes_the_id_of_its_resource_index() {
        let d = dev();
        let s = d
            .new_sampler_state(&SamplerDesc {
                resource_index: Some(7),
                ..SamplerDesc::default()
            })
            .unwrap()
            .unwrap();
        assert_eq!((s.gpu_resource_id(), s.resource_index()), (7, 7));
        let any = d
            .new_sampler_state(&SamplerDesc {
                min_filter: Some(LINEAR),
                ..SamplerDesc::default()
            })
            .unwrap()
            .unwrap();
        assert_eq!(any.resource_index(), any.gpu_resource_id());
        assert_ne!(any.gpu_resource_id(), 7);
    }

    /// Descriptors Metal's validation stops, of which Metal makes a state, make one: an anisotropy
    /// of 0, a minimum LOD clamp past the maximum, and a border colour no enumerator is. One of
    /// an anisotropy of 17, which Metal samples with 32 and the device with its largest, 16,
    /// dies at its creation (`capped-anisotropy`, [`crate::answer::DIES`]); any other raw value
    /// no enumerator is dies at its setter
    #[test]
    fn sampler_values_metal_takes_and_bad_values() {
        let d = dev();
        for desc in [
            SamplerDesc {
                max_anisotropy: Some(0),
                ..SamplerDesc::default()
            },
            SamplerDesc {
                lod_min_clamp: Some(4.0),
                lod_max_clamp: Some(2.0),
                ..SamplerDesc::default()
            },
            SamplerDesc {
                border_color: Some(3),
                ..SamplerDesc::default()
            },
            SamplerDesc {
                border_color: Some(7),
                ..SamplerDesc::default()
            },
        ] {
            assert!(d.new_sampler_state(&desc).unwrap().is_some(), "{desc:?}");
        }
        let (sel, line) = death_line(|| {
            d.new_sampler_state(&SamplerDesc {
                max_anisotropy: Some(17),
                ..SamplerDesc::default()
            })
        });
        assert_eq!(sel, "newSamplerStateWithDescriptor:", "{line}");
        assert!(line.contains("capped-anisotropy"), "{line}");
        let site = catch_death(|| {
            d.new_sampler_state(&SamplerDesc {
                min_filter: Some(9),
                ..SamplerDesc::default()
            })
        })
        .unwrap_err();
        assert_eq!(site.file, "setMinFilter:");
    }

    /// Runs `f`, which dies, and answers its site's selector and the line the death carried.
    fn death_line<T>(f: impl FnOnce() -> T) -> (&'static str, String) {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(f).err().expect("it dies");
        assert_eq!(site.kind, reims_vgpu_core::raise::death::DeathKind::Abort);
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        (site.file, line)
    }

    /// Any depth-stencil value is a state; its resource ID raises below Apple10, as Metal's
    /// does on this host
    #[test]
    fn depth_stencil_states_are_made_and_their_ids_raise() {
        let d = dev();
        let face = StencilDesc {
            stencil_compare_function: 3,
            stencil_failure_operation: 1,
            depth_failure_operation: 2,
            depth_stencil_pass_operation: 7,
            read_mask: 0xf0,
            write_mask: 0x0f,
        };
        let s = d
            .new_depth_stencil_state(&DepthStencilDesc {
                depth_compare_function: Some(1),
                depth_write_enabled: Some(true),
                front: Some(face),
                back: None,
            })
            .unwrap()
            .unwrap();
        assert!(s.gpu_resource_id().is_err());
        assert!(s.resource_index().is_err());
        let site = catch_death(|| {
            d.new_depth_stencil_state(&DepthStencilDesc {
                depth_compare_function: Some(8),
                ..DepthStencilDesc::default()
            })
        })
        .unwrap_err();
        assert_eq!(site.file, "setDepthCompareFunction:");
    }

    /// The runtime's stop of a state whose attachment is in a pixel format the device's Metal
    /// describes none of (`AttachmentNotAFormat`: a colour attachment of 630 to 650, a
    /// depth or stencil attachment of `Depth24Unorm_Stencil8` or `X24_Stencil8`), where Metal's
    /// descriptor validation asserts, is no `NSError`: it dies at the creator. No creator here
    /// reaches it, as a descriptor holds such a format raw and its state dies first
    /// (`gpu::device`'s `raw_pipeline_formats_die_where_states_are_made`)
    #[test]
    fn an_attachment_of_no_format_is_no_nserror() {
        let format = |raw| crate::gpu::runtime_pixel_format(raw).unwrap();
        for (attachment, raw) in [
            (rt::Attachment::Color(0), 630),
            (rt::Attachment::Depth, 255),
            (rt::Attachment::Stencil, 262),
        ] {
            let problem = rt::Problem::AttachmentNotAFormat {
                attachment,
                format: format(raw),
            };
            let e = rt::Error::Invalid(rt::Invalid::new("new_render_pipeline_state", problem));
            assert!(pipeline_error(&e).is_none(), "{e:?}");
        }
    }

    /// A refusal the runtime cannot prove Metal's answer for (`Error::Unproven`) dies at the
    /// creator, its line naming the call, the stage and the rule for an indirect-command
    /// argument's, of every rule, and the problem as the runtime states it for any other; a
    /// `MetalError` of any problem is the `NSError` of its code
    #[test]
    fn unproven_answers_die_naming_the_stage_and_the_rule() {
        use metal2vulkan::pipeline::Stage;
        use metal2vulkan::reflection::LinkedUseGap as G;
        let sel = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
        let call = "new_render_pipeline_state";
        let descriptor = || {
            let d = RenderPipelineDescriptor::new(cpu_stage());
            (
                PrimitiveTopologyClass::Unspecified,
                RenderDescriptor::Render(Box::new(d)),
            )
        };
        let line = |problem: rt::Problem| {
            *crate::answer::LAST_LINE.lock() = None;
            let e = rt::Error::Unproven(rt::Unproven::new(call, problem));
            let site = catch_death(|| {
                let _ = made(sel, descriptor(), Err(e));
            })
            .unwrap_err();
            assert_eq!(site.file, sel);
            crate::answer::LAST_LINE.lock().take().unwrap_or_default()
        };
        let gaps = [
            G::EndlessLoop,
            G::StackMemory,
            G::GatedOffMeshBuiltin,
            G::UnmeasuredColorWrite,
        ];
        for (stage, name) in [(Stage::Vertex, "vertex"), (Stage::Mesh, "mesh")] {
            for reason in gaps {
                let problem = rt::Problem::IndirectCommandsDirectArgumentUnproven { stage, reason };
                let line = line(problem);
                let named = format!(
                    "whose {name} function uses a direct texture or sampler argument only by a \
                     rule the runtime keeps beyond Metal ({reason:?})"
                );
                assert!(line.contains(&named), "{line}");
                assert!(
                    line.contains(&format!("is unmeasured, in {call}")),
                    "{line}"
                );
            }
        }
        let line = line(rt::Problem::UnrasterizedVertexReturn);
        let named = "the Vulkan runtime cannot hold UnrasterizedVertexReturn, whose answer";
        assert!(line.starts_with(named), "{line}");
        let e = rt::Error::MetalError(rt::MetalError::new(
            call,
            rt::Problem::UnrasterizedVertexReturn,
            "domain",
            3,
            "text",
        ));
        let answer = made(sel, descriptor(), Err(e)).unwrap().unwrap_err();
        assert_eq!(answer.code, 3);
        let unknown_intrinsic =
            rt::Error::Unsupported(metal2vulkan::error::Unsupported::not_implemented(
                metal2vulkan::error::Feature::UnknownIntrinsic,
            ));
        assert!(pipeline_error(&unknown_intrinsic).is_none());
        let relaxed_error = crate::answer::relaxed(|| pipeline_error(&unknown_intrinsic));
        assert_eq!(relaxed_error.map(|err| err.code), Some(METAL_REFUSES));
    }

    /// A stage of the platform's own vertex function, from its metallib on the CPU: the lines of
    /// a descriptor's formats read no stage.
    fn cpu_stage() -> StageSource {
        use metal2vulkan::library::Library;
        use metal2vulkan::pipeline::FunctionDescriptor;
        use reims_vgpu_core::display::pipeline::VERTEX_FUNCTION;
        let library = Library::from_metallib(crate::gpu::shaders::METALLIB).unwrap();
        StageSource::Function(FunctionDescriptor::new(
            library.function(VERTEX_FUNCTION).unwrap(),
        ))
    }

    /// A refused pipeline's death line names its descriptor's formats, numeric and named: a
    /// render descriptor's colour attachments with their blend states and write masks, its depth
    /// and stencil and its vertex attributes (the XR10 state 26.6.2's compositor makes), a tile
    /// descriptor's colours, and a compute descriptor's stage-in attributes
    #[test]
    fn a_refused_pipeline_names_its_descriptors_formats() {
        use metal2vulkan::metal::{
            AttributeFormat, BlendFactor, BlendOperation, ColorWriteMask, VertexFormat,
        };
        use metal2vulkan::pipeline::{
            StageInputAttributeDescriptor, StageInputDescriptor, VertexAttributeDescriptor,
            VertexDescriptor,
        };
        use metal2vulkan_runtime::{Error, RuntimeGap};

        use crate::answer::refusal_describing;

        let attachment = |format, mask| {
            let mut c = ColorAttachmentDescriptor::new(format);
            c.write_mask = mask;
            c
        };
        let mut d = RenderPipelineDescriptor::new(cpu_stage());
        d.fragment = Some(cpu_stage());
        d.color_attachments[0] = Some(attachment(PixelFormat::RGB10A82PXR10, ColorWriteMask::ALL));
        {
            let c = d.color_attachments[0].as_mut().unwrap();
            c.blending_enabled = true;
            c.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            c.destination_alpha_blend_factor = BlendFactor::OneMinusSource1Alpha;
            c.alpha_blend_operation = BlendOperation::Max;
        }
        d.color_attachments[1] = Some(attachment(PixelFormat::RG16Float, ColorWriteMask::NONE));
        d.color_attachments[1].as_mut().unwrap().rgb_blend_operation = BlendOperation::Subtract;
        d.color_attachments[2] = Some(attachment(PixelFormat::RGB10A82PXR10, ColorWriteMask::NONE));
        d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        let xr10 = "550 (MTLPixelFormatRGB10A8_2P_XR10)";
        let packed = "260 (MTLPixelFormatDepth32Float_Stencil8)";
        let (one, zero, add) = (
            "1 (MTLBlendFactorOne)",
            "0 (MTLBlendFactorZero)",
            "0 (MTLBlendOperationAdd)",
        );
        let unblended = format!(
            "blending off rgb source {one} destination {zero} operation {add} alpha source {one} \
             destination {zero} operation {add}"
        );
        assert_eq!(
            refusal_describing(
                &Error::NotImplemented(RuntimeGap::Textures),
                Some(&formats(&RenderDescriptor::Render(Box::new(d.clone()))))
            ),
            format!(
                "{}; the descriptor's formats: colour 0 {xr10} blending on rgb source 4 \
                 (MTLBlendFactorSourceAlpha) destination 5 (MTLBlendFactorOneMinusSourceAlpha) \
                 operation {add} alpha source {one} destination 18 \
                 (MTLBlendFactorOneMinusSource1Alpha) operation 4 (MTLBlendOperationMax) write \
                 mask 0xf, colour 1 65 (MTLPixelFormatRG16Float) blending off rgb source {one} \
                 destination {zero} operation 1 (MTLBlendOperationSubtract) alpha source {one} \
                 destination {zero} operation {add} write mask 0x0, colour 2 {xr10} {unblended} \
                 write mask 0x0, depth {packed}, stencil {packed}, no vertex descriptor",
                crate::answer::refusal(&Error::NotImplemented(RuntimeGap::Textures)),
            )
        );
        let mut v = VertexDescriptor::new();
        v.attributes.insert(
            3,
            VertexAttributeDescriptor::new(VertexFormat::Float2, 0, 0),
        );
        let mut d = RenderPipelineDescriptor::new(cpu_stage());
        d.vertex_descriptor = Some(v);
        assert_eq!(
            render_formats(&d),
            "the descriptor's formats: no colour attachments, no depth, no stencil, vertex \
             attribute 3 29 (MTLVertexFormatFloat2)"
        );
        d.vertex_descriptor = Some(VertexDescriptor::new());
        assert!(render_formats(&d).ends_with(", no vertex attributes"));

        let mut t = TileRenderPipelineDescriptor::new(cpu_stage());
        assert_eq!(
            formats(&RenderDescriptor::Tile(Box::new(t.clone()))),
            "the descriptor's formats: no colour attachments"
        );
        t.color_attachment_formats[1] = Some(PixelFormat::BGRA8Unorm);
        assert_eq!(
            tile_formats(&t),
            "the descriptor's formats: colour 1 80 (MTLPixelFormatBGRA8Unorm)"
        );

        let mut c = ComputePipelineDescriptor::new(cpu_stage());
        assert_eq!(
            compute_formats(&c),
            "the descriptor's formats: no stage input descriptor"
        );
        let mut s = StageInputDescriptor::new();
        s.attributes.insert(
            0,
            StageInputAttributeDescriptor::new(AttributeFormat::Float2, 0, 1),
        );
        c.stage_input = Some(s);
        assert_eq!(
            compute_formats(&c),
            "the descriptor's formats: stage-in attribute 0 29 (MTLAttributeFormatFloat2)"
        );
        assert_eq!(
            refusal_describing(&Error::DeviceLost, None),
            crate::answer::refusal(&Error::DeviceLost)
        );
    }

    #[test]
    fn raw_format_returns_error_when_relaxed() {
        let raw = serialized::RawFormat {
            attachment: "colour attachment 0".to_owned(),
            raw: 999,
        };
        let err = crate::answer::relaxed(|| {
            raw_format_or_die(
                "newRenderPipelineStateWithDescriptor:options:reflection:error:",
                &raw,
            )
        })
        .unwrap_err();
        assert_eq!(err.code, METAL_REFUSES);
        assert!(err.description.contains("colour attachment 0"));
        assert!(err.description.contains("999"));
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
