//! `id<MTLRenderCommandEncoder>`: core's `RenderEncoder` over the runtime's, one runtime call per
//! `RenderCmd`, and `renderCommandEncoderWithDescriptor:`, which opens it.
//!
//! - **Every runtime refusal dies at its selector** ([`Policy::Dies`]): a refusal is the
//!   runtime's validation or a form it lacks, and Metal raises at none of the encoder's commands,
//!   going on or asserting instead.
//! - **The encoder's creation** answers nil where Metal does (a pass with no attachment and no
//!   target size), dies where Metal aborts (a pass of no attachment with a target size and no
//!   default raster sample count, a memoryless depth or stencil attachment resolving at one
//!   sample with no resolve texture, more sample positions than a multisampled pass's samples,
//!   an encoder over an open one), drops the depth and stencil resolve textures a non-resolving
//!   store action leaves unused, as Metal ignores them, and dies at the runtime's other refusals.
//!   Among them, a `renderTargetArrayLength` past an attachment's slices (`render_pass_actions`:
//!   slice 1 of a 2D array of 2, at a length of 2), which Metal runs without validation. Measured
//!   on this host (a 4x4 2D array of 2 slices, plain and linear over a buffer as core makes one,
//!   cleared and stored at slices 0 and 1 and lengths 1 to 5): Metal renders min(length, the
//!   texture's slices) layers from the attachment's slice, those past the texture into the memory
//!   after it, at its slice stride (the buffer's bytes at `0x100` for slice 2 of a linear one,
//!   whose slices are `0x80` apart), or into a heap neighbour (the runtime's measurement); a
//!   draw that writes no layer renders the base slice alone. Nothing here writes past a texture,
//!   so such a pass dies at its creation. So does a pass into a YCbCr colour attachment the
//!   runtime refuses (a format refusal of a two-plane or 4:2:2 texture), its line naming the
//!   format and what Metal does there ([`crate::gpu::ycbcr::pass_refused`]): a packed 10-bit
//!   one's, which Metal fails with an error and bytes that vary from run to run, and the shapes
//!   of 500, 502, 503, 505, 506 and 507 and the other formats whose stores the runtime does not
//!   hold.
//! - **Store actions** Metal goes on with: a late one equal to the descriptor's known action, or
//!   of an attachment without a texture, encodes nothing; another over a known one dies, as its
//!   effect is not measured; one the descriptor left `Unknown` and nothing decided stores nothing
//!   or resolves at `endEncoding`, as on Metal (`DontCare`, its resolve target cleared).
//! - **A wait for a fence the encoder updated** goes to the runtime, which waits for the update
//!   before the encoder, as Metal completes it. **A barrier after the fragment stage and before
//!   the vertex stage** encodes nothing: this GPU runs a pass's vertex stages first, and Metal's
//!   barrier does not split the pass, where the runtime's would.
//! - **A nil pipeline state** has no runtime form: it is kept here, and what next runs the
//!   encoder's pipeline state dies, as Metal crashes at a draw with no pipeline state
//!   (`display/render.rs`'s nil pipeline of an integer target), where the runtime would run the
//!   state set before: a draw, direct or indirect, a tile dispatch, and an execution of an
//!   indirect command buffer whose commands inherit the pipeline state. **A nil depth-stencil
//!   state** is Metal's default state, with no depth or stencil test, as the runtime's nil is.
//! - **A draw of an index type no `MTLIndexType` is, or of no instances,** encodes nothing, as
//!   Metal draws nothing (`render_draw_indexed`: index type 2). The runtime would check an empty
//!   draw's state as any draw's before it records nothing, and refuse what it cannot bind, where
//!   Metal draws nothing. A patch draw of no instances, plain or indexed, goes to the runtime,
//!   which checks it so and records nothing.
//! - **Metal's private primitive types** 5 to 8 draw a line loop, a triangle fan, quads and a
//!   quad strip, measured against Metal pixel for pixel at every count from 0 to 10, with flat
//!   attributes, blending and instances: a non-indexed draw of one is an indexed draw of the
//!   public type over a buffer of the indices its primitives take ([`Private::indices`]),
//!   relative to its first vertex, which the indexed draw takes as its base vertex. One over
//!   indices or indirect arguments dies, as the GPU alone reads them. The index buffer lies
//!   outside the Metal address space ([`crate::gpu::own_buffer`]), so no guest buffer's
//!   `gpuAddress` depends on when the draw's command buffer completes.
//! - **A draw over indices of the platform's own**, a private type's and a rotated list or strip
//!   of triangles at the last vertex ([`last_first`]), takes its first vertex as its base vertex.
//!   Metal's non-indexed draws, of every type, give `[[base_vertex]]` as their first vertex and
//!   `[[vertex_id]]` as each vertex's (measured on the M3 Max, from 0 to `0xfffffffd`), and so
//!   does such a draw, whose base vertex the runtime gives as `[[base_vertex]]`.
//!   One from past `i32::MAX`, which the runtime's base vertex cannot hold and Metal draws from,
//!   dies: the limit is this platform's. Its vertex function runs as many times at each vertex
//!   as under Metal's draw, so one that writes memory writes as often: on the M3 Max both run it
//!   once for each primitive the vertex is in, Metal's non-indexed draws of every type as the
//!   indexed draws here do (a strip's vertices up to 3 times, a fan's first once a triangle;
//!   measured at every type, 6 to 1000 vertices, 1 to 3 instances, at either provoking vertex).
//! - **Work of a pipeline state that can encode indirect commands** (the runtime's
//!   `can_encode_indirect_commands`) dies in an encoder that made resident for writing an indirect
//!   command buffer whose ID is not the runtime's, as a compute encoder's does
//!   ([`crate::gpu::encode_compute`]): a draw of something, a tile dispatch, and an execution of
//!   commands that inherit the state at its call ([`VulkanRenderEncoder::encodes`]); an
//!   execution whose commands name their own at the commit, which reads them. Elsewhere such
//!   work may write the records of each buffer the encoder made resident for writing, for good
//!   (`executions::gpu_writes`).
//! - **Primitive type 9**, which Metal's GPU hangs at, fails the command buffer as Metal's GPU
//!   does ([`VulkanRenderEncoder::hangs`]), in flight as long as Metal's; over indices or
//!   indirect arguments, or of no vertices or instances, which nothing measured, it dies. A
//!   command buffer committed on its queue while it is in flight, which Metal runs nothing of
//!   and completes without an error, dies at its commit ([`crate::gpu::command`]).
//! - **Apple's private commands** compose public ones where their meaning is measured:
//!   `setDepthClipModeSPI:` is `setDepthClipMode:`; the late resolve setters set the runtime's
//!   resolve targets, `yInvert` dropped, which Metal ignores, and one the attachment cannot use
//!   encodes nothing, as Metal resolves nothing there: a colour one where the pass descriptor
//!   named no resolve texture, or any under a store action the descriptor knew and that does not
//!   resolve. A colour one of an attachment whose resolve into a texture of another format the
//!   pass was made without, as plain Metal leaves that texture untouched ([`open`]), dies, as
//!   nothing measured it, and so does a depth or stencil one in such a pass that may resolve.
//!   `setDepthCleared` and `setStencilCleared`, which change nothing Metal draws
//!   (`mutgap_dec_render_depth_stencil_cleared`), encode nothing. The rest go to the runtime,
//!   which answers each at Metal's default or refuses it.
//! - **The draw-state setters** encode nothing and mark the encoder; each draw then draws what
//!   Metal draws under them, or dies where it has no exact form ([`VulkanRenderEncoder::rotates`]).
//!   A draw of no primitive, patch or instance, and an execution of no command, draw nothing
//!   whatever they left, as Metal's do, and go on. Measured on Metal on this host, against a 16
//!   by 16 target:
//!   - `setPointSize:` changes nothing drawn at any size (-1 to 1000, and NaN): a point is the
//!     size its vertex function writes. It is not marked. Under a vertex function that writes
//!     no `[[point_size]]`, Metal draws a point of size 1, as one that writes 1, only in a state
//!     of the point input primitive topology class; in any other its points are undefined,
//!     whatever the setter set: most draws draw nothing, and some draw a few of the points, or a
//!     box whose size changes from one run to the next (the unspecified, line and triangle
//!     classes, 6 runs each in two processes). So a draw of points there dies, direct, over
//!     indices or from indirect arguments, but one of no vertices ([`VulkanRenderEncoder::sized`]);
//!     and so does an execution's, at the commit, which reads its commands' records as the
//!     runtime runs them: under a state its commands inherit, or each command's own
//!     ([`crate::gpu::states::points_sized`]), or one released since. An execution of a buffer
//!     whose records the GPU may have written dies there whatever its draws, as the runtime
//!     reads those only once the work before it completed ([`crate::gpu::command::executions`]).
//!   - `setLineWidth:` of anything but 1 (0, 0.5 to 8, -1, NaN) widens line and line-strip
//!     draws, and triangles filled as lines; points and filled triangles are unchanged. No line
//!     drawn here is wider than 1, so those draws, and executions, die after another width.
//!   - `setProvokingVertexMode:` of 0 is the first vertex, Metal's default and Vulkan's; every
//!     other mode is the last: a list triangle's third vertex, strip triangle i's vertex i + 2,
//!     an indexed list's third index, a line segment's second vertex. Smooth inputs interpolate
//!     alike in every vertex order (200 random perspective triangles, pixel for pixel). A draw
//!     under a pipeline state that reads no flat input (`has_flat_fragment_inputs`) and whose
//!     vertex function writes no layer or viewport index (below) draws as it stands; points draw
//!     as they stand; under one that reads a flat input, a list or strip of triangles over its
//!     vertices in order draws as a list over its triangles' vertices last first
//!     ([`last_first`]), Vulkan's first vertex then Metal's last, and the rest dies. Metal also
//!     routes a primitive to the layer and the viewport its provoking vertex writes
//!     (`[[render_target_array_index]]`, `[[viewport_array_index]]`: a triangle whose vertices
//!     write layers 0, 0 and 1 lands in layer 0 at mode 0 and in layer 1 at the last, and so for
//!     every mix and for viewports). So a state whose vertex function writes either
//!     (`writes_render_target_array_index`, `writes_viewport_array_index`) is taken as one that
//!     reads a flat input: a list or strip of triangles over its vertices in order draws
//!     rotated, last vertex first, and the rest dies but points.
//!   - `[[barycentric_coord]]` under the last vertex: Metal orders a list's as at mode 0, over
//!     its vertices and over indices, and moves a strip's odd triangles' one place, `(b2, b0,
//!     b1)` of mode 0's, its even ones unchanged. So a list drawn as it stands is exact, and a
//!     strip, or any rotated draw, is not: under a state whose fragment function reads it
//!     (`reads_barycentric_coord`), a draw at the last vertex dies but a list of triangles over
//!     its vertices or indices under a state that reads no flat input.
//!   - `setPrimitiveRestartEnabled:index:` sets the index an indexed strip restarts at, `k`,
//!     whatever `enabled` says (`{0, k}` draws as `{1, k}`, at `k` of 0, 3, 7, 15, `0xffff`
//!     and `0xffffffff`). Without it a strip restarts at its index type's all-ones index, as
//!     Vulkan's does: 16-bit `0xffff`, 32-bit `0xffffffff`, and a 32-bit `0xffff` is a vertex.
//!     After it a strip restarts at `k` alone, 16- and 32-bit alike, and draws the all-ones
//!     index as a vertex. Only line and triangle strips restart, drawn directly or from
//!     indirect arguments; points and lists draw every index as a vertex. A new encoder's
//!     strips restart at the all-ones index again. An execution of an indirect command buffer's
//!     indexed strips, 16- and 32-bit, and indexed patches draw alike with and without it.
//!     The runtime's encoder takes the setter and draws so (`set_primitive_restart_enabled`,
//!     measured on an M4): an indexed strip that restarts elsewhere than at the all-ones index
//!     draws as the list a GPU pass writes from its indices before the pass, and what that pass
//!     cannot draw as Metal does the runtime refuses, which dies at the draw. An indirect
//!     strip's list the command buffer's index pool cannot hold dies as the command buffer
//!     completes (`crate::gpu::command`). A line loop draws as without the setter, as Metal's
//!     does: its line strip's indices are this platform's ([`VulkanRenderEncoder::primitives`]).
//!     The guest's `k` is below `maxViewportCount` (core refuses the rest), and
//!     `mutgap2_rdec_restart_transform` alone draws a strip after it.
//!   - `setVertexAmplificationMode:value:` sets the amplification count that
//!     `setVertexAmplificationCount:viewMappings:` sets, and keeps the view mappings that one
//!     set; whichever of the two comes last decides, and a new encoder starts at 1. In every
//!     mode but 1 (0 to 8, 15, 16, 255, `u32::MAX` and `u64::MAX`) a value from 1 to 8 draws
//!     that many copies, amplification IDs 0 up; in mode 1 the number of the value's set bits
//!     does (1 to 9, 15 to 17, 255 to 257). Value 0 draws copies at random: 8 under a state of
//!     no amplification, and 0, 48 or 255 under one of `maxVertexAmplificationCount` 2, from
//!     one run to the next; 9 draws 1, and 16 and 32 draw 8. So the setter encodes nothing, and
//!     where it leaves a count the runtime does not draw (the last
//!     `setVertexAmplificationCount:viewMappings:`'s, or 1), or one not measured, draws and
//!     executions die until the next `setVertexAmplificationCount:viewMappings:`.
//! - **`useHeaps:count:` and `useResources:count:usage:`** reach the vertex and fragment stages,
//!   as they do on Metal's render encoder. A nil entry of `useResources:count:usage:` is skipped,
//!   as Metal goes on past one (`interplay_decodewalk_6`: an entry naming the render pipeline is
//!   nil, and the walk runs on); one of the stages form, or a nil heap, which nothing measured,
//!   dies.
//! - **Executing an indirect command buffer** runs its draws, with the encoder's render state, as
//!   their records stand when the command buffer is committed, or, for a buffer the GPU may have
//!   written (a blit wrote it, or an encoder made it resident for writing), once the work before
//!   the execution completed (the runtime's host-side step), where Metal's GPU reads them as it
//!   reaches them. The runtime's refusal of the records dies naming the execution, at the commit
//!   or as the command buffer completes ([`crate::gpu::command`]); a refusal at the call (a range
//!   past the buffer's size, which Metal runs without validation) dies there. An empty range
//!   within the size runs nothing and encodes nothing, as Metal's does under any pipeline state
//!   (`render_draw_icb`: one made without `supportIndirectCommandBuffers`, which the runtime
//!   refuses for a buffer that inherits it).
//! - **The pass's rate map** is taken as it is at the encoder's creation, where Metal's pass reads
//!   it as the GPU runs it: a reset in between dies ([`crate::gpu::rate_map`]). A pass of a
//!   mutable map not yet reset rasterizes nothing, as on Metal: its vertex functions run and no
//!   fragment does. Its scissor rectangles are empty from the encoder's creation, and each
//!   scissor setter, the runtime having checked the rectangles it is given, sets as many empty
//!   ones. A tile dispatch in it dies: the runtime's runs over every tile, scissors or not, and
//!   only Metal's draws were measured there. A pass of a map of no layers loads, clears and
//!   stores as Metal's does, but Metal's draws in it rasterize by what earlier maps left, where
//!   the runtime's draw nothing: a draw and an execution of indirect commands in it die, but one
//!   of no primitive or command, which draws nothing on either (the encoder's `layered`). A tile
//!   dispatch in it dies too, as nothing measured Metal's there (the encoder's `tiled`). A pass of
//!   a map whose rates vary (`rasterization-rate-map`, [`Regions`]) loads, clears and stores as
//!   Metal's does too, but the runtime's draws in it rasterize by regions whose lines are the
//!   map's tiles' exact lines, where Metal's are its fixed-point ones: a draw and an execution of
//!   indirect commands that is not empty die there, and so does a tile dispatch.
//! - **What the runtime has no form for dies**: an index or a count past `u32`, a raw value no
//!   Metal enumerator is, a slice shorter than its range, a nil heap or resource (but in
//!   `useResources:count:usage:`), and a command after `endEncoding`. A rotated texture is the
//!   runtime's, as an attachment, a resolve texture and bound to a stage
//!   ([`crate::gpu::resource::TextureObject::texels`]), and so are its refusals.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use metal2vulkan::metal::{
    BarrierScope, CullMode, DepthClipMode, IndexType, PixelFormat, PrimitiveTopologyClass,
    PrimitiveType, RenderStages, ResourceUsage, TextureType, TriangleFillMode, Winding,
};
use metal2vulkan::pipeline::RenderPipelineDescriptor;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::Handle as _;
use reims_vgpu_core::platform::blit::ResourceRef;
use reims_vgpu_core::platform::icb::IndirectCommandBuffer as _;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder};
use reims_vgpu_core::platform::resource::RenderPipeline as _;
use reims_vgpu_core::platform::types::{MtlScissorRect, MtlViewport, NsRange};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::Mutex;

use crate::answer::{
    Policy, decide, get, metal_aborts, plain, plain_at, refused, send, unrepresentable,
};
use crate::gpu::command::executions::{Execution, Points};
use crate::gpu::command::{HANG, VulkanCommandBuffer};
use crate::gpu::desc::{Over, Source, Sources, VulkanRenderPassDescriptor};
use crate::gpu::encode_blit::EncoderObject;
use crate::gpu::icb::{Inherits, VulkanIcb};
use crate::gpu::rate_map::Regions;
use crate::gpu::resource::{
    Named, RenderDescriptor, VulkanBuffer, VulkanRenderPipeline, VulkanTexture, reach,
};
use crate::gpu::writes::{
    FRAGMENT, Kept, Memory, Reflected, Reflection, Slot, TILE, VERTEX, Whether,
};
use crate::gpu::{metal_value, region, size};
use crate::platform::{Obj, VulkanPlatform};

/// The colour formats whose resolve the device averages and rounds (the runtime's
/// `DeviceRoundedResolve`: a format without the runtime's own resolve rules) measured to resolve
/// as Metal's here, at each of [`DEVICE_RESOLVE_SAMPLES`]: at modes 0 and 2, fully covered and a
/// quarter pixel in, by the drawing pass and by a later one that loads the samples
/// (`mode_2_multisampled_half_float_attachments_resolve_as_metals`, on the M3 Max, macOS 26.7).
const DEVICE_RESOLVES_MEASURED: [PixelFormat; 3] = [
    PixelFormat::R16Float,
    PixelFormat::RG16Float,
    PixelFormat::RGBA16Float,
];

/// The sample counts [`DEVICE_RESOLVES_MEASURED`] was measured at.
const DEVICE_RESOLVE_SAMPLES: [u64; 2] = [2, 4];

/// Relays at `sel`, or dies there in strict mode ([`crate::answer::inexact`]), the end of the
/// pass `pass` at which the runtime announced a colour resolve the device averages and rounds
/// (`DeviceRoundedResolve`), unless every multisampled colour
/// attachment of a format and a sample count without the runtime's own resolve rules (of its
/// direct Vulkan format, `ColorResolveFormat`: a 16-bit float format's hold at 2 samples alone)
/// is of a format and a sample count measured to resolve as Metal's
/// ([`DEVICE_RESOLVES_MEASURED`]), and there is one: the runtime names only the first such
/// attachment, and its average of any other format may round a code apart from Metal's.
fn device_rounded(sel: &'static str, pass: &rt::RenderPassDescriptor) {
    use metal2vulkan::bind::ColorResolveFormat;
    use metal2vulkan::metal::format::direct_format;
    let ruled = |pf, samples: u64| {
        direct_format(pf)
            .and_then(ColorResolveFormat::of)
            .is_some_and(|r| u32::try_from(samples).is_ok_and(|n| r.resolves(n)))
    };
    let rounded: Vec<(PixelFormat, u64)> = plain(|| {
        (pass.color_attachments.iter())
            .filter_map(|c| c.texture.as_ref())
            .map(|t| (t.pixel_format(), t.sample_count()))
            .filter(|&(pf, samples)| samples > 1 && !ruled(pf, samples))
            .collect()
    });
    let measured = |&(pf, samples): &(PixelFormat, u64)| {
        DEVICE_RESOLVES_MEASURED.contains(&pf) && DEVICE_RESOLVE_SAMPLES.contains(&samples)
    };
    let what = match rounded.iter().find(|r| !measured(r)) {
        None if !rounded.is_empty() => return,
        None => "a colour resolve of an attachment it does not name".to_owned(),
        Some((pf, samples)) => {
            format!("a resolve of a {pf:?} colour attachment of {samples} samples")
        }
    };
    crate::answer::inexact(
        sel,
        crate::answer::Inexact::RoundedResolve,
        &format!(
            "{what}, which the device averages and rounds: only R16Float, RG16Float and \
             RGBA16Float at 2 and 4 samples were measured to resolve as Metal's here"
        ),
    );
}

/// The first colour attachment of `d` that the state's fragment function writes (`written`, the
/// runtime's `fragment_color_outputs`), and its format, where a `half` colour output stores a
/// value further from Metal's than the bound the runtime's rounding of it is relayed within (one
/// code of a unorm, snorm or packed float channel, one unit in
/// the last place of a 16-bit float channel, a 32-bit float channel's own value). The runtime
/// rounds such an output to `half` before the blend and the format's conversion
/// (`HalfOutputPrecision`, a module's without fast math; the runtime notes a fast-math module's
/// `HalfOutputFastMathBlend`, relayed), where Metal may carry it at `float`
/// precision into the blend. Measured against Metal on the M3 Max under fast math
/// (`half_outputs_store_within_the_bound_or_die_named`: a `half4` converted from `float`, over
/// two sweeps from -2 to 2, onto loaded texels; a `float4` output under the same state stores
/// Metal's bytes in every case), the attachments past the bound are:
/// - `R16Float`, `RG16Float` and `RGBA16Float`, blended by other than `One`, `Zero` and `Add`, or
///   `Min` or `Max`: `One` and `One` 384 `half` units in the last place apart in `R16Float` and
///   3072 in `RG16Float`, `Zero` and `OneMinusSourceAlpha` 3468 in `RGBA16Float`,
///   `DestinationColor` and `OneMinusDestinationColor` 2213, where the blend's terms cancel to
///   less than the source;
/// - `R32Float`, `RG32Float` and `RGBA32Float`, blended by a factor that reads the source beside
///   a factor neither `Zero` nor `One`: in `RGBA32Float`, `SourceAlpha` and `OneMinusSourceAlpha`
///   1 float unit in the last place apart, `OneMinusSourceColor` twice 8.6e8 (`R32Float` and
///   `RG32Float` stored Metal's bits at the blends measured, and take the class's rule);
/// - the 8-bit `_sRGB` formats, blended by `DestinationColor` and `OneMinusDestinationColor`: 2
///   codes.
///
/// Every other format and blend measured stores within the bound: the 8-, 10- and 16-bit unorm
/// formats, `RG11B10Float` and `RGB9E5Float` at every blend measured, the `_sRGB` formats at the
/// other blends, and the float formats unblended or otherwise blended (premultiplied `One` and
/// `OneMinusSourceAlpha` stores Metal's bits in `RGBA16Float`, `RG16Float` and `RGBA32Float`, but
/// not in `R16Float`). The extended-range formats die of `extended-range-blend-source` first when
/// blended. A factor left to a specialization counts as one that reads the source.
fn half_apart(d: &RenderPipelineDescriptor, written: &[u32]) -> Option<(usize, PixelFormat)> {
    use metal2vulkan::metal::{BlendFactor as F, BlendOperation as O};
    let reads_the_source = |f: F| {
        matches!(
            f,
            F::SourceColor
                | F::OneMinusSourceColor
                | F::SourceAlpha
                | F::OneMinusSourceAlpha
                | F::SourceAlphaSaturated
                | F::Source1Color
                | F::OneMinusSource1Color
                | F::Source1Alpha
                | F::OneMinusSource1Alpha
                | F::Unspecialized
        )
    };
    let constant = |f: F| matches!(f, F::Zero | F::One);
    written.iter().find_map(|&i| {
        let i = usize::try_from(i).ok()?;
        let c = d.color_attachments.get(i)?.as_ref()?;
        let channels = [
            (
                c.source_rgb_blend_factor,
                c.destination_rgb_blend_factor,
                c.rgb_blend_operation,
            ),
            (
                c.source_alpha_blend_factor,
                c.destination_alpha_blend_factor,
                c.alpha_blend_operation,
            ),
        ];
        let blended = channels
            .into_iter()
            .filter(|&(.., op)| c.blending_enabled && !matches!(op, O::Min | O::Max));
        let apart = match c.pixel_format {
            PixelFormat::R16Float | PixelFormat::RG16Float | PixelFormat::RGBA16Float => {
                { blended }.any(|ch| !matches!(ch, (F::One, F::Zero, O::Add)))
            }
            PixelFormat::R32Float | PixelFormat::RG32Float | PixelFormat::RGBA32Float => {
                { blended }.any(|(s, t, _)| {
                    (reads_the_source(s) && !constant(t)) || (reads_the_source(t) && !constant(s))
                })
            }
            PixelFormat::R8UnormSRGB
            | PixelFormat::RG8UnormSRGB
            | PixelFormat::RGBA8UnormSRGB
            | PixelFormat::BGRA8UnormSRGB => { blended }
                .any(|ch| matches!(ch, (F::DestinationColor, F::OneMinusDestinationColor, _))),
            _ => false,
        };
        apart.then_some((i, c.pixel_format))
    })
}

/// Relays at `sel`, or dies there in strict mode ([`crate::answer::inexact`]), a draw at which
/// the runtime announced a `half` colour output of the state's fragment function
/// (`HalfOutputPrecision`), whose colour attachment `index` of `format` stores it past the bound
/// the rounding is relayed within ([`half_apart`]).
fn half_output_apart(sel: &'static str, (index, format): (usize, PixelFormat)) {
    crate::answer::inexact(
        sel,
        crate::answer::Inexact::HalfOutputBlend,
        &format!(
            "a fragment function's half colour output into colour attachment {index}, \
             {format:?}, as blended or converted there: the runtime rounds it to half first, \
             where Metal carries it at float precision, and the stored value lies past the \
             bound that rounding is relayed within ({})",
            rt::Degradation::HalfOutputPrecision.code()
        ),
    );
}

/// An open runtime render encoder, and what Metal's encoder holds that the runtime's has no form
/// for.
#[derive(Debug)]
pub struct RenderEncoderObject {
    /// The runtime's encoder, until it ends.
    pub encoder: EncoderObject<rt::RenderCommandEncoder>,
    /// `setRenderPipelineState:` of nil since the last state set: the next draw dies.
    nil_pipeline: Mutex<bool>,
    /// The pass the runtime's encoder was made of: the descriptor as core set it, with the rate
    /// map [`open`] took, without the colour resolves [`untouched_resolves`] dropped and the
    /// unused depth and stencil resolve textures [`without_unused_resolves`] removes. Its
    /// attachments' store actions and resolve textures decide what a late setter does.
    pass: rt::RenderPassDescriptor,
    /// The colour attachments whose resolves [`untouched_resolves`] dropped: a late resolve
    /// setter of one dies, and a late store action is compared with the descriptor's own; where
    /// there is one, a late depth or stencil resolve texture that may resolve dies
    /// ([`VulkanRenderEncoder::resolves_beside_dropped`]).
    dropped: Vec<Dropped>,
    /// The device of the command buffer, which makes the buffers of the encoder's own draws.
    device: rt::Device,
    /// The attachments whose store action the descriptor left `Unknown` and no late setter has
    /// decided yet, each with why a resolve into the resolve texture it names now, the
    /// descriptor's or a late resolve setter's, is never sent ([`Unsent`]): a late store action
    /// that resolves into such a one dies ([`VulkanRenderEncoder::late_store`]).
    undecided: Mutex<Vec<(Attachment, Option<Unsent>)>>,
    /// The store action a late setter gave each attachment whose store action the descriptor
    /// left `Unknown`, the last one set: whether a late depth or stencil resolve texture beside a
    /// dropped colour resolve may resolve ([`VulkanRenderEncoder::resolves_beside_dropped`]).
    late_stores: Mutex<Vec<(Attachment, rt::StoreAction)>>,
    /// What Metal's encoder holds of how its draws draw that the runtime's has no form for.
    marks: Mutex<Marks>,
    /// Whether the pass rasterizes: one of a mutable rate map not yet reset does not, and keeps
    /// its scissor rectangles empty.
    rasterizes: bool,
    /// Whether the pass's rate map has no layers, whose draws Metal rasterizes by what earlier
    /// maps left (`layered`).
    no_layers: bool,
    /// Whether the pass's rate map is one whose rates vary, whose regions the runtime's draws
    /// rasterize by its tiles' exact lines, where Metal's are its fixed-point ones (`layered`).
    varying_rates: bool,
    /// What its draws may write through what it binds ([`Reach`]).
    reach: Mutex<Reach>,
    /// The state set, where it rounds texture writes as the runtime does not, and the fragment
    /// stage's textures of a format whose writes it may round so: a draw that may write one dies
    /// ([`RoundedTextures`]).
    rounded: Mutex<RoundedTextures>,
    /// The pass's bfloat colour attachment it loads and stores, where it has one: a draw, an
    /// execution of render commands or a tile dispatch in the pass dies ([`bfloat_loaded`]).
    bfloat_loaded: Option<String>,
    /// Whether it drew primitive type 9, which Metal's GPU hangs at: its end holds what comes
    /// after it ([`VulkanRenderEncoder::hangs`]).
    hung: AtomicBool,
    /// Whether the pipeline state it set last can encode indirect commands (the runtime's
    /// `can_encode_indirect_commands`): its functions may write the commands of the indirect
    /// command buffers it made resident for writing ([`VulkanRenderEncoder::encodes`]).
    encodes_icbs: AtomicBool,
    /// The indirect command buffers it made resident for writing.
    writable_icbs: Mutex<Vec<VulkanIcb>>,
    /// Metal's ID of the first of those whose ID is not the runtime's
    /// ([`crate::gpu::icb::IcbObject::reaches_its_records`]).
    icb_written: Mutex<Option<u64>>,
}

/// What a draw under a render state of `textureWriteRoundingMode` 2 whose fragment function may
/// write a texture may write that Metal's mode 2 writes otherwise than the runtime, or that
/// nothing measured at the mode ([`crate::gpu::serialized::rounded_texture_writes`]): the
/// state's reflection, and the fragment stage's slots holding a texture of one of
/// [`crate::gpu::serialized::MODE_2_TEXTURE_FORMATS`], as the guest binds them, a view by its own
/// format.
#[derive(Debug, Default, Clone)]
struct RoundedTextures {
    /// The reflection of the state set, where it rounds texture writes otherwise than the
    /// runtime ([`crate::gpu::resource::RenderPipelineObject::rounds_texture_writes`]).
    state: Option<Arc<Reflection>>,
    /// The fragment stage's slots holding a texture of one of
    /// [`crate::gpu::serialized::MODE_2_TEXTURE_FORMATS`], each with Metal's name of it.
    formats: BTreeMap<u64, &'static str>,
}

impl RoundedTextures {
    /// `textures` bound to the fragment stage from slot `first`.
    fn bind(&mut self, first: u64, textures: &[Option<&VulkanTexture>]) {
        for (slot, t) in (first..).zip(textures) {
            self.formats.remove(&slot);
            let format = t
                .map(|t| t.raw.pixel_format())
                .filter(|f| crate::gpu::serialized::MODE_2_TEXTURE_FORMATS.contains(f));
            if let Some(format) = format {
                self.formats.insert(slot, format.name());
            }
        }
    }

    /// The pipeline state `state` set.
    fn set(&mut self, state: &VulkanRenderPipeline) {
        self.state = state
            .rounds_texture_writes()
            .then(|| Arc::clone(&state.reflection));
    }

    /// What a draw, or an execution whose commands inherit the state set (`execution`), may
    /// write under the state set that Metal's mode 2 writes otherwise than the runtime, `None`
    /// where it writes no texture of one of [`crate::gpu::serialized::MODE_2_TEXTURE_FORMATS`]
    /// ([`crate::gpu::serialized::rounded_texture_writes`]).
    /// An execution's commands bind no texture, and reach one only through an argument buffer,
    /// whose formats nothing tells.
    fn rounded(&self, execution: bool) -> Option<String> {
        let reflection = self.state.as_ref()?;
        if execution {
            return Some(
                "indirect commands that inherit it, whose fragment function may write the \
                 textures their argument buffers hold, whose formats nothing here tells"
                    .into(),
            );
        }
        let fragment = reflection.slots().and_then(|slots| {
            slots
                .get(FRAGMENT)
                .ok_or("a state of no fragment function's")
        });
        crate::gpu::serialized::rounded_texture_writes(fragment, &self.formats)
    }
}

/// An attachment of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Attachment {
    /// The colour attachment at an index.
    Color(u32),
    /// The depth attachment.
    Depth,
    /// The stencil attachment.
    Stencil,
}

/// Whether `attachment` is a colour attachment past the eighth.
fn past_the_eighth(attachment: Attachment) -> bool {
    matches!(attachment, Attachment::Color(i)
        if usize::try_from(i).map_or(true, |i| i >= rt::MAX_COLOR_ATTACHMENTS))
}

/// What the private setters left of how the encoder's draws draw, and what of the pipeline state
/// and fill mode decides whether a draw under them is exact ([`VulkanRenderEncoder::rotates`]).
#[derive(Debug, Default, Clone, Copy)]
struct Marks {
    /// The arguments of the last `setPrimitiveRestartEnabled:index:`, which the runtime's
    /// encoder takes too: Metal restarts an indexed strip at its index alone after it, but not
    /// the line strip a line loop draws as here ([`VulkanRenderEncoder::primitives`]).
    restart: Option<(bool, u64)>,
    /// `setLineWidth:` set a width other than 1, which widens lines on Metal and which no line
    /// drawn here has.
    wide_lines: bool,
    /// `setTriangleFillMode:` set `Lines`: a triangle draws its edges, at the line width.
    line_fill: bool,
    /// `setProvokingVertexMode:` set a mode other than 0: a primitive's last vertex provokes.
    last_vertex: bool,
    /// The pipeline state set reads a flat fragment input, which the provoking vertex decides.
    flat: bool,
    /// The pipeline state set reads `[[barycentric_coord]]`, which Metal orders by the vertices
    /// as drawn, not by the provoking vertex (the module docs).
    barycentric: bool,
    /// The pipeline state set's vertex function writes `[[render_target_array_index]]` or
    /// `[[viewport_array_index]]`, which Metal takes from the provoking vertex (the module docs).
    routed: bool,
    /// A pipeline state was set.
    pipeline: bool,
    /// The pipeline state set sizes its points as Metal does: its vertex function writes
    /// `[[point_size]]`, or its input primitive topology is the point class (the module docs).
    points_sized: bool,
    /// The amplification count the last `setVertexAmplificationCount:viewMappings:` set, which
    /// the runtime draws; `None` for none, Metal's 1.
    amplification: Option<u64>,
    /// `setVertexAmplificationMode:value:` left a count other than the runtime's, or one not
    /// measured ([`amplification_count`]).
    amplified: bool,
    /// The colour attachment of the pipeline state set, and its format, where a `half` colour
    /// output of its fragment function stores a value past the bound ([`half_apart`]).
    half_apart: Option<(usize, PixelFormat)>,
}

/// The amplification count `setVertexAmplificationMode:value:` sets, where it was measured (the
/// module docs): `value` in every mode but 1, the number of its set bits in mode 1, from 1 to 8;
/// `None` elsewhere: 0, which Metal draws copies of at random, and counts past 8.
fn amplification_count(mode: u64, value: u64) -> Option<u64> {
    let count = if mode == 1 {
        u64::from(value.count_ones())
    } else {
        value
    };
    (1..=8).contains(&count).then_some(count)
}

/// What a draw draws, for what the private setters left ([`VulkanRenderEncoder::rotates`]).
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// Primitives of a type, over the vertices in order, or over indices (`indexed`), from
    /// indirect arguments or not (`indirect`), the GPU reading both.
    Primitives {
        topology: Topology,
        indexed: bool,
        indirect: bool,
    },
    /// Patches, over a control-point index buffer or not.
    Patches { indexed: bool },
    /// An indirect command buffer's draws, whose types its records hold: their pipeline state
    /// the encoder's where they inherit it.
    Execution { inherits_pipeline: bool },
}

impl Shape {
    /// A draw of the public type `p`.
    fn public(p: PrimitiveType, indexed: bool, indirect: bool) -> Shape {
        Shape::Primitives {
            topology: Topology::Public(p),
            indexed,
            indirect,
        }
    }

    /// Whether it draws lines: a line or line strip, or Metal's line loop.
    fn lines(self) -> bool {
        matches!(
            self,
            Shape::Primitives {
                topology: Topology::Public(PrimitiveType::Line | PrimitiveType::LineStrip)
                    | Topology::Private(Private::LineLoop),
                ..
            }
        )
    }

    /// Whether it draws triangles: a list or strip of them, Metal's fan, quads and quad strip,
    /// and patches.
    fn triangles(self) -> bool {
        match self {
            Shape::Primitives { topology, .. } => matches!(
                topology,
                Topology::Public(PrimitiveType::Triangle | PrimitiveType::TriangleStrip)
                    | Topology::Private(Private::TriangleFan | Private::Quads | Private::QuadStrip)
            ),
            Shape::Patches { .. } => true,
            Shape::Execution { .. } => false,
        }
    }
}

/// Whether a store action resolves, or may (`Unknown`, which the encoder decides).
fn may_resolve(store: rt::StoreAction) -> bool {
    matches!(
        store,
        rt::StoreAction::MultisampleResolve
            | rt::StoreAction::StoreAndMultisampleResolve
            | rt::StoreAction::Unknown
    )
}

/// `d` without the depth and stencil resolve textures its store actions leave unused: Metal
/// resolves nothing into them and stores the attachment as its action says
/// (`render_pass_memoryless`'s unused resolve textures on depth and stencil), where the runtime
/// refuses the pass. A colour attachment's goes to the runtime, which leaves it untouched, as
/// Metal does.
fn without_unused_resolves(d: &rt::RenderPassDescriptor) -> rt::RenderPassDescriptor {
    let mut out = d.clone();
    if !may_resolve(out.depth_attachment.store_action) {
        out.depth_attachment.resolve_texture = None;
    }
    if !may_resolve(out.stencil_attachment.store_action) {
        out.stencil_attachment.resolve_texture = None;
    }
    out
}

/// Whether the colour resolve of an attachment of `from` into a texture of `into` is the
/// runtime's to decide, as it was before any resolve was dropped here: the same format; an sRGB
/// pair (`R8Unorm`, `RG8Unorm`, `RGBA8Unorm`, `BGRA8Unorm`, `BGR10_XR` and `BGRA10_XR`, each with
/// its `_sRGB` twin: the colour-renderable pairs among the runtime's twins), which the runtime
/// resolves from a multisampled attachment in the attachment's encoding, but for the
/// extended-range ones, whose resolve it does not model; and the 8-bit four-channel family across
/// its channel orders (`RGBA8Unorm` and `BGRA8Unorm`, and their twins), which it does not build.
/// The runtime refuses every other pair (`ResolveFormat`, or an extended-range format's
/// resolve), which [`fate`] takes.
fn runtime_resolves(from: PixelFormat, into: PixelFormat) -> bool {
    use PixelFormat as P;
    const TWINS: [(PixelFormat, PixelFormat); 6] = [
        (P::R8Unorm, P::R8UnormSRGB),
        (P::RG8Unorm, P::RG8UnormSRGB),
        (P::RGBA8Unorm, P::RGBA8UnormSRGB),
        (P::BGRA8Unorm, P::BGRA8UnormSRGB),
        (P::BGR10XR, P::BGR10XRSRGB),
        (P::BGRA10XR, P::BGRA10XRSRGB),
    ];
    let eight_bit = |f| {
        matches!(
            f,
            P::RGBA8Unorm | P::RGBA8UnormSRGB | P::BGRA8Unorm | P::BGRA8UnormSRGB
        )
    };
    from == into
        || (eight_bit(from) && eight_bit(into))
        || TWINS.contains(&(from, into))
        || TWINS.contains(&(into, from))
}

/// What plain Metal was measured to do with the resolve of a colour attachment of `t` into a
/// texture of `r`, by their formats and the attachment's sample count alone, on the M3 Max
/// (`resolves_into_other_formats_are_metals_or_die_named`): a 4-sample `BGRA8Unorm` or
/// `BGRA8Unorm_sRGB` attachment's into `RGBA16Float`, `BGR10_XR` or `BGR10_XR_sRGB`, the formats
/// a drawable takes beside the 8-bit ones, leaves the texture untouched in the shapes
/// [`unmeasured_shape`] states; into `RGB10A2Unorm`, `BGR10A2Unorm`, `BGRA10_XR` or
/// `BGRA10_XR_sRGB` it writes into the texture, measured from a Private 2D multisample texture of
/// its own into a Shared single-level 2D texture of its own, both 16 by 16. `None` for any other
/// pair.
fn measured_pair(t: &rt::Texture, r: &rt::Texture) -> Option<Pair> {
    use PixelFormat as P;
    if !matches!(t.pixel_format(), P::BGRA8Unorm | P::BGRA8UnormSRGB) || t.sample_count() != 4 {
        return None;
    }
    match r.pixel_format() {
        P::RGBA16Float | P::BGR10XR | P::BGR10XRSRGB => Some(Pair::Untouched),
        P::RGB10A2Unorm | P::BGR10A2Unorm | P::BGRA10XR | P::BGRA10XRSRGB => Some(Pair::Written),
        _ => None,
    }
}

/// What plain Metal was measured to do with a resolve into a texture of another format
/// ([`measured_pair`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pair {
    /// It leaves the texture untouched.
    Untouched,
    /// It writes into the texture.
    Written,
}

/// What of the pass `d` lies outside the shapes plain Metal was measured to leave the resolve
/// texture `r` of colour attachment `index`, of texture `t`, untouched in ([`Pair::Untouched`]),
/// if anything: `source` holds the attachment's texture and resolve texture as core set them
/// ([`Over`]), and `mapped` is the rate map the pass takes ([`PassMap`]). Metal's validation
/// stops such a pass ("does not match Resolve PixelFormat"); a guest runs without it, and macOS
/// 26.6.2's Maps resolves its `BGRA8Unorm_sRGB` samples into a texture of another format, its
/// resolving passes loading them by `Load` or `DontCare`. Measured on the M3 Max
/// (`resolves_into_other_formats_are_metals_or_die_named`), Metal writes nothing into the
/// texture, `StoreAndMultisampleResolve` stores the samples as `Store` does, and
/// `MultisampleResolve` leaves them as they were before the pass, as `DontCare` does, for:
/// - an attachment of a Private 2D multisample texture of its own, compressed where Metal may
///   (`allowGPUOptimizedContents`) and not, of usage `RenderTarget`, alone or with `ShaderRead`
///   or `PixelFormatView`, at its level 0, slice 0 and depth plane 0, without store action
///   options;
/// - cleared by the pass, or loaded by `Load` or `DontCare` after a pass that cleared and stored
///   it, and drawn, or not: with no draw call, with only the state a draw takes, or with a draw
///   call of no vertices, of no instances, of two vertices of a triangle list, or indirect of no
///   vertices;
/// - resolved by `MultisampleResolve` or `StoreAndMultisampleResolve` into level 0, slice 0 and
///   depth plane 0 of an unrotated single-level, single-sample 2D texture of its size (16 by 16,
///   and 96 by 37; 1920 by 1080 and 1023 by 771 into a Shared one of its own, drawn and with no
///   draw call): one of its own (Shared, Managed, or Private compressed where Metal may and
///   not) of usage `ShaderRead` and `RenderTarget` (Shared and Private ones also of
///   `RenderTarget` alone, and with `ShaderWrite`, `PixelFormatView` or both), a Private
///   shareable one or one opened from its handle, or one over an IOSurface without planes
///   (Shared or Managed, each also presented as a framebuffer-only drawable);
/// - in a pass beside no other attachment, a depth one (`Depth32Float`, memoryless and not
///   stored or Private and stored; `Depth32Float_Stencil8` as both depth and stencil attachment,
///   memoryless, or Private cleared or not loaded, and not stored), or another 4-sample colour
///   attachment (`RGBA8Unorm` at index 0 or 1 or `BGRA8Unorm` at 2, resolving into a texture of
///   its format, or `R32Float` at 2, stored), or with a rate map of one sample of 1 on each axis.
///
/// A pass that draws nothing is no exception: Metal leaves the samples of one that clears the
/// attachment and resolves it by `MultisampleResolve` with no draw call as they were, compressed
/// or not, where a resolve into the attachment's own format leaves the clear in a compressed
/// one's (alike here). Measured on 2026-10-10, each as this platform's
/// samples and texture: 216 passes with no draw call (each load action, both actions, the
/// attachment compressed and not, into each format, 16 by 16 and 96 by 37 into a Shared texture
/// and 16 by 16 into a Private one); 112 of 120 with only the state a draw takes or a draw call
/// that draws nothing (into `RGBA16Float`, 16 by 16; the other 8 below); and 144 at 1920 by 1080
/// and 1023 by 771 (each load action, both actions, into each format, drawn and with no draw
/// call), of which the test holds 22. As Metal's `MultisampleResolve` stores none of the pass's
/// tiles into the samples, drawn or not, and its `StoreAndMultisampleResolve` stores them as
/// `Store` does, what writes the tiles (draws of any kind, an execution of render commands, a
/// tile dispatch) does not decide the drop: the pass made without the resolve leaves or stores
/// the same tiles, but where the runtime splits it, storing each part but the last, its
/// `DontCareStore`, relayed as for any `DontCare` store ([`crate::answer`]). Where the pass
/// loads by `DontCare`, stores by `StoreAndMultisampleResolve` and its draw calls draw nothing
/// on the GPU (two vertices of a triangle list, an indirect draw of no vertices), Metal keeps
/// the samples where the runtime's `Store` clears their tiles to zero: the runtime's
/// `DontCareLoadTiles`, announced and relayed as for any pass that loads by `DontCare` and
/// stores, not the drop's.
///
/// The drop takes those shapes at any size, of any usage, whatever the pass draws, beside any
/// depth and stencil attachment that does not resolve and any other colour attachment whose
/// resolve is the runtime's ([`runtime_resolves`]), and with any rate map the runtime takes as
/// the identity inside the screen ([`PassMap::Identity`]). Any other shape, which the runtime
/// may refuse for its own reasons too (a memoryless resolve texture, a level or slice past it),
/// is not measured: it is named. So is a depth or stencil resolve texture set late in such a
/// pass where it may resolve ([`VulkanRenderEncoder::resolves_beside_dropped`]).
fn unmeasured_shape(
    d: &rt::RenderPassDescriptor,
    index: usize,
    (t, r): (&rt::Texture, &rt::Texture),
    source: &[Source; 2],
    mapped: PassMap,
) -> Option<String> {
    use rt::StorageMode as S;
    let a = &d.color_attachments[index];
    let [texture, resolve] = source;
    if t.texture_type() != TextureType::Type2DMultisample {
        return Some(format!("the attachment is of {:?}", t.texture_type()));
    }
    if t.storage_mode() != S::Private {
        return Some(format!("the attachment is {:?}", t.storage_mode()));
    }
    if texture.over() != Some(Over::Own) {
        return Some(format!("the attachment is {}", Over::name(texture.over())));
    }
    if (a.level, a.slice, a.depth_plane) != (0, 0, 0) {
        return Some(format!(
            "the attachment is at level {}, slice {} and depth plane {}",
            a.level, a.slice, a.depth_plane
        ));
    }
    if a.store_action_options != rt::StoreActionOptions::NONE {
        return Some(format!(
            "the attachment's store action options are {:?}",
            a.store_action_options
        ));
    }
    let kind = (r.texture_type(), r.sample_count(), r.mipmap_level_count());
    if kind != (TextureType::Type2D, 1, 1) {
        return Some(format!(
            "the resolve texture is of {:?}, {} samples and {} levels",
            kind.0, kind.1, kind.2
        ));
    }
    if r.rotation() != 0 {
        return Some(format!(
            "the resolve texture is of rotation {}",
            r.rotation()
        ));
    }
    if (r.width(), r.height()) != (t.width(), t.height()) {
        return Some(format!(
            "the resolve texture is {} by {}, the attachment {} by {}",
            r.width(),
            r.height(),
            t.width(),
            t.height()
        ));
    }
    if (a.resolve_level, a.resolve_slice, a.resolve_depth_plane) != (0, 0, 0) {
        return Some(format!(
            "the resolve is into level {}, slice {} and depth plane {}",
            a.resolve_level, a.resolve_slice, a.resolve_depth_plane
        ));
    }
    match (resolve.over(), r.storage_mode()) {
        (Some(Over::Own), S::Shared | S::Managed | S::Private)
        | (Some(Over::Shareable), S::Private)
        | (Some(Over::Surface), S::Shared | S::Managed) => {}
        (over, storage) => {
            return Some(format!(
                "the resolve texture is {} of storage mode {storage:?}",
                Over::name(over)
            ));
        }
    }
    match mapped {
        PassMap::Unmapped | PassMap::Identity => {}
        PassMap::Varying => {
            return Some("the pass has a rasterization rate map whose rates vary".to_owned());
        }
        PassMap::Unset => {
            return Some(
                "the pass has a mutable rasterization rate map not yet reset, and rasterizes \
                 nothing"
                    .to_owned(),
            );
        }
    }
    let (depth, stencil) = (&d.depth_attachment, &d.stencil_attachment);
    let depth_resolves = [
        (&depth.texture, &depth.resolve_texture, depth.store_action),
        (
            &stencil.texture,
            &stencil.resolve_texture,
            stencil.store_action,
        ),
    ]
    .into_iter()
    .any(|(t, r, store)| t.is_some() && r.is_some() && may_resolve(store));
    if depth_resolves {
        return Some("the pass's depth or stencil attachment resolves".to_owned());
    }
    let other = (d.color_attachments.iter().enumerate()).find(|&(i, o)| {
        i != index
            && match (&o.texture, &o.resolve_texture) {
                (Some(t), Some(r)) => {
                    may_resolve(o.store_action)
                        && !runtime_resolves(t.pixel_format(), r.pixel_format())
                }
                _ => false,
            }
    });
    other.map(|(i, _)| {
        format!(
            "colour attachment {i} resolves into a texture of another format too, which the \
             runtime does not resolve into"
        )
    })
}

/// The rasterization rate map a pass takes, as far as its colour resolves tell maps apart
/// ([`unmeasured_shape`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassMap {
    /// None.
    Unmapped,
    /// A map the runtime takes as the identity inside the screen ([`Regions::Identity`]).
    Identity,
    /// A map whose rates vary ([`Regions::Varying`]).
    Varying,
    /// A mutable map not yet reset, whose pass rasterizes nothing.
    Unset,
}

/// What becomes of the resolve of a colour attachment ([`fate`]).
#[derive(Debug)]
enum Fate {
    /// The runtime's to resolve or refuse: there is none, its store action does not resolve,
    /// or the runtime decides the pair of formats ([`runtime_resolves`]).
    Runtimes,
    /// Dropped, as plain Metal leaves the texture untouched ([`untouched_resolves`]).
    Dropped,
    /// The pass dies at its creation, named so.
    Dies(String),
}

/// What the death of a resolve plain Metal writes into ([`Pair::Written`]) says Metal was
/// measured to do.
const WRITTEN: &str = "plain Metal writes into such a texture, measured from a Private 2D \
                       multisample texture of its own into a Shared single-level 2D texture of \
                       its own, both 16 by 16";

/// What becomes of the resolve of colour attachment `index` of the pass `d`, `sources` holding
/// the textures as core set them and `mapped` being the rate map the pass takes ([`PassMap`]):
/// the runtime's ([`Fate::Runtimes`]); dropped where plain Metal was measured to leave the
/// texture untouched ([`measured_pair`], [`unmeasured_shape`]); else, by a store action that
/// resolves or is left `Unknown` into a texture of another format the runtime refuses
/// (`ResolveFormat`, or an extended-range format's resolve), the pass dies, named with what plain
/// Metal was measured to do there. A store action left `Unknown` resolves by its late store
/// action, which nothing measured here, and the runtime counts it as resolving: it refuses such
/// a pass as it is made.
fn fate(d: &rt::RenderPassDescriptor, index: usize, sources: &Sources, mapped: PassMap) -> Fate {
    let a = &d.color_attachments[index];
    let (Some(t), Some(r)) = (&a.texture, &a.resolve_texture) else {
        return Fate::Runtimes;
    };
    let (from, into) = (t.pixel_format(), r.pixel_format());
    if !may_resolve(a.store_action) || runtime_resolves(from, into) {
        return Fate::Runtimes;
    }
    let metal = match (a.store_action, measured_pair(t, r)) {
        (rt::StoreAction::Unknown, _) => {
            "its store action left Unknown, whose late resolve plain Metal is not measured at"
                .to_owned()
        }
        (_, Some(Pair::Written)) => WRITTEN.to_owned(),
        (_, Some(Pair::Untouched)) => {
            let source = &sources.color[index];
            match unmeasured_shape(d, index, (t, r), source, mapped) {
                None => return Fate::Dropped,
                Some(what) => format!(
                    "plain Metal leaves such a texture untouched, but {what}, a shape it is not \
                     measured at"
                ),
            }
        }
        (_, None) => "a pair plain Metal is not measured at".to_owned(),
    };
    Fate::Dies(format!(
        "a resolve of colour attachment {index} ({}, {} samples) by {:?} into a texture of {} \
         ({} samples), another format, which the runtime does not resolve into: {metal}",
        from.name(),
        t.sample_count(),
        a.store_action,
        into.name(),
        r.sample_count()
    ))
}

/// A colour attachment whose resolve [`untouched_resolves`] dropped, as plain Metal leaves its
/// resolve texture untouched.
#[derive(Debug, Clone, Copy)]
struct Dropped {
    /// The attachment's index.
    index: u32,
    /// The store action the pass descriptor gave it, which a late store action is compared with
    /// ([`VulkanRenderEncoder::late_store`]).
    store: rt::StoreAction,
    /// The resolve texture's format.
    into: PixelFormat,
}

/// `d` without the colour resolves plain Metal leaves untouched ([`Fate::Dropped`], `sources`
/// holding the textures as core set them and `mapped` being the rate map the pass takes), and
/// those attachments: each stores as its action does without the resolve, as plain Metal was
/// measured to (`MultisampleResolve` as `DontCare`, which leaves the samples as they were, the
/// pass drawn or not, and `StoreAndMultisampleResolve` as `Store`), and the resolve texture
/// keeps its texels ([`unmeasured_shape`]). A late resolve setter of such an attachment dies, a
/// late store action is compared with the descriptor's own ([`Dropped`]), and a late depth or
/// stencil resolve texture of the pass that may resolve dies
/// ([`VulkanRenderEncoder::resolves_beside_dropped`]). Each attachment's fate is decided on `d`
/// as it is given, before any resolve is dropped.
fn untouched_resolves(
    mut d: rt::RenderPassDescriptor,
    sources: &Sources,
    mapped: PassMap,
) -> (rt::RenderPassDescriptor, Vec<Dropped>) {
    let drops: Vec<bool> = (0..d.color_attachments.len())
        .map(|i| matches!(fate(&d, i, sources, mapped), Fate::Dropped))
        .collect();
    let mut dropped = Vec::new();
    for ((index, a), drop) in (0u32..).zip(&mut d.color_attachments).zip(drops) {
        if !drop {
            continue;
        }
        let Some(r) = a.resolve_texture.take() else {
            continue;
        };
        dropped.push(Dropped {
            index,
            store: a.store_action,
            into: r.pixel_format(),
        });
        a.store_action = match a.store_action {
            rt::StoreAction::StoreAndMultisampleResolve => rt::StoreAction::Store,
            _ => rt::StoreAction::DontCare,
        };
    }
    (d, dropped)
}

/// The first colour resolve of the pass `d` that dies ([`Fate::Dies`]), named, if any, `sources`
/// holding the textures as core set them and `mapped` being the rate map the pass takes.
fn refused_resolve_format(
    d: &rt::RenderPassDescriptor,
    sources: &Sources,
    mapped: PassMap,
) -> Option<String> {
    (0..d.color_attachments.len()).find_map(|i| match fate(d, i, sources, mapped) {
        Fate::Dies(what) => Some(what),
        Fate::Runtimes | Fate::Dropped => None,
    })
}

/// The attachments of `d` with a texture whose store action is `Unknown`, each with `None`,
/// which [`VulkanRenderEncoder::undecided_dropped`] sets where a resolve into its resolve texture
/// is never sent ([`Unsent`]).
fn undecided(d: &rt::RenderPassDescriptor) -> Vec<(Attachment, Option<Unsent>)> {
    let unknown = |t: &Option<rt::Texture>, store| t.is_some() && store == rt::StoreAction::Unknown;
    let colors = (0u32..).zip(&d.color_attachments);
    let mut out: Vec<_> = colors
        .filter(|(_, c)| unknown(&c.texture, c.store_action))
        .map(|(i, _)| (Attachment::Color(i), None))
        .collect();
    if unknown(&d.depth_attachment.texture, d.depth_attachment.store_action) {
        out.push((Attachment::Depth, None));
    }
    if unknown(
        &d.stencil_attachment.texture,
        d.stencil_attachment.store_action,
    ) {
        out.push((Attachment::Stencil, None));
    }
    out
}

/// Why a pass's write into a texture is never sent: the runtime imports its memory through a
/// writable twin of read-only pages, where the write would land
/// ([`crate::gpu::writes::kept_write`]), or it lies on a surface whose GPU writes never reach its
/// memory ([`crate::os::dropped_write`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unsent {
    /// Over memory the runtime imports through a writable twin of read-only pages.
    Kept,
    /// On a surface whose GPU writes never reach its memory.
    Dropped,
}

impl Unsent {
    /// Why a write into `t` is never sent, if it is not: the twin first, which the write would
    /// reach.
    fn of(t: &VulkanTexture) -> Option<Unsent> {
        if crate::gpu::writes::texture_kept(t) {
            Some(Unsent::Kept)
        } else if t.gpu_read_only() {
            Some(Unsent::Dropped)
        } else {
            None
        }
    }

    /// Dies at `sel`, `what` naming a write into a texture, which the death names it lies in.
    fn dies(self, sel: &'static str, what: &str) -> ! {
        match self {
            Unsent::Kept => crate::gpu::writes::kept_write(sel, &format!("{what} over")),
            Unsent::Dropped => crate::os::dropped_write(sel, &format!("{what} on")),
        }
    }
}

/// What a pass of `d` may write in memory the runtime imports through a writable twin of
/// read-only pages ([`crate::gpu::writes::kept_write`]), as `sources` says of its attachments: an
/// attachment's texture, whatever the pass's load and store actions, as the runtime's GPU may
/// write the texels of one it loads and stores unchanged, or whose store it does not care for,
/// which Vulkan leaves undefined; or a resolve texture its known store action resolves into.
/// `None` for none. A resolve texture of an attachment whose store action is `Unknown` is written
/// only where a late store action resolves into it ([`VulkanRenderEncoder::late_store`]), and a
/// visibility result buffer only where a mode is set ([`Reach::visibility`]).
fn kept_attachment(d: &rt::RenderPassDescriptor, sources: &Sources) -> Option<&'static str> {
    let (depth, stencil) = (&d.depth_attachment, &d.stencil_attachment);
    let colors = d.color_attachments.iter().map(|a| a.store_action);
    let stores = colors.chain([depth.store_action, stencil.store_action]);
    let held = (sources.color.iter()).chain([&sources.depth, &sources.stencil]);
    for (store, [texture, resolve]) in stores.zip(held) {
        if texture.writes_kept() {
            return Some(
                "a pass, which the runtime's GPU may write whatever its load and store actions, \
                 whose attachment lies over",
            );
        }
        if store.resolves() && resolve.writes_kept() {
            return Some("a pass that resolves into a texture over");
        }
    }
    None
}

/// What [`metal_aborts`] says of a pass without attachments whose `defaultRasterSampleCount` is
/// `count`, which the runtime refuses as plain Metal stops it
/// (`RenderPassProblem::DefaultRasterSampleCount`): one whose low 8 bits are
/// 8 to 15 or 128 to 255, or 0 in a count of 256 or more, where plain Metal ends the process
/// making the encoder with `abort()`, nothing on stderr (on this host's M3 Max under macOS 26.7,
/// at every such count the runtime's sample-count probe tried there: 8 to 15, 128, 255 and 256;
/// on an M4 under macOS 27.0.1, at every count to 256 and the larger ones the runtime measured).
/// A count of 0 is [`metal_aborts_at`]'s.
fn stopped_default_samples(count: u64) -> String {
    format!(
        "a pass without attachments of defaultRasterSampleCount {count}, a count plain Metal ends \
         the process at"
    )
}

/// What Metal aborts at in a pass of `d`, which the runtime would refuse or make: a pass of no
/// attachment with a target size and no default raster sample count (`render_pass_no_target`:
/// Metal aborts once the encoder is made; the runtime refuses its count of 0, and
/// the death is this one), a memoryless depth or stencil attachment stored with
/// `MultisampleResolve` at one sample with no resolve texture (`render_pass_memoryless`,
/// `render_pass_memoryless_stencil_resolve`), and more programmable sample positions than a
/// multisampled pass's samples (`render_pass_msaa`: 8 on a 4-sample pass).
fn metal_aborts_at(d: &rt::RenderPassDescriptor) -> Option<&'static str> {
    let textures: Vec<&rt::Texture> = d
        .color_attachments
        .iter()
        .filter_map(|c| c.texture.as_ref())
        .chain(d.depth_attachment.texture.as_ref())
        .chain(d.stencil_attachment.texture.as_ref())
        .collect();
    if textures.is_empty()
        && d.render_target_width != 0
        && d.render_target_height != 0
        && d.default_raster_sample_count == 0
    {
        return Some(
            "a pass of no attachment with a target size and no default raster sample count",
        );
    }
    let resolves_from_memory = |t: &Option<rt::Texture>, store, resolve: &Option<rt::Texture>| {
        t.as_ref().is_some_and(|t| {
            t.storage_mode() == rt::StorageMode::Memoryless
                && t.sample_count() == 1
                && store == rt::StoreAction::MultisampleResolve
                && resolve.is_none()
        })
    };
    let (depth, stencil) = (&d.depth_attachment, &d.stencil_attachment);
    if resolves_from_memory(&depth.texture, depth.store_action, &depth.resolve_texture)
        || resolves_from_memory(
            &stencil.texture,
            stencil.store_action,
            &stencil.resolve_texture,
        )
    {
        return Some(
            "a memoryless depth or stencil attachment resolving at one sample with no resolve texture",
        );
    }
    let samples = textures
        .first()
        .map_or(d.default_raster_sample_count, |t| t.sample_count());
    let positions = u64::try_from(d.sample_positions.len()).unwrap_or(u64::MAX);
    if samples > 1 && positions > samples {
        return Some("more programmable sample positions than the pass's samples");
    }
    None
}

/// What a pass of `d` writes in the memory of a surface whose GPU writes never reach it
/// ([`crate::os::dropped_write`]), as `sources` says of its attachments: an attachment's
/// texture, a resolve texture its known store action resolves into, or its visibility result
/// buffer; `None` for none. A resolve texture of an attachment whose store action is `Unknown`
/// is written only where a late store action resolves into it
/// ([`VulkanRenderEncoder::late_store`]).
fn dropped_attachment(d: &rt::RenderPassDescriptor, sources: &Sources) -> Option<&'static str> {
    let (depth, stencil) = (&d.depth_attachment, &d.stencil_attachment);
    let colors = d.color_attachments.iter().map(|a| a.store_action);
    let stores = colors.chain([depth.store_action, stencil.store_action]);
    let held = (sources.color.iter()).chain([&sources.depth, &sources.stencil]);
    for (store, [texture, resolve]) in stores.zip(held) {
        if texture.gpu_read_only() {
            return Some("a pass whose attachment lies on");
        }
        if store.resolves() && resolve.gpu_read_only() {
            return Some("a pass that resolves into a texture on");
        }
    }
    let visibility = sources.visibility.as_ref();
    visibility
        .is_some_and(|b| b.gpu_read_only())
        .then_some("a pass whose visibility result buffer lies on")
}

/// `id<MTLRenderCommandEncoder>`.
pub type VulkanRenderEncoder = Obj<RenderEncoderObject>;

/// `renderCommandEncoderWithDescriptor:` of `descriptor` on `command_buffer`: nil for a pass with
/// no attachment and no target size, as Metal answers; the runtime's other refusals (Metal's
/// validation's, an encoder open, a committed buffer) die at the send, as Metal raises at none
/// of them: it goes on or asserts (`render_pass_actions`: slice 1 of a two-slice array at an
/// array length of 2, which the runtime refuses, completes on Metal). A pass without attachments
/// takes Metal's answer to its `defaultRasterSampleCount` by the count's low 8 bits (this host's
/// answers in a run of the runtime's probe): 1 to 7 run, 0, 8 to
/// 15 and 128 to 255 die as Metal ends the process there ([`metal_aborts_at`],
/// [`stopped_default_samples`]), and 16 to 127 make the encoder and fail the command buffer
/// with Metal's code ([`crate::gpu::command`]). A pass of a rotated
/// texture is the runtime's, which stores its texels turned and loads them as Metal does, but
/// for the 32 by 32 tiles that none of the draws reach of a pass that loads a texture of
/// rotation 4 and draws, which Metal keeps in place and the runtime lands mirrored as the others
/// (`rotated-load-tiles`, [`crate::gpu::desc`]), so such a pass dies at `endEncoding`, naming
/// the kind; a pass that draws nothing is Metal's. One whose attachment or resolve texture lies
/// over memory the runtime imports through a writable twin of read-only pages dies first
/// ([`kept_attachment`]).
/// One whose attachment, resolve texture or visibility result buffer lies on a surface whose GPU
/// writes never reach its memory dies ([`crate::os::dropped_write`]): Metal faults the command
/// buffer of such a pass that stores what it cleared or drew, and completes one that stores
/// nothing, or draws nothing. One that clears a bfloat attachment to a value the runtime stores
/// otherwise than Metal dies too ([`inexact_bfloat_clear`]), and so does one that clears an
/// attachment to a NaN the runtime is not known to store as Metal does ([`nan_clear`]): a
/// multisampled one to any NaN, which the runtime's multisampled clear drops, and a
/// single-sample one to any but the plain quiet NaN, or of an extended-range format to any.
/// A colour attachment that resolves into a texture of another format the runtime refuses is
/// made without the resolve where plain Metal was measured to leave the texture untouched, in
/// the shapes measured ([`untouched_resolves`]), and dies naming both formats and what plain
/// Metal was measured to do otherwise ([`refused_resolve_format`]), both decided by the rate map
/// the pass takes, as it takes it; the pairs the runtime decides itself are its own
/// ([`runtime_resolves`]). A resolve texture on
/// a surface whose GPU writes never reach its memory dies as a dropped write before that, as no
/// such pass was measured.
/// Its creation, nil or not, ends the exact `uniqueIdentifier` of later indirect command buffers
/// ([`crate::gpu::icb::encoder_made`]).
pub(crate) fn open(
    command_buffer: &VulkanCommandBuffer,
    _on: OnFault,
    descriptor: &VulkanRenderPassDescriptor,
) -> RaiseResult<Option<VulkanRenderEncoder>> {
    crate::gpu::icb::encoder_made();
    command_buffer.protected_creation("renderCommandEncoderWithDescriptor:");
    if let Some((attachment, setter, raw)) = descriptor.raw_value() {
        crate::answer::metal_aborts(
            "renderCommandEncoderWithDescriptor:",
            &format!(
                "a pass whose {attachment} holds {raw:#x} from {setter}, which Metal's validation \
                 refuses"
            ),
        );
    }
    let (d, map) = descriptor.pass();
    if let Some(what) = inexact_bfloat_clear(&d).or_else(|| nan_clear(&d)) {
        crate::answer::inexact(
            "renderCommandEncoderWithDescriptor:",
            crate::answer::Inexact::Clear,
            &what,
        );
    }
    let sources = descriptor.sources();
    if let Some(visibility) = &sources.visibility {
        visibility.used("renderCommandEncoderWithDescriptor:", None);
    }
    if crate::os::TWINS_READ_ONLY
        && let Some(what) = kept_attachment(&d, &sources)
    {
        crate::gpu::writes::kept_write("renderCommandEncoderWithDescriptor:", what);
    }
    if let Some(what) = dropped_attachment(&d, &sources) {
        crate::os::dropped_write("renderCommandEncoderWithDescriptor:", what);
    }
    // The colour resolves into a texture of another format, decided by the rate map the pass
    // takes, as it takes it.
    let settled = |d: rt::RenderPassDescriptor, mapped: PassMap| {
        if let Some(what) = refused_resolve_format(&d, &sources, mapped) {
            unrepresentable("renderCommandEncoderWithDescriptor:", &what);
        }
        untouched_resolves(d, &sources, mapped)
    };
    let made = match map {
        None => {
            let (d, dropped) = settled(d, PassMap::Unmapped);
            make(command_buffer, d, dropped, true, false)
        }
        Some(map) => map.for_pass(command_buffer, |taken| {
            let rasterizes = taken.is_some();
            let mapped = match &taken {
                None => PassMap::Unset,
                Some((_, Regions::Identity)) => PassMap::Identity,
                Some((_, Regions::Varying)) => PassMap::Varying,
            };
            let (mut d, dropped) = settled(d, mapped);
            d.rasterization_rate_map = taken.map(|(map, _)| map);
            make(
                command_buffer,
                d,
                dropped,
                rasterizes,
                mapped == PassMap::Varying,
            )
        }),
    };
    if let Ok(Some(e)) = &made {
        e.undecided_dropped(&sources);
        let visibility = (sources.visibility.as_ref())
            .map_or(Kept::No, |b| crate::gpu::writes::buffer_memory(b).kept);
        plain(|| e.reach.lock().visibility = visibility);
    }
    made
}

/// The clear of a multisampled bfloat colour attachment (`R16Bfloat`, `RG16Bfloat`,
/// `RGBA16Bfloat`) of the pass `d` that the runtime is not measured to store as Metal does, if
/// any: one to a value of a channel of the format that `half` and bfloat do not both hold exactly
/// ([`half_and_bfloat_hold`]), a NaN among them. Metal converts a clear's `float` to bfloat by one
/// of two routes, by the attachment and the pass: straight to the nearest bfloat where it keeps
/// the level losslessly compressed and the pass records no draw call, else through `half` (on
/// the M3 Max, -3e38 stores `ff62` straight and `ff80` through `half`, 1e-40 `0001` and `0000`,
/// 70000 `4789` and `7f80`, and 1.0039063692092896 `3f81` and `3f80`, rounded twice). The
/// runtime takes Metal's route, measured at one sample on an M4 and
/// on the M3 Max (`rgbx`'s `clears_and_bfloat_fills_are_metals`); no multisampled bfloat clear
/// is measured. A value both hold converts to its own code either way, as Metal's: 0, -0, ±1,
/// 0.5, -2.5, the infinities among them.
fn inexact_bfloat_clear(d: &rt::RenderPassDescriptor) -> Option<String> {
    d.color_attachments
        .iter()
        .enumerate()
        .find_map(|(index, a)| {
            let t = a.texture.as_ref()?;
            let channels = match t.pixel_format() {
                PixelFormat::R16Bfloat => 1,
                PixelFormat::RG16Bfloat => 2,
                PixelFormat::RGBA16Bfloat => 4,
                _ => return None,
            };
            if a.load_action != rt::LoadAction::Clear || t.sample_count() <= 1 {
                return None;
            }
            let c = a.clear_color;
            let values = [c.red, c.green, c.blue, c.alpha];
            let value = values[..channels]
                .iter()
                .find(|&&v| !half_and_bfloat_hold(v))?;
            Some(format!(
                "a clear of {}-sample bfloat colour attachment {index} to {values:?}, whose \
                 {value:e} the runtime's multisampled clear is not measured to store as Metal \
                 does",
                t.sample_count()
            ))
        })
}

/// The clear of a colour attachment of the pass `d` to a NaN the runtime is not known to store
/// as Metal does, in a channel its format has ([`unkept_nan`]), if any. The runtime's
/// multisampled clear drops any NaN: the samples do not hold its bits, and their resolve reads 0
/// in that channel (the runtime's lane on MoltenVK), where
/// Metal's answer follows the target's size, as whether it compresses it does: where both sides
/// are 16 or more, its samples keep the NaN, its sign too, and so does their resolve (a 4-sample
/// `RGBA16Float` clear to (NaN, 1, 0.5, NaN) resolves to `7e00` in red and alpha, a negative NaN
/// to `fe00`); below 16 a side its samples drop the sign (`7e00`) and their resolve stores 0 in
/// those channels (4 by 4 on the M3 Max: samples `7e00 3c00 3800 7e00`, resolve `0000 3c00 3800
/// 0000`). A single-sample clear of a half, float or packed float format to the plain quiet NaN
/// is Metal's (the runtime's lanes); one to another NaN is not measured, where Metal's half
/// clear keeps a negative NaN's sign at both sides 16 or more (`fe00`) and drops it below
/// (`7e00`), and neither is any clear of an extended-range format to a NaN. Neither rule needs
/// the size: a multisampled clear to any NaN dies, and a single-sample one to a negative NaN. A
/// multisampled bfloat attachment dies at any NaN, and a single-sample one's clear takes Metal's
/// route ([`inexact_bfloat_clear`]). Each such death is strict mode's: by default the pass opens,
/// relaying the clear as [`crate::answer::Inexact::Clear`].
fn nan_clear(d: &rt::RenderPassDescriptor) -> Option<String> {
    d.color_attachments
        .iter()
        .enumerate()
        .find_map(|(index, a)| {
            let t = a.texture.as_ref()?;
            if a.load_action != rt::LoadAction::Clear {
                return None;
            }
            let c = a.clear_color;
            let values = [c.red, c.green, c.blue, c.alpha];
            let (format, samples) = (t.pixel_format(), t.sample_count());
            let channel = unkept_nan(format, samples, values)?;
            let name = format.name();
            Some(match samples {
                ..=1 => format!(
                    "a clear of colour attachment {index} ({name}) to {values:?}, whose channel \
                 {channel} is a NaN the runtime's single-sample clear is not measured to store as \
                 Metal does: any NaN in an extended-range format, in another one other than the \
                 plain quiet NaN"
                ),
                _ => format!(
                    "a clear of {samples}-sample colour attachment {index} ({name}) to {values:?}, \
                 whose NaN in channel {channel} the runtime's multisampled clear drops: the \
                 samples hold no NaN and their resolve reads 0 there, where Metal keeps the NaN"
                ),
            })
        })
}

/// The first channel of `colour` (red, green, blue, alpha) among the channels of `format` that
/// is a NaN a clear of a texture of `samples` samples to `colour` is not known to store as
/// Metal does: any NaN, of either sign, at more than one sample; at one, any NaN in an
/// extended-range format, and in a half, float or packed float one a NaN other than the plain
/// quiet one (a double's `0x7ff8000000000000`, a float's `0x7fc00000`): negative, signalling or
/// of a payload. The formats: half and float `R16Float`, `RG16Float`, `RGBA16Float`, `R32Float`,
/// `RG32Float`, `RGBA32Float`, and `RGBX16Float` and `RGBX32Float`, whose four channels count,
/// the X field's among them, as no clear of theirs is measured; packed float `RG11B10Float` and
/// `RGB9E5Float` (red, green and blue); and extended-range `RGB10A8_2P_XR10`, `BGRA10_XR`,
/// `BGR10_XR` (red, green and blue), their `_sRGB` twins, and `RGBA16Float_XR`, a half format
/// whose clears are not measured. A bfloat format is [`inexact_bfloat_clear`]'s.
fn unkept_nan(format: PixelFormat, samples: u64, colour: [f64; 4]) -> Option<usize> {
    /// The plain quiet NaN, as a double.
    const QUIET: u64 = 0x7ff8_0000_0000_0000;
    // The format's channels, and whether a single-sample clear of it keeps the plain quiet NaN.
    let (channels, quiet_kept) = match format {
        PixelFormat::R16Float | PixelFormat::R32Float => (1, true),
        PixelFormat::RG16Float | PixelFormat::RG32Float => (2, true),
        PixelFormat::RG11B10Float | PixelFormat::RGB9E5Float => (3, true),
        PixelFormat::RGBA16Float
        | PixelFormat::RGBA32Float
        | PixelFormat::RGBX16Float
        | PixelFormat::RGBX32Float => (4, true),
        PixelFormat::BGR10XR | PixelFormat::BGR10XRSRGB => (3, false),
        PixelFormat::BGRA10XR
        | PixelFormat::BGRA10XRSRGB
        | PixelFormat::RGB10A82PXR10
        | PixelFormat::RGB10A82PXR10SRGB
        | PixelFormat::RGBA16FloatXR => (4, false),
        _ => return None,
    };
    let unkept = |v: &f64| v.is_nan() && (samples > 1 || !quiet_kept || v.to_bits() != QUIET);
    colour[..channels].iter().position(unkept)
}

/// The colour attachment of the pass `d` of a bfloat format (`R16Bfloat`, `RG16Bfloat`,
/// `RGBA16Bfloat`) that the pass loads (`Load`) and stores (any store action but `DontCare`), if
/// any, named: a draw (of anything, or of nothing), an execution of render commands or a tile
/// dispatch in the pass dies ([`RenderEncoderObject::bfloat_drawn`]), where Metal's draws. The
/// runtime draws into a bfloat attachment, holding its codes as `half` values in
/// the pass as Metal's tile memory does, but what a pass leaves in the tiles is not Metal's:
/// - its load pass moves each code of the attachment through `half` as a clear converts a value
///   (NaNs to `0x7fc0`, magnitudes past 65504 to the infinity, those below 2^-14 to `half`'s
///   steps) in every 32 by 32 tile until a draw's reach is known, announcing
///   `unreached-tiles-rewritten`, where Metal's tile store rewrites only the tiles a draw's
///   primitives reach, keeping the others' codes;
/// - one its load pass does not take (multisampled, held as a shadow, sparse) keeps its codes,
///   where Metal moves the codes of the tiles a draw reaches through `half`
///   (`extended-range-bits-kept`).
///
/// Both kinds are relayed for the extended-range formats on their guest reach
/// ([`crate::answer`]), where only bytes no Metal write leaves tell them apart. Neither ground
/// holds for bfloat: a kernel's write or a fill leaves codes `half` does not hold, and no guest is
/// known to draw into a bfloat attachment. Which tiles Metal rewrites was measured on
/// extended-range and padded attachments alone (the runtime's measurement: every tile where another
/// colour attachment is cleared and stored, else those a draw's primitives reach), so a pass that
/// also clears and stores another attachment dies too. Each such death is strict mode's: by default
/// the draw relays it ([`crate::answer::Inexact::BfloatLoaded`]).
fn bfloat_loaded(d: &rt::RenderPassDescriptor) -> Option<String> {
    d.color_attachments
        .iter()
        .enumerate()
        .find_map(|(index, a)| {
            let format = a.texture.as_ref()?.pixel_format();
            let bfloat = matches!(
                format,
                PixelFormat::R16Bfloat | PixelFormat::RG16Bfloat | PixelFormat::RGBA16Bfloat
            );
            let kept = a.load_action == rt::LoadAction::Load
                && a.store_action != rt::StoreAction::DontCare;
            (bfloat && kept).then(|| format!("colour attachment {index} ({})", format.name()))
        })
}

/// Whether `v` is a value both `half` and bfloat hold exactly: an infinity, or a finite `float`
/// whose low 16 bits are zero (bfloat) that is a multiple of `half`'s step at its binade (2^-24
/// below 2^-14, else 2^(e - 10) in [2^e, 2^(e + 1))) no greater than 65504. A NaN is not, its
/// payload's bits unmeasured.
fn half_and_bfloat_hold(v: f64) -> bool {
    if v.is_infinite() {
        return true;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the clear's channel as Metal converts it to a float, compared back below"
    )]
    let f = v as f32;
    if v.is_nan() || f64::from(f) != v || f.to_bits() & 0xffff != 0 {
        return false;
    }
    let a = f64::from(f.abs());
    if a == 0.0 {
        return true;
    }
    if a > 65504.0 {
        return false;
    }
    // The float's binade, from its exponent field: a value of half's normal range is a normal
    // float.
    let binade = ((f.to_bits() >> 23) & 0xff).cast_signed() - 127;
    let step = if a < 2f64.powi(-14) {
        2f64.powi(-24)
    } else {
        2f64.powi(binade - 10)
    };
    (a / step).fract() == 0.0
}

/// Whether the runtime answers nil for the pass `d`, as Metal does: no attachment has a texture
/// and the target width or height is 0 (`RenderPassProblem::NoTargets`). A pass whose tile
/// size, per-sample storage, threadgroup memory or tile memory is past the device's is nil too
/// ([`make`]).
fn answers_nil(d: &rt::RenderPassDescriptor) -> bool {
    d.color_attachments.iter().all(|a| a.texture.is_none())
        && d.depth_attachment.texture.is_none()
        && d.stencil_attachment.texture.is_none()
        && (d.render_target_width == 0 || d.render_target_height == 0)
}

/// What a render command binds or makes resident, by the host memory of each object it names,
/// which a draw may write through ([`Reach`]).
enum Bind {
    /// Buffers bound to a stage ([`VERTEX`], [`FRAGMENT`], [`TILE`]) from a slot.
    Buffers(usize, u64, Vec<Memory>),
    /// Textures bound to a stage from a slot.
    Textures(usize, u64, Vec<Memory>),
    /// The textures among resources made resident, the buffers among them, and whether they
    /// were made resident with write usage, with any of them lying in any memory.
    Used(Vec<Memory>, Vec<Memory>, bool),
}

/// The buffers `cmd` names, each with the bytes it reaches where they are exact
/// ([`crate::gpu::resource::Named`]): an indexed draw's indices ([`indices`]), the arguments of
/// an indirect draw (`MTLDrawPrimitivesIndirectArguments` and `MTLDrawPatchIndirectArguments`,
/// 16 bytes; `MTLDrawIndexedPrimitivesIndirectArguments`, 20) and an indirect execution range
/// (`MTLIndirectCommandBufferExecutionRange`, 8); all of a bound buffer, of a patch or control
/// point index buffer, of the indices of an indirect indexed draw, whose count the GPU reads, and
/// of a buffer made resident. Each is a use of those bytes
/// ([`crate::gpu::resource::BufferObject::used`]).
fn named_buffers<'a>(cmd: &RenderCmd<'a, VulkanPlatform>) -> Vec<Named<'a>> {
    match *cmd {
        RenderCmd::DrawIndexedPatchesIndirect {
            patch_index_buffer,
            control_point_index_buffer,
            indirect_buffer,
            indirect_buffer_offset,
            ..
        } => vec![
            (patch_index_buffer, None),
            (Some(control_point_index_buffer), None),
            (Some(indirect_buffer), reach(indirect_buffer_offset, 16)),
        ],
        RenderCmd::DrawIndexedPatches {
            patch_index_buffer,
            control_point_index_buffer,
            ..
        } => vec![
            (patch_index_buffer, None),
            (Some(control_point_index_buffer), None),
        ],
        RenderCmd::DrawIndexedPrimitives {
            index_count,
            index_type,
            index_buffer,
            index_buffer_offset,
            ..
        }
        | RenderCmd::DrawIndexedPrimitivesInstances {
            index_count,
            index_type,
            index_buffer,
            index_buffer_offset,
            ..
        }
        | RenderCmd::DrawIndexedPrimitivesBase {
            index_count,
            index_type,
            index_buffer,
            index_buffer_offset,
            ..
        } => vec![(
            Some(index_buffer),
            indices(index_type, index_buffer_offset, index_count),
        )],
        RenderCmd::DrawIndexedPrimitivesIndirect {
            index_buffer,
            indirect_buffer,
            indirect_buffer_offset,
            ..
        } => vec![
            (Some(index_buffer), None),
            (Some(indirect_buffer), reach(indirect_buffer_offset, 20)),
        ],
        RenderCmd::DrawPatchesIndirect {
            patch_index_buffer,
            indirect_buffer,
            indirect_buffer_offset,
            ..
        } => vec![
            (patch_index_buffer, None),
            (Some(indirect_buffer), reach(indirect_buffer_offset, 16)),
        ],
        RenderCmd::DrawPatches {
            patch_index_buffer, ..
        } => vec![(patch_index_buffer, None)],
        RenderCmd::DrawPrimitivesIndirect {
            indirect_buffer,
            indirect_buffer_offset,
            ..
        } => vec![(Some(indirect_buffer), reach(indirect_buffer_offset, 16))],
        RenderCmd::ExecuteCommandsIndirect {
            indirect_range_buffer,
            indirect_buffer_offset,
            ..
        } => vec![(
            Some(indirect_range_buffer),
            reach(indirect_buffer_offset, 8),
        )],
        RenderCmd::SetFragmentBuffer { buffer, .. }
        | RenderCmd::SetTessellationFactorBuffer { buffer, .. }
        | RenderCmd::SetVertexBuffer { buffer, .. } => vec![(buffer, None)],
        RenderCmd::SetFragmentBuffers { buffers, .. }
        | RenderCmd::SetTileBuffers { buffers, .. }
        | RenderCmd::SetVertexBuffersAttributeStrides { buffers, .. }
        | RenderCmd::SetVertexBuffers { buffers, .. } => {
            buffers.iter().map(|&b| (b, None)).collect()
        }
        RenderCmd::UseResources { resources, .. }
        | RenderCmd::UseResourcesStages { resources, .. } => {
            crate::gpu::encode_compute::resident_buffers(resources)
        }
        _ => Vec::new(),
    }
}

/// The bytes `count` indices of `index_type` read from `offset`: 2 each for
/// `MTLIndexTypeUInt16` (0), 4 for `MTLIndexTypeUInt32` (1); all of the buffer for any other
/// type, which the draw refuses. An indexed draw, the encoder's own and an indirect command's,
/// reads from the offset's low 32 bits, as Metal's does: an offset of 2^32 + 16 into a buffer
/// past 4 GiB reads the index at 16, where its validation checks the whole offset, as the
/// runtime's refusals do.
pub(crate) fn indices(index_type: u64, offset: u64, count: u64) -> Option<std::ops::Range<u64>> {
    let size = match index_type {
        0 => 2,
        1 => 4,
        _ => return None,
    };
    reach(offset & 0xffff_ffff, count.checked_mul(size)?)
}

/// What `cmd` binds or makes resident ([`Bind`]); `None` for any other command.
fn binds(cmd: &RenderCmd<'_, VulkanPlatform>) -> Option<Bind> {
    use crate::gpu::writes::{buffer_memory, texture_memory};
    let texture = |t: &Option<&VulkanTexture>| t.map(|t| texture_memory(t)).unwrap_or_default();
    let buffer = |b: &Option<&VulkanBuffer>| b.map(|b| buffer_memory(b)).unwrap_or_default();
    let buffers = |list: &[Option<&VulkanBuffer>]| list.iter().map(buffer).collect();
    let textures = |list: &[Option<&VulkanTexture>]| list.iter().map(texture).collect();
    Some(match cmd {
        RenderCmd::SetFragmentBuffer {
            buffer: b, index, ..
        } => Bind::Buffers(FRAGMENT, *index, vec![buffer(b)]),
        RenderCmd::SetVertexBuffer {
            buffer: b, index, ..
        } => Bind::Buffers(VERTEX, *index, vec![buffer(b)]),
        RenderCmd::SetFragmentBuffers {
            buffers: list,
            range,
            ..
        } => Bind::Buffers(FRAGMENT, range.location, buffers(list)),
        RenderCmd::SetTileBuffers {
            buffers: list,
            range,
            ..
        } => Bind::Buffers(TILE, range.location, buffers(list)),
        RenderCmd::SetVertexBuffers {
            buffers: list,
            range,
            ..
        }
        | RenderCmd::SetVertexBuffersAttributeStrides {
            buffers: list,
            range,
            ..
        } => Bind::Buffers(VERTEX, range.location, buffers(list)),
        RenderCmd::SetFragmentTexture { texture: t, index } => {
            Bind::Textures(FRAGMENT, *index, vec![texture(t)])
        }
        RenderCmd::SetFragmentTextures {
            textures: list,
            range,
        } => Bind::Textures(FRAGMENT, range.location, textures(list)),
        RenderCmd::SetTileTextures {
            textures: list,
            range,
        } => Bind::Textures(TILE, range.location, textures(list)),
        RenderCmd::SetVertexTextures {
            textures: list,
            range,
        } => Bind::Textures(VERTEX, range.location, textures(list)),
        RenderCmd::UseResources { resources, usage }
        | RenderCmd::UseResourcesStages {
            resources, usage, ..
        } => {
            let write = *usage & u64::from(ResourceUsage::WRITE.as_raw()) != 0;
            let any = (resources.iter().flatten()).any(|r| match r {
                ResourceRef::Texture(t) => !texture_memory(t).is_empty(),
                ResourceRef::Buffer(b) => !buffer_memory(b).is_empty(),
                ResourceRef::Icb(_) => false,
            });
            Bind::Used(
                (resources.iter().flatten())
                    .filter_map(|r| match r {
                        ResourceRef::Texture(t) => Some(texture_memory(t)),
                        _ => None,
                    })
                    .collect(),
                (resources.iter().flatten())
                    .filter_map(|r| match r {
                        ResourceRef::Buffer(b) => Some(buffer_memory(b)),
                        _ => None,
                    })
                    .collect(),
                write && any,
            )
        }
        _ => return None,
    })
}

/// What a pass's draws may write through what it binds and made resident: the host memory of
/// the buffers and textures bound at each stage's slots, of the textures made resident with
/// write usage, and the pipeline state's facts of which slots its functions write. A draw, a
/// tile dispatch or an execution reads it ([`Reach::unsent`]).
#[derive(Debug, Default)]
struct Reach {
    /// Each stage's buffers by slot, in [`VERTEX`], [`FRAGMENT`] and [`TILE`] order.
    buffers: [BTreeMap<u64, Memory>; 3],
    /// Each stage's textures by slot.
    textures: [BTreeMap<u64, Memory>; 3],
    /// The textures made resident with write usage.
    used_textures: Vec<Memory>,
    /// The buffers made resident with write usage.
    used_buffers: Vec<Memory>,
    /// Whether the pass's visibility result buffer lies over memory the runtime imports through
    /// a writable twin of read-only pages: a mode other than `Disabled` dies, as the pass writes
    /// a result at its offset, a draw after it or not ([`crate::gpu::writes::kept_write`]).
    visibility: Kept,
    /// The reflection of the pipeline state set, which this holds as Metal's encoder holds the
    /// state, and whether its functions write textures, which tells which texture slots they
    /// may write where the reflection cannot: the vertex function's where it writes any memory,
    /// and the fragment and tile functions' where they write textures.
    pipeline: Option<(Arc<Reflection>, [bool; 3])>,
}

impl Reach {
    /// `memory`, one entry an object, bound to `stage`'s buffer slots (`buffers`) or texture
    /// slots from `first`.
    fn bind(&mut self, stage: usize, buffers: bool, first: u64, memory: Vec<Memory>) {
        let slots = if buffers {
            &mut self.buffers[stage]
        } else {
            &mut self.textures[stage]
        };
        for (slot, m) in (first..).zip(memory) {
            if m.is_empty() {
                slots.remove(&slot);
            } else {
                slots.insert(slot, m);
            }
        }
    }

    /// The pipeline state `state` set.
    fn set(&mut self, state: &VulkanRenderPipeline) {
        let raw = &state.raw;
        let facts = plain(|| {
            [
                raw.vertex_writes_memory(),
                raw.fragment_writes_textures(),
                raw.tile_writes_textures(),
            ]
        });
        self.pipeline = Some((Arc::clone(&state.reflection), facts));
    }

    /// Whether a texture bound or made resident for writing, or a buffer bound to the tile
    /// stage, lies on a surface whose GPU writes never reach its memory
    /// ([`crate::os::gpu_read_only`]). A vertex or fragment function's stores into a buffer on
    /// one are dropped, as on Metal, and are sent ([`crate::os::dropped_write`]).
    fn unsent_dropped(&self) -> bool {
        let bound = (self.textures.iter().chain([&self.buffers[TILE]])).flat_map(BTreeMap::values);
        bound
            .chain(&self.used_textures)
            .any(|m| m.dropped || m.refused.is_some())
    }

    /// What a draw, a tile dispatch, or an execution whose commands take `inherits` of the
    /// encoder, may write that is never sent where it lies on a surface whose GPU writes never
    /// reach its memory ([`VulkanRenderEncoder::issues`]): the textures bound at the
    /// slots the pipeline state's functions write, the textures made resident with write usage,
    /// and, but for an execution, whose commands run no tile function, the buffers bound at the
    /// slots the tile function writes.
    fn unsent(&self, inherits: Option<Inherits>) -> Vec<(Whether, Memory)> {
        let mut written: Vec<_> = (self.used_textures.iter())
            .map(|m| (Whether::Writes, m.clone()))
            .collect();
        written.extend(self.at_slots(inherits, [false, false, inherits.is_none()]));
        written
    }

    /// How a draw, a tile dispatch, or an execution whose commands take `inherits` of the encoder,
    /// writes through an object over memory the runtime imports through a writable twin of
    /// read-only pages ([`crate::gpu::writes::kept_write`]), the first found: a texture or buffer
    /// made resident with write usage, and, as [`Self::at_slots`] reads them, a texture bound at
    /// a slot a function writes and a buffer bound at one, of every stage, but for an execution,
    /// whose commands run no tile function and take the encoder's buffers only where they inherit
    /// them. `None` where it writes none.
    fn kept(&self, inherits: Option<Inherits>) -> Option<Whether> {
        let used = (self.used_textures.iter()).chain(&self.used_buffers);
        let mut written: Vec<_> = (used.filter(|m| m.kept.holds()))
            .map(|_| Whether::Writes)
            .collect();
        let buffers = inherits.is_none_or(|i| i.buffers);
        let slots = self.at_slots(inherits, [buffers, buffers, inherits.is_none()]);
        written.extend((slots.into_iter()).filter_map(|(w, m)| m.kept.holds().then_some(w)));
        written.into_iter().find(Whether::holds)
    }

    /// What a draw, or an execution whose commands take `inherits` of the encoder, may write at
    /// the stages' slots, one entry an object, with whether it does: the textures bound at the
    /// slots the pipeline state's functions write, by its reflection, and the buffers too at the
    /// stages `buffers` names, in [`VERTEX`], [`FRAGMENT`] and [`TILE`] order. Where the
    /// reflection cannot tell, every function may write every buffer slot of its stage, as a
    /// kernel may, a vertex function that writes memory every texture slot of its stage, and a
    /// fragment or tile function that writes textures every texture slot of its stage. A vertex
    /// function that writes no memory, by the runtime's translation of it (its
    /// `vertex_writes_memory`), writes no slot of its stage, whatever its reflection answers of
    /// an argument: a `device` pointer that is not `const` is read-write there, stored through
    /// or not. An execution whose commands run pipeline states of their own, which no
    /// reflection here tells, writes every bound buffer it takes, and no bound texture: Metal
    /// makes no state that supports indirect command buffers of a function with a texture
    /// argument, and such commands reach a texture only through an argument buffer, whose
    /// textures are made resident. Measured on the M3 Max (macOS 26.7):
    /// `supportIndirectCommandBuffers` with a vertex, fragment or kernel function taking a
    /// texture to read, sample or write answers no state ("Fragment shader cannot be used with
    /// indirect command buffers"), with one taking buffers or an argument buffer of a texture a
    /// state; an execution that inherits a state made without it, of such a function, faults
    /// its command buffer (`PageFault`), the texture bound unwritten.
    fn at_slots(&self, inherits: Option<Inherits>, buffers: [bool; 3]) -> Vec<(Whether, Memory)> {
        let mut written = Vec::new();
        let state = match inherits {
            Some(i) if !i.pipeline => None,
            _ => match &self.pipeline {
                Some(p) => Some(p),
                // No state for the draw, which the runtime refuses.
                None => return written,
            },
        };
        for stage in [VERTEX, FRAGMENT, TILE] {
            if stage == VERTEX && state.is_some_and(|(_, facts)| !facts[VERTEX]) {
                continue;
            }
            let at = |texture: bool, slot: u64| match state {
                Some((reflection, facts)) => Whether::Slot(
                    Arc::clone(reflection),
                    Slot {
                        stage,
                        texture,
                        slot,
                        untold: !texture || facts[stage],
                    },
                ),
                None => Whether::Writes,
            };
            for (&slot, m) in (self.buffers[stage].iter()).filter(|_| buffers[stage]) {
                written.push((at(false, slot), m.clone()));
            }
            for (&slot, m) in (self.textures[stage].iter()).filter(|_| state.is_some()) {
                written.push((at(true, slot), m.clone()));
            }
        }
        written
    }
}

/// The encoder of the pass `d` on `command_buffer`, with its rate map as [`open`] took it and
/// without the colour resolves it `dropped` ([`untouched_resolves`]); `rasterizes` is false for a
/// pass of a mutable rate map not yet reset, and `varying_rates` true for one of a map whose
/// rates vary ([`Regions`]). One made is work for Metal's tiler
/// ([`crate::gpu::command::CommandBufferObject::tiler_work`]).
fn make(
    command_buffer: &VulkanCommandBuffer,
    d: rt::RenderPassDescriptor,
    dropped: Vec<Dropped>,
    rasterizes: bool,
    varying_rates: bool,
) -> RaiseResult<Option<VulkanRenderEncoder>> {
    let sel = "renderCommandEncoderWithDescriptor:";
    let made = without_unused_resolves(&d);
    match plain_at(sel, || command_buffer.raw.render_command_encoder(&made)) {
        Ok(raw) => {
            command_buffer.tiler_work(crate::gpu::metal_space::FirstUse::RenderEncoder);
            opened(
                command_buffer,
                raw,
                (&d, made, dropped),
                rasterizes,
                varying_rates,
            )
            .map(Some)
        }
        Err(rt::Error::Nil(n))
            if n.problem == rt::Problem::RenderPass(rt::RenderPassProblem::NoTargets) =>
        {
            debug_assert!(answers_nil(&d), "a nil pass answers_nil missed");
            Ok(None)
        }
        Err(rt::Error::Invalid(i)) if i.problem == rt::Problem::EncoderOpen => metal_aborts(
            sel,
            "a command buffer with an open encoder (\"with uncommitted encoder\")",
        ),
        Err(e) => match metal_aborts_at(&d) {
            Some(what) => metal_aborts(sel, what),
            None => match e {
                // What plain Metal answers its encoder nil for: a tile size, per-sample storage,
                // threadgroup memory or tile memory past the device's.
                rt::Error::Nil(_) => Ok(None),
                rt::Error::Invalid(i)
                    if matches!(
                        i.problem,
                        rt::Problem::RenderPass(
                            rt::RenderPassProblem::DefaultRasterSampleCount { .. }
                        )
                    ) =>
                {
                    metal_aborts(sel, &stopped_default_samples(d.default_raster_sample_count))
                }
                rt::Error::Invalid(i)
                    if matches!(
                        i.problem,
                        rt::Problem::RenderPass(
                            rt::RenderPassProblem::ArrayLengthPastSlices { .. }
                        )
                    ) =>
                {
                    unrepresentable(
                        sel,
                        &format!(
                            "{:?}: a renderTargetArrayLength past an attachment's slices, where \
                             Metal renders min(length, the texture's slices) layers from the \
                             attachment's slice, those past the texture into the memory after \
                             it at its slice stride (a buffer's bytes, for a texture over one) \
                             or into a heap neighbour, and a draw that writes no layer renders \
                             the base slice alone; nothing here writes past a texture",
                            i.problem
                        ),
                    )
                }
                e => {
                    if crate::gpu::desc::is_format_refusal(&e) {
                        crate::gpu::ycbcr::pass_refused(sel, &d, &e);
                    }
                    decide(sel, Policy::Dies, e)
                }
            },
        },
    }
}

/// The encoder of the runtime's `raw`, just made on `command_buffer` of `made`: the pass `d`
/// without its unused resolves ([`without_unused_resolves`]), `d` itself made without the colour
/// resolves `dropped` lists ([`untouched_resolves`]). It is in its handle before anything can
/// stop this thread: a thread of an isolated task that ended stops at the next send with the
/// encoder open, as Metal's is left, where the runtime's encoder dropped as it unwinds would
/// abandon the command buffer. What Metal aborts at in the pass dies then; a pass that
/// rasterizes nothing empties its scissor rectangle.
fn opened(
    command_buffer: &VulkanCommandBuffer,
    raw: rt::RenderCommandEncoder,
    (d, made, dropped): (
        &rt::RenderPassDescriptor,
        rt::RenderPassDescriptor,
        Vec<Dropped>,
    ),
    rasterizes: bool,
    varying_rates: bool,
) -> RaiseResult<VulkanRenderEncoder> {
    let sel = "renderCommandEncoderWithDescriptor:";
    let e = Obj::new(RenderEncoderObject {
        encoder: EncoderObject::new(raw, command_buffer),
        nil_pipeline: Mutex::new(false),
        device: command_buffer.device.raw.clone(),
        undecided: Mutex::new(undecided(d)),
        late_stores: Mutex::new(Vec::new()),
        marks: Mutex::new(Marks::default()),
        no_layers: (d.rasterization_rate_map.as_ref()).is_some_and(|m| m.layer_count() == 0),
        pass: made,
        dropped,
        rasterizes,
        varying_rates,
        reach: Mutex::new(Reach::default()),
        rounded: Mutex::new(RoundedTextures::default()),
        bfloat_loaded: bfloat_loaded(d),
        hung: AtomicBool::new(false),
        encodes_icbs: AtomicBool::new(false),
        writable_icbs: Mutex::new(Vec::new()),
        icb_written: Mutex::new(None),
    });
    if let Some(what) = metal_aborts_at(d) {
        metal_aborts(sel, what);
    }
    if !rasterizes {
        e.run(sel, &mut |r| r.set_scissor_rect(EMPTY))?;
    }
    Ok(e)
}

/// A binding index or a count core passes as `NSUInteger`, which the runtime takes as `u32`.
fn small(sel: &'static str, value: u64) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| unrepresentable(sel, &format!("{value} as a u32")))
}

/// The first index of `range`, and the first `range.length` elements of `slice`: a slice shorter
/// than its range dies, as Metal's reads past it.
fn ranged<'a, T>(sel: &'static str, range: NsRange, slice: &'a [T]) -> (u32, &'a [T]) {
    let n = usize::try_from(range.length)
        .ok()
        .filter(|&n| n <= slice.len())
        .unwrap_or_else(|| {
            unrepresentable(sel, &format!("{} elements for {range:?}", slice.len()))
        });
    (small(sel, range.location), &slice[..n])
}

/// Core's buffers, as the runtime's.
fn buffers<'a>(list: &[Option<&'a VulkanBuffer>]) -> Vec<Option<&'a rt::Buffer>> {
    list.iter().map(|b| b.map(|b| &b.raw)).collect()
}

/// Core's textures, as the runtime's, bound by `sel`
/// ([`crate::gpu::resource::TextureObject::texels`]).
fn textures<'a>(
    sel: &'static str,
    list: &[Option<&'a VulkanTexture>],
) -> Vec<Option<&'a rt::Texture>> {
    list.iter().map(|t| t.map(|t| t.texels(sel))).collect()
}

/// Core's sampler states, as the runtime's.
fn samplers<'a>(
    list: &[Option<&'a crate::gpu::resource::VulkanSamplerState>],
) -> Vec<Option<&'a rt::SamplerState>> {
    list.iter().map(|s| s.map(|s| &**s)).collect()
}

/// The runtime's form of core's resources, for `sel`: a nil one dies.
fn resources<'a>(
    sel: &'static str,
    list: &'a [Option<ResourceRef<'a, VulkanPlatform>>],
) -> Vec<rt::ResourceRef<'a>> {
    list.iter()
        .map(|r| {
            r.as_ref()
                .map_or_else(|| unrepresentable(sel, "a nil resource"), resource)
        })
        .collect()
}

/// The runtime's form of core's resources, the nil ones skipped.
fn present<'a>(list: &'a [Option<ResourceRef<'a, VulkanPlatform>>]) -> Vec<rt::ResourceRef<'a>> {
    list.iter().flatten().map(resource).collect()
}

/// The runtime's form of a resource of core's.
fn resource<'a>(r: &'a ResourceRef<'a, VulkanPlatform>) -> rt::ResourceRef<'a> {
    match r {
        ResourceRef::Buffer(b) => rt::ResourceRef::Buffer(&b.raw),
        ResourceRef::Texture(t) => rt::ResourceRef::Texture(&t.raw),
        ResourceRef::Icb(icb) => rt::ResourceRef::IndirectCommandBuffer(&icb.raw),
    }
}

/// Core's heaps, as the runtime's: a nil one dies. Each is made resident
/// ([`crate::gpu::resource::HeapObject::made_resident`]).
fn heaps<'a>(
    sel: &'static str,
    list: &[Option<&'a crate::gpu::resource::VulkanHeap>],
) -> Vec<&'a rt::Heap> {
    let list: Vec<&crate::gpu::resource::VulkanHeap> = list
        .iter()
        .map(|h| h.unwrap_or_else(|| unrepresentable(sel, "a nil heap")))
        .collect();
    for heap in &list {
        heap.made_resident(sel);
    }
    list.into_iter().map(|h| &h.raw).collect()
}

/// The stages `useHeaps:count:` and `useResources:count:usage:` reach on a render encoder.
const VERTEX_AND_FRAGMENT: RenderStages = RenderStages::VERTEX.union(RenderStages::FRAGMENT);

/// Core's `MTLViewport`.
fn viewport(v: MtlViewport) -> rt::Viewport {
    rt::Viewport::new(v.origin_x, v.origin_y, v.width, v.height, v.znear, v.zfar)
}

/// Core's `MTLScissorRect`.
fn scissor(r: MtlScissorRect) -> rt::ScissorRect {
    rt::ScissorRect::new(r.x, r.y, r.width, r.height)
}

/// The scissor rectangle of a pass that rasterizes nothing.
const EMPTY: rt::ScissorRect = rt::ScissorRect {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
};

/// The line of a draw's death in a pass of a rate map whose rates vary ([`Regions`]).
const VARYING_RATES_DRAW: &str = "a draw in a pass of a rasterization rate map whose rates vary, \
     which the runtime rasterizes by regions whose lines are the map's tiles' exact lines, where \
     Metal's are its fixed-point ones (rasterization-rate-map)";

/// The line of a tile dispatch's death in a pass of a rate map whose rates vary ([`Regions`]).
const VARYING_RATES_TILES: &str = "a tile dispatch in a pass of a rasterization rate map whose \
     rates vary, which Metal is not measured at (rasterization-rate-map)";

/// The line of a draw's death, direct or an execution's command, of points under a state whose
/// points Metal sizes none of ([`VulkanRenderEncoder::sized`]).
pub(crate) const UNSIZED_POINTS: &str = "points under a vertex function that writes no \
     [[point_size]], outside the point input primitive topology class, whose points Metal does \
     not size: most such draws draw nothing, and some a box whose size changes from one run to \
     the next";

/// A raw store action, sent with `sel`.
fn store_action(sel: &'static str, raw: u64) -> rt::StoreAction {
    rt::StoreAction::from_raw(raw)
        .unwrap_or_else(|_| unrepresentable(sel, &format!("store action {raw}")))
}

/// A raw store action option set, sent with `sel`.
fn store_action_options(sel: &'static str, raw: u64) -> rt::StoreActionOptions {
    rt::StoreActionOptions::from_raw(raw)
        .unwrap_or_else(|_| unrepresentable(sel, &format!("store action options {raw}")))
}

/// A late resolve target of a texture at its level, slice and depth plane; `None` for nil.
fn resolve_target(
    texture: Option<&rt::Texture>,
    (slice, depth_plane, level): (u64, u64, u64),
) -> Option<rt::ResolveTarget> {
    texture.map(|t| {
        let mut target = rt::ResolveTarget::new(t.clone());
        target.slice = slice;
        target.depth_plane = depth_plane;
        target.level = level;
        target
    })
}

/// How a draw's raw primitive type draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Topology {
    /// A public `MTLPrimitiveType`, drawn as it is.
    Public(PrimitiveType),
    /// One of Metal's private types 5 to 8.
    Private(Private),
}

/// Metal's private primitive types 5 to 8, which Metal draws on this host (`render_draw_odd_types`,
/// and probes of every vertex count from 0 to 12, from vertex 0 to 2, at random positions, one
/// instance and three from base instance 1): each draws exactly what an indexed draw of a public
/// type over [`Private::indices`] draws, every pixel's flat vertex ID alike, with and without
/// additive blending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Private {
    /// 5: a line loop, the line strip of its vertices and its first again.
    LineLoop,
    /// 6: a triangle fan, the triangles (first, i, i + 1).
    TriangleFan,
    /// 7: quads, each four vertices a, b, c, d the triangles (a, b, d) and (b, c, d); the vertices
    /// past the last whole quad draw nothing.
    Quads,
    /// 8: a quad strip, each pair c, d after a pair a, b the triangles (a, b, d) and (c, a, d); an
    /// odd last vertex draws nothing.
    QuadStrip,
}

/// The most indices an emulated draw makes: 64 MiB of them. Drawing them in chunks would take no
/// less, as each chunk's buffer lives until the command buffer completes.
const MAX_EMULATED_INDICES: u64 = 1 << 24;

impl Private {
    /// The public type its indices draw.
    fn public(self) -> PrimitiveType {
        match self {
            Private::LineLoop => PrimitiveType::LineStrip,
            Private::TriangleFan | Private::Quads | Private::QuadStrip => PrimitiveType::Triangle,
        }
    }

    /// How many indices it draws `count` vertices as.
    fn index_count(self, count: u64) -> u64 {
        match self {
            Private::LineLoop if count >= 2 => count + 1,
            Private::TriangleFan if count >= 3 => 3 * (count - 2),
            Private::Quads => 6 * (count / 4),
            Private::QuadStrip if count >= 4 => 6 * ((count - 2) / 2),
            _ => 0,
        }
    }

    /// The indices of the vertices `start..start + count` it draws as [`Private::public`], in
    /// order, relative to `start`, which the draw takes as its base vertex
    /// ([`VulkanRenderEncoder::primitives`]); none where it draws nothing. Vertex IDs past
    /// `u32`, and more indices than [`MAX_EMULATED_INDICES`], die at `sel`. No index is
    /// `0xffffffff`, the restart index of the line strip a line loop draws as.
    fn indices(self, sel: &'static str, start: u32, count: u32) -> Vec<u32> {
        let n = self.index_count(u64::from(count));
        let last = start.checked_add(count.saturating_sub(1));
        if n > MAX_EMULATED_INDICES || (count > 0 && last.is_none()) {
            unrepresentable(
                sel,
                &format!("{self:?} of {count} vertices from {start} as an indexed draw"),
            );
        }
        let at = |i: u32| i;
        let mut out = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
        match self {
            Private::LineLoop if count >= 2 => {
                out.extend((0..count).map(at));
                out.push(at(0));
            }
            Private::TriangleFan => {
                for i in 1..count.saturating_sub(1) {
                    out.extend([at(0), at(i), at(i + 1)]);
                }
            }
            Private::Quads => {
                for q in (0..count / 4).map(|q| 4 * q) {
                    let (a, b, c, d) = (at(q), at(q + 1), at(q + 2), at(q + 3));
                    out.extend([a, b, d, b, c, d]);
                }
            }
            Private::QuadStrip if count >= 4 => {
                for q in (0..(count - 2) / 2).map(|q| 2 * q) {
                    let (a, b, c, d) = (at(q), at(q + 1), at(q + 2), at(q + 3));
                    out.extend([a, b, d, c, a, d]);
                }
            }
            _ => {}
        }
        out
    }
}

/// The indices of the vertices `start..start + count` of a list (`p` `Triangle`) or strip
/// (`TriangleStrip`) of triangles, relative to `start`, which the draw takes as its base vertex
/// ([`VulkanRenderEncoder::primitives`]), as a list of its triangles each with its last vertex
/// first, in its winding: a list's (a, b, c) as (c, a, b), a strip's triangle i, (i, i + 1,
/// i + 2) when i is even and (i + 1, i, i + 2) when odd, as (i + 2, i, i + 1) and (i + 2, i + 1,
/// i). Drawn from its first vertex, each triangle takes its flat inputs from the vertex Metal's
/// last-vertex mode takes them from (`setProvokingVertexMode:`), and its smooth ones as Metal
/// draws them (the module docs). Vertex IDs past `u32`, and more indices than
/// [`MAX_EMULATED_INDICES`], die at `sel`; another type has no such list.
fn last_first(sel: &'static str, p: PrimitiveType, start: u32, count: u32) -> Vec<u32> {
    let triangles = match p {
        PrimitiveType::Triangle => count / 3,
        PrimitiveType::TriangleStrip => count.saturating_sub(2),
        _ => unrepresentable(sel, &format!("{p:?} with its last vertex first")),
    };
    let n = 3 * u64::from(triangles);
    let last = start.checked_add(count.saturating_sub(1));
    if n > MAX_EMULATED_INDICES || (count > 0 && last.is_none()) {
        unrepresentable(
            sel,
            &format!("{p:?} of {count} vertices from {start}, last vertex first"),
        );
    }
    let at = |i: u32| i;
    let mut out = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
    for t in 0..triangles {
        match p {
            PrimitiveType::Triangle => out.extend([at(3 * t + 2), at(3 * t), at(3 * t + 1)]),
            _ if t % 2 == 0 => out.extend([at(t + 2), at(t), at(t + 1)]),
            _ => out.extend([at(t + 2), at(t + 1), at(t)]),
        }
    }
    out
}

/// Whether a draw of `count` vertices or indices of `p` makes no primitive, which Metal draws
/// nothing of whatever the private setters left ([`VulkanRenderEncoder::rotates`]).
fn draws_none(p: PrimitiveType, count: u32) -> bool {
    let least = match p {
        PrimitiveType::Point => 1,
        PrimitiveType::Line | PrimitiveType::LineStrip => 2,
        PrimitiveType::Triangle | PrimitiveType::TriangleStrip => 3,
    };
    count < least
}

/// The primitive type Metal's GPU hangs at ([`VulkanRenderEncoder::hangs`]).
const HANGING_TYPE: u64 = 9;

/// A draw's raw primitive type: a public one, or one of the private ones Metal draws (5 to 8).
/// Type 9 dies here: where Metal's GPU fails the command buffer at it is the non-indexed draws'
/// ([`VulkanRenderEncoder::hangs`]); a type past it no `MTLPrimitiveType` is.
fn topology(sel: &'static str, raw: u64) -> Topology {
    match raw {
        5 => Topology::Private(Private::LineLoop),
        6 => Topology::Private(Private::TriangleFan),
        7 => Topology::Private(Private::Quads),
        8 => Topology::Private(Private::QuadStrip),
        HANGING_TYPE => unrepresentable(
            sel,
            "primitive type 9 over indices or indirect arguments, which nothing measured",
        ),
        _ => Topology::Public(metal_value(sel, "primitive type", raw)),
    }
}

/// A draw's raw primitive type where only a public one has a form: an indexed or indirect draw,
/// whose vertices the GPU reads at execution, so no draw of the runtime's draws a private type
/// over them exactly.
fn public_type(sel: &'static str, raw: u64) -> PrimitiveType {
    match topology(sel, raw) {
        Topology::Public(p) => p,
        Topology::Private(t) => unrepresentable(
            sel,
            &format!(
                "primitive type {raw} ({t:?}) over indices or indirect arguments, which only the \
                 GPU reads"
            ),
        ),
    }
}

/// The primitive and index type of an indexed draw: `None` for an index type no `MTLIndexType`
/// is, which Metal draws nothing for. A private primitive type dies ([`public_type`]).
fn indexed_kinds(
    sel: &'static str,
    primitive_type: u64,
    raw_index_type: u64,
) -> Option<(PrimitiveType, IndexType)> {
    let p = public_type(sel, primitive_type);
    let i = index_type(raw_index_type)?;
    Some((p, i))
}

/// A draw's raw index type: `None` for one no `MTLIndexType` is, which Metal draws nothing for.
fn index_type(raw: u64) -> Option<IndexType> {
    u32::try_from(raw)
        .ok()
        .and_then(|raw| IndexType::try_from(raw).ok())
}

/// The range of an `NSRange`, for `sel`.
fn range_of(sel: &'static str, r: NsRange) -> Range<u64> {
    let end = r
        .location
        .checked_add(r.length)
        .unwrap_or_else(|| unrepresentable(sel, &format!("{r:?}")));
    r.location..end
}

/// A command on the open runtime encoder.
type Call<'c> = &'c mut dyn FnMut(&mut rt::RenderCommandEncoder) -> Result<(), rt::Error>;

impl VulkanRenderEncoder {
    /// `bind`, of a command: what a draw may write through what it binds or makes resident is
    /// read at the draw ([`Reach`]).
    fn bound(&self, bind: Bind) {
        plain(|| {
            let mut reach = self.reach.lock();
            match bind {
                Bind::Buffers(stage, first, memory) => reach.bind(stage, true, first, memory),
                Bind::Textures(stage, first, memory) => reach.bind(stage, false, first, memory),
                Bind::Used(textures, buffers, true) => {
                    reach.used_textures.extend(textures);
                    reach.used_buffers.extend(buffers);
                }
                Bind::Used(_, _, false) => {}
            }
        });
    }

    /// Dies at `sel`, a draw, a tile dispatch or an execution the encoder issues, `inherits` what
    /// its commands take of the encoder, where it may write the memory of a surface whose GPU
    /// writes never reach it through a texture, one the pass binds at a slot a function writes,
    /// by its reflection, or one made resident for writing, or through a buffer the tile function
    /// writes, which nothing measured ([`Reach::unsent`], [`crate::os::dropped_write`]), naming
    /// how ([`Whether::through`]). A vertex or fragment function's stores into a buffer on such a
    /// surface are sent, and dropped as on Metal. Before that, it dies where it may write through
    /// an object over memory the runtime imports through a writable twin of read-only pages,
    /// whose pages the write would reach ([`Reach::kept`], [`crate::gpu::writes::kept_write`]).
    /// One that issues nothing, which Metal writes nothing of, is not asked.
    fn issues(&self, sel: &'static str, inherits: Option<Inherits>) {
        if crate::os::TWINS_READ_ONLY
            && let Some(whether) = plain(|| self.reach.lock().kept(inherits))
        {
            let what = format!("a draw's write through {}, into", whether.through());
            crate::gpu::writes::kept_write(sel, &what);
        }
        let written = plain(|| {
            let reach = self.reach.lock();
            reach.unsent_dropped().then(|| reach.unsent(inherits))
        });
        let hit = (written.iter().flatten())
            .find(|(w, m)| (m.dropped || m.refused.is_some()) && w.holds());
        if let Some((whether, m)) = hit {
            let what = format!("a draw's write through {}", whether.through());
            match m.refused.filter(|_| !m.dropped) {
                None => crate::os::dropped_write(sel, &format!("{what}, on")),
                Some(pf) => crate::gpu::ycbcr::refused_write(sel, &format!("{what}, into"), pf),
            }
        }
    }

    /// Relays at `sel`, or dies there in strict mode ([`crate::answer::inexact`]), a draw, or an
    /// execution whose commands inherit the encoder's state (`execution`), under a state that
    /// rounds texture writes otherwise than the runtime, where it may write a texture of one of
    /// [`crate::gpu::serialized::MODE_2_TEXTURE_FORMATS`] ([`RoundedTextures::rounded`]). One
    /// that draws nothing, which writes nothing, is not asked.
    fn rounds(&self, sel: &'static str, execution: bool) {
        if let Some(what) = plain(|| self.rounded.lock().rounded(execution)) {
            crate::gpu::serialized::rounded_texture_writes_inexact(sel, &what);
        }
    }

    /// Runs `call` on the open encoder for `sel`: a refusal dies at the send, naming the
    /// runtime's error, as Metal raises at none of these commands.
    fn run(&self, sel: &'static str, call: Call<'_>) -> RaiseResult<()> {
        send(sel, Policy::Dies, || self.encoder.with(sel, |e| call(e)))
    }

    /// Dies at `sel`, a draw or an execution of indirect commands that is not empty, in a pass
    /// of a rate map of no layers: Metal's rasterizes it by what earlier maps left, which no call
    /// tells, where the runtime's draws nothing ([`crate::gpu::rate_map`]). In a pass of a map
    /// whose rates vary, whose regions the runtime's rasterizes by the tiles' exact lines, where
    /// Metal's are their fixed-point ones ([`Regions`]), it draws, relayed, and dies in strict
    /// mode ([`crate::answer::Inexact::RateMap`]). The pass's loads, clears and
    /// stores are Metal's, so one that draws nothing runs. A tile dispatch there dies by its own
    /// line ([`Self::tiled`]).
    fn layered(&self, sel: &'static str) {
        if self.no_layers {
            unrepresentable(
                sel,
                "a draw in a pass of a rasterization rate map of no layers, which Metal \
                 rasterizes by what earlier maps left, where the runtime's draws nothing",
            );
        }
        if self.varying_rates {
            crate::answer::inexact(sel, crate::answer::Inexact::RateMap, VARYING_RATES_DRAW);
        }
    }

    /// After a scissor setter `sel` of `count` rectangles: in a pass that rasterizes nothing,
    /// as many empty rectangles.
    fn unrasterized(&self, sel: &'static str, count: usize) -> RaiseResult<()> {
        if self.rasterizes {
            return Ok(());
        }
        let empty = vec![EMPTY; count];
        self.run(sel, &mut |e| e.set_scissor_rects(&empty))
    }

    /// A draw `sel` of `shape` with the pipeline state set: one after a nil state dies, as Metal
    /// crashes there ([`Self::pipelined`]), and so does one in a pass that loads and stores a
    /// bfloat colour attachment, empty or not ([`Self::bfloat_drawn`]). So do one of points
    /// Metal sizes none of ([`Self::sized`]) and one the private setters leave no exact form of
    /// ([`Self::rotates`]), unless it is `empty`, of no primitive, patch or instance, which
    /// Metal draws nothing of whatever they left; `None` for one Metal completes without
    /// drawing, which encodes nothing.
    fn draw<K: Copy>(
        &self,
        sel: &'static str,
        (drawn, empty): (Option<(Shape, K)>, bool),
        mut call: impl FnMut(&mut rt::RenderCommandEncoder, K) -> Result<(), rt::Error>,
    ) -> RaiseResult<()> {
        self.pipelined(sel);
        match drawn {
            Some((shape, k)) => {
                self.bfloat_drawn(sel);
                if !empty {
                    self.layered(sel);
                    self.sized(sel, shape);
                    self.issues(sel, None);
                    self.rounds(sel, false);
                    self.encodes(sel, "a draw");
                }
                if !empty && self.rotates(sel, shape) {
                    crate::answer::inexact(
                        sel,
                        crate::answer::Inexact::ProvokingVertex,
                        "a draw the provoking vertex rotates, as it stands",
                    );
                }
                self.binds(sel, &mut |e| call(e, k))
            }
            None => self.encoder.nothing(sel),
        }
    }

    /// Runs `call`, a draw `sel` that binds the pipeline state set, which the runtime announces
    /// a `half` colour output of the state's fragment function at (`HalfOutputPrecision`): the
    /// draw relays it, or in strict mode dies, where an attachment the function writes stores it
    /// past the bound ([`half_apart`]). An execution's commands announce it at the commit,
    /// outside the call.
    fn binds(&self, sel: &'static str, call: Call<'_>) -> RaiseResult<()> {
        let (drawn, heard) = crate::answer::hearing(|| self.run(sel, call));
        if heard.contains(&rt::Degradation::HalfOutputPrecision)
            && let Some(apart) = plain(|| self.marks.lock().half_apart)
        {
            half_output_apart(sel, apart);
        }
        drawn
    }

    /// Relays at `sel`, or dies there in strict mode ([`crate::answer::inexact`]), a draw (of
    /// anything, or of nothing), an execution of render commands or a tile dispatch, in a pass
    /// that loads and stores a bfloat colour attachment ([`bfloat_loaded`]): the tiles the pass
    /// leaves of it are not Metal's.
    fn bfloat_drawn(&self, sel: &'static str) {
        if let Some(attachment) = &self.bfloat_loaded {
            crate::answer::inexact(
                sel,
                crate::answer::Inexact::BfloatLoaded,
                &format!(
                    "a draw in a pass that loads and stores the bfloat {attachment}, whose codes \
                     the runtime moves through half in tiles no draw reaches \
                     (unreached-tiles-rewritten), or keeps in tiles a draw reaches \
                     (extended-range-bits-kept), where Metal's tile store moves those a draw \
                     reaches and keeps the rest; which tiles Metal rewrites is not measured for \
                     a bfloat attachment"
                ),
            );
        }
    }

    /// Dies at a draw `sel` of points, `shape`, under a pipeline state whose points Metal sizes
    /// none of: its vertex function writes no `[[point_size]]`, and its input primitive
    /// topology is not the point class (the module docs).
    fn sized(&self, sel: &'static str, shape: Shape) {
        let points = matches!(
            shape,
            Shape::Primitives {
                topology: Topology::Public(PrimitiveType::Point),
                ..
            }
        );
        let m = *plain(|| self.marks.lock());
        if points && m.pipeline && !m.points_sized {
            unrepresentable(sel, UNSIZED_POINTS);
        }
    }

    /// Dies at `sel`, which runs the encoder's pipeline state, after `setRenderPipelineState:`
    /// of nil: Metal crashes at a draw with no pipeline state, where the runtime would run the
    /// state set before the nil (a draw, direct or indirect, a tile dispatch, and an indirect
    /// command buffer's execution whose commands inherit the encoder's state).
    fn pipelined(&self, sel: &'static str) {
        if *plain(|| self.nil_pipeline.lock()) {
            unrepresentable(
                sel,
                "a draw after a nil pipeline state, which Metal crashes at",
            );
        }
    }

    /// A non-indexed draw `sel` of primitive type 9, `count` vertices of `instances` instances,
    /// which Metal's GPU hangs at: it fails the command buffer with [`HANG`] after about 40 ms,
    /// from this encoder on (none of it runs, its clear included, nor any encoder after it,
    /// where those before it run), its event signals firing (measured on the M3 Max at 3
    /// vertices and 1 instance, 5 runs alike; `render_draw_type9`,
    /// `mutgap2_rdec_primitive_type_9`). It encodes nothing, and the commit fails the command
    /// buffer so ([`crate::gpu::command::CommandBufferObject::fail`]). After
    /// `setRenderPipelineState:` of nil it dies as every draw does ([`Self::pipelined`]); with no
    /// pipeline state set, or of no vertices or instances, which nothing measured, it dies; and
    /// one that passes those dies in a pass of a rate map of no layers or of one whose rates
    /// vary, as a draw does ([`Self::layered`]): Metal's hang was measured in no pass over a rate
    /// map.
    /// The encoder's end holds what comes after it until the time Metal's GPU fails the command
    /// buffer ([`crate::gpu::command::CommandBufferObject::hold_after_hang`]), so the hang is in
    /// flight as long as Metal's, and a command buffer committed on its queue meanwhile, which
    /// Metal runs nothing of and completes without an error, dies at its commit
    /// ([`crate::gpu::command`]).
    fn hangs(&self, sel: &'static str, count: u32, instances: u32) -> RaiseResult<()> {
        self.pipelined(sel);
        if !plain(|| self.marks.lock().pipeline) {
            unrepresentable(
                sel,
                "primitive type 9 with no pipeline state set, which nothing measured",
            );
        }
        if count == 0 || instances == 0 {
            unrepresentable(
                sel,
                &format!(
                    "primitive type 9 of {count} vertices and {instances} instances, which \
                     nothing measured"
                ),
            );
        }
        self.layered(sel);
        self.encoder.nothing(sel)?;
        self.encoder.fail(HANG, true);
        self.hung.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// An indirect command buffer's execution `sel`, of `icb`'s commands `range` (`None`: a range
    /// a buffer holds): it draws, so it dies as a draw does where the private setters leave it no
    /// exact form ([`Self::rotates`]), unless it is `empty`, of no command, and after a nil
    /// pipeline state where its commands inherit the encoder's ([`Self::pipelined`]). One that is
    /// not empty dies where it may write a surface whose GPU writes never reach its memory
    /// ([`Self::issues`]). Where its commands inherit the encoder's state, that state decides
    /// what a draw's would ([`Self::encodes`]); its commit reads its commands where they name
    /// their own, or inherit a state whose points Metal sizes none of
    /// ([`crate::gpu::command::executions`]).
    fn executable(
        &self,
        sel: &'static str,
        (icb, range): (&VulkanIcb, Option<Range<u64>>),
        empty: bool,
    ) -> Option<Execution> {
        if icb.inherits.pipeline {
            self.pipelined(sel);
        }
        self.bfloat_drawn(sel);
        if !empty {
            self.layered(sel);
            self.issues(sel, Some(icb.inherits));
            if icb.inherits.pipeline {
                self.rounds(sel, true);
            }
        }
        let inherits_pipeline = icb.inherits.pipeline;
        if !empty && self.rotates(sel, Shape::Execution { inherits_pipeline }) {
            crate::answer::inexact(
                sel,
                crate::answer::Inexact::ProvokingVertex,
                "an execution the provoking vertex rotates",
            );
        }
        if empty {
            return None;
        }
        let execution = if inherits_pipeline {
            self.encodes(sel, "an execution of commands that inherit");
            let m = *plain(|| self.marks.lock());
            Execution {
                sel,
                icb: icb.clone_ref(),
                range,
                guest_id: None,
                points: if m.pipeline && !m.points_sized {
                    Points::Unsized
                } else {
                    Points::Unasked
                },
                writable: Vec::new(),
            }
        } else {
            Execution {
                sel,
                icb: icb.clone_ref(),
                range,
                guest_id: plain(|| *self.icb_written.lock()),
                points: Points::Own,
                writable: plain(|| {
                    let writable = self.writable_icbs.lock();
                    writable.iter().map(|icb| icb.clone_ref()).collect()
                }),
            }
        };
        execution.asks().then_some(execution)
    }

    /// `execution`, which the runtime took, is checked at its command buffer's commit
    /// ([`crate::gpu::command::CommandBufferObject::checked_at_commit`]).
    fn checked_at_commit(&self, execution: Option<Execution>) {
        if let Some(execution) = execution
            && let Some(cb) = self.encoder.command_buffer()
        {
            cb.checked_at_commit(execution);
        }
    }

    /// Work `sel`, `what`, of the pipeline state the encoder set last: where that state can encode
    /// indirect commands, it dies in an encoder that made resident for writing an indirect command
    /// buffer whose ID is not the runtime's
    /// ([`crate::gpu::encode_compute::encodes_through_guest_id`]), and its functions may write the
    /// records of each buffer the encoder made resident for writing
    /// ([`crate::gpu::command::executions::gpu_writes`]).
    fn encodes(&self, sel: &'static str, what: &str) {
        if !plain(|| self.encodes_icbs.load(Ordering::SeqCst)) {
            return;
        }
        if let Some(id) = plain(|| *self.icb_written.lock()) {
            let what = format!("{what} of a pipeline state that can encode indirect commands");
            crate::gpu::encode_compute::encodes_through_guest_id(sel, &what, id);
        }
        let writable: Vec<VulkanIcb> = plain(|| {
            let writable = self.writable_icbs.lock();
            writable.iter().map(|icb| icb.clone_ref()).collect()
        });
        for icb in &writable {
            crate::gpu::command::executions::gpu_writes(sel, icb);
        }
    }

    /// `useResource` forms made `icbs` resident for writing: the encoder's functions whose state
    /// can encode indirect commands may write their records ([`Self::encodes`]), and the first
    /// whose ID is not the runtime's is the one such a function would reach another buffer's
    /// records through.
    fn writable(&self, icbs: &[VulkanIcb]) {
        plain(|| {
            let mut writable = self.writable_icbs.lock();
            writable.extend(icbs.iter().map(|icb| icb.clone_ref()));
        });
        if let Some(icb) = icbs.iter().find(|icb| !icb.reaches_its_records()) {
            let id = icb.gpu_resource_id();
            plain(|| {
                self.icb_written.lock().get_or_insert(id);
            });
        }
    }

    /// A tile dispatch `sel`: one after a nil pipeline state dies ([`Self::pipelined`]). One in a
    /// pass of a rate map of no layers, where what Metal's does is not measured (only its draws
    /// were, [`Self::layered`]), and one in a pass of a map whose rates vary, where it is not
    /// measured either ([`Regions`]), run, relayed, and die in strict mode
    /// ([`crate::answer::Inexact::RateMap`]). One in a pass that rasterizes nothing, whose
    /// empty scissor rectangles the runtime's tile dispatch does not read, so its tile function
    /// would run over every tile, where what Metal's does there is not measured either.
    fn tiled(&self, sel: &'static str) {
        self.pipelined(sel);
        self.bfloat_drawn(sel);
        self.encodes(sel, "a tile dispatch");
        if self.no_layers {
            crate::answer::inexact(
                sel,
                crate::answer::Inexact::RateMap,
                "a tile dispatch in a pass of a rasterization rate map of no layers, which Metal \
                 is not measured at (only its draws were)",
            );
        }
        if self.varying_rates {
            crate::answer::inexact(sel, crate::answer::Inexact::RateMap, VARYING_RATES_TILES);
        }
        self.issues(sel, None);
        if !self.rasterizes {
            unrepresentable(
                sel,
                "a tile dispatch in a pass of a mutable rate map not yet reset, which Metal is \
                 not measured at",
            );
        }
    }

    /// How a draw `sel` of `shape` draws what Metal draws after the private setters ([`Marks`]):
    /// as it stands (`false`), or over its triangles' vertices rotated so each one's last comes
    /// first (`true`), which only a list or strip of triangles over its vertices in order takes.
    /// What has no exact form is drawn as it stands, relayed ([`crate::answer::inexact`]), but
    /// for the amplification count, which dies; in strict mode each dies:
    /// - at a line width other than 1, lines, triangles and patches filled as lines, and an
    ///   execution: no line drawn here is wider than 1;
    /// - at the last vertex provoking, under a pipeline state that reads a flat input or whose
    ///   vertex function writes a layer or viewport index, what the first vertex's draw has no
    ///   rotation for: lines, Metal's private types, a draw over indices or indirect arguments,
    ///   patches, and an execution; points, and every draw under a state that reads no flat input
    ///   and whose vertex function writes no layer or viewport index, draw as they stand, but
    ///   where the state reads `[[barycentric_coord]]`, which only a list of triangles over its
    ///   vertices or indices keeps, and no rotated draw;
    /// - after `setVertexAmplificationMode:value:` left a count the runtime does not draw, or
    ///   one not measured, every draw and execution.
    fn rotates(&self, sel: &'static str, shape: Shape) -> bool {
        self.rotation(sel, shape, true)
    }

    /// [`Self::rotates`], relaying what it relays only where `relay` holds: a caller that draws
    /// by [`Self::draw`] where the draw is not rotated, which asks again and relays then, asks
    /// so outside strict mode, so a draw relays each kind once.
    fn rotation(&self, sel: &'static str, shape: Shape, relay: bool) -> bool {
        let m = *plain(|| self.marks.lock());
        if m.amplified {
            unrepresentable(
                sel,
                "a draw after setVertexAmplificationMode:value: left an amplification count the \
                 runtime does not draw",
            );
        }
        let (indexed, indirect) = match shape {
            Shape::Primitives {
                indexed, indirect, ..
            } => (indexed, indirect),
            Shape::Patches { indexed } => (indexed, false),
            Shape::Execution { .. } => (true, true),
        };
        let execution = matches!(shape, Shape::Execution { .. });
        if relay
            && m.wide_lines
            && (execution || shape.lines() || (m.line_fill && shape.triangles()))
        {
            crate::answer::inexact(
                sel,
                crate::answer::Inexact::LineWidth,
                "lines at a line width other than 1, which no line drawn here has",
            );
        }
        if !m.last_vertex {
            return false;
        }
        let topology = match shape {
            Shape::Primitives { topology, .. } => Some(topology),
            _ => None,
        };
        if topology == Some(Topology::Public(PrimitiveType::Point)) {
            return false;
        }
        // A list of triangles over its vertices or indices, which Metal's barycentrics order
        // alike at either provoking vertex.
        let list = topology == Some(Topology::Public(PrimitiveType::Triangle)) && !indirect;
        let rotatable = matches!(
            topology,
            Some(Topology::Public(
                PrimitiveType::Triangle | PrimitiveType::TriangleStrip
            ))
        ) && !indexed
            && !indirect;
        let inherited = matches!(
            shape,
            Shape::Execution {
                inherits_pipeline: true
            }
        );
        // The provoking vertex decides what the state draws: a flat input's value, or the layer
        // and viewport a primitive lands in.
        let provoked = m.flat || m.routed;
        if (!provoked && (!execution || inherited)) && (!m.barycentric || list) {
            return false;
        }
        if provoked && rotatable && !m.barycentric {
            return true;
        }
        if !relay {
            return false;
        }
        crate::answer::inexact(
            sel,
            crate::answer::Inexact::ProvokingVertex,
            "a draw with the last vertex provoking (setProvokingVertexMode:) under a pipeline \
             state that reads a flat input or barycentric weights, or whose vertex function \
             writes a layer or viewport index, which this draw's first-vertex form has no exact \
             rotation for",
        );
        false
    }
}

impl RenderEncoder<VulkanPlatform> for VulkanRenderEncoder {
    #[expect(clippy::too_many_lines, reason = "one arm per command")]
    fn encode(&self, _on: OnFault, cmd: RenderCmd<'_, VulkanPlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        for (buffer, bytes) in named_buffers(&cmd) {
            if let Some(buffer) = buffer {
                buffer.used(sel, bytes);
            }
        }
        let encoded = |call: Call<'_>| self.run(sel, call);
        if let Some(bind) = binds(&cmd)
            && self.encoder.command_buffer().is_some()
        {
            self.bound(bind);
        }
        match cmd {
            RenderCmd::DispatchThreadsPerTile { threads_per_tile } => {
                self.tiled(sel);
                encoded(&mut |e| e.dispatch_threads_per_tile(size(threads_per_tile)))
            }
            RenderCmd::DispatchThreadsPerTileInRegion {
                threads_per_tile,
                region: r,
            } => {
                self.tiled(sel);
                encoded(&mut |e| {
                    e.dispatch_threads_per_tile_in_region(size(threads_per_tile), region(r), None)
                })
            }
            RenderCmd::DispatchThreadsPerTileInRegionIndex {
                threads_per_tile,
                region: r,
                render_target_array_index,
            } => {
                self.tiled(sel);
                encoded(&mut |e| {
                    e.dispatch_threads_per_tile_in_region(
                        size(threads_per_tile),
                        region(r),
                        Some(render_target_array_index),
                    )
                })
            }
            RenderCmd::DrawIndexedPatchesIndirect {
                number_of_patch_control_points,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let points = small(sel, number_of_patch_control_points);
                let shape = Shape::Patches { indexed: true };
                self.draw(sel, (Some((shape, ())), false), |e, ()| {
                    e.draw_indexed_patches_indirect(
                        points,
                        patch_index_buffer.map(|b| &b.raw),
                        patch_index_buffer_offset,
                        &control_point_index_buffer.raw,
                        control_point_index_buffer_offset,
                        &indirect_buffer.raw,
                        indirect_buffer_offset,
                    )
                })
            }
            RenderCmd::DrawIndexedPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                instance_count,
                base_instance,
            } => {
                let points = small(sel, number_of_patch_control_points);
                let (start, count) = (small(sel, patch_start), small(sel, patch_count));
                let (instances, base) = (small(sel, instance_count), small(sel, base_instance));
                let shape = Shape::Patches { indexed: true };
                let empty = count == 0 || instances == 0;
                self.draw(sel, (Some((shape, ())), empty), |e, ()| {
                    e.draw_indexed_patches(
                        points,
                        start,
                        count,
                        patch_index_buffer.map(|b| &b.raw),
                        patch_index_buffer_offset,
                        &control_point_index_buffer.raw,
                        control_point_index_buffer_offset,
                        instances,
                        base,
                    )
                })
            }
            RenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type: raw_index_type,
                index_buffer,
                index_buffer_offset,
            } => self.indexed(
                sel,
                (primitive_type, raw_index_type),
                (index_count, 1, 0, 0),
                (index_buffer, index_buffer_offset),
            ),
            RenderCmd::DrawIndexedPrimitivesInstances {
                primitive_type,
                index_count,
                index_type: raw_index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
            } => self.indexed(
                sel,
                (primitive_type, raw_index_type),
                (index_count, instance_count, 0, 0),
                (index_buffer, index_buffer_offset),
            ),
            RenderCmd::DrawIndexedPrimitivesBase {
                primitive_type,
                index_count,
                index_type: raw_index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            } => self.indexed(
                sel,
                (primitive_type, raw_index_type),
                (index_count, instance_count, base_vertex, base_instance),
                (index_buffer, index_buffer_offset),
            ),
            RenderCmd::DrawIndexedPrimitivesIndirect {
                primitive_type,
                index_type: raw_index_type,
                index_buffer,
                index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let kinds = indexed_kinds(sel, primitive_type, raw_index_type);
                let drawn = kinds.map(|(p, i)| (Shape::public(p, true, true), (p, i)));
                self.draw(sel, (drawn, false), |e, (p, i)| {
                    e.draw_indexed_primitives_indirect(
                        p,
                        i,
                        &index_buffer.raw,
                        index_buffer_offset,
                        &indirect_buffer.raw,
                        indirect_buffer_offset,
                    )
                })
            }
            RenderCmd::DrawPatchesIndirect {
                number_of_patch_control_points,
                patch_index_buffer,
                patch_index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let points = small(sel, number_of_patch_control_points);
                let shape = Shape::Patches { indexed: false };
                self.draw(sel, (Some((shape, ())), false), |e, ()| {
                    e.draw_patches_indirect(
                        points,
                        patch_index_buffer.map(|b| &b.raw),
                        patch_index_buffer_offset,
                        &indirect_buffer.raw,
                        indirect_buffer_offset,
                    )
                })
            }
            RenderCmd::DrawPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                instance_count,
                base_instance,
            } => {
                let points = small(sel, number_of_patch_control_points);
                let (start, count) = (small(sel, patch_start), small(sel, patch_count));
                let (instances, base) = (small(sel, instance_count), small(sel, base_instance));
                let shape = Shape::Patches { indexed: false };
                let empty = count == 0 || instances == 0;
                self.draw(sel, (Some((shape, ())), empty), |e, ()| {
                    e.draw_patches(
                        points,
                        start,
                        count,
                        patch_index_buffer.map(|b| &b.raw),
                        patch_index_buffer_offset,
                        instances,
                        base,
                    )
                })
            }
            RenderCmd::DrawPrimitivesIndirect {
                primitive_type,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let p = public_type(sel, primitive_type);
                let drawn = Some((Shape::public(p, false, true), p));
                self.draw(sel, (drawn, false), |e, p| {
                    e.draw_primitives_indirect(p, &indirect_buffer.raw, indirect_buffer_offset)
                })
            }
            RenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
            } => self.primitives(sel, primitive_type, (vertex_start, vertex_count, 1, 0)),
            RenderCmd::DrawPrimitivesInstances {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
            } => self.primitives(
                sel,
                primitive_type,
                (vertex_start, vertex_count, instance_count, 0),
            ),
            RenderCmd::DrawPrimitivesBase {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => self.primitives(
                sel,
                primitive_type,
                (vertex_start, vertex_count, instance_count, base_instance),
            ),
            RenderCmd::EndEncoding => {
                // A store action left `Unknown` stores nothing and resolves nothing, as on Metal
                // (`render_pass_unknown_store`; `mutgap2_rdec_depth_stencil_resolve`: a depth
                // left `Unknown` over a resolve texture resolves nothing): the attachment's
                // resolve target is cleared, as the runtime refuses `DontCare` over one, then it
                // stores `DontCare`. Both are set while the encoder is in its handle, so a thread
                // that stops or dies there leaves it untouched; the end takes it after the one
                // check of the task's end ([`EncoderObject::end`]).
                for (a, ..) in plain(|| std::mem::take(&mut *self.undecided.lock())) {
                    let dont_care = rt::StoreAction::DontCare;
                    self.run(sel, &mut |e| match a {
                        Attachment::Color(i) => e
                            .set_color_resolve_target(None, i)
                            .and_then(|()| e.set_color_store_action(dont_care, i)),
                        Attachment::Depth => e
                            .set_depth_resolve_target(None)
                            .and_then(|()| e.set_depth_store_action(dont_care)),
                        Attachment::Stencil => e
                            .set_stencil_resolve_target(None)
                            .and_then(|()| e.set_stencil_store_action(dont_care)),
                    })?;
                }
                // The runtime decides the pass's resolves as it ends, and announces a colour
                // resolve the device averages and rounds there ([`device_rounded`]).
                let (ended, heard) = crate::answer::hearing(|| {
                    self.encoder
                        .end(sel, rt::RenderCommandEncoder::end_encoding)
                });
                if heard.contains(&rt::Degradation::DeviceRoundedResolve) {
                    device_rounded(sel, &self.pass);
                }
                if ended.is_ok()
                    && self.hung.load(Ordering::SeqCst)
                    && let Some(cb) = self.encoder.command_buffer()
                {
                    cb.hold_after_hang(sel);
                }
                ended
            }
            RenderCmd::ExecuteCommandsIndirect {
                indirect_commandbuffer,
                indirect_range_buffer,
                indirect_buffer_offset,
            } => {
                let checked = self.executable(sel, (indirect_commandbuffer, None), false);
                let progress = self.encoder.progress(sel);
                indirect_commandbuffer.executed(sel, None, progress.as_ref());
                encoded(&mut |e| {
                    e.execute_commands_in_buffer_indirect(
                        &indirect_commandbuffer.raw,
                        &indirect_range_buffer.raw,
                        indirect_buffer_offset,
                    )
                })?;
                self.checked_at_commit(checked);
                if let Some(progress) = progress {
                    progress.executes(sel, indirect_commandbuffer.clone_ref());
                }
                Ok(())
            }
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer,
                execution_range,
            } => {
                let range = range_of(sel, execution_range);
                let checked = self.executable(
                    sel,
                    (indirect_command_buffer, Some(range.clone())),
                    range.is_empty(),
                );
                let progress = self.encoder.progress(sel);
                indirect_command_buffer.executed(sel, Some(range.clone()), progress.as_ref());
                // An empty range runs nothing, as Metal's does under any pipeline state
                // (`render_draw_icb`), where the runtime refuses an inherited state made without
                // `supportIndirectCommandBuffers` first: it is not asked. One past the size is.
                if range.is_empty() && range.end <= indirect_command_buffer.size() {
                    self.encoder.nothing(sel)?;
                } else {
                    encoded(&mut |e| {
                        e.execute_commands_in_buffer(&indirect_command_buffer.raw, range.clone())
                    })?;
                    self.checked_at_commit(checked);
                }
                if let Some(progress) = progress {
                    progress.executes(sel, indirect_command_buffer.clone_ref());
                }
                Ok(())
            }
            RenderCmd::MemoryBarrierWithScope {
                scope,
                after,
                before,
            } => {
                let scope: BarrierScope = metal_value(sel, "barrier scope", scope);
                let after: RenderStages = metal_value(sel, "render stages", after);
                let before: RenderStages = metal_value(sel, "render stages", before);
                // A barrier after the fragment stage and before the vertex stage orders nothing
                // on this GPU, which runs a pass's vertex stages first: it does not split the
                // pass (`mutgap_dec_render_fragment_vertex_barrier`: the later draw's vertices
                // see none of the earlier draw's fragment writes), where the runtime splits it.
                if after == RenderStages::FRAGMENT && before == RenderStages::VERTEX {
                    return self.encoder.nothing(sel);
                }
                encoded(&mut |e| {
                    e.memory_barrier_with_scope(scope, after, before);
                    Ok(())
                })
            }
            RenderCmd::SetAlphaTestReferenceValue { reference_value } => {
                encoded(&mut |e| e.set_alpha_test_reference_value(reference_value))
            }
            RenderCmd::SetBlendColorRed {
                red,
                green,
                blue,
                alpha,
            } => encoded(&mut |e| {
                e.set_blend_color(red, green, blue, alpha);
                Ok(())
            }),
            RenderCmd::SetColorResolveTexture {
                texture,
                slice,
                depth_plane,
                level,
                y_invert: _,
                index,
            } => {
                let at = usize::try_from(index)
                    .ok()
                    .filter(|&i| i < rt::MAX_COLOR_ATTACHMENTS)
                    .unwrap_or_else(|| unrepresentable(sel, &format!("colour attachment {index}")));
                let index = small(sel, index);
                if let Some(d) = self.dropped_resolve(index) {
                    // Relayed, it stays dropped: the texture is left untouched, as plain Metal
                    // leaves the descriptor's.
                    crate::answer::inexact(
                        sel,
                        crate::answer::Inexact::LateResolve,
                        &format!(
                            "a late resolve texture of colour attachment {index}, whose resolve \
                             by {:?} into {} the pass was made without, as plain Metal leaves \
                             that texture untouched; a late resolve texture there is not measured",
                            d.store,
                            d.into.name()
                        ),
                    );
                    return self.encoder.nothing(sel);
                }
                if self.pass.color_attachments[at].resolve_texture.is_none()
                    || self.resolve_unused(Attachment::Color(index))
                {
                    return self.encoder.nothing(sel);
                }
                let at = (slice, depth_plane, level);
                let target = self.late_resolve(sel, Attachment::Color(index), texture, at);
                encoded(&mut |e| e.set_color_resolve_target(target.clone(), index))
            }
            RenderCmd::SetColorStoreAction {
                store_action: raw,
                color_attachment_index,
            } => {
                let (action, index) = (store_action(sel, raw), small(sel, color_attachment_index));
                self.late_store(sel, Attachment::Color(index), action, &mut |e| {
                    e.set_color_store_action(action, index)
                })
            }
            RenderCmd::SetColorStoreActionOptions {
                store_action_options: raw,
                color_attachment_index,
            } => {
                let options = store_action_options(sel, raw);
                let index = small(sel, color_attachment_index);
                self.late_options(sel, Attachment::Color(index), options, &mut |e| {
                    e.set_color_store_action_options(options, index)
                })
            }
            RenderCmd::SetCullMode { cull_mode } => {
                let mode: CullMode = metal_value(sel, "cull mode", cull_mode);
                encoded(&mut |e| {
                    e.set_cull_mode(mode);
                    Ok(())
                })
            }
            RenderCmd::SetDepthBias {
                depth_bias,
                slope_scale,
                clamp,
            } => encoded(&mut |e| e.set_depth_bias(depth_bias, slope_scale, clamp)),
            // The cleared hints change nothing drawn on this host, over loaded values too
            // (`mutgap_dec_render_depth_stencil_cleared`): they encode nothing.
            RenderCmd::SetDepthCleared | RenderCmd::SetStencilCleared => self.encoder.nothing(sel),
            RenderCmd::SetDepthClipModeSpi { mode } => {
                let mode: DepthClipMode = metal_value(sel, "depth clip mode", mode);
                encoded(&mut |e| e.set_depth_clip_mode(mode))
            }
            RenderCmd::SetDepthResolveTexture {
                texture,
                slice,
                depth_plane,
                level,
                y_invert: _,
            } => {
                if self.resolve_unused(Attachment::Depth) {
                    return self.encoder.nothing(sel);
                }
                if texture.is_some() {
                    self.resolves_beside_dropped(sel, Attachment::Depth);
                }
                let at = (slice, depth_plane, level);
                let target = self.late_resolve(sel, Attachment::Depth, texture, at);
                encoded(&mut |e| e.set_depth_resolve_target(target.clone()))
            }
            RenderCmd::SetDepthStencilState {
                depth_stencil_state,
            } => encoded(&mut |e| e.set_depth_stencil_state(depth_stencil_state.map(|s| &**s))),
            RenderCmd::SetDepthStoreAction { store_action: raw } => {
                let action = store_action(sel, raw);
                self.late_store(sel, Attachment::Depth, action, &mut |e| {
                    e.set_depth_store_action(action)
                })
            }
            RenderCmd::SetFragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let index = small(sel, index);
                self.run(sel, &mut |e| {
                    e.set_fragment_buffer(buffer.map(|b| &b.raw), offset, index)
                })
            }
            RenderCmd::SetFragmentBufferOffset { offset, index } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_fragment_buffer_offset(offset, index))
            }
            RenderCmd::SetFragmentBuffers {
                buffers: list,
                offsets,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, offsets) = ranged(sel, range, offsets);
                let list = buffers(list);
                encoded(&mut |e| e.set_fragment_buffers(&list, offsets, at))
            }
            RenderCmd::SetFragmentSamplerStatesLodClamps {
                samplers: list,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, min) = ranged(sel, range, lod_min_clamps);
                let (_, max) = ranged(sel, range, lod_max_clamps);
                let list = samplers(list);
                encoded(&mut |e| e.set_fragment_sampler_states(&list, Some((min, max)), at))
            }
            RenderCmd::SetFragmentSamplerStates {
                samplers: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let list = samplers(list);
                encoded(&mut |e| e.set_fragment_sampler_states(&list, None, at))
            }
            RenderCmd::SetFragmentTexture { texture, index } => {
                plain(|| self.rounded.lock().bind(index, &[texture]));
                let index = small(sel, index);
                let texture = texture.map(|t| t.texels(sel));
                self.run(sel, &mut |e| e.set_fragment_texture(texture, index))
            }
            RenderCmd::SetFragmentTextures {
                textures: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                plain(|| self.rounded.lock().bind(u64::from(at), list));
                let list = textures(sel, list);
                encoded(&mut |e| e.set_fragment_textures(&list, at))
            }
            RenderCmd::SetFrontFacingWinding {
                front_facing_winding,
            } => {
                let winding: Winding = metal_value(sel, "winding", front_facing_winding);
                encoded(&mut |e| {
                    e.set_front_facing_winding(winding);
                    Ok(())
                })
            }
            // A width of 1 is every line's here; another is marked, and the draws it widens
            // die ([`Self::rotates`]).
            RenderCmd::SetLineWidth { line_width } => {
                self.encoder.nothing(sel)?;
                #[expect(clippy::float_cmp, reason = "Metal's default width, exactly")]
                let wide = line_width != 1.0;
                plain(|| self.marks.lock().wide_lines = wide);
                Ok(())
            }
            // Metal draws alike at every point size on this host, and a vertex function's
            // `[[point_size]]` decides a point's: it encodes nothing.
            RenderCmd::SetPointSize { .. } => self.encoder.nothing(sel),
            // Metal's indexed strips restart at the index set, until the encoder ends, and
            // nothing else changes (the module docs), as the runtime's do.
            RenderCmd::SetPrimitiveRestartEnabled { enabled, index } => {
                plain(|| self.marks.lock().restart = Some((enabled, index)));
                encoded(&mut |e| {
                    e.set_primitive_restart_enabled(enabled, index);
                    Ok(())
                })
            }
            // Mode 0 is Metal's default and the runtime's, the first vertex provoking
            // (`mutgap2_rdec_draw_state`); every other mode is the last vertex, which the draws
            // after it take as they stand, rotated, or die ([`Self::rotates`]).
            RenderCmd::SetProvokingVertexMode { mode } => {
                self.encoder.nothing(sel)?;
                plain(|| self.marks.lock().last_vertex = mode != 0);
                Ok(())
            }
            RenderCmd::SetRenderPipelineState { pipeline_state } => match pipeline_state {
                None => {
                    self.encoder.nothing(sel)?;
                    plain(|| *self.nil_pipeline.lock() = true);
                    self.encoder.pipeline.store(0, Ordering::Relaxed);
                    self.encodes_icbs.store(false, Ordering::SeqCst);
                    Ok(())
                }
                Some(state) => {
                    self.run(sel, &mut |e| e.set_render_pipeline_state(&state.raw))?;
                    plain(|| *self.nil_pipeline.lock() = false);
                    plain(|| self.reach.lock().set(state));
                    plain(|| self.rounded.lock().set(state));
                    self.encoder
                        .pipeline
                        .store(state.unique_identifier(), Ordering::Relaxed);
                    let flat = plain(|| state.raw.has_flat_fragment_inputs());
                    let barycentric = plain(|| state.raw.reads_barycentric_coord());
                    let routed = plain(|| {
                        state.raw.writes_render_target_array_index()
                            || state.raw.writes_viewport_array_index()
                    });
                    let points_sized = plain(|| state.raw.writes_point_size())
                        || state.input_primitive_topology == PrimitiveTopologyClass::Point;
                    let encodes = plain(|| state.raw.can_encode_indirect_commands());
                    self.encodes_icbs.store(encodes, Ordering::SeqCst);
                    let apart = match state.reflection.descriptor() {
                        Some(Reflected::Render(RenderDescriptor::Render(d))) => {
                            half_apart(d, &plain(|| state.raw.fragment_color_outputs()))
                        }
                        _ => None,
                    };
                    plain(|| {
                        let mut m = self.marks.lock();
                        m.flat = flat;
                        m.barycentric = barycentric;
                        m.routed = routed;
                        m.pipeline = true;
                        m.points_sized = points_sized;
                        m.half_apart = apart;
                    });
                    Ok(())
                }
            },
            RenderCmd::SetScissorRect { rect } => {
                let rect = scissor(rect);
                encoded(&mut |e| e.set_scissor_rect(rect))?;
                self.unrasterized(sel, 1)
            }
            RenderCmd::SetScissorRects { scissor_rects } => {
                let rects: Vec<_> = scissor_rects.iter().copied().map(scissor).collect();
                encoded(&mut |e| e.set_scissor_rects(&rects))?;
                self.unrasterized(sel, rects.len())
            }
            RenderCmd::SetStencilFrontReferenceValue {
                front_reference_value,
                back_reference_value,
            } => encoded(&mut |e| {
                e.set_stencil_reference_values(front_reference_value, back_reference_value);
                Ok(())
            }),
            RenderCmd::SetStencilResolveTexture {
                texture,
                slice,
                depth_plane,
                level,
                y_invert: _,
            } => {
                if self.resolve_unused(Attachment::Stencil) {
                    return self.encoder.nothing(sel);
                }
                if texture.is_some() {
                    self.resolves_beside_dropped(sel, Attachment::Stencil);
                }
                let at = (slice, depth_plane, level);
                let target = self.late_resolve(sel, Attachment::Stencil, texture, at);
                encoded(&mut |e| e.set_stencil_resolve_target(target.clone()))
            }
            RenderCmd::SetStencilStoreAction { store_action: raw } => {
                let action = store_action(sel, raw);
                self.late_store(sel, Attachment::Stencil, action, &mut |e| {
                    e.set_stencil_store_action(action)
                })
            }
            RenderCmd::SetStencilStoreActionOptions {
                store_action_options: raw,
            } => {
                let options = store_action_options(sel, raw);
                self.late_options(sel, Attachment::Stencil, options, &mut |e| {
                    e.set_stencil_store_action_options(options)
                })
            }
            RenderCmd::SetTessellationFactorBuffer {
                buffer,
                offset,
                instance_stride,
            } => encoded(&mut |e| {
                e.set_tessellation_factor_buffer(buffer.map(|b| &b.raw), offset, instance_stride)
            }),
            RenderCmd::SetTessellationFactorScale { scale } => encoded(&mut |e| {
                e.set_tessellation_factor_scale(scale);
                Ok(())
            }),
            RenderCmd::SetThreadgroupMemoryLength {
                length,
                offset,
                index,
            } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_threadgroup_memory_length(length, offset, index))
            }
            RenderCmd::SetTileBufferOffset { offset, index } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_tile_buffer_offset(offset, index))
            }
            RenderCmd::SetTileBuffers {
                buffers: list,
                offsets,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, offsets) = ranged(sel, range, offsets);
                let list = buffers(list);
                encoded(&mut |e| e.set_tile_buffers(&list, offsets, at))
            }
            RenderCmd::SetTileSamplerStatesLodClamps {
                samplers: list,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, min) = ranged(sel, range, lod_min_clamps);
                let (_, max) = ranged(sel, range, lod_max_clamps);
                let list = samplers(list);
                encoded(&mut |e| e.set_tile_sampler_states(&list, Some((min, max)), at))
            }
            RenderCmd::SetTileSamplerStates {
                samplers: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let list = samplers(list);
                encoded(&mut |e| e.set_tile_sampler_states(&list, None, at))
            }
            RenderCmd::SetTileTextures {
                textures: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let list = textures(sel, list);
                encoded(&mut |e| e.set_tile_textures(&list, at))
            }
            RenderCmd::SetTriangleFillMode { fill_mode } => {
                let mode: TriangleFillMode = metal_value(sel, "triangle fill mode", fill_mode);
                encoded(&mut |e| e.set_triangle_fill_mode(mode))?;
                plain(|| self.marks.lock().line_fill = mode == TriangleFillMode::Lines);
                Ok(())
            }
            RenderCmd::SetVertexAmplificationCount { view_mappings } => {
                let count = u64::try_from(view_mappings.len()).unwrap_or_else(|_| {
                    unrepresentable(sel, &format!("{} view mappings", view_mappings.len()))
                });
                let mappings: Vec<_> = view_mappings
                    .iter()
                    .map(|m| {
                        rt::VertexAmplificationViewMapping::new(
                            m.viewport_array_index_offset,
                            m.render_target_array_index_offset,
                        )
                    })
                    .collect();
                let mappings = (!mappings.is_empty()).then_some(mappings.as_slice());
                encoded(&mut |e| e.set_vertex_amplification_count(count, mappings))?;
                plain(|| {
                    let mut m = self.marks.lock();
                    m.amplification = Some(count);
                    m.amplified = false;
                });
                Ok(())
            }
            // The count it sets, which keeps the view mappings set before, is the runtime's
            // where it equals the runtime's (the module docs).
            RenderCmd::SetVertexAmplificationMode { mode, value } => {
                plain(|| {
                    let mut m = self.marks.lock();
                    let runtime = m.amplification.unwrap_or(1);
                    m.amplified = amplification_count(mode, value) != Some(runtime);
                });
                self.encoder.nothing(sel)
            }
            RenderCmd::SetVertexBuffer {
                buffer,
                offset,
                index,
            } => {
                let index = small(sel, index);
                self.run(sel, &mut |e| {
                    e.set_vertex_buffer(buffer.map(|b| &b.raw), offset, index)
                })
            }
            RenderCmd::SetVertexBufferOffset { offset, index } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_vertex_buffer_offset(offset, index))
            }
            RenderCmd::SetVertexBufferOffsetAttributeStride {
                offset,
                stride,
                index,
            } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_vertex_buffer_offset_with_stride(offset, stride, index))
            }
            RenderCmd::SetVertexBuffersAttributeStrides {
                buffers: list,
                offsets,
                strides,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, offsets) = ranged(sel, range, offsets);
                let (_, strides) = ranged(sel, range, strides);
                let list = buffers(list);
                encoded(&mut |e| e.set_vertex_buffers_with_strides(&list, offsets, strides, at))
            }
            RenderCmd::SetVertexBuffers {
                buffers: list,
                offsets,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, offsets) = ranged(sel, range, offsets);
                let list = buffers(list);
                encoded(&mut |e| e.set_vertex_buffers(&list, offsets, at))
            }
            RenderCmd::SetVertexSamplerStatesLodClamps {
                samplers: list,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, min) = ranged(sel, range, lod_min_clamps);
                let (_, max) = ranged(sel, range, lod_max_clamps);
                let list = samplers(list);
                encoded(&mut |e| e.set_vertex_sampler_states(&list, Some((min, max)), at))
            }
            RenderCmd::SetVertexSamplerStates {
                samplers: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let list = samplers(list);
                encoded(&mut |e| e.set_vertex_sampler_states(&list, None, at))
            }
            RenderCmd::SetVertexTextures {
                textures: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let list = textures(sel, list);
                encoded(&mut |e| e.set_vertex_textures(&list, at))
            }
            RenderCmd::SetViewport { viewport: v } => {
                let v = viewport(v);
                self.run(sel, &mut |e| e.set_viewport(v))
            }
            RenderCmd::SetViewportTransformEnabled { enabled } => {
                encoded(&mut |e| e.set_viewport_transform_enabled(enabled))
            }
            RenderCmd::SetViewports { viewports } => {
                let list: Vec<_> = viewports.iter().copied().map(viewport).collect();
                encoded(&mut |e| e.set_viewports(&list))
            }
            RenderCmd::SetVisibilityResultMode { mode, offset } => {
                let mode = rt::VisibilityResultMode::from_raw(mode).unwrap_or_else(|_| {
                    unrepresentable(sel, &format!("visibility result mode {mode}"))
                });
                if mode != rt::VisibilityResultMode::Disabled
                    && plain(|| self.reach.lock().visibility.holds())
                {
                    crate::gpu::writes::kept_write(
                        sel,
                        "a visibility result mode whose result the pass writes into a buffer over",
                    );
                }
                encoded(&mut |e| e.set_visibility_result_mode(mode, offset))
            }
            RenderCmd::UpdateFence { fence, stages } => {
                let stages: RenderStages = metal_value(sel, "render stages", stages);
                encoded(&mut |e| e.update_fence(fence, stages))
            }
            RenderCmd::UseHeaps { heaps: list } => {
                let list = heaps(sel, list);
                encoded(&mut |e| e.use_heaps(&list, VERTEX_AND_FRAGMENT))
            }
            RenderCmd::UseHeapsStages {
                heaps: list,
                stages,
            } => {
                let list = heaps(sel, list);
                let stages: RenderStages = metal_value(sel, "render stages", stages);
                encoded(&mut |e| e.use_heaps(&list, stages))
            }
            RenderCmd::UseResources {
                resources: list,
                usage,
            } => {
                let some = present(list);
                let usage: ResourceUsage = metal_value(sel, "resource usage", usage);
                let icbs = crate::gpu::icb::made_resident(list);
                if !icbs.is_empty() {
                    // A function of the encoder's may copy any of their commands.
                    let progress = self.encoder.progress(sel);
                    for icb in &icbs {
                        icb.resident(sel, progress.as_ref());
                    }
                }
                encoded(&mut |e| e.use_resources(&some, usage, VERTEX_AND_FRAGMENT))?;
                if usage.contains(ResourceUsage::WRITE) {
                    // A function of the encoder's may write their records.
                    for icb in &icbs {
                        icb.may_be_written(self.encoder.progress(sel));
                    }
                    self.writable(&icbs);
                }
                if !icbs.is_empty() {
                    self.encoder.hold(sel, icbs);
                }
                Ok(())
            }
            RenderCmd::UseResourcesStages {
                resources: list,
                usage,
                stages,
            } => {
                let some = resources(sel, list);
                let usage: ResourceUsage = metal_value(sel, "resource usage", usage);
                let stages: RenderStages = metal_value(sel, "render stages", stages);
                let icbs = crate::gpu::icb::made_resident(list);
                if !icbs.is_empty() {
                    // A function of the encoder's may copy any of their commands.
                    let progress = self.encoder.progress(sel);
                    for icb in &icbs {
                        icb.resident(sel, progress.as_ref());
                    }
                }
                encoded(&mut |e| e.use_resources(&some, usage, stages))?;
                if usage.contains(ResourceUsage::WRITE) {
                    // A function of the encoder's may write their records.
                    for icb in &icbs {
                        icb.may_be_written(self.encoder.progress(sel));
                    }
                    self.writable(&icbs);
                }
                if !icbs.is_empty() {
                    self.encoder.hold(sel, icbs);
                }
                Ok(())
            }
            RenderCmd::WaitForFence { fence, stages } => {
                let stages: RenderStages = metal_value(sel, "render stages", stages);
                // A wait for a fence this encoder updated before completes on Metal
                // (`render_objects_fence`'s update then wait), as the runtime's waits for the
                // update before the encoder.
                encoded(&mut |e| e.wait_for_fence(fence, stages))
            }
        }
    }

    fn tile_height(&self) -> u64 {
        let sel = "tileHeight";
        get(sel, || self.encoder.with(sel, |e| e.tile_height()))
    }

    fn tile_width(&self) -> u64 {
        let sel = "tileWidth";
        get(sel, || self.encoder.with(sel, |e| e.tile_width()))
    }
}

impl VulkanRenderEncoder {
    /// The descriptor's store action and options of `attachment`, as core set them (a dropped
    /// resolve's action too, [`Dropped`]); `None` for one without a texture, or past the eighth
    /// colour attachment.
    fn stored(&self, attachment: Attachment) -> Option<(rt::StoreAction, rt::StoreActionOptions)> {
        let p = &self.pass;
        let (texture, action, options) = match attachment {
            Attachment::Color(i) => {
                let c = p.color_attachments.get(usize::try_from(i).ok()?)?;
                let action = self.dropped_resolve(i).map_or(c.store_action, |d| d.store);
                (&c.texture, action, c.store_action_options)
            }
            Attachment::Depth => {
                let a = &p.depth_attachment;
                (&a.texture, a.store_action, a.store_action_options)
            }
            Attachment::Stencil => {
                let a = &p.stencil_attachment;
                (&a.texture, a.store_action, a.store_action_options)
            }
        };
        texture.as_ref().map(|_| (action, options))
    }

    /// The colour attachment at `index` whose resolve [`untouched_resolves`] dropped, if it is
    /// one.
    fn dropped_resolve(&self, index: u32) -> Option<Dropped> {
        self.dropped.iter().find(|d| d.index == index).copied()
    }

    /// Marks each undecided attachment ([`Self::late_store`]) with why a resolve into its resolve
    /// texture, as `sources` holds the pass descriptor's, is never sent ([`Unsent`]).
    fn undecided_dropped(&self, sources: &Sources) {
        plain(|| {
            for (a, unsent) in self.undecided.lock().iter_mut() {
                let resolve = match *a {
                    Attachment::Color(i) => usize::try_from(i)
                        .ok()
                        .and_then(|i| sources.color.get(i))
                        .map(|[_, r]| r),
                    Attachment::Depth => Some(&sources.depth[1]),
                    Attachment::Stencil => Some(&sources.stencil[1]),
                };
                *unsent = resolve.and_then(Source::texture).and_then(Unsent::of);
            }
        });
    }

    /// Relays at `sel`, or dies there in strict mode ([`crate::answer::Inexact::LateResolve`]), a
    /// late resolve texture of the depth or stencil `attachment`, whose store
    /// action resolves or is left `Unknown` ([`Self::resolve_unused`]), in a pass made without a
    /// colour resolve plain Metal leaves untouched ([`Dropped`]): the attachment may then resolve,
    /// a shape the drop is not measured beside, at which the pass's creation dies where the
    /// descriptor names the resolve texture ([`unmeasured_shape`]). One of a pass without that
    /// attachment goes on, and so does one whose action a late setter decided and that does not
    /// resolve ([`Self::late_store`]): nothing resolves beside the drop, so the setter goes to
    /// the runtime, as in a pass without a dropped resolve (which refuses a resolve texture under
    /// an action that does not resolve, `ResolveStoreAction`, and that refusal dies there).
    fn resolves_beside_dropped(&self, sel: &'static str, attachment: Attachment) {
        let Some(d) = self.dropped.first() else {
            return;
        };
        if self.stored(attachment).is_none() {
            return;
        }
        let late = plain(|| {
            let late = self.late_stores.lock();
            late.iter()
                .find(|&&(a, _)| a == attachment)
                .map(|&(_, action)| action)
        });
        if late.is_some_and(|action| !may_resolve(action)) {
            return;
        }
        let name = match attachment {
            Attachment::Depth => "depth",
            _ => "stencil",
        };
        crate::answer::inexact(
            sel,
            crate::answer::Inexact::LateResolve,
            &format!(
                "a late {name} resolve texture in a pass made without the resolve of colour \
                 attachment {} by {:?} into {}, as plain Metal leaves that texture untouched: \
                 the pass's depth or stencil attachment resolves, a shape that drop is not \
                 measured beside",
                d.index,
                d.store,
                d.into.name()
            ),
        );
    }

    /// Whether `attachment` resolves nowhere whatever its resolve texture: the descriptor knew
    /// its store action, and the action does not resolve, so Metal ignores a late resolve setter
    /// there as it ignores the descriptor's resolve texture ([`without_unused_resolves`]).
    fn resolve_unused(&self, attachment: Attachment) -> bool {
        matches!(self.stored(attachment), Some((action, _))
            if action != rt::StoreAction::Unknown && !may_resolve(action))
    }

    /// A late store action `action` of `attachment`, set by `call`. Where the descriptor knew
    /// the action, or named no texture, Metal goes on (`render_pass_eight`): the same action
    /// again, and any action of an attachment without a texture, encode nothing; another action
    /// over a known one dies, as what Metal does with it is not measured. The known action is the
    /// descriptor's as core set it, a dropped resolve's too ([`Dropped`]): its resolving action
    /// again encodes nothing, as plain Metal resolves nothing into the texture
    /// (`resolves_into_other_formats_are_metals_or_die_named`). A colour index past the eighth is
    /// the runtime's to refuse. One that resolves an undecided attachment into a resolve texture
    /// whose GPU writes never reach its memory dies ([`crate::os::dropped_write`]).
    fn late_store(
        &self,
        sel: &'static str,
        attachment: Attachment,
        action: rt::StoreAction,
        call: Call<'_>,
    ) -> RaiseResult<()> {
        if past_the_eighth(attachment) {
            return self.run(sel, call);
        }
        match self.stored(attachment) {
            Some((rt::StoreAction::Unknown, _)) => {
                let named = plain(|| {
                    let undecided = self.undecided.lock();
                    undecided
                        .iter()
                        .find(|(a, _)| *a == attachment)
                        .and_then(|&(_, unsent)| unsent)
                });
                if let Some(unsent) = named
                    && action.resolves()
                {
                    unsent.dies(sel, "a store action that resolves into a texture");
                }
                self.run(sel, call)?;
                plain(|| {
                    self.undecided.lock().retain(|&(a, _)| a != attachment);
                    let mut late = self.late_stores.lock();
                    late.retain(|&(a, _)| a != attachment);
                    late.push((attachment, action));
                });
                Ok(())
            }
            Some((known, _)) if known != action => unrepresentable(
                sel,
                &format!("a late store action {action:?} over the known {known:?}"),
            ),
            _ => self.encoder.nothing(sel),
        }
    }

    /// A late resolve target of `texture` at `at`, its slice, depth plane and level, for
    /// `attachment`, which `sel` sets. Where the attachment's store action is undecided, whether
    /// the texture's GPU writes never reach its memory is kept for the late store action that
    /// decides whether it resolves ([`Self::late_store`]); elsewhere the action resolves, so one
    /// whose GPU writes never reach its memory dies ([`crate::os::dropped_write`]).
    fn late_resolve(
        &self,
        sel: &'static str,
        attachment: Attachment,
        texture: Option<&VulkanTexture>,
        at: (u64, u64, u64),
    ) -> Option<rt::ResolveTarget> {
        let unsent = texture.and_then(Unsent::of);
        let kept = plain(|| {
            let mut undecided = self.undecided.lock();
            let named = undecided.iter_mut().find(|(a, _)| *a == attachment);
            named.map(|(_, never)| *never = unsent)
        });
        if kept.is_none()
            && let Some(unsent) = unsent
        {
            unsent.dies(sel, "a resolve into a texture");
        }
        match kept {
            Some(()) => resolve_target(texture.map(|t| &t.raw), at),
            None => resolve_target(texture.map(|t| t.texels(sel)), at),
        }
    }

    /// Late store action options `options` of `attachment`, set by `call`: as
    /// [`late_store`](Self::late_store), the descriptor's own options again encoding nothing.
    fn late_options(
        &self,
        sel: &'static str,
        attachment: Attachment,
        options: rt::StoreActionOptions,
        call: Call<'_>,
    ) -> RaiseResult<()> {
        if past_the_eighth(attachment) {
            return self.run(sel, call);
        }
        match self.stored(attachment) {
            Some((rt::StoreAction::Unknown, _)) => self.run(sel, call),
            Some((_, known)) if known != options => unrepresentable(
                sel,
                &format!("late store action options {options:?} over the known {known:?}"),
            ),
            _ => self.encoder.nothing(sel),
        }
    }

    /// A non-indexed draw of a raw primitive type: `(start, count, instances, base instance)`; one
    /// of no instances encodes nothing, as Metal draws nothing, before the runtime's checks of its
    /// state (the module docs). A private type draws as its public one over the indices of its
    /// vertices ([`Private::indices`]), and a list or strip of triangles the provoking vertex
    /// rotates ([`Self::rotates`]) as a list over its triangles' vertices last first
    /// ([`last_first`]), each in a buffer of its own that the command buffer keeps until it
    /// completes; one of too few vertices draws nothing, as on Metal. The indices are relative to
    /// `start`, the indexed draw's base vertex, so `[[base_vertex]]` reads `start` and
    /// `[[vertex_id]]` each vertex's own, as Metal's non-indexed draws give them (the module
    /// docs); a `start` past `i32::MAX`, which the runtime's base vertex cannot hold and Metal
    /// draws from, dies. Such a draw restarts at the all-ones index alone after
    /// `setPrimitiveRestartEnabled:index:` too, as a line loop's line strip has no index where
    /// Metal's loop restarts.
    fn primitives(
        &self,
        sel: &'static str,
        primitive_type: u64,
        (start, count, instances, base): (u64, u64, u64, u64),
    ) -> RaiseResult<()> {
        let (start, count) = (small(sel, start), small(sel, count));
        let (instances, base) = (small(sel, instances), small(sel, base));
        if primitive_type == HANGING_TYPE {
            return self.hangs(sel, count, instances);
        }
        let topology = topology(sel, primitive_type);
        if instances == 0 {
            return self.draw(sel, (None::<(Shape, ())>, true), |_, ()| Ok(()));
        }
        let shape = Shape::Primitives {
            topology,
            indexed: false,
            indirect: false,
        };
        let (public, indices) = match topology {
            Topology::Public(p) => {
                self.pipelined(sel);
                let empty = draws_none(p, count);
                if empty || !self.rotation(sel, shape, crate::answer::strict()) {
                    return self.draw(sel, (Some((shape, p)), empty), |e, p| {
                        e.draw_primitives(p, start, count, instances, base)
                    });
                }
                (PrimitiveType::Triangle, last_first(sel, p, start, count))
            }
            Topology::Private(t) => (t.public(), t.indices(sel, start, count)),
        };
        if indices.is_empty() {
            return self.draw(sel, (None::<(Shape, ())>, true), |_, ()| Ok(()));
        }
        let base_vertex = i32::try_from(start).unwrap_or_else(|_| {
            unrepresentable(sel, &format!("vertex start {start} as a base vertex"))
        });
        let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_ne_bytes()).collect();
        let options = rt::ResourceOptions::new(
            rt::StorageMode::Shared,
            rt::CpuCacheMode::WriteCombined,
            rt::HazardTrackingMode::Default,
        );
        let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        let contents = rt::BufferContents::Copy(&bytes);
        let buffer = plain(|| crate::gpu::own_buffer(&self.device, length, options, contents))
            .unwrap_or_else(|e| refused(sel, &e));
        let n = small(sel, u64::try_from(indices.len()).unwrap_or(u64::MAX));
        self.pipelined(sel);
        self.issues(sel, None);
        self.rounds(sel, false);
        if matches!(topology, Topology::Private(_)) && self.rotates(sel, shape) {
            crate::answer::inexact(
                sel,
                crate::answer::Inexact::ProvokingVertex,
                "a private type the provoking vertex rotates",
            );
        }
        // These indices are this platform's, not the guest's: the draw restarts at the
        // all-ones index alone, which none of them is, and the guest's index holds again
        // after it.
        let restart = plain(|| self.marks.lock().restart);
        self.binds(sel, &mut |e| {
            let i = IndexType::UInt32;
            if restart.is_some() {
                e.set_primitive_restart_enabled(false, u64::from(u32::MAX));
            }
            let drawn =
                e.draw_indexed_primitives(public, n, i, &buffer, 0, instances, base_vertex, base);
            if let Some((enabled, index)) = restart {
                e.set_primitive_restart_enabled(enabled, index);
            }
            drawn
        })
    }

    /// An indexed draw of a raw primitive and index type: `(count, instances, base vertex, base
    /// instance)` from the index buffer at its offset; one of no instances encodes nothing, as
    /// Metal draws nothing, before the runtime's checks of its state (the module docs).
    fn indexed(
        &self,
        sel: &'static str,
        (primitive_type, raw_index_type): (u64, u64),
        (count, instances, base_vertex, base_instance): (u64, u64, i64, u64),
        (buffer, offset): (&VulkanBuffer, u64),
    ) -> RaiseResult<()> {
        let kinds = indexed_kinds(sel, primitive_type, raw_index_type);
        let (count, instances) = (small(sel, count), small(sel, instances));
        let kinds = kinds.filter(|_| instances != 0);
        let base_vertex = i32::try_from(base_vertex).unwrap_or_else(|_| {
            unrepresentable(sel, &format!("base vertex {base_vertex} as an i32"))
        });
        let base_instance = small(sel, base_instance);
        let empty = kinds.is_some_and(|(p, _)| draws_none(p, count));
        let drawn = kinds.map(|(p, i)| (Shape::public(p, true, false), (p, i)));
        self.draw(sel, (drawn, empty), |e, (p, i)| {
            e.draw_indexed_primitives(
                p,
                count,
                i,
                &buffer.raw,
                offset,
                instances,
                base_vertex,
                base_instance,
            )
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use reims_vgpu_core::display::pipeline::{DisplayPipelineCache, PipelineKey};
    use reims_vgpu_core::platform::blit::ResourceRef;
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
    use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::resource::{Buffer, Texture};
    use reims_vgpu_core::platform::types::{
        ClearColor, MtlOrigin, MtlRegion, MtlScissorRect, MtlSize, NsRange,
    };
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, TextureDesc};
    use reims_vgpu_core::platform::{Handle, HostRange, Platform};
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::VulkanTexture;
    use crate::platform::tests::platform;

    const BGRA8: u64 = 80;
    const RGBA16F: u64 = 115;
    /// `MTLTextureUsageShaderRead` and `RenderTarget`.
    const READ: u64 = 1;
    const TARGET: u64 = 4;
    /// `MTLLoadActionClear`, `MTLStoreActionStore`, `MTLPrimitiveTypeTriangleStrip`.
    const CLEAR: u64 = 2;
    const STORE: u64 = 1;
    const STRIP: u64 = 4;

    /// The native-endian bytes of `floats`.
    fn bytes(floats: &[f32]) -> Vec<u8> {
        floats.iter().flat_map(|f| f.to_ne_bytes()).collect()
    }

    /// A buffer of `length` bytes of `options`, holding `contents` from its start.
    fn buffer<P: Platform>(d: &P::Device, length: u64, options: u64, contents: &[u8]) -> P::Buffer {
        let b = d
            .new_buffer_of_length(OnFault::Die, length, options)
            .unwrap()
            .unwrap();
        // SAFETY: a CPU-visible buffer of at least `contents.len()` bytes, which no command uses
        // yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents.as_ptr(),
                b.contents().as_ptr::<u8>(),
                contents.len(),
            )
        };
        b
    }

    /// A 2D texture of `format`, `usage`, `width` × `height`, holding `texels` (rows of
    /// `bytes_per_row`).
    fn texture<P: Platform>(
        p: &P,
        d: &P::Device,
        format: u64,
        usage: u64,
        (width, height): (u64, u64),
        texels: Option<(&[u8], u64)>,
    ) -> P::Texture {
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: format,
            width,
            height,
            usage,
            ..TextureDesc::default()
        });
        let t = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        if let Some((texels, row)) = texels {
            let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(width, height, 1));
            t.replace_region(region, 0, 0, HostRange::of_slice(texels), row, 0)
                .unwrap();
        }
        t
    }

    /// Sixteen vertices at scattered positions inside clip space, as the present pass's
    /// position (float4) and texture-coordinate (float2) buffers hold them.
    fn scattered() -> (Vec<f32>, Vec<f32>) {
        let mut state = 7919u32;
        let mut next = move || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            f32::from(u16::try_from((state >> 8) & 0xffff).unwrap()) / 65536.0
        };
        let (mut pos, mut tex) = (Vec::new(), Vec::new());
        for _ in 0..16 {
            pos.extend([next() * 1.9 - 0.95, next() * 1.9 - 0.95, 0.0, 1.0]);
            tex.extend([next(), next()]);
        }
        (pos, tex)
    }
    /// `MTLIndexTypeUInt16`.
    const U16: u64 = 0;

    /// The clear values `half` and bfloat both hold exactly, which a bfloat attachment's clear
    /// stores as Metal's, and the ones they do not, which die
    #[test]
    fn bfloat_clears_take_what_half_and_bfloat_both_hold() {
        for v in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -2.5,
            0.25,
            0.75,
            3.0,
            0.375,
            61440.0,
            2f64.powi(-24),
            -2f64.powi(-20),
            2f64.powi(-14),
            32768.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(half_and_bfloat_hold(v), "{v:e}");
        }
        for v in [
            -3.0e38,
            1.0e-40,
            70_000.0,
            1.003_906_369_209_289_6,
            1.003_906_25,
            1.0e-3,
            65536.0,
            2f64.powi(-25),
            0.1,
            f64::NAN,
            -f64::NAN,
        ] {
            assert!(!half_and_bfloat_hold(v), "{v:e}");
        }
    }

    /// A clear's NaN is found in each channel its format has: any NaN at 4 samples, of either
    /// sign; at 1, any NaN in an extended-range format, and in a half, float or packed float one
    /// a negative, signalling or payload NaN, and not the plain quiet one. In no other channel or
    /// format
    #[test]
    fn clears_find_the_nans_the_runtime_does_not_keep() {
        let (nan, neg) = (f64::NAN, -f64::NAN);
        let payload = f64::from_bits(0x7ff8_0000_2000_0000);
        let low = f64::from_bits(0x7ff8_0000_0000_0001);
        let signalling = f64::from_bits(0x7ff0_0000_2000_0000);
        let cases = [
            (
                PixelFormat::RGBA16Float,
                [nan, 1.0, 0.5, nan],
                Some(0),
                None,
            ),
            (
                PixelFormat::RGBA16Float,
                [0.0, 1.0, 0.5, neg],
                Some(3),
                Some(3),
            ),
            (
                PixelFormat::RGBA16Float,
                [nan, payload, 0.5, 1.0],
                Some(0),
                Some(1),
            ),
            (
                PixelFormat::RGBA16Float,
                [nan, 1.0, signalling, 1.0],
                Some(0),
                Some(2),
            ),
            (
                PixelFormat::RGBA16Float,
                [0.0, low, 0.5, 1.0],
                Some(1),
                Some(1),
            ),
            (PixelFormat::RGBA16Float, [0.25, 1.0, 0.5, -2.0], None, None),
            (PixelFormat::R16Float, [0.5, neg, neg, neg], None, None),
            (
                PixelFormat::R16Float,
                [neg, 0.0, 0.0, 0.0],
                Some(0),
                Some(0),
            ),
            (PixelFormat::RG16Float, [0.5, nan, 0.0, 0.0], Some(1), None),
            (PixelFormat::RG16Float, [0.5, 0.5, neg, neg], None, None),
            (PixelFormat::R32Float, [nan, 0.0, 0.0, 0.0], Some(0), None),
            (
                PixelFormat::RG32Float,
                [0.0, neg, 0.0, 0.0],
                Some(1),
                Some(1),
            ),
            (
                PixelFormat::RGBA32Float,
                [0.0, 0.0, nan, payload],
                Some(2),
                Some(3),
            ),
            (
                PixelFormat::RG11B10Float,
                [0.0, 0.0, neg, 0.0],
                Some(2),
                Some(2),
            ),
            (
                PixelFormat::RG11B10Float,
                [0.0, nan, 0.0, 0.0],
                Some(1),
                None,
            ),
            (PixelFormat::RG11B10Float, [0.0, 0.0, 0.0, neg], None, None),
            (
                PixelFormat::RGB9E5Float,
                [nan, 0.0, 0.0, 0.0],
                Some(0),
                None,
            ),
            (
                PixelFormat::RGB9E5Float,
                [0.0, 0.0, neg, 0.0],
                Some(2),
                Some(2),
            ),
            (PixelFormat::RGB9E5Float, [0.0, 0.0, 0.0, nan], None, None),
            (
                PixelFormat::RGBX16Float,
                [0.0, 0.0, 0.0, nan],
                Some(3),
                None,
            ),
            (
                PixelFormat::RGBX32Float,
                [neg, 0.0, 0.0, 0.0],
                Some(0),
                Some(0),
            ),
            (
                PixelFormat::BGRA10XR,
                [nan, 0.0, 0.0, 0.0],
                Some(0),
                Some(0),
            ),
            (
                PixelFormat::BGRA10XRSRGB,
                [0.0, 0.0, 0.0, neg],
                Some(3),
                Some(3),
            ),
            (PixelFormat::BGRA10XR, [0.25, 0.5, 1.25, -0.5], None, None),
            (PixelFormat::BGR10XR, [0.0, 0.0, nan, 0.0], Some(2), Some(2)),
            (PixelFormat::BGR10XRSRGB, [0.0, 0.0, 0.0, nan], None, None),
            (
                PixelFormat::RGB10A82PXR10,
                [0.0, nan, 0.0, 0.0],
                Some(1),
                Some(1),
            ),
            (
                PixelFormat::RGB10A82PXR10SRGB,
                [0.0, 0.0, 0.0, nan],
                Some(3),
                Some(3),
            ),
            (
                PixelFormat::RGBA16FloatXR,
                [0.0, 0.0, 0.0, nan],
                Some(3),
                Some(3),
            ),
            (PixelFormat::RGBA8Unorm, [neg, neg, neg, neg], None, None),
            (PixelFormat::R16Bfloat, [neg, neg, neg, neg], None, None),
        ];
        for (format, colour, four, one) in cases {
            let case = format!("{format:?} {colour:?}");
            assert_eq!(unkept_nan(format, 4, colour), four, "{case} at 4 samples");
            assert_eq!(unkept_nan(format, 1, colour), one, "{case} at 1");
        }
    }

    /// An indexed draw's indices are read from its offset's low 32 bits, as Metal's are
    #[test]
    fn indexed_draws_read_from_their_offsets_low_32_bits() {
        assert_eq!(indices(1, (1 << 32) + 16, 2), Some(16..24));
        assert_eq!(indices(0, (1 << 40) + 6, 3), Some(6..12));
        assert_eq!(indices(1, 64, 1), Some(64..68));
        assert_eq!(indices(2, (1 << 32) + 16, 1), None);
        assert_eq!(indices(1, 0, u64::MAX), None);
    }

    /// Each private type's indices: the vertices of the public type's primitives, in order,
    /// relative to the first; vertex IDs past `u32` die, and a loop through vertex 0xffffffff
    /// takes no restart index
    #[test]
    fn private_types_index_their_vertices() {
        let sel = "drawPrimitives:vertexStart:vertexCount:";
        assert_eq!(Private::LineLoop.indices(sel, 2, 3), [0, 1, 2, 0]);
        assert!(Private::LineLoop.indices(sel, 2, 1).is_empty());
        assert_eq!(
            Private::TriangleFan.indices(sel, 0, 5),
            [0, 1, 2, 0, 2, 3, 0, 3, 4]
        );
        assert_eq!(
            Private::Quads.indices(sel, 1, 9),
            [0, 1, 3, 1, 2, 3, 4, 5, 7, 5, 6, 7]
        );
        assert_eq!(
            Private::QuadStrip.indices(sel, 0, 7),
            [0, 1, 3, 2, 0, 3, 2, 3, 5, 4, 2, 5]
        );
        assert!(Private::QuadStrip.indices(sel, 0, 3).is_empty());
        for t in [
            Private::LineLoop,
            Private::TriangleFan,
            Private::Quads,
            Private::QuadStrip,
        ] {
            for count in 0..12 {
                let n = t.indices(sel, 0, count).len();
                assert_eq!(u64::try_from(n).unwrap(), t.index_count(u64::from(count)));
            }
        }
        let site = catch_death(|| Private::Quads.indices(sel, u32::MAX, 8)).unwrap_err();
        assert_eq!(site.file, sel);
        assert_eq!(
            Private::LineLoop.indices(sel, u32::MAX - 2, 3),
            [0, 1, 2, 0]
        );
        assert_eq!(
            Private::Quads.indices(sel, u32::MAX - 3, 4),
            [0, 1, 3, 1, 2, 3]
        );
    }

    /// An open encoder on a target, on this platform.
    fn open_encoder() -> (VulkanRenderEncoder, VulkanTexture) {
        let (p, d) = (platform(), dev());
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        (e, dst)
    }

    /// What the runtime has no form for, and what it refuses where Metal goes on, dies at the
    /// selector
    #[test]
    fn render_refusals_die() {
        let (e, dst) = open_encoder();
        let d = dev();
        let indices = d
            .new_buffer_of_length(OnFault::Die, 256, 0)
            .unwrap()
            .unwrap();
        let die = |cmd: RenderCmd<'_, VulkanPlatform>| {
            catch_death(|| e.encode(OnFault::Return, cmd))
                .unwrap_err()
                .file
        };
        let indexed = |primitive_type, index_type| RenderCmd::DrawIndexedPrimitives {
            primitive_type,
            index_count: 3,
            index_type,
            index_buffer: &indices,
            index_buffer_offset: 0,
        };
        let draw_indexed =
            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:";
        assert_eq!(
            die(RenderCmd::DrawPrimitives {
                primitive_type: 9,
                vertex_start: 0,
                vertex_count: 3,
            }),
            "drawPrimitives:vertexStart:vertexCount:"
        );
        assert_eq!(die(indexed(6, U16)), draw_indexed);
        assert_eq!(
            die(RenderCmd::DrawPrimitivesIndirect {
                primitive_type: 8,
                indirect_buffer: &indices,
                indirect_buffer_offset: 0,
            }),
            "drawPrimitives:indirectBuffer:indirectBufferOffset:"
        );
        assert_eq!(
            die(RenderCmd::SetFragmentTextures {
                textures: &[Some(&dst)],
                range: NsRange::new(0, 2),
            }),
            "setFragmentTextures:withRange:"
        );
        assert_eq!(
            die(RenderCmd::SetColorResolveTexture {
                texture: None,
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
                index: 8,
            }),
            "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:"
        );
        assert_eq!(
            die(RenderCmd::SetFragmentBufferOffset {
                offset: 0,
                index: 99,
            }),
            "setFragmentBufferOffset:atIndex:"
        );
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// The tests' render functions that read `[[barycentric_coord]]` (`barycentric.metal`).
    const BARYCENTRIC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/barycentric.metallib"));

    /// The tests' render functions that read `[[vertex_id]]` (`vertex_ids.metal`).
    const VERTEX_IDS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vertex_ids.metallib"));

    /// The tests' functions that use SIMD-group operations in each stage (`simd_stages.metal`).
    const SIMD_STAGES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/simd_stages.metallib"));

    /// A function of the suite's fixtures on `d`: `name` of `fixtures/shaders/<file>.metallib`,
    /// given as `file:name`, or as `name` where the two are one; of the files `barycentric`,
    /// `vertex_ids` and `simd_stages`, the crate's own [`BARYCENTRIC`], [`VERTEX_IDS`] and
    /// [`SIMD_STAGES`].
    fn fixture<P: Platform>(d: &P::Device, function: &str) -> P::Function {
        use reims_vgpu_core::platform::resource::Library as _;
        let (file, name) = function.split_once(':').unwrap_or((function, function));
        let bytes = if file == "barycentric" {
            BARYCENTRIC.to_vec()
        } else if file == "vertex_ids" {
            VERTEX_IDS.to_vec()
        } else if file == "simd_stages" {
            SIMD_STAGES.to_vec()
        } else {
            let path = format!(
                "{}/../../tests/suite/fixtures/shaders/{file}.metallib",
                env!("CARGO_MANIFEST_DIR")
            );
            std::fs::read(path).unwrap()
        };
        let library = d.new_library(&bytes).unwrap().unwrap().unwrap();
        library.new_function(OnFault::Die, name).unwrap().unwrap()
    }

    /// The state of the fixtures `vertex` and `fragment` on `d`, for a BGRA8 target.
    fn fixture_state<P: Platform>(
        d: &P::Device,
        vertex: &str,
        fragment: &str,
    ) -> P::RenderPipeline {
        use reims_vgpu_core::platform::values::RenderPipelineDesc;
        let (v, f) = (fixture::<P>(d, vertex), fixture::<P>(d, fragment));
        d.new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &v,
            fragment: &f,
            color0_pixel_format: BGRA8,
            resource_index: None,
        })
        .unwrap()
        .unwrap()
    }

    /// The colours of [`flat_pass`]'s vertices: sixteen, each its own.
    fn vertex_colours() -> Vec<f32> {
        (0..16u8)
            .flat_map(|i| {
                let c = |k: u8| f32::from(k) / 3.0;
                [c(i % 4), c((i / 4) % 4), c((i * 3) % 4), 1.0]
            })
            .collect()
    }

    /// [`scattered`]'s positions in a buffer, then [`vertex_colours`] in another, padded to `n`
    /// vertices, each past the sixteenth white and off the target.
    fn flat_vertices<P: Platform>(d: &P::Device, n: usize) -> (P::Buffer, P::Buffer) {
        let (mut pos, _) = scattered();
        let mut colours = vertex_colours();
        pos.resize(n * 4, 0.0);
        colours.resize(n * 4, 1.0);
        for vertex in pos.as_chunks_mut::<4>().0.iter_mut().skip(16) {
            *vertex = [3.0, 3.0, 0.0, 1.0];
        }
        let length = |data: &[f32]| u64::try_from(data.len() * 4).unwrap();
        (
            buffer::<P>(d, length(&pos), 0, &bytes(&pos)),
            buffer::<P>(d, length(&colours), 0, &bytes(&colours)),
        )
    }

    /// The pixels of a 16 by 16 BGRA8 target, cleared to black, after `cmds` and `EndEncoding`
    /// in a render encoder of it.
    fn pass_pixels<P: Platform>(p: &P, d: &P::Device, cmds: Vec<RenderCmd<'_, P>>) -> Vec<[u8; 4]> {
        let (w, h) = (16, 16);
        let (dst, pass) = pass_of(p, d, (w, h));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in cmds.into_iter().chain([RenderCmd::EndEncoding]) {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let out = buffer::<P>(d, w * h * 4, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        blit.encode(
            OnFault::Die,
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(w, h, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: w * 4,
                destination_bytes_per_image: w * h * 4,
            },
        )
        .unwrap();
        blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        let n = usize::try_from(w * h * 4).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the blit completed.
        let texels = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) };
        texels.as_chunks::<4>().0.to_vec()
    }

    /// A texture of protection options 0x1, as a macOS 26 guest's WindowServer on the x86 device
    /// asks for once protected video is on screen (`BGRA8Unorm` and `RGBA16Float`, usage 0x5,
    /// `Private`), is made by `newTextureWithDescriptor:` and `newSharedTextureWithDescriptor:`
    /// as without them, each relayed and counted, where Metal answers nil
    /// ([`crate::gpu::desc::TextureFields::made_unprotected`]): it answers protection options 0,
    /// and a pass clears it and a blit copies it out. In strict mode each creator dies there
    #[test]
    fn a_protected_texture_is_made_unprotected() {
        let (p, d) = (platform(), dev());
        let (w, h) = (64, 16);
        let relayed = || relayed_of(crate::answer::Inexact::ProtectedContent);
        for (format, texel) in [(BGRA8, 4), (RGBA16F, 8)] {
            let desc = p.texture_descriptor(&TextureDesc {
                pixel_format: format,
                width: w,
                height: h,
                usage: READ | TARGET,
                resource_options: Some(0x20),
                protection_options: Some(0x1),
                ..TextureDesc::default()
            });
            type Made = RaiseResult<Option<VulkanTexture>>;
            let creators: [(&str, &dyn Fn() -> Made); 2] = [
                ("newTextureWithDescriptor:", &|| {
                    d.new_texture(OnFault::Return, &desc)
                }),
                ("newSharedTextureWithDescriptor:", &|| {
                    d.new_shared_texture(&desc)
                }),
            ];
            let mut made = Vec::new();
            for (sel, make) in creators {
                *crate::answer::LAST_LINE.lock() = None;
                let site = catch_death(|| drop(make())).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!(site.file, sel, "{line}");
                assert!(line.contains("protection options 0x1"), "{line}");
                let before = relayed();
                let t = crate::answer::relaxed(make).unwrap();
                let t = t.unwrap_or_else(|| panic!("{format}: {sel} made nothing"));
                assert!(relayed() > before, "{format}: {sel}");
                assert_eq!(t.protection_options(), 0, "{format}: {sel}");
                made.push(t);
            }
            let pass = p.render_pass_descriptor();
            pass.set_color_attachment(
                OnFault::Die,
                0,
                &ColorAttachmentSet {
                    texture: Some(Some(&made[0])),
                    load_action: Some(CLEAR),
                    store_action: Some(STORE),
                    clear_color: Some(ClearColor {
                        red: 0.25,
                        green: 0.5,
                        blue: 0.75,
                        alpha: 1.0,
                    }),
                    ..ColorAttachmentSet::default()
                },
            )
            .unwrap();
            let q = d.new_command_queue().unwrap();
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            let out = buffer::<VulkanPlatform>(&d, w * h * texel, 0, &[]);
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            blit.encode(
                OnFault::Die,
                BlitCmd::CopyTextureToBuffer {
                    source_texture: &made[0],
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::new(0, 0, 0),
                    source_size: MtlSize::new(w, h, 1),
                    destination_buffer: &out,
                    destination_offset: 0,
                    destination_bytes_per_row: w * texel,
                    destination_bytes_per_image: w * h * texel,
                },
            )
            .unwrap();
            blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert!(cb.error().is_none(), "{format}: {:?}", cb.error());
            let n = usize::try_from(w * h * texel).unwrap();
            // SAFETY: a Shared buffer's bytes, read after the blit completed.
            let bytes = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) };
            // 0.25, 0.5, 0.75 and 1.0: BGRA8Unorm's bytes, and RGBA16Float's halves.
            let cleared: &[u8] = match format {
                BGRA8 => &[0xbf, 0x80, 0x40, 0xff],
                _ => &[0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x3c],
            };
            assert!(
                bytes.chunks(cleared.len()).all(|t| t == cleared),
                "{format}"
            );
        }
    }

    /// The fixtures of a state whose colour is each vertex's, flat.
    const FLAT: (&str, &str) = ("rdec_flat_v", "rdec_flat_f");

    /// The functions of a state that sizes its points, writing `[[point_size]]`, and reads a
    /// flat input; its vertex function reads no buffer.
    const SIZED: (&str, &str) = ("vertex_ids:sized_vertex", "vertex_ids:ids_fragment");

    /// A draw of primitive type 9 encodes nothing, and the command buffer fails as Metal's GPU
    /// fails it: with `MTLCommandBufferErrorTimeout` (status 5), the blit before the pass run,
    /// neither the pass's clear nor the blit after it, an event signal after them firing. The
    /// commands after the draw in its encoder are taken
    #[test]
    fn a_type_9_draw_fails_its_command_buffer_as_metals_gpu_does() {
        use reims_vgpu_core::platform::event::EventRef;
        use reims_vgpu_core::platform::types::NsRange;

        let (p, d) = (platform(), dev());
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        crate::gpu::resource::tests::fill(&dst, &[0x11; 64]);
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let (before, after) = (
            buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]),
            buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]),
        );
        let event = d.new_shared_event().unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let fill = |b, value| {
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            let range = NsRange::new(0, 16);
            let fill = BlitCmd::FillBuffer {
                buffer: b,
                range,
                value,
            };
            blit.encode(OnFault::Die, fill).unwrap();
            blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        };
        fill(&before, 0x5a);
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        send_all(
            &e,
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                three(9, None),
                three(3, None),
                RenderCmd::EndEncoding,
            ],
        );
        fill(&after, 0x6b);
        cb.encode_signal(Some(EventRef::Shared(&event)), 5).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.status().0, 5);
        assert_eq!(cb.error().map(|e| e.code), Some(2));
        let bytes = |b: &VulkanBuffer| {
            // SAFETY: a Shared buffer's bytes, read after its command buffer completed.
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), 16) }.to_vec()
        };
        assert_eq!(
            (bytes(&before), bytes(&after)),
            (vec![0x5a; 16], vec![0; 16])
        );
        assert_eq!(crate::gpu::resource::tests::read_back(&dst), vec![0x11; 64]);
        assert_eq!(event.signaled_value(), 5);
    }

    /// A command buffer of a draw of primitive type 9 is in flight as long as Metal's: its signal
    /// after the hang fires, and it completes, no sooner than `HANG_TIME` after its commit. A
    /// command buffer committed on its queue meanwhile, which Metal runs nothing of and completes
    /// without an error, dies at its commit; one of another queue runs then, and one of its
    /// queue once it completed (`metal_runs_after_a_type_9_hang_here` measures Metal's)
    #[test]
    fn a_type_9_hang_is_in_flight_as_long_as_metals() {
        use reims_vgpu_core::platform::event::EventRef;
        use reims_vgpu_core::platform::os::Clock as _;
        use reims_vgpu_core::platform::types::NsRange;

        let (p, d) = (platform(), dev());
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let (q, other) = (
            d.new_command_queue().unwrap(),
            d.new_command_queue().unwrap(),
        );
        let event = d.new_shared_event().unwrap();
        let hang = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = hang.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        send_all(
            &e,
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                three(9, None),
                RenderCmd::EndEncoding,
            ],
        );
        hang.encode_signal(Some(EventRef::Shared(&event)), 1)
            .unwrap();
        // A command buffer of `q` that fills a buffer with 0x5a, and the buffer.
        let filling = |q: &crate::gpu::command::VulkanCommandQueue| {
            let b = buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]);
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            let fill = BlitCmd::FillBuffer {
                buffer: &b,
                range: NsRange::new(0, 16),
                value: 0x5a,
            };
            blit.encode(OnFault::Die, fill).unwrap();
            blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            (cb, b)
        };
        let runs = |q| {
            let (cb, b) = filling(q);
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert_eq!(cb.status().0, 4);
            assert!(cb.error().is_none(), "{:?}", cb.error());
            // SAFETY: a Shared buffer's bytes, read after its command buffer completed.
            let first = unsafe { *b.contents().as_ptr::<u8>() };
            assert_eq!(first, 0x5a);
        };
        let clock = crate::os::OsClock::new();
        let since = |start| clock.now_mono().saturating_duration_since(start);
        let started = clock.now_mono();
        hang.commit(OnFault::Die).unwrap();
        runs(&other);
        let (cb, _b) = filling(&q);
        let site = catch_death(|| cb.commit(OnFault::Die)).unwrap_err();
        assert_eq!(site.file, "commit");
        assert!(event.wait_until_signaled_value(1, std::time::Duration::from_secs(5)));
        assert!(since(started) >= crate::gpu::command::HANG_TIME);
        hang.wait_until_completed(OnFault::Die).unwrap();
        assert!(since(started) >= crate::gpu::command::HANG_TIME);
        assert_eq!(
            (hang.status().0, hang.error().map(|e| e.code)),
            (5, Some(2))
        );
        runs(&q);
    }

    /// A command buffer Metal's GPU would fail in two ways, a draw of primitive type 9 in a
    /// render encoder made under protection options, dies at its commit, as nothing measured it
    #[test]
    fn a_type_9_draw_under_protection_options_dies_at_its_commit() {
        let (p, d) = (platform(), dev());
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        cb.set_protection_options(OnFault::Die, 2).unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        send_all(
            &e,
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                three(9, None),
                RenderCmd::EndEncoding,
            ],
        );
        let site = catch_death(|| cb.commit(OnFault::Die)).unwrap_err();
        assert_eq!(site.file, "commit");
    }

    /// A draw of primitive type 9 over indices or indirect arguments, of no vertices or no
    /// instances, or with no pipeline state set dies, as nothing measured it
    #[test]
    fn unmeasured_type_9_draws_die() {
        let (e, indices) = drawing_encoder("vert:tri_vert", "frag:tri_frag");
        let draw = "drawPrimitives:vertexStart:vertexCount:";
        assert_eq!(
            dies_at(&e, three(9, Some(&indices))),
            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:"
        );
        let none = RenderCmd::DrawPrimitives {
            primitive_type: 9,
            vertex_start: 0,
            vertex_count: 0,
        };
        assert_eq!(dies_at(&e, none), draw);
        let no_instances = RenderCmd::DrawPrimitivesInstances {
            primitive_type: 9,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 0,
        };
        assert_eq!(
            dies_at(&e, no_instances),
            "drawPrimitives:vertexStart:vertexCount:instanceCount:"
        );
        let (e, _dst) = open_encoder();
        assert_eq!(dies_at(&e, three(9, None)), draw);
    }

    /// The sweep library, `rounding.metal`, compiled with fast math, the compiler's default.
    pub(crate) const ROUNDING: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/rounding.metallib"));

    /// A guest's render state of `textureWriteRoundingMode` `mode` of [`ROUNDING`]'s
    /// `sweep_vertex` and `fragment`, colour 0 of `RGBA8Unorm`, of `supportIndirectCommandBuffers`
    /// where `icb`, decoded from a blob as core decodes one: its creator's answer.
    fn texture_writing_state<P: Platform>(
        d: &P::Device,
        mode: u32,
        fragment: &str,
        icb: bool,
    ) -> RaiseResult<Result<Option<P::RenderPipeline>, reims_vgpu_core::platform::types::NsErrorText>>
    {
        use reims_vgpu_core::platform::resource::Library as _;
        use reims_vgpu_core::rt::Hold;

        use crate::gpu::serialized::tests::rounded_render_of;

        let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
        let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
        let functions = [function("sweep_vertex"), function(fragment)];
        let mut resolve = |_: Hold<'_>, r: u64| {
            let at = usize::try_from(r - 1).unwrap();
            Ok(Some(functions[at].clone_ref()))
        };
        let mut entries = vec!["02 04 02000000"];
        if icb {
            entries.push("19 04 01000000");
        }
        let blob = rounded_render_of(&entries, &[&[(0, 0), (1, 70)]], mode);
        let desc = d
            .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
            .unwrap()
            .unwrap();
        d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
    }

    /// Under a render state of `textureWriteRoundingMode` 2 whose fragment function writes
    /// texture 0, an indexed draw, an indirect one and one of the private type 7 die at their
    /// selectors with an `RGBA16Float` texture bound there, as a direct draw does, and are encoded
    /// with an `RGBA8Unorm` one
    #[test]
    fn mode_2_half_float_texture_writes_die_at_every_kind_of_draw() {
        use crate::gpu::serialized::NEAREST_EVEN;

        let (p, d) = (platform(), dev());
        let state =
            texture_writing_state::<VulkanPlatform>(&d, NEAREST_EVEN, "write_fragment", false)
                .unwrap()
                .unwrap()
                .unwrap();
        let kinds = buffer::<VulkanPlatform>(&d, 4, 0, &0u32.to_le_bytes());
        let arguments: Vec<u8> = [3u32, 1, 0, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let indirect = buffer::<VulkanPlatform>(&d, 16, 0, &arguments);
        let indices = buffer::<VulkanPlatform>(&d, 6, 0, &u16s(&[0, 1, 2]));
        let target = texture(&p, &d, 70, READ | TARGET, (16, 16), None);
        let q = d.new_command_queue().unwrap();
        let draws = || {
            [
                (
                    three(3, Some(&indices)),
                    "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
                ),
                (
                    RenderCmd::DrawPrimitivesIndirect {
                        primitive_type: 3,
                        indirect_buffer: &indirect,
                        indirect_buffer_offset: 0,
                    },
                    "drawPrimitives:indirectBuffer:indirectBufferOffset:",
                ),
                (
                    RenderCmd::DrawPrimitives {
                        primitive_type: 7,
                        vertex_start: 0,
                        vertex_count: 4,
                    },
                    "drawPrimitives:vertexStart:vertexCount:",
                ),
            ]
        };
        for format in [115, 70] {
            let t = texture(&p, &d, format, READ | 2, (16, 16), None);
            for (draw, sel) in draws() {
                let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
                let pass = p.render_pass_descriptor();
                let attachment = ColorAttachmentSet {
                    texture: Some(Some(&target)),
                    load_action: Some(CLEAR),
                    store_action: Some(STORE),
                    ..ColorAttachmentSet::default()
                };
                pass.set_color_attachment(OnFault::Die, 0, &attachment)
                    .unwrap();
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                send_all(
                    &e,
                    vec![
                        RenderCmd::SetRenderPipelineState {
                            pipeline_state: Some(&state),
                        },
                        RenderCmd::SetFragmentBuffer {
                            buffer: Some(&kinds),
                            offset: 0,
                            index: 0,
                        },
                        RenderCmd::SetFragmentTexture {
                            texture: Some(&t),
                            index: 0,
                        },
                    ],
                );
                if format == 115 {
                    assert_eq!(dies_at(&e, draw), sel);
                } else {
                    send_all(&e, vec![draw, RenderCmd::EndEncoding]);
                }
            }
        }
    }

    /// A pass whose colour resolve the device averages and rounds, which the runtime announces
    /// (`DeviceRoundedResolve`), dies at `endEncoding` where the format was not measured to
    /// resolve as Metal's, `RG11B10Float` (92) at 4 samples, naming it; one of a measured format,
    /// `RGBA16Float` at 4 samples, ends ([`device_rounded`]). MoltenVK announces neither and
    /// both end; a native device that announces the rounding dies at the unmeasured pass's
    /// end. The descriptors check that same decision independently on either host.
    #[test]
    fn an_unmeasured_device_resolve_dies_at_the_end() {
        use reims_vgpu_core::raise::death::DeathKind;

        /// `MTLTextureType2DMultisample` and `MTLStoreActionMultisampleResolve`.
        const MULTISAMPLE: u64 = 4;
        const RESOLVE: u64 = 2;
        let (p, d) = (platform(), dev());
        let end = |format| {
            let samples = d
                .new_texture(
                    OnFault::Die,
                    &p.texture_descriptor(&TextureDesc {
                        texture_type: MULTISAMPLE,
                        pixel_format: format,
                        width: 16,
                        height: 16,
                        sample_count: 4,
                        usage: TARGET,
                        resource_options: Some(0x20),
                        ..TextureDesc::default()
                    }),
                )
                .unwrap()
                .unwrap();
            let target = texture(&p, &d, format, READ | TARGET, (16, 16), None);
            let pass = p.render_pass_descriptor();
            let attachment = ColorAttachmentSet {
                texture: Some(Some(&samples)),
                resolve_texture: Some(Some(&target)),
                load_action: Some(CLEAR),
                store_action: Some(RESOLVE),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &attachment)
                .unwrap();
            let q = d.new_command_queue().unwrap();
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            let result = catch_death(|| e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap());
            let mut ended = rt::RenderPassDescriptor::default();
            ended.color_attachments[0].texture = Some(samples.raw.clone());
            (ended, result)
        };
        let (measured, measured_end) = end(RGBA16F);
        assert!(measured_end.is_ok());
        let (unmeasured, unmeasured_end) = end(92);
        let relayed = crate::answer::RELAYED.lock().clone();
        if relayed.contains(&rt::Degradation::DeviceRoundedResolve) {
            let site = unmeasured_end.unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, "endEncoding"));
        } else {
            assert!(unmeasured_end.is_ok());
        }
        device_rounded("endEncoding", &measured);
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| device_rounded("endEncoding", &unmeasured)).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "endEncoding"),
            "{line}"
        );
        assert!(
            line.contains("RG11B10Float colour attachment of 4 samples"),
            "{line}"
        );
    }

    /// Each triangle's indices, its last vertex first, in its winding
    #[test]
    fn last_first_rotates_each_triangle() {
        let sel = "drawPrimitives:vertexStart:vertexCount:";
        assert_eq!(
            last_first(sel, PrimitiveType::Triangle, 1, 7),
            [2, 0, 1, 5, 3, 4]
        );
        assert_eq!(
            last_first(sel, PrimitiveType::TriangleStrip, 2, 5),
            [2, 0, 1, 3, 2, 1, 4, 2, 3]
        );
        assert!(last_first(sel, PrimitiveType::TriangleStrip, 0, 2).is_empty());
        let site = catch_death(|| last_first(sel, PrimitiveType::Line, 0, 4)).unwrap_err();
        assert_eq!(site.file, sel);
        let site =
            catch_death(|| last_first(sel, PrimitiveType::Triangle, u32::MAX, 3)).unwrap_err();
        assert_eq!(site.file, sel);
    }

    /// An open encoder on a BGRA8 target with the state of the fixtures `vertex` and `fragment`
    /// set, and an index buffer.
    fn drawing_encoder(vertex: &str, fragment: &str) -> (VulkanRenderEncoder, VulkanBuffer) {
        open_drawing(&fixture_state::<VulkanPlatform>(&dev(), vertex, fragment))
    }

    /// An open encoder on a BGRA8 target with `state` set, and an index buffer.
    fn open_drawing(
        state: &crate::gpu::resource::VulkanRenderPipeline,
    ) -> (VulkanRenderEncoder, VulkanBuffer) {
        let (e, _dst) = open_encoder();
        e.encode(
            OnFault::Die,
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
            },
        )
        .unwrap();
        let indices = buffer::<VulkanPlatform>(&dev(), 64, 0, &[0; 64]);
        (e, indices)
    }

    /// Sends each of `cmds` to `e`.
    fn send_all(e: &VulkanRenderEncoder, cmds: Vec<RenderCmd<'_, VulkanPlatform>>) {
        for cmd in cmds {
            let sel = cmd.selector();
            e.encode(OnFault::Return, cmd)
                .unwrap_or_else(|r| panic!("{sel}: {r:?}"));
        }
    }

    /// How many times this process relayed `kind` ([`crate::answer::inexact_counts`]).
    fn relayed_of(kind: crate::answer::Inexact) -> u64 {
        (crate::answer::inexact_counts().into_iter())
            .find(|&(k, _)| k == kind)
            .map_or(0, |(_, n)| n)
    }

    /// A draw of `primitive_type`, three vertices or indices of `indices`.
    fn three(primitive_type: u64, indices: Option<&VulkanBuffer>) -> RenderCmd<'_, VulkanPlatform> {
        match indices {
            None => RenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start: 0,
                vertex_count: 3,
            },
            Some(index_buffer) => RenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count: 3,
                index_type: U16,
                index_buffer,
                index_buffer_offset: 0,
            },
        }
    }

    /// The selector a death of `cmd` names.
    fn dies_at(e: &VulkanRenderEncoder, cmd: RenderCmd<'_, VulkanPlatform>) -> &'static str {
        catch_death(|| e.encode(OnFault::Return, cmd))
            .unwrap_err()
            .file
    }

    /// An indirect strip of 200,000 16- or 32-bit indices that restarts at the setter's index
    /// ([`long_restarted_strips`]) draws as the same strip drawn directly: the command buffer's
    /// index pool, a sixty-fourth of the device's largest device-local heap, holds the list the
    /// runtime rewrites it into before the pass. A list past the pool still dies as the command
    /// buffer's failure is read ([`crate::gpu::command`])
    #[test]
    fn a_long_restarted_strip_draws_within_the_index_pool() {
        let (p, d) = (platform(), dev());
        for (index_type, indices) in long_restarted_strips() {
            let drawn = (STRIP, index_type, &indices[..]);
            let restart = Some((true, 3));
            let indirect = restart_draw(&p, &d, drawn, restart, (true, 0));
            let direct = restart_draw(&p, &d, drawn, restart, (false, 0));
            assert_eq!(indirect, direct, "{index_type}");
        }
    }

    /// Strips of 200,000 indices, 16-bit and 32-bit, each its type's: 0, 1, 2 and 3 over and
    /// over, so a strip that restarts at 3 draws the triangle (0, 1, 2) again after each.
    fn long_restarted_strips() -> [(u64, Vec<u8>); 2] {
        let pattern: Vec<u32> = [0, 1, 2, 3].into_iter().cycle().take(200_000).collect();
        let narrow: Vec<u16> = pattern.iter().map(|&i| u16::try_from(i).unwrap()).collect();
        [(U16, u16s(&narrow)), (U32, u32s(&pattern))]
    }

    /// Little-endian `u16` indices.
    fn u16s(indices: &[u16]) -> Vec<u8> {
        indices.iter().flat_map(|i| i.to_le_bytes()).collect()
    }

    /// Little-endian `u32` indices.
    fn u32s(indices: &[u32]) -> Vec<u8> {
        indices.iter().flat_map(|i| i.to_le_bytes()).collect()
    }

    /// `MTLIndexTypeUInt32`.
    const U32: u64 = 1;

    /// The pixels of [`pass_pixels`] after a draw of `primitive_type` over `indices` of
    /// `index_type`, under [`FLAT`]'s state over [`flat_vertices`] padded to 65536, so every
    /// 16-bit index reads a vertex: drawn directly, or from indirect arguments where
    /// `indirect`, after `restart`'s `setPrimitiveRestartEnabled:index:` where it has one. The
    /// indices lie `offset` bytes into their buffer, after bytes of index 5 (a 16-bit one where
    /// `offset` is 2), which a draw reading them would show.
    fn restart_draw<P: Platform>(
        p: &P,
        d: &P::Device,
        (primitive_type, index_type, indices): (u64, u64, &[u8]),
        restart: Option<(bool, u64)>,
        (indirect, offset): (bool, u64),
    ) -> Vec<[u8; 4]> {
        let state = fixture_state::<P>(d, FLAT.0, FLAT.1);
        let (pos, colours) = flat_vertices::<P>(d, 0x10000);
        let length = u64::try_from(indices.len()).unwrap();
        let mut bytes: Vec<u8> = [5u8, 0, 0, 0]
            .into_iter()
            .cycle()
            .take(usize::try_from(offset).unwrap())
            .collect();
        bytes.extend_from_slice(indices);
        let index_buffer = buffer::<P>(d, offset + length, 0, &bytes);
        let index_count = length / if index_type == U16 { 2 } else { 4 };
        // `MTLDrawIndexedPrimitivesIndirectArguments`: the count, one instance, and zeros.
        let arguments: Vec<u8> = [u32::try_from(index_count).unwrap(), 1, 0, 0, 0]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let arguments = buffer::<P>(d, 20, 0, &arguments);
        let mut cmds = vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&pos),
                offset: 0,
                index: 0,
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&colours),
                offset: 0,
                index: 1,
            },
        ];
        cmds.extend(
            restart
                .map(|(enabled, index)| RenderCmd::SetPrimitiveRestartEnabled { enabled, index }),
        );
        cmds.push(match indirect {
            false => RenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer: &index_buffer,
                index_buffer_offset: offset,
            },
            true => RenderCmd::DrawIndexedPrimitivesIndirect {
                primitive_type,
                index_type,
                index_buffer: &index_buffer,
                index_buffer_offset: offset,
                indirect_buffer: &arguments,
                indirect_buffer_offset: 0,
            },
        });
        pass_pixels(p, d, cmds)
    }

    /// `setLineWidth:` of anything but 1 encodes nothing, and after it lines and triangles
    /// filled as lines die in strict mode, as no line drawn here is wider than 1, and draw 1
    /// pixel wide by default, relayed; filled triangles and points draw, and a width of 1 again
    /// draws lines
    #[test]
    fn a_line_width_kills_only_lines() {
        let (e, indices) = drawing_encoder(SIZED.0, SIZED.1);
        let draw = "drawPrimitives:vertexStart:vertexCount:";
        for width in [2.0, 0.0, -1.0, f32::NAN] {
            send_all(&e, vec![RenderCmd::SetLineWidth { line_width: width }]);
            send_all(
                &e,
                vec![three(3, None), three(STRIP, Some(&indices)), three(0, None)],
            );
            for primitive_type in [1, 2, 5] {
                assert_eq!(dies_at(&e, three(primitive_type, None)), draw, "{width}");
            }
        }
        send_all(&e, vec![RenderCmd::SetTriangleFillMode { fill_mode: 1 }]);
        assert_eq!(dies_at(&e, three(3, None)), draw);
        assert_eq!(dies_at(&e, three(6, None)), draw);
        // By default each draws, 1 pixel wide, relayed and counted.
        let widths = || relayed_of(crate::answer::Inexact::LineWidth);
        let before = widths();
        crate::answer::relaxed(|| send_all(&e, vec![three(1, None), three(3, None)]));
        assert_eq!(widths(), before + 2);
        send_all(
            &e,
            vec![
                RenderCmd::SetLineWidth { line_width: 1.0 },
                three(3, None),
                three(1, None),
                RenderCmd::SetLineWidth { line_width: 3.0 },
                RenderCmd::SetTriangleFillMode { fill_mode: 0 },
                three(3, None),
                RenderCmd::SetPointSize { point_size: 4.0 },
                three(0, None),
            ],
        );
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// After a nonzero provoking vertex mode, a state that reads no flat input draws everything
    /// as it stands; one that does dies in strict mode at what its first-vertex form has no
    /// rotation for, and by default draws it as it stands, relayed, and draws points, and mode 0
    /// again draws all
    #[test]
    fn the_last_vertex_kills_only_flat_draws_it_cannot_rotate() {
        let (e, indices) = drawing_encoder("vert:tri_vert", "frag:tri_frag");
        let indirect = |primitive_type| RenderCmd::DrawPrimitivesIndirect {
            primitive_type,
            indirect_buffer: &indices,
            indirect_buffer_offset: 0,
        };
        send_all(
            &e,
            vec![
                RenderCmd::SetProvokingVertexMode { mode: 1 },
                three(3, Some(&indices)),
                three(STRIP, Some(&indices)),
                three(1, None),
                three(6, None),
                indirect(3),
            ],
        );
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let (e, indices) = drawing_encoder("rdec_flat_v", "rdec_flat_f");
        let draw = "drawPrimitives:vertexStart:vertexCount:";
        let indexed = "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:";
        send_all(
            &e,
            vec![
                RenderCmd::SetProvokingVertexMode { mode: u64::MAX },
                three(3, None),
                three(STRIP, None),
            ],
        );
        // Points, under a state that reads a flat input and sizes its points.
        let (sized, sized_indices) = drawing_encoder(SIZED.0, SIZED.1);
        send_all(
            &sized,
            vec![
                RenderCmd::SetProvokingVertexMode { mode: u64::MAX },
                three(0, None),
                three(0, Some(&sized_indices)),
                RenderCmd::EndEncoding,
            ],
        );
        assert_eq!(dies_at(&e, three(3, Some(&indices))), indexed);
        assert_eq!(dies_at(&e, three(STRIP, Some(&indices))), indexed);
        assert_eq!(dies_at(&e, three(1, None)), draw);
        assert_eq!(dies_at(&e, three(2, None)), draw);
        assert_eq!(dies_at(&e, three(6, None)), draw);
        assert_eq!(
            dies_at(
                &e,
                RenderCmd::DrawPrimitivesIndirect {
                    primitive_type: 3,
                    indirect_buffer: &indices,
                    indirect_buffer_offset: 0,
                }
            ),
            "drawPrimitives:indirectBuffer:indirectBufferOffset:"
        );
        // By default each draws as it stands, its first vertex provoking, relayed and counted.
        let rotations = || relayed_of(crate::answer::Inexact::ProvokingVertex);
        let before = rotations();
        crate::answer::relaxed(|| {
            send_all(
                &e,
                vec![three(3, Some(&indices)), three(1, None), three(6, None)],
            );
        });
        assert_eq!(rotations(), before + 3);
        send_all(
            &e,
            vec![
                RenderCmd::SetProvokingVertexMode { mode: 0 },
                three(3, Some(&indices)),
                three(1, None),
                three(6, None),
            ],
        );
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// A draw of no primitive draws nothing whatever the private setters left, as on Metal: of
    /// too few vertices or indices for one, and an execution of no command, after a restart, a
    /// line width of 2, triangles filled as lines and the last vertex provoking under a state
    /// that reads a flat input; one of a primitive dies
    #[test]
    fn empty_draws_go_on_after_the_private_setters() {
        use reims_vgpu_core::platform::values::IcbDesc;

        let (e, indices) = drawing_encoder(FLAT.0, FLAT.1);
        let icb = dev()
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap();
        let drawn = |primitive_type, vertex_count| RenderCmd::DrawPrimitives {
            primitive_type,
            vertex_start: 0,
            vertex_count,
        };
        let indexed = |primitive_type, index_count| RenderCmd::DrawIndexedPrimitives {
            primitive_type,
            index_count,
            index_type: U16,
            index_buffer: &indices,
            index_buffer_offset: 0,
        };
        send_all(
            &e,
            vec![
                RenderCmd::SetPrimitiveRestartEnabled {
                    enabled: true,
                    index: 0xffff,
                },
                RenderCmd::SetLineWidth { line_width: 2.0 },
                RenderCmd::SetTriangleFillMode { fill_mode: 1 },
                RenderCmd::SetProvokingVertexMode { mode: 1 },
            ],
        );
        let mut empty = vec![drawn(0, 0), indexed(0, 0)];
        for (primitive_type, least) in [(1, 2), (2, 2), (3, 3), (STRIP, 3)] {
            for count in 0..least {
                empty.extend([drawn(primitive_type, count), indexed(primitive_type, count)]);
            }
        }
        empty.push(RenderCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &icb,
            execution_range: NsRange::new(0, 0),
        });
        send_all(&e, empty);
        let draw = "drawPrimitives:vertexStart:vertexCount:";
        assert_eq!(dies_at(&e, drawn(1, 2)), draw);
        assert_eq!(dies_at(&e, drawn(3, 3)), draw);
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// `setVertexAmplificationMode:value:` encodes nothing; draws go on where it leaves the
    /// runtime's count, and where it leaves another, or 0, which Metal draws copies of at random,
    /// draws and executions die until `setVertexAmplificationCount:viewMappings:` sets one, as
    /// `render_state_misc` sends them
    #[test]
    fn an_amplification_mode_kills_draws_only_where_it_changes_the_count() {
        use reims_vgpu_core::platform::types::VertexAmplificationViewMapping;
        use reims_vgpu_core::platform::values::IcbDesc;

        let (e, _indices) = drawing_encoder("vert:tri_vert", "frag:tri_frag");
        let icb = dev()
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap();
        let mode = |mode, value| RenderCmd::SetVertexAmplificationMode { mode, value };
        let none = VertexAmplificationViewMapping {
            viewport_array_index_offset: 0,
            render_target_array_index_offset: 0,
        };
        let one = [none];
        let counted = RenderCmd::SetVertexAmplificationCount {
            view_mappings: &one,
        };
        send_all(
            &e,
            vec![
                mode(0, 1),
                mode(1, 4),
                mode(7, 1),
                three(3, None),
                mode(0, 0),
                counted,
                three(3, None),
            ],
        );
        let draw = "drawPrimitives:vertexStart:vertexCount:";
        for (m, value) in [(0, 0), (0, 2), (1, 3), (1, 0), (2, 9), (u64::MAX, 0)] {
            send_all(&e, vec![mode(m, value)]);
            assert_eq!(dies_at(&e, three(3, None)), draw, "{m} {value}");
        }
        assert_eq!(
            dies_at(
                &e,
                RenderCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: &icb,
                    execution_range: NsRange::new(0, 1),
                }
            ),
            "executeCommandsInBuffer:withRange:"
        );
        let empty = RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 0,
        };
        let counted = RenderCmd::SetVertexAmplificationCount {
            view_mappings: &one,
        };
        send_all(
            &e,
            vec![empty, counted, three(3, None), RenderCmd::EndEncoding],
        );
    }

    /// The state commands the runtime answers are taken
    #[test]
    fn state_commands_are_taken() {
        let (e, dst) = open_encoder();
        let d = dev();
        let sampler = d
            .new_sampler_state(&reims_vgpu_core::platform::values::SamplerDesc::default())
            .unwrap()
            .unwrap();
        let buf = d
            .new_buffer_of_length(OnFault::Die, 256, 0)
            .unwrap()
            .unwrap();
        let (two, one) = ([Some(&buf), None], [Some(&buf)]);
        let (texture, sampled) = ([Some(&dst)], [Some(&sampler)]);
        let used = [Some(ResourceRef::Buffer(&buf))];
        let cmds: Vec<RenderCmd<'_, VulkanPlatform>> = vec![
            RenderCmd::SetBlendColorRed {
                red: 0.5,
                green: 0.25,
                blue: 0.0,
                alpha: 1.0,
            },
            RenderCmd::SetCullMode { cull_mode: 2 },
            RenderCmd::SetFrontFacingWinding {
                front_facing_winding: 1,
            },
            RenderCmd::SetTriangleFillMode { fill_mode: 1 },
            RenderCmd::SetDepthBias {
                depth_bias: 1.0,
                slope_scale: 0.5,
                clamp: 0.0,
            },
            RenderCmd::SetDepthClipModeSpi { mode: 1 },
            RenderCmd::SetStencilFrontReferenceValue {
                front_reference_value: 3,
                back_reference_value: 4,
            },
            RenderCmd::SetScissorRects {
                scissor_rects: &[MtlScissorRect {
                    x: 0,
                    y: 0,
                    width: 2,
                    height: 2,
                }],
            },
            RenderCmd::SetVertexBuffers {
                buffers: &two,
                offsets: &[0, 0],
                range: NsRange::new(2, 2),
            },
            RenderCmd::SetVertexBufferOffset {
                offset: 16,
                index: 2,
            },
            RenderCmd::SetFragmentBuffers {
                buffers: &one,
                offsets: &[32],
                range: NsRange::new(1, 1),
            },
            RenderCmd::SetVertexTextures {
                textures: &texture,
                range: NsRange::new(0, 1),
            },
            RenderCmd::SetFragmentSamplerStates {
                samplers: &sampled,
                range: NsRange::new(0, 1),
            },
            RenderCmd::SetVertexSamplerStatesLodClamps {
                samplers: &sampled,
                lod_min_clamps: &[0.0],
                lod_max_clamps: &[4.0],
                range: NsRange::new(1, 1),
            },
            RenderCmd::SetVisibilityResultMode { mode: 0, offset: 0 },
            RenderCmd::UseResources {
                resources: &used,
                usage: 1,
            },
            RenderCmd::MemoryBarrierWithScope {
                scope: 1,
                after: 2,
                before: 1,
            },
            RenderCmd::SetDepthCleared,
            RenderCmd::SetStencilCleared,
            RenderCmd::SetProvokingVertexMode { mode: 0 },
            RenderCmd::SetPrimitiveRestartEnabled {
                enabled: true,
                index: 0,
            },
        ];
        for cmd in cmds {
            let sel = cmd.selector();
            e.encode(OnFault::Return, cmd)
                .unwrap_or_else(|r| panic!("{sel}: {r:?}"));
        }
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// A late resolve setter under a store action the descriptor knew and that does not
    /// resolve encodes nothing, as Metal ignores the resolve texture there: colour, depth and
    /// stencil, over a pass that named resolve textures for each
    #[test]
    fn late_resolves_under_known_stores_encode_nothing() {
        /// `MTLTextureType2DMultisample`, `Depth32Float_Stencil8`.
        const MULTISAMPLE: u64 = 4;
        const DEPTH32F_S8: u64 = 260;
        let (p, d) = (platform(), dev());
        let made = |texture_type, format, sample_count| {
            let desc = p.texture_descriptor(&TextureDesc {
                texture_type,
                pixel_format: format,
                width: 4,
                height: 4,
                sample_count,
                usage: TARGET,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            });
            d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
        };
        let (color, color_resolve) = (made(MULTISAMPLE, BGRA8, 4), made(2, BGRA8, 1));
        let (depth, depth_resolve) = (made(MULTISAMPLE, DEPTH32F_S8, 4), made(2, DEPTH32F_S8, 1));
        let pass = p.render_pass_descriptor();
        pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&color)),
                resolve_texture: Some(Some(&color_resolve)),
                load_action: Some(CLEAR),
                store_action: Some(STORE),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        pass.set_depth_attachment(&reims_vgpu_core::platform::values::DepthAttachmentSet {
            texture: Some(Some(&depth)),
            resolve_texture: Some(Some(&depth_resolve)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..Default::default()
        });
        pass.set_stencil_attachment(&reims_vgpu_core::platform::values::StencilAttachmentSet {
            texture: Some(Some(&depth)),
            resolve_texture: Some(Some(&depth_resolve)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..Default::default()
        });
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetColorResolveTexture {
                texture: Some(&color_resolve),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
                index: 0,
            },
            RenderCmd::SetDepthResolveTexture {
                texture: Some(&depth_resolve),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
            },
            RenderCmd::SetStencilResolveTexture {
                texture: Some(&depth_resolve),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    }

    /// A target of `w` × `h` cleared and stored, and a pass of it.
    pub(crate) fn pass_of<P: Platform>(
        p: &P,
        d: &P::Device,
        (w, h): (u64, u64),
    ) -> (P::Texture, P::RenderPassDescriptor) {
        let dst = texture(p, d, BGRA8, TARGET | READ, (w, h), None);
        let pass = p.render_pass_descriptor();
        pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&dst)),
                load_action: Some(CLEAR),
                store_action: Some(STORE),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        (dst, pass)
    }

    /// Each set reaches the runtime's descriptor, and the attachment states answer what was set
    #[test]
    fn pass_sets_reach_the_runtime_descriptor() {
        let p = platform();
        let d = dev();
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        pass.set_color_attachment(
            OnFault::Die,
            2,
            &ColorAttachmentSet {
                texture: Some(Some(&dst)),
                load_action: Some(1),
                store_action: Some(0),
                clear_color: Some(ClearColor {
                    red: 0.25,
                    green: 0.5,
                    blue: 0.75,
                    alpha: 1.0,
                }),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        pass.set_pass(&reims_vgpu_core::platform::values::PassSet {
            render_target_width: Some(16),
            render_target_height: Some(8),
            default_raster_sample_count: Some(1),
            ..Default::default()
        });
        let r = pass.descriptor();
        let c2 = &r.color_attachments[2];
        assert!(c2.texture.is_some());
        assert_eq!(
            (c2.load_action, c2.store_action),
            (rt::LoadAction::Load, rt::StoreAction::DontCare)
        );
        assert_eq!(
            (
                c2.clear_color.red,
                c2.clear_color.green,
                c2.clear_color.blue
            ),
            (0.25, 0.5, 0.75)
        );
        assert_eq!(
            (
                r.render_target_width,
                r.render_target_height,
                r.default_raster_sample_count
            ),
            (16, 8, 1)
        );
        assert!(r.color_attachments[1].texture.is_none());
        let state = pass.color_attachment_state(2).unwrap();
        assert_eq!((state.load_action, state.store_action), (1, 0));
        let state = pass.color_attachment_state(0).unwrap();
        assert_eq!((state.load_action, state.store_action), (CLEAR, STORE));
    }

    /// A colour attachment past the eighth is Metal's raise; a default colour sample count, which
    /// changes nothing on this host, is taken
    #[test]
    fn pass_refusals_raise() {
        let p = platform();
        let pass = p.render_pass_descriptor();
        let r = pass.set_color_attachment(OnFault::Return, 8, &ColorAttachmentSet::default());
        assert!(r.is_err());
        assert!(pass.color_attachment_state(8).is_err());
        pass.set_pass(&reims_vgpu_core::platform::values::PassSet {
            default_color_sample_count: Some(4),
            ..Default::default()
        });
    }

    /// Every creation of a render pipeline takes a new nonzero identifier, even for an equal
    /// request the runtime answers its cached state for; every reference answers its handle's
    #[test]
    fn every_render_pipeline_creation_takes_a_new_identifier() {
        use reims_vgpu_core::platform::resource::RenderPipeline as _;
        let d = dev();
        let key = PipelineKey {
            format: BGRA8,
            gamma: false,
        };
        let first = DisplayPipelineCache::<VulkanPlatform>::new(d.clone_ref()).unwrap();
        let second = DisplayPipelineCache::<VulkanPlatform>::new(d.clone_ref()).unwrap();
        let a = first.pipeline_for(key).unwrap();
        let again = first.pipeline_for(key).unwrap();
        let b = second.pipeline_for(key).unwrap();
        assert_ne!(a.unique_identifier(), 0);
        assert_eq!(again.unique_identifier(), a.unique_identifier());
        assert_ne!(b.unique_identifier(), a.unique_identifier());
    }

    /// A pass of no attachment and no target size has no encoder, as Metal answers nil
    #[test]
    fn a_pass_without_targets_has_no_encoder() {
        let p = platform();
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let pass = p.render_pass_descriptor();
        assert!(cb.render_encoder(OnFault::Die, &pass).unwrap().is_none());
    }

    /// A pass without attachments of a 4x4 target answers this host's Metal at each default
    /// raster sample count the runtime's probe ran on it (the M3 Max under macOS 26.7): 0, 8 to
    /// 15, 128, 255 and 256 die at the encoder's creation, as Metal ends the process there; 16
    /// and 127 make the encoder and fail the command buffer with `MTLCommandBufferErrorInternal`
    /// (code 1); 1, 4 and 7 complete it
    #[test]
    fn a_pass_without_attachments_answers_metals_default_sample_counts() {
        use reims_vgpu_core::platform::types::CommandBufferStatus;
        use reims_vgpu_core::platform::values::PassSet;
        use reims_vgpu_core::raise::death::DeathKind;

        let p = platform();
        let q = dev().new_command_queue().unwrap();
        let pass = |count| {
            let pass = p.render_pass_descriptor();
            pass.set_pass(&PassSet {
                render_target_width: Some(4),
                render_target_height: Some(4),
                default_raster_sample_count: Some(count),
                ..Default::default()
            });
            pass
        };
        let ended = |count| {
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb
                .render_encoder(OnFault::Die, &pass(count))
                .unwrap()
                .unwrap();
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            (cb.status(), cb.error().map(|e| e.code))
        };
        for count in [0, 8, 9, 10, 11, 12, 13, 14, 15, 128, 255, 256] {
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            *crate::answer::LAST_LINE.lock() = None;
            let site =
                catch_death(|| cb.render_encoder(OnFault::Return, &pass(count))).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, "renderCommandEncoderWithDescriptor:"),
                "{count}: {line}"
            );
            assert!(line.starts_with("Metal aborts at "), "{count}: {line}");
        }
        for count in [16, 127] {
            assert_eq!(
                ended(count),
                (CommandBufferStatus::ERROR, Some(1)),
                "{count}"
            );
        }
        for count in [1, 4, 7] {
            assert_eq!(
                ended(count),
                (CommandBufferStatus::COMPLETED, None),
                "{count}"
            );
        }
    }

    /// A nil pipeline state is taken; what runs the encoder's pipeline state after it dies, as
    /// Metal crashes at a draw with no pipeline state: a draw, direct or indirect, a tile
    /// dispatch, and an execution of an indirect command buffer whose commands inherit the
    /// pipeline state. The execution of one whose commands set their own is taken
    #[test]
    fn what_runs_the_pipeline_after_a_nil_one_dies() {
        use reims_vgpu_core::platform::values::IcbDesc;

        let p = platform();
        let d = dev();
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let args = buffer::<VulkanPlatform>(&d, 64, 0, &[]);
        let icb = |inherit_pipeline_state| {
            d.new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    inherit_pipeline_state,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap()
        };
        let (inheriting, own) = (icb(true), icb(false));
        e.encode(
            OnFault::Die,
            RenderCmd::SetRenderPipelineState {
                pipeline_state: None,
            },
        )
        .unwrap();
        let dies = |cmd: RenderCmd<'_, VulkanPlatform>| {
            let sel = cmd.selector();
            let site = catch_death(|| e.encode(OnFault::Die, cmd)).unwrap_err();
            assert_eq!(site.file, sel);
        };
        dies(RenderCmd::DrawPrimitives {
            primitive_type: STRIP,
            vertex_start: 0,
            vertex_count: 4,
        });
        dies(RenderCmd::DrawPrimitivesIndirect {
            primitive_type: STRIP,
            indirect_buffer: &args,
            indirect_buffer_offset: 0,
        });
        dies(RenderCmd::DispatchThreadsPerTile {
            threads_per_tile: MtlSize::new(1, 1, 1),
        });
        dies(RenderCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &inheriting,
            execution_range: NsRange::new(0, 1),
        });
        dies(RenderCmd::ExecuteCommandsIndirect {
            indirect_commandbuffer: &inheriting,
            indirect_range_buffer: &args,
            indirect_buffer_offset: 0,
        });
        e.encode(
            OnFault::Die,
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &own,
                execution_range: NsRange::new(0, 1),
            },
        )
        .unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// Runs, on a new device, a pass that executes a command of an indirect command buffer whose
    /// pipeline state draws to RGBA16Float in a pass of a BGRA8 target, which the runtime refuses,
    /// the buffer made resident with `usage`.
    fn execute_a_mismatched_draw(usage: u64) {
        use metal2vulkan::metal::PixelFormat;
        use metal2vulkan::pipeline::{
            ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
        };
        use reims_vgpu_core::platform::icb::{
            IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
        };
        use reims_vgpu_core::platform::values::IcbDesc;

        const VERT: &[u8] =
            include_bytes!("../../../../tests/suite/fixtures/shaders/vert.metallib");
        const FRAG: &[u8] =
            include_bytes!("../../../../tests/suite/fixtures/shaders/frag.metallib");
        let p = platform();
        let d = dev();
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
        };
        let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
            function(VERT, "tri_vert").function_descriptor(),
        ));
        desc.fragment = Some(StageSource::Function(
            function(FRAG, "tri_frag").function_descriptor(),
        ));
        desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
        desc.support_indirect_command_buffers = true;
        let pipeline = crate::gpu::resource::render_pipeline(
            d.raw
                .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                .unwrap(),
        );
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap();
        let c = icb.render_command(0).unwrap().unwrap();
        for cmd in [
            IcbRenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&pipeline),
            },
            IcbRenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        ] {
            c.encode(cmd).unwrap();
        }
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::UseResources {
                resources: &[Some(ResourceRef::Icb(&icb))],
                usage,
            },
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &icb,
                execution_range: NsRange::new(0, 1),
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        panic!("ran: {:?}", (cb.status(), cb.error()));
    }

    /// The records of an indirect command buffer that the runtime refuses die naming the
    /// execution: at the commit where only the CPU wrote them, and as the command buffer
    /// completes where the GPU may have (an encoder made the buffer resident for writing), whose
    /// failure no handler of core's learns of
    #[test]
    fn refused_icb_records_die_naming_the_execution() {
        use reims_vgpu_core::raise::death::{DeathKind, DeathSite};
        use reims_vgpu_core::testkit::capture::expect_death;

        let site = catch_death(|| execute_a_mismatched_draw(1)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "commit"));
        let site = DeathSite {
            kind: DeathKind::Abort,
            file: "executeCommandsInBuffer:withRange:",
            line: 0,
        };
        expect_death(site, || execute_a_mismatched_draw(3));
    }

    /// The texels of a 4 by 4 BGRA8 target, cleared to black, after `setters` and then each of
    /// `draws`, `(indexed, patch_count, instance_count)`: a patch draw of `render_tess`'s
    /// triangle, which covers the target in `tri_frag`'s colour, plain or over the 32-bit
    /// control point indices 0, 1 and 2, under a state of its kind.
    fn patch_texels(
        setters: Vec<RenderCmd<'_, VulkanPlatform>>,
        draws: &[(bool, u64, u64)],
    ) -> Vec<u8> {
        use metal2vulkan::metal::{PixelFormat, TessellationControlPointIndexType};
        use metal2vulkan::pipeline::{
            ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
            TessellationDescriptor,
        };

        const TESS: &[u8] =
            include_bytes!("../../../../tests/suite/fixtures/shaders/render_tess.metallib");
        const FRAG: &[u8] =
            include_bytes!("../../../../tests/suite/fixtures/shaders/frag.metallib");
        let p = platform();
        let d = dev();
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
        };
        let pipeline = |indices| {
            let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
                function(TESS, "render_tess").function_descriptor(),
            ));
            desc.fragment = Some(StageSource::Function(
                function(FRAG, "tri_frag").function_descriptor(),
            ));
            desc.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
            let mut tessellation = TessellationDescriptor::default();
            tessellation.control_point_index_type = indices;
            desc.tessellation = Some(tessellation);
            crate::gpu::resource::render_pipeline(
                d.raw
                    .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                    .unwrap(),
            )
        };
        let (plain, indexed) = (
            pipeline(TessellationControlPointIndexType::None),
            pipeline(TessellationControlPointIndexType::UInt32),
        );
        let points =
            buffer::<VulkanPlatform>(&d, 64, 0, &bytes(&[-3.0, -3.0, 3.0, -3.0, 0.0, 3.0]));
        let factors = buffer::<VulkanPlatform>(&d, 64, 0, &[0x00, 0x3c].repeat(4));
        let indices = buffer::<VulkanPlatform>(&d, 64, 0, &u32s(&[0, 1, 2]));
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let mut cmds = setters;
        for &(by_index, patch_count, instance_count) in draws {
            cmds.extend([
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(if by_index { &indexed } else { &plain }),
                },
                RenderCmd::SetVertexBuffer {
                    buffer: Some(&points),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetTessellationFactorBuffer {
                    buffer: Some(&factors),
                    offset: 0,
                    instance_stride: 0,
                },
            ]);
            cmds.push(if by_index {
                RenderCmd::DrawIndexedPatches {
                    number_of_patch_control_points: 3,
                    patch_start: 0,
                    patch_count,
                    patch_index_buffer: None,
                    patch_index_buffer_offset: 0,
                    control_point_index_buffer: &indices,
                    control_point_index_buffer_offset: 0,
                    instance_count,
                    base_instance: 0,
                }
            } else {
                RenderCmd::DrawPatches {
                    number_of_patch_control_points: 3,
                    patch_start: 0,
                    patch_count,
                    patch_index_buffer: None,
                    patch_index_buffer_offset: 0,
                    instance_count,
                    base_instance: 0,
                }
            });
        }
        cmds.push(RenderCmd::EndEncoding);
        for cmd in cmds {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let out = buffer::<VulkanPlatform>(&d, 64, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(4, 4, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: 16,
                destination_bytes_per_image: 64,
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
        // SAFETY: the Shared buffer's 64 bytes, read after the blit completed.
        unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 64) }.to_vec()
    }

    /// The selector of [`patch_texels`]'s plain patch draw.
    const DRAW_PATCHES: &str = "drawPatches:patchStart:patchCount:patchIndexBuffer:\
                                patchIndexBufferOffset:instanceCount:baseInstance:";

    /// The selector of [`patch_texels`]'s indexed patch draw.
    const DRAW_INDEXED_PATCHES: &str = "drawIndexedPatches:patchStart:patchCount:\
        patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:\
        controlPointIndexBufferOffset:instanceCount:baseInstance:";

    /// A patch draw of no instances or no patches, plain or indexed, draws nothing and the pass
    /// completes, as Metal's does, whatever the private setters left (a restart, a line width of
    /// 2 and triangles filled as lines). One of a patch and an instance dies at its selector
    /// naming `tessellation-factor-edge-cases` ([`crate::answer::DIES`]): the runtime announces
    /// the kind at every draw of a tessellation pipeline that draws a patch, whose factors it does
    /// not read, though these factors are 1
    #[test]
    fn empty_patch_draws_draw_nothing() {
        let setters = vec![
            RenderCmd::SetPrimitiveRestartEnabled {
                enabled: true,
                index: 0xffff,
            },
            RenderCmd::SetLineWidth { line_width: 2.0 },
            RenderCmd::SetTriangleFillMode { fill_mode: 1 },
        ];
        let empty = [(false, 1, 0), (false, 0, 1), (true, 1, 0), (true, 0, 1)];
        assert_eq!(patch_texels(setters, &empty), [0, 0, 0, 0xff].repeat(16));
        let kind = rt::Degradation::TessellationFactorEdgeCases;
        for (draw, at) in [
            ((false, 1, 1), DRAW_PATCHES),
            ((true, 1, 1), DRAW_INDEXED_PATCHES),
        ] {
            let site = crate::answer::dies_of(kind, || patch_texels(Vec::new(), &[draw]));
            assert_eq!(site.file, at, "{draw:?}");
        }
    }

    /// Indexed patches drawn after `setPrimitiveRestartEnabled:index:`, enabled or not, and
    /// without it, each die at the draw naming `tessellation-factor-edge-cases`
    /// ([`crate::answer::DIES`]), which the runtime announces at every draw of a tessellation
    /// pipeline that draws a patch, whose factors it does not read, though these factors are 1
    #[test]
    fn indexed_patches_draw_after_a_restart_set_as_without_it() {
        let kind = rt::Degradation::TessellationFactorEdgeCases;
        let site = crate::answer::dies_of(kind, || patch_texels(Vec::new(), &[(true, 1, 1)]));
        assert_eq!(site.file, DRAW_INDEXED_PATCHES);
        for index in [0, 1, 2] {
            for enabled in [false, true] {
                let set = vec![RenderCmd::SetPrimitiveRestartEnabled { enabled, index }];
                let site = crate::answer::dies_of(kind, || patch_texels(set, &[(true, 1, 1)]));
                assert_eq!(site.file, DRAW_INDEXED_PATCHES, "{enabled} {index}");
            }
        }
    }

    /// A render pass executes an indirect command buffer's draws: the triangle of a command's
    /// pipeline state covers the target in its fragment's colour; an execution past the buffer's
    /// size dies at its selector, naming the runtime's refusal
    #[test]
    fn a_pass_executes_an_icbs_draws() {
        use metal2vulkan::metal::PixelFormat;
        use metal2vulkan::pipeline::{
            ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
        };
        use reims_vgpu_core::platform::icb::{
            IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
        };
        use reims_vgpu_core::platform::values::IcbDesc;

        const VERT: &[u8] =
            include_bytes!("../../../../tests/suite/fixtures/shaders/vert.metallib");
        const FRAG: &[u8] =
            include_bytes!("../../../../tests/suite/fixtures/shaders/frag.metallib");
        let p = platform();
        let d = dev();
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
        };
        let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
            function(VERT, "tri_vert").function_descriptor(),
        ));
        desc.fragment = Some(StageSource::Function(
            function(FRAG, "tri_frag").function_descriptor(),
        ));
        desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        desc.support_indirect_command_buffers = true;
        let pipeline = crate::gpu::resource::render_pipeline(
            d.raw
                .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                .unwrap(),
        );
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    ..IcbDesc::default()
                },
                2,
                0,
            )
            .unwrap()
            .unwrap();
        let c = icb.render_command(1).unwrap().unwrap();
        for cmd in [
            IcbRenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&pipeline),
            },
            IcbRenderCmd::ClearBarrier,
            IcbRenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        ] {
            c.encode(cmd).unwrap();
        }
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::UseResources {
                resources: &[Some(ResourceRef::Icb(&icb))],
                usage: 1,
            },
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &icb,
                execution_range: NsRange::new(0, 2),
            },
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let site = catch_death(|| {
            e.encode(
                OnFault::Return,
                RenderCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: &icb,
                    execution_range: NsRange::new(1, 2),
                },
            )
        })
        .unwrap_err();
        assert_eq!(site.file, "executeCommandsInBuffer:withRange:");
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let out = buffer::<VulkanPlatform>(&d, 64, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(4, 4, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: 16,
                destination_bytes_per_image: 64,
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
        // SAFETY: the Shared buffer's 64 bytes, read after the blit completed.
        let texels = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 64) };
        assert_eq!(texels, [0x40, 0x80, 0xff, 0xff].repeat(16));
    }

    /// An execution whose commands run pipeline states of their own writes no texture bound on
    /// the encoder ([`Reach::at_slots`]): with a texture on a surface made read-only for the GPU
    /// bound at fragment texture 0, and made resident for reading, it goes on; with the texture
    /// made resident for writing, it dies at the execution
    #[test]
    fn an_execution_of_states_of_its_own_writes_no_bound_texture() {
        use metal2vulkan::metal::PixelFormat;
        use metal2vulkan::pipeline::{
            ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
        };
        use reims_vgpu_core::host::PhysicalMemoryRange;
        use reims_vgpu_core::platform::icb::{
            IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
        };
        use reims_vgpu_core::platform::os::surface::SurfaceSpec;
        use reims_vgpu_core::platform::os::vm::HostVm as _;
        use reims_vgpu_core::platform::values::IcbDesc;

        const VERT: &[u8] =
            include_bytes!("../../../../tests/suite/fixtures/shaders/vert.metallib");
        const FRAG: &[u8] =
            include_bytes!("../../../../tests/suite/fixtures/shaders/frag.metallib");
        let p = platform();
        let d = dev();
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
        };
        let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
            function(VERT, "tri_vert").function_descriptor(),
        ));
        desc.fragment = Some(StageSource::Function(
            function(FRAG, "tri_frag").function_descriptor(),
        ));
        desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        desc.support_indirect_command_buffers = true;
        let pipeline = crate::gpu::resource::render_pipeline(
            d.raw
                .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                .unwrap(),
        );
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap();
        let c = icb.render_command(0).unwrap().unwrap();
        for cmd in [
            IcbRenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&pipeline),
            },
            IcbRenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        ] {
            c.encode(cmd).unwrap();
        }
        let vm = crate::os::OsVm::new();
        let page = vm.page_size();
        let guest = crate::os::allocate_aliasable(page).unwrap();
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: page,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            client_direction: Some(2),
            client_address_ranges: Some(vec![PhysicalMemoryRange {
                physical_address: reims_vgpu_core::num::ToU64::to_u64(guest.addr()),
                physical_length: page,
            }]),
            ..SurfaceSpec::default()
        };
        let s = p.surfaces().create(&spec).unwrap();
        let read = p.texture_descriptor(&TextureDesc {
            pixel_format: BGRA8,
            width: 16,
            height: 16,
            usage: READ,
            ..TextureDesc::default()
        });
        let t = d
            .new_texture_on_surface(&read, Some(&s), 0)
            .unwrap()
            .unwrap();
        assert!(t.gpu_read_only());
        let execute = |usage: u64| {
            let (_dst, pass) = pass_of(&p, &d, (4, 4));
            let q = d.new_command_queue().unwrap();
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            let (listed, used) = (
                [Some(ResourceRef::Icb(&icb))],
                [Some(ResourceRef::Texture(&t))],
            );
            for cmd in [
                RenderCmd::SetFragmentTexture {
                    texture: Some(&t),
                    index: 0,
                },
                RenderCmd::UseResources {
                    resources: &listed,
                    usage: 1,
                },
                RenderCmd::UseResources {
                    resources: &used,
                    usage,
                },
                RenderCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: &icb,
                    execution_range: NsRange::new(0, 1),
                },
                RenderCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            cb.status().0
        };
        assert_eq!(execute(1), 4);
        let site = catch_death(|| execute(3)).unwrap_err();
        assert_eq!(site.file, "executeCommandsInBuffer:withRange:");
        drop((t, s));
        assert!(vm.deallocate(guest, page).is_success());
    }

    /// A pass of a mutable rate map not yet reset rasterizes nothing, a scissor rectangle set in
    /// it too: the colour and depth attachments keep their clears and the fragment function's
    /// write does not land, as on Metal; a tile dispatch in it dies. A reset before such a pass
    /// has run dies, where Metal's
    /// pass would read the reset map; once it has run the reset compiles the map, and an encoder
    /// of the same descriptor, which holds the map, draws by it
    #[test]
    fn a_pass_of_a_mutable_rate_map_not_yet_reset_rasterizes_nothing() {
        use metal2vulkan::metal::PixelFormat;
        use metal2vulkan::pipeline::{
            ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
        };
        use reims_vgpu_core::platform::rate_map::{
            RateMap as _, RateMapDescriptor as _, RateMapLayerDescriptor as _,
        };
        use reims_vgpu_core::platform::values::{
            DepthAttachmentSet, DepthStencilDesc, PassSet, RateMapDesc,
        };

        const VERT: &[u8] =
            include_bytes!("../../../../tests/suite/fixtures/shaders/vert.metallib");
        const FRAG: &[u8] =
            include_bytes!("../../../../tests/suite/fixtures/shaders/mutdec_fwrite.metallib");
        /// `MTLPixelFormatDepth32Float`, `MTLCompareFunctionAlways`.
        const DEPTH32F: u64 = 252;
        const ALWAYS: u64 = 7;
        const SIDE: u64 = 32;
        const TEXELS: usize = (SIDE * SIDE) as usize;
        let p = platform();
        let d = dev();
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
        };
        let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
            function(VERT, "tri_vert").function_descriptor(),
        ));
        desc.fragment = Some(StageSource::Function(
            function(FRAG, "mutdec_fwrite").function_descriptor(),
        ));
        desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        desc.depth_format = Some(PixelFormat::Depth32Float);
        let pipeline = crate::gpu::resource::render_pipeline(
            d.raw
                .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                .unwrap(),
        );
        let writes_depth = d
            .new_depth_stencil_state(&DepthStencilDesc {
                depth_compare_function: Some(ALWAYS),
                depth_write_enabled: Some(true),
                ..DepthStencilDesc::default()
            })
            .unwrap()
            .unwrap();
        let map_descriptor = |mutability| {
            let m = p.rate_map_descriptor(&RateMapDesc {
                screen_size: MtlSize::new(SIDE, SIDE, 1),
                mutability: Some(mutability),
                ..RateMapDesc::default()
            });
            let layer = p
                .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
                .unwrap()
                .unwrap();
            for at in [
                layer.horizontal_sample_storage(),
                layer.vertical_sample_storage(),
            ] {
                // SAFETY: the layer's storage holds one sample on each axis.
                unsafe { *at.as_ptr::<f32>() = 1.0 };
            }
            m.set_layer(OnFault::Die, &layer, 0).unwrap();
            m
        };
        let map = d.new_rate_map(&map_descriptor(1)).unwrap().unwrap();
        let (dst, pass) = pass_of(&p, &d, (SIDE, SIDE));
        let depth = texture(&p, &d, DEPTH32F, TARGET, (SIDE, SIDE), None);
        pass.set_depth_attachment(&DepthAttachmentSet {
            texture: Some(Some(&depth)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            clear_depth: Some(0.5),
            ..DepthAttachmentSet::default()
        });
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&map)),
            ..Default::default()
        });
        let q = d.new_command_queue().unwrap();
        let read = |cb: &VulkanCommandBuffer, texture: &VulkanTexture, out: &VulkanBuffer| {
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            for cmd in [
                BlitCmd::CopyTextureToBuffer {
                    source_texture: texture,
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::new(0, 0, 0),
                    source_size: MtlSize::new(SIDE, SIDE, 1),
                    destination_buffer: out,
                    destination_offset: 0,
                    destination_bytes_per_row: SIDE * 4,
                    destination_bytes_per_image: SIDE * SIDE * 4,
                },
                BlitCmd::EndEncoding,
            ] {
                blit.encode(OnFault::Die, cmd).unwrap();
            }
        };
        let words = |b: &VulkanBuffer| {
            // SAFETY: the Shared buffer's words, read after its command buffer completed.
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u32>(), TEXELS).to_vec() }
        };
        let run = || {
            let flag = buffer::<VulkanPlatform>(&d, 4, 0, &[0; 4]);
            let colors = buffer::<VulkanPlatform>(&d, SIDE * SIDE * 4, 0, &[]);
            let depths = buffer::<VulkanPlatform>(&d, SIDE * SIDE * 4, 0, &[]);
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            for cmd in [
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&pipeline),
                },
                RenderCmd::SetDepthStencilState {
                    depth_stencil_state: Some(&writes_depth),
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(&flag),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetScissorRect {
                    rect: MtlScissorRect {
                        x: 0,
                        y: 0,
                        width: SIDE,
                        height: SIDE,
                    },
                },
                RenderCmd::DrawPrimitives {
                    primitive_type: 3,
                    vertex_start: 0,
                    vertex_count: 3,
                },
                RenderCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            read(&cb, &dst, &colors);
            read(&cb, &depth, &depths);
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert_eq!(cb.error(), None);
            (words(&colors), words(&depths), words(&flag)[0])
        };
        let clear = u32::from_le_bytes([0, 0, 0, 0xff]);
        let green = u32::from_le_bytes([0, 0xff, 0, 0xff]);
        assert_eq!(
            run(),
            (vec![clear; TEXELS], vec![0.5f32.to_bits(); TEXELS], 0)
        );
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        // A tile dispatch in the pass, which the runtime's would run over every tile, dies.
        let site = catch_death(|| {
            e.encode(
                OnFault::Die,
                RenderCmd::DispatchThreadsPerTile {
                    threads_per_tile: MtlSize::new(1, 1, 1),
                },
            )
        })
        .unwrap_err();
        assert_eq!(site.file, "dispatchThreadsPerTile:");
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let site = catch_death(|| map.reset(&map_descriptor(2))).unwrap_err();
        assert_eq!(site.file, "resetUsingDescriptor:");
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        map.reset(&map_descriptor(2));
        assert_eq!(
            run(),
            (vec![green; TEXELS], vec![0.0f32.to_bits(); TEXELS], 1)
        );
    }

    /// A rate map descriptor of a `side` by `side` screen, of `mutability`, with one layer of
    /// one sample of 1 on each axis.
    fn rate_map_descriptor(
        p: &VulkanPlatform,
        side: u64,
        mutability: u64,
    ) -> crate::gpu::rate_map::VulkanRateMapDescriptor {
        use reims_vgpu_core::platform::rate_map::{
            RateMapDescriptor as _, RateMapLayerDescriptor as _,
        };
        use reims_vgpu_core::platform::values::RateMapDesc;

        let m = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(side, side, 1),
            mutability: Some(mutability),
            ..RateMapDesc::default()
        });
        let layer = p
            .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
            .unwrap()
            .unwrap();
        for at in [
            layer.horizontal_sample_storage(),
            layer.vertical_sample_storage(),
        ] {
            // SAFETY: the layer's storage holds one sample on each axis.
            unsafe { *at.as_ptr::<f32>() = 1.0 };
        }
        m.set_layer(OnFault::Die, &layer, 0).unwrap();
        m
    }

    /// A reset of a rate map dies while a committed pass that took it may still run, after core
    /// dropped its handles of the command buffer and the encoder, as it does right after the
    /// commit; once that command buffer completed, the reset is taken. A pass whose creation
    /// answers no encoder leaves the map free to reset
    #[test]
    fn a_reset_waits_for_a_committed_pass_core_let_go_of() {
        use std::sync::mpsc;

        use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
        use reims_vgpu_core::platform::rate_map::RateMap as _;
        use reims_vgpu_core::platform::values::PassSet;

        let p = platform();
        let d = dev();
        let map = d
            .new_rate_map(&rate_map_descriptor(&p, 32, 1))
            .unwrap()
            .unwrap();
        let (_dst, pass) = pass_of(&p, &d, (32, 32));
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&map)),
            ..Default::default()
        });
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let (tx, rx) = mpsc::channel();
        cb.add_completed_handler(OnFault::Die, Box::new(move |_| tx.send(()).unwrap()))
            .unwrap();
        cb.commit(OnFault::Die).unwrap();
        drop((e, cb));
        let site = catch_death(|| map.reset(&rate_map_descriptor(&p, 32, 2))).unwrap_err();
        assert_eq!(site.file, "resetUsingDescriptor:");
        event.set_signaled_value(1).unwrap();
        rx.recv().unwrap();
        map.reset(&rate_map_descriptor(&p, 32, 2));
        let empty = p.render_pass_descriptor();
        empty.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&map)),
            ..Default::default()
        });
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        assert!(cb.render_encoder(OnFault::Die, &empty).unwrap().is_none());
        map.reset(&rate_map_descriptor(&p, 32, 2));
    }

    /// A rate map whose rates vary is made and relayed (`rasterization-rate-map`), at an
    /// immutable map's creation and at a mutable map's reset, and a pass over it that only clears
    /// and stores runs; a draw and a tile dispatch in it die at their selectors, naming the kind.
    /// A map whose tiles span as many screen pixels as physical ones announces nothing, at its
    /// creation or a reset, and a draw in a pass over it is taken
    #[test]
    fn draws_over_a_map_of_varying_rates_die() {
        use reims_vgpu_core::platform::rate_map::{
            RateMap as _, RateMapDescriptor as _, RateMapLayerDescriptor as _,
        };
        use reims_vgpu_core::platform::values::{PassSet, RateMapDesc};
        use reims_vgpu_core::raise::death::DeathKind;

        use crate::gpu::rate_map::{VulkanRateMap, VulkanRateMapDescriptor};

        const SIDE: u64 = 256;
        /// `MTLMutabilityMutable`, `MTLMutabilityImmutable`.
        const MUTABLE: u64 = 1;
        const IMMUTABLE: u64 = 2;
        const VARYING: [f32; 2] = [0.25, 1.0];
        const IDENTITY: [f32; 2] = [1.0, 1.0];

        /// A descriptor of a `SIDE` by `SIDE` screen of `mutability`, with one layer of two
        /// samples on each axis, `rates`.
        fn of_rates(
            p: &VulkanPlatform,
            mutability: u64,
            rates: [f32; 2],
        ) -> VulkanRateMapDescriptor {
            let desc = p.rate_map_descriptor(&RateMapDesc {
                screen_size: MtlSize::new(SIDE, SIDE, 1),
                mutability: Some(mutability),
                ..RateMapDesc::default()
            });
            let layer = p
                .new_rate_map_layer(OnFault::Die, MtlSize::new(2, 2, 0))
                .unwrap()
                .unwrap();
            for at in [
                layer.horizontal_sample_storage(),
                layer.vertical_sample_storage(),
            ] {
                // SAFETY: the layer's storage holds two samples on each axis.
                unsafe { std::ptr::copy_nonoverlapping(rates.as_ptr(), at.as_ptr::<f32>(), 2) };
            }
            desc.set_layer(OnFault::Die, &layer, 0).unwrap();
            desc
        }

        /// How many times the runtime announced the kind in this process.
        fn announced() -> usize {
            (crate::answer::RELAYED.lock().iter())
                .filter(|&&k| k == rt::Degradation::RasterizationRateMap)
                .count()
        }

        let p = platform();
        let d = dev();
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let q = d.new_command_queue().unwrap();
        let draw = || RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        };
        // A pass over `map` running `cmds` after the state, committed and completed.
        let run = |map: &VulkanRateMap, cmds: Vec<RenderCmd<'_, VulkanPlatform>>| {
            let (_dst, pass) = pass_of(&p, &d, (SIDE, SIDE));
            pass.set_pass(&PassSet {
                rasterization_rate_map: Some(Some(map)),
                ..Default::default()
            });
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            let set = RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            };
            for cmd in std::iter::once(set).chain(cmds) {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert!(cb.error().is_none(), "{:?}", cb.error());
        };

        let before = announced();
        let identity = d
            .new_rate_map(&of_rates(&p, IMMUTABLE, IDENTITY))
            .unwrap()
            .unwrap();
        let reset = d
            .new_rate_map(&of_rates(&p, MUTABLE, IDENTITY))
            .unwrap()
            .unwrap();
        reset.reset(&of_rates(&p, IMMUTABLE, IDENTITY));
        assert_eq!(announced(), before, "an identity map announced");
        run(&identity, vec![draw()]);
        run(&reset, vec![draw()]);
        let made = d
            .new_rate_map(&of_rates(&p, IMMUTABLE, VARYING))
            .unwrap()
            .unwrap();
        assert_eq!(announced(), before + 1);
        reset.reset(&of_rates(&p, IMMUTABLE, VARYING));
        assert_eq!(announced(), before + 2);
        for map in [&made, &reset] {
            run(map, Vec::new());
            let (_dst, pass) = pass_of(&p, &d, (SIDE, SIDE));
            pass.set_pass(&PassSet {
                rasterization_rate_map: Some(Some(map)),
                ..Default::default()
            });
            let tile = RenderCmd::DispatchThreadsPerTile {
                threads_per_tile: MtlSize::new(1, 1, 1),
            };
            for cmd in [draw(), tile] {
                let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                e.encode(
                    OnFault::Die,
                    RenderCmd::SetRenderPipelineState {
                        pipeline_state: Some(&state),
                    },
                )
                .unwrap();
                *crate::answer::LAST_LINE.lock() = None;
                let sel = cmd.selector();
                let site = catch_death(|| e.encode(OnFault::Die, cmd)).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
                assert!(line.contains("(rasterization-rate-map)"), "{line}");
            }
        }
    }

    /// endEncoding in a thread of an isolated task that has ended stops before it touches the
    /// encoder, which stays in its handle, its store action left `Unknown` unset, as Metal's is
    /// left; outside a task the same pass ends, storing nothing for that attachment
    #[test]
    fn an_ended_tasks_end_encoding_leaves_the_encoder() {
        use reims_vgpu_core::tasks::isolation::TaskLife;

        /// `MTLStoreActionUnknown`.
        const UNKNOWN: u64 = 4;
        let p = platform();
        let d = dev();
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&dst)),
                store_action: Some(UNKNOWN),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let life = TaskLife::new();
        let stopped = life.run(|| {
            life.exit();
            e.encode(OnFault::Return, RenderCmd::EndEncoding)
        });
        assert!(stopped.is_none());
        assert!(
            e.encoder.raw.lock().is_some(),
            "the render encoder was taken"
        );
        assert_eq!(e.undecided.lock().len(), 1);
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        assert!(e.encoder.raw.lock().is_none());
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
    }

    /// An encoder's creation in a thread of an isolated task that ended, stopping at the send
    /// that empties the scissor rectangle of a pass that rasterizes nothing, leaves the runtime's
    /// encoder in its handle, open, as Metal's is left: the command buffer then has an encoder
    /// open, not one abandoned
    #[test]
    fn an_ended_tasks_encoder_creation_leaves_the_encoder_open() {
        use reims_vgpu_core::tasks::isolation::TaskLife;

        let p = platform();
        let d = dev();
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let desc = pass.descriptor();
        let made = without_unused_resolves(&desc);
        let raw = cb.raw.render_command_encoder(&made).unwrap();
        let life = TaskLife::new();
        let stopped = life.run(|| {
            life.exit();
            opened(&cb, raw, (&desc, made, Vec::new()), false, false)
        });
        assert!(stopped.is_none());
        let refused = cb.raw.blit_command_encoder().unwrap_err();
        assert!(
            matches!(&refused, rt::Error::Invalid(i) if i.problem == rt::Problem::EncoderOpen),
            "{refused:?}"
        );
    }

    /// A nil entry of useResources:count:usage: is skipped and the pass runs, as on Metal; one of
    /// useResources:count:usage:stages:, which nothing measured, dies at its selector
    #[test]
    fn a_nil_resource_is_skipped_where_metal_goes_on() {
        let d = dev();
        let (e, _dst) = open_encoder();
        let b = buffer::<VulkanPlatform>(&d, 16, 0, &[]);
        e.encode(
            OnFault::Die,
            RenderCmd::UseResources {
                resources: &[None, Some(ResourceRef::Buffer(&b)), None],
                usage: 1,
            },
        )
        .unwrap();
        e.encode(
            OnFault::Die,
            RenderCmd::UseResources {
                resources: &[None],
                usage: 1,
            },
        )
        .unwrap();
        let site = catch_death(|| {
            e.encode(
                OnFault::Die,
                RenderCmd::UseResourcesStages {
                    resources: &[None, Some(ResourceRef::Buffer(&b))],
                    usage: 1,
                    stages: 1,
                },
            )
        })
        .unwrap_err();
        assert_eq!(site.file, "useResources:count:usage:stages:");
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// A command after `endEncoding` dies at its selector
    #[test]
    fn a_command_after_end_encoding_dies() {
        let p = platform();
        let d = dev();
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let site = catch_death(|| {
            e.encode(
                OnFault::Die,
                RenderCmd::SetViewport {
                    viewport: MtlViewport {
                        origin_x: 0.0,
                        origin_y: 0.0,
                        width: 4.0,
                        height: 4.0,
                        znear: 0.0,
                        zfar: 1.0,
                    },
                },
            )
        })
        .unwrap_err();
        assert_eq!(site.file, "setViewport:");
    }

    #[cfg(target_os = "macos")]
    pub(crate) mod metal_tests;
}
