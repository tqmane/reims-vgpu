//! The compute decoder (encoder type 1): each segment that does not continue the previous one
//! opens a compute encoder from a compute pass descriptor, and each record is one command on it.
//!
//! # Contract
//!
//! - **The pass prefix (0xdb)**: a new segment's descriptor takes its dispatch type from the
//!   segment's first record when that record is 0xdb, which it then consumes. The first record
//!   is read and its length checked whatever its id, before the encoder exists, so a 1-7 byte tail
//!   raises where the record loop would stop; 0xdb past the first record is unrecognized
//!   (`tests::compute_pass_descriptor`).
//! - **The encoder** is made from the command buffer for every segment that does not continue the
//!   previous one. A carried decoder reused for a new segment leaks the encoder it had; no
//!   encoder raises before any record is read (`tests::decode::a_reopened_segment_leaks_the_old_encoder`).
//! - **Per record**: the record, then the check against the device's limit or support, then the
//!   lookups in record order (each list entry read, then looked up), then the one send. The
//!   stride setters check dynamic attribute stride support after the entries or the index
//!   (`tests::set_buffer_offset::with_stride_checks_index_then_support`). Counts and ranges are
//!   `u32` record fields widened to `u64`, compared with a wrapping sum.
//! - **Dispatches** need a pipeline set in this pass; an indirect dispatch sends a nil buffer to
//!   Metal as it is (`tests::dispatch_indirect::a_nil_buffer_is_passed_to_metal`).
//! - **The resource and heap lists** (0x86, 0x87, 0xd6) refuse a count past the record's bytes
//!   before reading any, and run under their own handler frame. UseResources (0x87) in a
//!   concurrent encoder (0xdb 1) raises for a reference whose entry is empty, where Metal would
//!   end the process at the nil (`ManagerLookup::get_resource_raising_empty`); every other
//!   failed lookup of the untyped lists is nil
//!   (`tests::decode::a_concurrent_pass_looks_resources_up_raising_empty`).
//! - **A jump end** (`encodeEndDoWhile`, `encodeEndWhile`, `encodeEndIf`) that answers NO only
//!   logs (`tests::jumps::an_end_returning_no_only_logs`).
//! - **The end of a segment** not continued by the next ends the encoder, which stays held, and
//!   drops the pipeline.
//! - **A raise** out of a segment faults the decoder (ends the encoder, drops it and the pipeline)
//!   and goes on; a raise from that `endEncoding` goes on in its place and leaves both held.

use std::fmt;
use std::sync::{Arc, Weak};

use crate::deserializer::records::{
    CmdDispatchThreadgroups, CmdDispatchThreadgroupsIndirect, CmdDispatchThreads,
    CmdDispatchThreadsIndirect, CmdEncodeEndDoWhile, CmdEncodeStartIf, CmdEncodeStartWhile,
    CmdExecuteCommandsInBuffer, CmdExecuteCommandsInBufferRanged, CmdFenceOp, CmdHeader,
    CmdMemoryBarrierResources, CmdMemoryBarrierScope, CmdSetBufferEntry,
    CmdSetBufferEntryWithStride, CmdSetBufferOffset, CmdSetBufferOffsetWithStride, CmdSetBuffers,
    CmdSetBuffersWithStride, CmdSetComputePassDescriptorDispatchType, CmdSetComputeStageInRegion,
    CmdSetComputeStageInRegionIndirect, CmdSetImageblockWidth, CmdSetRenderPipelineState,
    CmdSetSamplerStates, CmdSetSamplerStatesLODClamp, CmdSetSamplerStatesLODClampEntry,
    CmdSetTextures, CmdSetThreadgroupMemoryLength, CmdUseHeaps, CmdUseResources, SegmentHeader,
};
use crate::deserializer::walk::{RECORD_HEADER_SIZE, next_record};
use crate::deserializer::{Deserializer, RaiseKind, upgrade};
use crate::memory::cursor::Cursor;
use crate::num::ToUsize;
use crate::platform::blit::ResourceRef;
use crate::platform::caps::{DeviceCaps, Flags};
use crate::platform::command::CommandBuffer;
use crate::platform::compute::{ComputeCmd, ComputeEncoder};
use crate::platform::device::GpuDevice;
use crate::platform::icb::IndirectCommandBuffer;
use crate::platform::types::{MtlOrigin, MtlRegion, MtlSize, NsRange};
use crate::platform::values::ComputePassDesc;
use crate::platform::{Handle, Platform};
use crate::raise::{Kept, OnFault, Raise, RaiseResult, keep_forever};
use crate::resources::resource::AnyResource;
use crate::rt::Hold;

/// The command id that sets the pass's dispatch type, as a segment's first record only.
const SET_DISPATCH_TYPE: u32 = 0xdb;

/// The compute decoder's own raises: its limit, support and state checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeRaise {
    /// A dispatch with no pipeline set in this pass.
    NoPipeline,
    /// A buffer range past the device's compute buffer count.
    BufferCount,
    /// A sampler range past the device's compute sampler count.
    SamplerCount,
    /// A texture range past the device's compute texture count.
    TextureCount,
    /// A resource or heap list whose count is past the record's bytes.
    ResourceCount,
    /// A buffer index not below the device's compute buffer count.
    BufferIndex,
    /// A stride setter on a device without dynamic attribute stride.
    DynamicAttributeStride,
    /// A pipeline state reference of 0.
    PipelineReference,
    /// A stage-in region's indirect buffer found nothing.
    IndirectBuffer,
    /// A stage-in region's indirect buffer offset is not 4-byte aligned.
    IndirectBufferOffset,
    /// A threadgroup memory index not below the device's local memory size count.
    ThreadgroupMemoryIndex,
    /// An imageblock size on a device without image blocks.
    ImageBlocks,
    /// A jump record on a device without command buffer jumps.
    CommandBufferJump,
    /// An ICB execution range past the ICB's size.
    IcbRange,
    /// An indirect ICB execution's range buffer found nothing.
    RangeBuffer,
    /// The command buffer answered no compute encoder.
    NoEncoder,
}

impl ComputeRaise {
    /// The dotted id.
    pub fn name(self) -> &'static str {
        match self {
            ComputeRaise::NoPipeline => "deserializer.compute.no_pipeline",
            ComputeRaise::BufferCount => "deserializer.compute.buffer_count",
            ComputeRaise::SamplerCount => "deserializer.compute.sampler_count",
            ComputeRaise::TextureCount => "deserializer.compute.texture_count",
            ComputeRaise::ResourceCount => "deserializer.compute.resource_count",
            ComputeRaise::BufferIndex => "deserializer.compute.buffer_index",
            ComputeRaise::DynamicAttributeStride => "deserializer.compute.dynamic_attribute_stride",
            ComputeRaise::PipelineReference => "deserializer.compute.pipeline_reference",
            ComputeRaise::IndirectBuffer => "deserializer.compute.indirect_buffer",
            ComputeRaise::IndirectBufferOffset => "deserializer.compute.indirect_buffer_offset",
            ComputeRaise::ThreadgroupMemoryIndex => "deserializer.compute.threadgroup_memory_index",
            ComputeRaise::ImageBlocks => "deserializer.compute.image_blocks",
            ComputeRaise::CommandBufferJump => "deserializer.compute.command_buffer_jump",
            ComputeRaise::IcbRange => "deserializer.compute.icb_range",
            ComputeRaise::RangeBuffer => "deserializer.compute.range_buffer",
            ComputeRaise::NoEncoder => "deserializer.compute.no_encoder",
        }
    }
}

impl fmt::Display for ComputeRaise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// `raise!` of a compute raise.
macro_rules! refuse {
    ($hold:expr, $kind:ident, $detail:expr) => {
        crate::raise!(RaiseKind::Compute(ComputeRaise::$kind), $detail)
    };
}

/// A looked-up resource as the encoder takes it. The lookup answers a buffer, a texture or an
/// ICB (a heap resource raises in the lookup), so a heap is never here; it would go as nil.
fn resource_ref<P: Platform>(r: &Option<AnyResource<P>>) -> Option<ResourceRef<'_, P>> {
    match r {
        Some(AnyResource::Buffer(b)) => Some(ResourceRef::Buffer(b)),
        Some(AnyResource::Texture(t)) => Some(ResourceRef::Texture(t)),
        Some(AnyResource::Icb(i)) => Some(ResourceRef::Icb(i)),
        Some(AnyResource::Heap(_)) | None => None,
    }
}

/// Refuses a list of `count` 4-byte references that `cursor` does not hold (a wrapping
/// product).
fn check_list(cursor: &Cursor, count: u32) -> RaiseResult<()> {
    if cursor.remaining() < u64::from(count).wrapping_mul(4) {
        refuse!(
            hold,
            ResourceCount,
            format_args!("reference count {count} runs past the record")
        );
    }
    Ok(())
}

/// A resource or heap list's own handler frame, with no cleanup (an empty `@finally`): `body`,
/// then its raise goes on when the frame ends. Every list is decoded under its segment's handler
/// ([`ComputeDecoder::decode`]), so the raise is `Err`, the segment's to catch.
fn list_frame(body: impl FnOnce() -> RaiseResult<()>) -> RaiseResult<()> {
    body()
}

/// Reads `count` references from `cursor`, each looked up with `get` as it is read.
fn read_references<T, E>(
    cursor: &mut Cursor,
    count: u32,
    mut get: impl FnMut(u32) -> Result<T, E>,
) -> RaiseResult<Vec<T>>
where
    Raise: From<E>,
{
    let mut out = Vec::with_capacity(count.to_usize());
    for _ in 0..count {
        let reference: u32 = cursor.read_or_raise(OnFault::Return)?;
        out.push(get(reference)?);
    }
    Ok(out)
}

/// The decoder of one compute segment, and of the segments it is carried into.
///
/// Its fields drop in declaration order: the encoder, the command buffer, then the pipeline.
pub struct ComputeDecoder<P: Platform> {
    /// The deserializer, weakly (the walk's contract).
    deserializer: Weak<Deserializer<P>>,
    /// The pass's encoder: made per new segment, taken by a fault.
    encoder: Option<P::ComputeEncoder>,
    /// The command buffer the encoders are made from; `None` makes none.
    command_buffer: Option<P::CommandBuffer>,
    /// The pipeline set in this pass: dropped at the pass's end and by a fault.
    pipeline: Option<P::ComputePipeline>,
    /// Whether the encoder's pass asked for concurrent dispatch (0xdb 1), set with each new
    /// segment's encoder.
    concurrent: bool,
}

impl<P: Platform> ComputeDecoder<P> {
    /// A decoder over `command_buffer`, with no encoder yet.
    pub fn new(
        deserializer: Weak<Deserializer<P>>,
        command_buffer: Option<P::CommandBuffer>,
    ) -> Self {
        ComputeDecoder {
            deserializer,
            encoder: None,
            command_buffer,
            pipeline: None,
            concurrent: false,
        }
    }

    /// The deserializer, for one record.
    fn deserializer(&self) -> Arc<Deserializer<P>> {
        upgrade(&self.deserializer)
    }

    /// Sends `cmd` to the encoder; nothing without one.
    fn send(&self, cmd: ComputeCmd<'_, P>) -> RaiseResult<()> {
        match &self.encoder {
            Some(encoder) => Ok(encoder.encode(OnFault::Return, cmd)?),
            None => Ok(()),
        }
    }

    /// A limit from the device's capabilities; 0 for a deserializer without a device.
    fn device_limit(&self, limit: impl FnOnce(&DeviceCaps) -> u64) -> u64 {
        self.deserializer()
            .device()
            .map_or(0, |device| limit(&device.caps()))
    }

    /// A flag from the device's capabilities; false for a deserializer without a device.
    fn device_supports(&self, flag: impl FnOnce(&Flags) -> bool) -> bool {
        self.deserializer()
            .device()
            .is_some_and(|device| flag(&device.caps().flags))
    }

    /// Decodes one segment. A raise out of it faults the decoder and goes on; a raise from the
    /// fault goes on in its place, and the segment's ends there.
    pub fn decode(
        &mut self,
        hold: Hold<'_>,
        header: &SegmentHeader,
        mut segment: Cursor,
    ) -> RaiseResult<()> {
        let mut decoded = self.decode_segment(hold, header, &mut segment);
        if decoded.is_err() {
            hold.cleanup_step(&mut decoded, self.fault())?;
        }
        decoded
    }

    /// Ends the encoder and drops it and the pipeline; a second fault has no encoder to end. A
    /// raise from `endEncoding` goes on and leaves both held.
    pub fn fault(&mut self) -> RaiseResult<()> {
        self.send(ComputeCmd::EndEncoding)?;
        self.encoder = None;
        self.pipeline = None;
        Ok(())
    }

    fn decode_segment(
        &mut self,
        hold: Hold<'_>,
        header: &SegmentHeader,
        segment: &mut Cursor,
    ) -> RaiseResult<()> {
        if header.continues_previous == 0 {
            let descriptor = self.decode_compute_pass_descriptor(segment)?;
            self.concurrent = descriptor.dispatch_type == Some(1);
            let encoder = match &self.command_buffer {
                Some(buffer) => buffer.compute_encoder_for_pass(&descriptor)?,
                None => None,
            };
            // A carried decoder reused for a new segment replaces its encoder without ending or
            // releasing it: it is kept for good, with its command buffer and what that holds.
            if let Some(old) = std::mem::replace(&mut self.encoder, encoder) {
                keep_forever(old, Kept::MetalNumbering);
            }
        }
        if self.encoder.is_none() {
            refuse!(hold, NoEncoder, "no compute encoder");
        }
        while let Some((record_header, mut record)) = next_record(segment)? {
            let c = &mut record;
            match record_header.command_id {
                0xc8 => self.decode_dispatch_threadgroups(c)?,
                0xc9 => self.decode_dispatch_threadgroups_indirect(hold, c)?,
                0xca => self.decode_dispatch_threads(c)?,
                0xcb => self.decode_set_buffers(hold, c)?,
                0xcc => self.decode_set_samplers(hold, c)?,
                0xcd => self.decode_set_samplers_lod_clamp(hold, c)?,
                0xce => self.decode_set_textures(hold, c)?,
                0xcf => self.decode_set_buffer_offset(c)?,
                0xd0 => self.decode_set_pipeline_state(hold, c)?,
                0xd1 => self.decode_set_stage_in_region(c)?,
                0xd2 => self.decode_set_stage_in_region_indirect(hold, c)?,
                0xd3 => self.decode_set_threadgroup_memory_length(c)?,
                0xd4 => self.decode_update_fence(hold, c)?,
                0xd5 => self.decode_wait_for_fence(hold, c)?,
                0xd6 => self.decode_barrier_resources(hold, c)?,
                0xd7 => self.decode_barrier_scope(c)?,
                0xd8 => self.decode_set_imageblock_width(c)?,
                0xd9 => self.decode_set_buffers_with_stride(hold, c)?,
                0xda => self.decode_set_buffer_offset_with_stride(c)?,
                0xdc => self.decode_encode_start_do_while(c)?,
                0xdd => self.decode_encode_end_do_while(hold, c)?,
                0xde => self.decode_encode_start_while(hold, c)?,
                0xdf => self.decode_encode_end_while(c)?,
                0xe0 => self.decode_encode_start_if(hold, c)?,
                0xe1 => self.decode_encode_start_else(c)?,
                0xe2 => self.decode_encode_end_if(c)?,
                0xe3 => self.decode_insert_compressed_texture_reinterpretation_flush(c)?,
                0xe4 => self.decode_execute_command_in_buffer_ranged(hold, c)?,
                0xe5 => self.decode_execute_command_in_buffer(hold, c)?,
                0xe6 => self.decode_dispatch_threads_indirect(hold, c)?,
                0x86 => self.decode_use_heaps(hold, c)?,
                0x87 => self.decode_use_resources(hold, c)?,
                // 0xdb past the first record lands here.
                id => crate::raise!(
                    RaiseKind::UnrecognizedCommand,
                    format_args!("unknown command {id}")
                ),
            }
        }
        if header.continued_by_next == 0 {
            // The encoder is ended, not released or dropped.
            self.send(ComputeCmd::EndEncoding)?;
            self.pipeline = None;
        }
        Ok(())
    }

    /// The pass descriptor of a new segment.
    ///
    /// Only a first record 0xdb sets the dispatch type, and the first record is read and its length
    /// checked (a 1-7 byte tail raises) before the encoder exists.
    fn decode_compute_pass_descriptor(&self, segment: &mut Cursor) -> RaiseResult<ComputePassDesc> {
        let mut descriptor = ComputePassDesc::default();
        if segment.remaining() != 0 {
            // A peek: the segment moves only past a consumed 0xdb.
            let mut peek = segment.clone();
            let header: CmdHeader = peek.read_or_raise(OnFault::Return)?;
            let Some(length) = u64::from(header.length).checked_sub(RECORD_HEADER_SIZE) else {
                crate::raise!(
                    RaiseKind::RecordUnderflow,
                    format_args!("record length {} is shorter than its header", header.length)
                );
            };
            let mut record = match peek.sub_cursor(length) {
                Ok(record) => record,
                Err(error) => crate::raise!(
                    RaiseKind::RecordOverrun,
                    format_args!("record not readable: {error}")
                ),
            };
            if header.command_id == SET_DISPATCH_TYPE {
                segment.advance(u64::from(header.length));
                let command: CmdSetComputePassDescriptorDispatchType =
                    record.read_or_raise(OnFault::Return)?;
                descriptor.dispatch_type = Some(u64::from(command.dispatch_type));
            }
        }
        Ok(descriptor)
    }

    /// Refuses a dispatch with no pipeline set.
    fn require_pipeline(&self) -> RaiseResult<()> {
        if self.pipeline.is_none() {
            refuse!(hold, NoPipeline, "dispatch without a pipeline state");
        }
        Ok(())
    }

    fn decode_dispatch_threadgroups(&mut self, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDispatchThreadgroups = cursor.read_or_raise(OnFault::Return)?;
        self.require_pipeline()?;
        self.send(ComputeCmd::DispatchThreadgroups {
            threadgroups_per_grid: MtlSize::from(command.threadgroups_per_grid),
            threads_per_threadgroup: MtlSize::from(command.threads_per_threadgroup),
        })
    }

    fn decode_dispatch_threadgroups_indirect(
        &mut self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDispatchThreadgroupsIndirect = cursor.read_or_raise(OnFault::Return)?;
        self.require_pipeline()?;
        let buffer = self
            .deserializer()
            .object_delegate()
            .get_buffer(hold, command.indirect_buffer_reference)?;
        // A nil buffer goes to Metal, which crashes.
        self.send(ComputeCmd::DispatchThreadgroupsIndirect {
            indirect_buffer: buffer.as_ref(),
            indirect_buffer_offset: command.indirect_buffer_offset,
            threads_per_threadgroup: MtlSize::from(command.threads_per_threadgroup),
        })
    }

    fn decode_dispatch_threads(&mut self, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDispatchThreads = cursor.read_or_raise(OnFault::Return)?;
        self.require_pipeline()?;
        self.send(ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::from(command.threads_per_grid),
            threads_per_threadgroup: MtlSize::from(command.threads_per_threadgroup),
        })
    }

    fn decode_dispatch_threads_indirect(
        &mut self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDispatchThreadsIndirect = cursor.read_or_raise(OnFault::Return)?;
        self.require_pipeline()?;
        let buffer = self
            .deserializer()
            .object_delegate()
            .get_buffer(hold, command.indirect_buffer_reference)?;
        // A nil buffer goes to Metal, which crashes.
        self.send(ComputeCmd::DispatchThreadsIndirect {
            indirect_buffer: buffer.as_ref(),
            indirect_buffer_offset: command.indirect_buffer_offset,
        })
    }

    fn decode_set_buffers(&mut self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetBuffers = cursor.read_or_raise(OnFault::Return)?;
        let range = NsRange::new(
            u64::from(command.range_location),
            u64::from(command.range_length),
        );
        let max = self.device_limit(|c| c.limits.max_compute_buffers);
        if range.location.wrapping_add(range.length) > max {
            refuse!(
                hold,
                BufferCount,
                format_args!(
                    "buffers {}+{} are past the device's {max}",
                    range.location, range.length
                )
            );
        }
        let count = command.range_length.to_usize();
        let mut buffers = Vec::with_capacity(count);
        let mut offsets = Vec::with_capacity(count);
        for _ in 0..command.range_length {
            let entry: CmdSetBufferEntry = cursor.read_or_raise(OnFault::Return)?;
            buffers.push(
                self.deserializer()
                    .object_delegate()
                    .get_buffer(hold, entry.buffer_reference)?,
            );
            offsets.push(entry.offset);
        }
        let buffers: Vec<Option<&P::Buffer>> = buffers.iter().map(Option::as_ref).collect();
        self.send(ComputeCmd::SetBuffers {
            buffers: &buffers,
            offsets: &offsets,
            range,
        })
    }

    fn decode_set_buffers_with_stride(
        &mut self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetBuffersWithStride = cursor.read_or_raise(OnFault::Return)?;
        let range = NsRange::new(
            u64::from(command.range_location),
            u64::from(command.range_length),
        );
        let max = self.device_limit(|c| c.limits.max_compute_buffers);
        if range.location.wrapping_add(range.length) > max {
            refuse!(
                hold,
                BufferCount,
                format_args!(
                    "buffers {}+{} are past the device's {max}",
                    range.location, range.length
                )
            );
        }
        let count = command.range_length.to_usize();
        let mut buffers = Vec::with_capacity(count);
        let mut offsets = Vec::with_capacity(count);
        let mut strides = Vec::with_capacity(count);
        for _ in 0..command.range_length {
            let entry: CmdSetBufferEntryWithStride = cursor.read_or_raise(OnFault::Return)?;
            buffers.push(
                self.deserializer()
                    .object_delegate()
                    .get_buffer(hold, entry.buffer_reference)?,
            );
            offsets.push(entry.offset);
            strides.push(entry.attribute_stride);
        }
        // Support is checked after every entry is read and looked up.
        if !self.device_supports(|f| f.supports_dynamic_attribute_stride) {
            refuse!(
                hold,
                DynamicAttributeStride,
                "the device does not support dynamic attribute strides"
            );
        }
        let buffers: Vec<Option<&P::Buffer>> = buffers.iter().map(Option::as_ref).collect();
        self.send(ComputeCmd::SetBuffersAttributeStrides {
            buffers: &buffers,
            offsets: &offsets,
            strides: &strides,
            range,
        })
    }

    fn decode_set_samplers(&mut self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetSamplerStates = cursor.read_or_raise(OnFault::Return)?;
        let range = NsRange::new(
            u64::from(command.range_location),
            u64::from(command.range_length),
        );
        let max = self.device_limit(|c| c.limits.max_compute_samplers);
        if range.location.wrapping_add(range.length) > max {
            refuse!(
                hold,
                SamplerCount,
                format_args!(
                    "samplers {}+{} are past the device's {max}",
                    range.location, range.length
                )
            );
        }
        let mut samplers = Vec::with_capacity(command.range_length.to_usize());
        for _ in 0..command.range_length {
            let reference: u32 = cursor.read_or_raise(OnFault::Return)?;
            samplers.push(
                self.deserializer()
                    .object_delegate()
                    .get_sampler_state(hold, reference)?,
            );
        }
        let samplers: Vec<Option<&P::SamplerState>> = samplers.iter().map(Option::as_ref).collect();
        self.send(ComputeCmd::SetSamplerStates {
            samplers: &samplers,
            range,
        })
    }

    fn decode_set_samplers_lod_clamp(
        &mut self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetSamplerStatesLODClamp = cursor.read_or_raise(OnFault::Return)?;
        let range = NsRange::new(
            u64::from(command.range_location),
            u64::from(command.range_length),
        );
        let max = self.device_limit(|c| c.limits.max_compute_samplers);
        if range.location.wrapping_add(range.length) > max {
            refuse!(
                hold,
                SamplerCount,
                format_args!(
                    "samplers {}+{} are past the device's {max}",
                    range.location, range.length
                )
            );
        }
        let count = command.range_length.to_usize();
        let mut samplers = Vec::with_capacity(count);
        let mut lod_min_clamps = Vec::with_capacity(count);
        let mut lod_max_clamps = Vec::with_capacity(count);
        for _ in 0..command.range_length {
            let entry: CmdSetSamplerStatesLODClampEntry = cursor.read_or_raise(OnFault::Return)?;
            samplers.push(
                self.deserializer()
                    .object_delegate()
                    .get_sampler_state(hold, entry.sampler_state_reference)?,
            );
            lod_min_clamps.push(entry.lod_min_clamp);
            lod_max_clamps.push(entry.lod_max_clamp);
        }
        let samplers: Vec<Option<&P::SamplerState>> = samplers.iter().map(Option::as_ref).collect();
        self.send(ComputeCmd::SetSamplerStatesLodClamps {
            samplers: &samplers,
            lod_min_clamps: &lod_min_clamps,
            lod_max_clamps: &lod_max_clamps,
            range,
        })
    }

    fn decode_set_textures(&mut self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetTextures = cursor.read_or_raise(OnFault::Return)?;
        let range = NsRange::new(
            u64::from(command.range_location),
            u64::from(command.range_length),
        );
        let max = self.device_limit(|c| c.limits.max_compute_textures);
        if range.location.wrapping_add(range.length) > max {
            refuse!(
                hold,
                TextureCount,
                format_args!(
                    "textures {}+{} are past the device's {max}",
                    range.location, range.length
                )
            );
        }
        let mut textures = Vec::with_capacity(command.range_length.to_usize());
        for _ in 0..command.range_length {
            let reference: u32 = cursor.read_or_raise(OnFault::Return)?;
            textures.push(
                self.deserializer()
                    .object_delegate()
                    .get_texture(hold, reference)?,
            );
        }
        let textures: Vec<Option<&P::Texture>> = textures.iter().map(Option::as_ref).collect();
        self.send(ComputeCmd::SetTextures {
            textures: &textures,
            range,
        })
    }

    fn decode_use_resources(&mut self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let this = &*self;
        // The list's own handler frame: a raise inside goes on when it ends.
        list_frame(|| {
            let command: CmdUseResources = cursor.read_or_raise(OnFault::Return)?;
            check_list(cursor, command.count)?;
            let deserializer = this.deserializer();
            let lookup = deserializer.object_delegate();
            let resources = read_references(cursor, command.count, |r| {
                if this.concurrent {
                    lookup.get_resource_raising_empty(hold, r)
                } else {
                    lookup.get_resource(hold, r)
                }
            })?;
            let refs: Vec<Option<ResourceRef<'_, P>>> =
                resources.iter().map(resource_ref).collect();
            this.send(ComputeCmd::UseResources {
                resources: &refs,
                usage: u64::from(command.usage),
            })
        })
    }

    fn decode_use_heaps(&mut self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let this = &*self;
        // The list's own handler frame: a raise inside goes on when it ends.
        list_frame(|| {
            let command: CmdUseHeaps = cursor.read_or_raise(OnFault::Return)?;
            check_list(cursor, command.count)?;
            let deserializer = this.deserializer();
            let heaps = read_references(cursor, command.count, |r| {
                deserializer.object_delegate().get_heap(hold, r)
            })?;
            let refs: Vec<Option<&P::Heap>> = heaps.iter().map(Option::as_ref).collect();
            this.send(ComputeCmd::UseHeaps { heaps: &refs })
        })
    }

    fn decode_barrier_resources(&mut self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let this = &*self;
        // The list's own handler frame: a raise inside goes on when it ends.
        list_frame(|| {
            let command: CmdMemoryBarrierResources = cursor.read_or_raise(OnFault::Return)?;
            check_list(cursor, command.count)?;
            let deserializer = this.deserializer();
            let resources = read_references(cursor, command.count, |r| {
                deserializer.object_delegate().get_resource(hold, r)
            })?;
            let refs: Vec<Option<ResourceRef<'_, P>>> =
                resources.iter().map(resource_ref).collect();
            this.send(ComputeCmd::MemoryBarrierWithResources { resources: &refs })
        })
    }

    fn decode_set_buffer_offset(&mut self, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetBufferOffset = cursor.read_or_raise(OnFault::Return)?;
        let index = u64::from(command.index);
        if index >= self.device_limit(|c| c.limits.max_compute_buffers) {
            refuse!(
                hold,
                BufferIndex,
                format_args!("buffer index {index} is past the device's limit")
            );
        }
        self.send(ComputeCmd::SetBufferOffset {
            offset: command.offset,
            index,
        })
    }

    fn decode_set_buffer_offset_with_stride(&mut self, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetBufferOffsetWithStride = cursor.read_or_raise(OnFault::Return)?;
        let index = u64::from(command.index);
        if index >= self.device_limit(|c| c.limits.max_compute_buffers) {
            refuse!(
                hold,
                BufferIndex,
                format_args!("buffer index {index} is past the device's limit")
            );
        }
        // Support is checked after the index.
        if !self.device_supports(|f| f.supports_dynamic_attribute_stride) {
            refuse!(
                hold,
                DynamicAttributeStride,
                "the device does not support dynamic attribute strides"
            );
        }
        self.send(ComputeCmd::SetBufferOffsetAttributeStride {
            offset: command.offset,
            stride: command.attribute_stride,
            index,
        })
    }

    fn decode_set_pipeline_state(
        &mut self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        // The render record's shape (4 bytes).
        let command: CmdSetRenderPipelineState = cursor.read_or_raise(OnFault::Return)?;
        if command.pipeline_state_reference == 0 {
            refuse!(
                hold,
                PipelineReference,
                "compute pipeline state reference 0"
            );
        }
        // The nullable lookup: nil is sent, and held, as nil.
        let pipeline = self
            .deserializer()
            .object_delegate()
            .get_compute_pipeline_state(hold, command.pipeline_state_reference)?;
        self.send(ComputeCmd::SetComputePipelineState {
            state: pipeline.as_ref(),
        })?;
        if pipeline.as_ref().map(Handle::id) != self.pipeline.as_ref().map(Handle::id) {
            self.pipeline = pipeline.as_ref().map(Handle::clone_ref);
        }
        Ok(())
    }

    fn decode_set_stage_in_region(&mut self, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetComputeStageInRegion = cursor.read_or_raise(OnFault::Return)?;
        // The record's size comes before its origin.
        let region = MtlRegion::new(
            MtlOrigin::from(command.region_origin),
            MtlSize::from(command.region_size),
        );
        self.send(ComputeCmd::SetStageInRegion { region })
    }

    fn decode_set_stage_in_region_indirect(
        &mut self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdSetComputeStageInRegionIndirect = cursor.read_or_raise(OnFault::Return)?;
        let buffer = self
            .deserializer()
            .object_delegate()
            .get_buffer(hold, command.indirect_buffer_reference)?;
        let Some(buffer) = buffer else {
            refuse!(
                hold,
                IndirectBuffer,
                format_args!("no indirect buffer for reference {}", {
                    command.indirect_buffer_reference
                })
            );
        };
        let offset = command.indirect_buffer_offset;
        if offset & 3 != 0 {
            refuse!(
                hold,
                IndirectBufferOffset,
                format_args!("indirect buffer offset {offset:#x} is not a multiple of 4")
            );
        }
        self.send(ComputeCmd::SetStageInRegionIndirect {
            indirect_buffer: &buffer,
            indirect_buffer_offset: offset,
        })
    }

    fn decode_set_threadgroup_memory_length(&mut self, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetThreadgroupMemoryLength = cursor.read_or_raise(OnFault::Return)?;
        let index = u64::from(command.index);
        if index >= self.device_limit(|c| c.limits.max_compute_local_memory_sizes) {
            refuse!(
                hold,
                ThreadgroupMemoryIndex,
                format_args!("threadgroup memory index {index} is past the device's limit")
            );
        }
        self.send(ComputeCmd::SetThreadgroupMemoryLength {
            length: command.length,
            index,
        })
    }

    fn decode_update_fence(&mut self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdFenceOp = cursor.read_or_raise(OnFault::Return)?;
        let fence = self
            .deserializer()
            .require_fence(hold, command.fence_reference)?;
        self.send(ComputeCmd::UpdateFence { fence: &fence })
    }

    fn decode_wait_for_fence(&mut self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdFenceOp = cursor.read_or_raise(OnFault::Return)?;
        let fence = self
            .deserializer()
            .require_fence(hold, command.fence_reference)?;
        self.send(ComputeCmd::WaitForFence { fence: &fence })
    }

    fn decode_barrier_scope(&mut self, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdMemoryBarrierScope = cursor.read_or_raise(OnFault::Return)?;
        // The low 16 bits, zero-extended.
        self.send(ComputeCmd::MemoryBarrierWithScope {
            scope: u64::from(command.scope()),
        })
    }

    fn decode_set_imageblock_width(&mut self, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdSetImageblockWidth = cursor.read_or_raise(OnFault::Return)?;
        if !self.device_supports(|f| f.supports_image_blocks) {
            refuse!(hold, ImageBlocks, "the device does not support imageblocks");
        }
        self.send(ComputeCmd::SetImageblockWidth {
            width: u64::from(command.width),
            height: u64::from(command.height),
        })
    }

    /// Refuses a jump record on a device without command buffer jumps.
    fn require_command_buffer_jump(&self) -> RaiseResult<()> {
        if !self.device_supports(|f| f.supports_command_buffer_jump) {
            refuse!(
                hold,
                CommandBufferJump,
                "the device does not support command buffer jumps"
            );
        }
        Ok(())
    }

    /// The payload is not read.
    fn decode_encode_start_do_while(&mut self, _cursor: &mut Cursor) -> RaiseResult<()> {
        self.require_command_buffer_jump()?;
        self.send(ComputeCmd::EncodeStartDoWhile)
    }

    fn decode_encode_end_do_while(
        &mut self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdEncodeEndDoWhile = cursor.read_or_raise(OnFault::Return)?;
        self.require_command_buffer_jump()?;
        let buffer = self
            .deserializer()
            .object_delegate()
            .get_buffer(hold, command.buffer_reference)?;
        let ended = match &self.encoder {
            Some(encoder) => encoder.encode_end_do_while(
                buffer.as_ref(),
                command.offset,
                u64::from(command.comparison),
                command.reference_value,
            )?,
            None => false,
        };
        if !ended {
            // A refusal is only a warning. The deserializer is still looked up for it, whether or
            // not the line is written: a gone one dies here.
            let _deserializer = self.deserializer();
            log::warn!("the compute encoder refused to end the do-while block");
        }
        Ok(())
    }

    fn decode_encode_start_while(
        &mut self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdEncodeStartWhile = cursor.read_or_raise(OnFault::Return)?;
        self.require_command_buffer_jump()?;
        let buffer = self
            .deserializer()
            .object_delegate()
            .get_buffer(hold, command.buffer_reference)?;
        self.send(ComputeCmd::EncodeStartWhile {
            buffer: buffer.as_ref(),
            offset: command.offset,
            comparison: u64::from(command.comparison),
            reference_value: command.reference_value,
        })
    }

    /// The payload is not read.
    fn decode_encode_end_while(&mut self, _cursor: &mut Cursor) -> RaiseResult<()> {
        self.require_command_buffer_jump()?;
        let ended = match &self.encoder {
            Some(encoder) => encoder.encode_end_while()?,
            None => false,
        };
        if !ended {
            // A refusal is only a warning. The deserializer is still looked up for it, whether or
            // not the line is written: a gone one dies here.
            let _deserializer = self.deserializer();
            log::warn!("the compute encoder refused to end the while block");
        }
        Ok(())
    }

    fn decode_encode_start_if(&mut self, hold: Hold<'_>, cursor: &mut Cursor) -> RaiseResult<()> {
        let command: CmdEncodeStartIf = cursor.read_or_raise(OnFault::Return)?;
        self.require_command_buffer_jump()?;
        let buffer = self
            .deserializer()
            .object_delegate()
            .get_buffer(hold, command.buffer_reference)?;
        self.send(ComputeCmd::EncodeStartIf {
            buffer: buffer.as_ref(),
            offset: command.offset,
            comparison: u64::from(command.comparison),
            reference_value: command.reference_value,
        })
    }

    /// The payload is not read.
    fn decode_encode_start_else(&mut self, _cursor: &mut Cursor) -> RaiseResult<()> {
        self.require_command_buffer_jump()?;
        self.send(ComputeCmd::EncodeStartElse)
    }

    /// The payload is not read.
    fn decode_encode_end_if(&mut self, _cursor: &mut Cursor) -> RaiseResult<()> {
        self.require_command_buffer_jump()?;
        let ended = match &self.encoder {
            Some(encoder) => encoder.encode_end_if()?,
            None => false,
        };
        if !ended {
            // A refusal is only a warning. The deserializer is still looked up for it, whether or
            // not the line is written: a gone one dies here.
            let _deserializer = self.deserializer();
            log::warn!("the compute encoder refused to end the if block");
        }
        Ok(())
    }

    /// No read, no check.
    fn decode_insert_compressed_texture_reinterpretation_flush(
        &mut self,
        _cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        self.send(ComputeCmd::CompressedTextureReinterpretationFlush)
    }

    fn decode_execute_command_in_buffer_ranged(
        &mut self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdExecuteCommandsInBufferRanged = cursor.read_or_raise(OnFault::Return)?;
        let icb = self
            .deserializer()
            .require_indirect_command_buffer(hold, command.indirect_command_buffer_reference)?;
        let range = NsRange::new(command.range_location, command.range_length);
        let size = icb.size();
        let (end, wrapped) = range.location.overflowing_add(range.length);
        if wrapped || end > size {
            refuse!(
                hold,
                IcbRange,
                format_args!(
                    "range {}+{} runs past the indirect command buffer's {size} commands",
                    range.location, range.length
                )
            );
        }
        self.send(ComputeCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &icb,
            execution_range: range,
        })
    }

    fn decode_execute_command_in_buffer(
        &mut self,
        hold: Hold<'_>,
        cursor: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdExecuteCommandsInBuffer = cursor.read_or_raise(OnFault::Return)?;
        let icb_reference = command.indirect_command_buffer_reference;
        let icb = self
            .deserializer()
            .require_indirect_command_buffer(hold, icb_reference)?;
        let range_buffer = self
            .deserializer()
            .object_delegate()
            .get_buffer(hold, command.indirect_range_buffer_reference)?;
        let Some(range_buffer) = range_buffer else {
            refuse!(
                hold,
                RangeBuffer,
                format_args!(
                    "no indirect range buffer for reference {}",
                    command.indirect_range_buffer_reference
                )
            );
        };
        self.send(ComputeCmd::ExecuteCommandsIndirect {
            indirect_commandbuffer: &icb,
            indirect_range_buffer: &range_buffer,
            indirect_buffer_offset: command.indirect_buffer_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::deserializer::blit::test_support::platform_site;
    use crate::deserializer::records::{MtlOriginRecord, MtlSizeRecord};
    use crate::deserializer::test_lookups::Lookups;
    use crate::host::HostPtr;
    use crate::platform::command::CommandQueue;
    use crate::raise::RaiseId;
    use crate::raise::death::DeathKind;
    use crate::raise::death::DeathSite;
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::Model;
    use crate::testkit::{Arg, FakePlatform, Fk, Step, kind};
    use zerocopy::IntoBytes;

    struct T {
        p: Arc<FakePlatform>,
        rec: Arc<Lookups>,
        cb: Fk<kind::CommandBuffer>,
        d: ComputeDecoder<FakePlatform>,
        /// The deserializer the decoder reaches unretained: the caller's reference.
        de: Arc<Deserializer<FakePlatform>>,
    }

    fn setup() -> T {
        let p = Arc::new(FakePlatform::new());
        let rec = Arc::new(Lookups::new(&p));
        let de = Deserializer::new(p.clone(), Some(p.device()), rec.lookup());
        let q = p.device().new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let d = ComputeDecoder::new(Arc::downgrade(&de), Some(cb.clone_ref()));
        p.call_log().clear();
        T { p, rec, cb, d, de }
    }

    impl T {
        /// Gives the decoder an encoder, as a first segment would.
        fn with_encoder(&mut self) -> Fk<kind::ComputeEncoder> {
            let e = self.p.world().new_obj::<kind::ComputeEncoder>(Model::None);
            self.d.encoder = Some(e.clone_ref());
            e
        }

        fn with_pipeline(&mut self) -> Fk<kind::ComputePipeline> {
            let pl = self.p.world().new_obj::<kind::ComputePipeline>(Model::None);
            self.d.pipeline = Some(pl.clone_ref());
            pl
        }

        fn obj<K: crate::testkit::objects::Kind>(&self, reference: u32) -> Fk<K> {
            let o = self.p.world().new_obj::<K>(Model::None);
            self.rec.provide(reference, &o);
            o
        }

        /// The ordered calls' selectors: lookups and Metal calls whose position is observable.
        fn sels(&self) -> Vec<&'static str> {
            self.p.call_log().sels()
        }

        /// Sets the device's capabilities: its defaults, changed by `f`.
        fn set_caps(&self, f: impl FnOnce(&mut crate::platform::caps::DeviceCaps)) {
            crate::deserializer::test_lookups::set_caps(&self.p, f);
        }

        fn last(&self, sel: &str) -> Vec<Arg> {
            self.p.call_log().calls_to(sel).last().unwrap().args.clone()
        }
    }

    /// A cursor over `bytes` (kept alive by the cursor).
    fn cursor(bytes: Vec<u8>) -> Cursor {
        let len = bytes.len() as u64;
        let data = Arc::new(bytes);
        let ptr = HostPtr::from_addr(data.as_ptr().addr());
        Cursor::with_pointer(data, ptr, len)
    }

    /// One record: its header (id, whole length) and payload.
    fn rec(id: u32, payload: &[u8]) -> Vec<u8> {
        let mut v = CmdHeader {
            command_id: id,
            length: 8 + payload.len() as u32,
        }
        .as_bytes()
        .to_vec();
        v.extend_from_slice(payload);
        v
    }

    fn seg(continues_previous: u8, continued_by_next: u8) -> SegmentHeader {
        SegmentHeader {
            length: 0,
            encoder_type: 1,
            continues_previous,
            continued_by_next,
            unknown07: 0,
        }
    }

    /// Whether `e` is the cursor's failed read of a record.
    fn read_failed(e: &Raise) -> bool {
        e.detail()
            .is_some_and(|d| d.starts_with("record not readable: read of "))
    }

    fn size(w: u64, h: u64, d: u64) -> MtlSizeRecord {
        MtlSizeRecord {
            width: w,
            height: h,
            depth: d,
        }
    }

    fn u32s(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn cat(parts: &[&[u8]]) -> Vec<u8> {
        parts.concat()
    }

    /// A record with no payload: every command that reads a fixed record fails the cursor's read
    /// and the raise leaves the segment; the commands without a record (start-do-while, end-while,
    /// start-else, end-if, the compressed-texture flush) run.
    /// Every opcode's short record
    #[test]
    fn every_opcode_with_a_short_record() {
        let no_record = [0xdc, 0xdf, 0xe1, 0xe2, 0xe3];
        for id in (0xc8..=0xe6u32).filter(|&i| i != 0xdb).chain([0x86, 0x87]) {
            let mut t = setup();
            t.with_encoder();
            t.with_pipeline();
            let r =
                t.d.decode_segment(Hold::NONE, &seg(1, 0), &mut cursor(rec(id, &[])));
            if no_record.contains(&id) {
                assert!(r.is_ok(), "{id:#x}");
            } else {
                let e = r.unwrap_err();
                assert!(read_failed(&e), "{id:#x}");
            }
        }
    }

    mod init_and_drop {
        use super::*;

        #[test]
        fn stores_both_and_makes_no_encoder() {
            let t = setup();
            assert!(t.d.encoder.is_none());
            assert!(t.d.pipeline.is_none());
            assert_eq!(t.d.command_buffer.as_ref().map(Handle::id), Some(t.cb.id()));
            assert!(t.sels().is_empty(), "no Metal call at init");
        }

        #[test]
        fn the_drop_releases_the_encoder_and_the_pipeline() {
            let mut t = setup();
            let e = t.with_encoder();
            let pl = t.with_pipeline();
            assert_eq!((e.retain_count(), pl.retain_count()), (2, 2));
            drop(t.d);
            assert_eq!((e.retain_count(), pl.retain_count()), (1, 1));
        }

        #[test]
        fn keeps_the_deserializer_unretained() {
            // The deserializer is held weakly (the drop releases the encoder and the pipeline
            // only). The task's release after the decode is the deserializer's last while the
            // decoder, its encoder open, is still held: the deserializer's drop runs there.
            let T {
                p,
                rec,
                cb: _cb,
                mut d,
                de,
            } = setup();
            let dev = p.device();
            d.decode(Hold::NONE, &seg(0, 1), cursor(Vec::new()))
                .unwrap();
            let device_refs = dev.retain_count();
            assert_eq!(Arc::strong_count(&de), 1);
            let enc = d.encoder.as_ref().unwrap().clone_ref();
            let gone = Arc::downgrade(&de);
            drop(de);
            assert!(gone.upgrade().is_none());
            assert_eq!(dev.retain_count(), device_refs - 1);
            assert_eq!(Arc::strong_count(&rec), 1);
            assert!(p.world().live("MTLSharedEventListener").is_empty());
            assert_eq!(enc.retain_count(), 2);
            drop(d);
            assert_eq!(enc.retain_count(), 1);
            assert_eq!(dev.retain_count(), device_refs - 1);
        }
    }

    mod dispatch {
        use super::*;

        fn payload() -> Vec<u8> {
            CmdDispatchThreadgroups {
                threadgroups_per_grid: size(1, 2, 3),
                threads_per_threadgroup: size(4, 5, 6),
            }
            .as_bytes()
            .to_vec()
        }

        #[test]
        fn threadgroups_sends_both_sizes() {
            let mut t = setup();
            t.with_encoder();
            t.with_pipeline();
            t.d.decode_dispatch_threadgroups(&mut cursor(payload()))
                .unwrap();
            assert_eq!(
                t.last("dispatchThreadgroups:threadsPerThreadgroup:"),
                vec![
                    Arg::val(&MtlSize::new(1, 2, 3)),
                    Arg::val(&MtlSize::new(4, 5, 6))
                ]
            );
        }

        #[test]
        fn threads_sends_both_sizes() {
            let mut t = setup();
            t.with_encoder();
            t.with_pipeline();
            t.d.decode_dispatch_threads(&mut cursor(payload())).unwrap();
            assert_eq!(
                t.last("dispatchThreads:threadsPerThreadgroup:"),
                vec![
                    Arg::val(&MtlSize::new(1, 2, 3)),
                    Arg::val(&MtlSize::new(4, 5, 6))
                ]
            );
        }

        #[test]
        fn a_short_record_fails_the_read_before_the_pipeline_check() {
            let mut t = setup();
            t.with_encoder();
            let e =
                t.d.decode_dispatch_threadgroups(&mut cursor(vec![0; 0x2f]))
                    .unwrap_err();
            assert!(read_failed(&e), "{e}");
            assert!(
                e.detail()
                    .unwrap_or("")
                    .starts_with("record not readable: ")
            );
        }

        #[test]
        fn no_pipeline_throws() {
            for which in [0, 1] {
                let mut t = setup();
                t.with_encoder();
                let e = ({
                    let mut c = cursor(payload());
                    if which == 0 {
                        t.d.decode_dispatch_threadgroups(&mut c)
                    } else {
                        t.d.decode_dispatch_threads(&mut c)
                    }
                })
                .unwrap_err();
                assert_eq!(
                    e.id(),
                    RaiseId::from(RaiseKind::Compute(ComputeRaise::NoPipeline))
                );
                assert_eq!(e.detail(), Some("dispatch without a pipeline state"));
                assert!(t.sels().is_empty());
            }
        }
    }

    mod dispatch_indirect {
        use super::*;

        fn tg(reference: u32) -> Vec<u8> {
            CmdDispatchThreadgroupsIndirect {
                threads_per_threadgroup: size(7, 8, 9),
                indirect_buffer_offset: 0x40,
                indirect_buffer_reference: reference,
            }
            .as_bytes()
            .to_vec()
        }

        fn th(reference: u32) -> Vec<u8> {
            CmdDispatchThreadsIndirect {
                indirect_buffer_offset: 0x80,
                indirect_buffer_reference: reference,
            }
            .as_bytes()
            .to_vec()
        }

        #[test]
        fn threadgroups_looks_the_buffer_up_then_dispatches() {
            let mut t = setup();
            t.with_encoder();
            t.with_pipeline();
            let b = t.obj::<kind::Buffer>(5);
            t.d.decode_dispatch_threadgroups_indirect(Hold::NONE, &mut cursor(tg(5)))
                .unwrap();
            assert_eq!(
                t.sels(),
                vec![
                    "get_buffer",
                    "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:"
                ]
            );
            assert_eq!(
                t.last(
                    "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:"
                ),
                vec![Arg::obj(&b), Arg::U64(0x40), Arg::val(&MtlSize::new(7, 8, 9))]
            );
        }

        #[test]
        fn threads_looks_the_buffer_up_then_dispatches() {
            let mut t = setup();
            t.with_encoder();
            t.with_pipeline();
            let b = t.obj::<kind::Buffer>(6);
            t.d.decode_dispatch_threads_indirect(Hold::NONE, &mut cursor(th(6)))
                .unwrap();
            assert_eq!(
                t.last("dispatchThreadsWithIndirectBuffer:indirectBufferOffset:"),
                vec![Arg::obj(&b), Arg::U64(0x80)]
            );
        }

        #[test]
        fn no_pipeline_throws_before_the_lookup() {
            for which in [0, 1] {
                let mut t = setup();
                t.with_encoder();
                let e = if which == 0 {
                    t.d.decode_dispatch_threadgroups_indirect(Hold::NONE, &mut cursor(tg(5)))
                } else {
                    t.d.decode_dispatch_threads_indirect(Hold::NONE, &mut cursor(th(5)))
                }
                .unwrap_err();
                assert_eq!(
                    e.id(),
                    RaiseId::from(RaiseKind::Compute(ComputeRaise::NoPipeline))
                );
                assert!(t.sels().is_empty(), "no lookup");
            }
        }
    }

    mod set_buffers {
        use super::*;

        fn entry(r: u32, off: u64) -> Vec<u8> {
            CmdSetBufferEntry {
                buffer_reference: r,
                offset: off,
            }
            .as_bytes()
            .to_vec()
        }

        fn head(loc: u32, len: u32) -> Vec<u8> {
            CmdSetBuffers {
                range_location: loc,
                range_length: len,
            }
            .as_bytes()
            .to_vec()
        }

        #[test]
        fn looks_up_each_entry_in_order_and_passes_nils() {
            let mut t = setup();
            t.with_encoder();
            let b1 = t.obj::<kind::Buffer>(1);
            let bytes = cat(&[&head(3, 2), &entry(1, 16), &entry(0, 32)]);
            t.d.decode_set_buffers(Hold::NONE, &mut cursor(bytes))
                .unwrap();
            assert_eq!(
                t.sels(),
                vec!["get_buffer", "get_buffer", "setBuffers:offsets:withRange:"]
            );
            assert_eq!(
                t.last("setBuffers:offsets:withRange:"),
                vec![
                    Arg::Objs(vec![Some(b1.id()), None]),
                    Arg::U64s(vec![16, 32]),
                    Arg::val(&NsRange::new(3, 2))
                ]
            );
        }

        #[test]
        fn the_limit_is_on_location_plus_length() {
            let mut t = setup();
            t.with_encoder();
            // 31 is the fake's maxComputeBuffers: 30 + 1 passes, 30 + 2 fails.
            t.d.decode_set_buffers(Hold::NONE, &mut cursor(cat(&[&head(30, 1), &entry(0, 0)])))
                .unwrap();
            let e =
                t.d.decode_set_buffers(Hold::NONE, &mut cursor(head(30, 2)))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::BufferCount))
            );
            assert_eq!(e.detail(), Some("buffers 30+2 are past the device's 31"));
            // u32 + u32 in u64: no wrap.
            let e =
                t.d.decode_set_buffers(Hold::NONE, &mut cursor(head(u32::MAX, 1)))
                    .unwrap_err();
            assert_eq!(
                e.detail(),
                Some("buffers 4294967295+1 are past the device's 31")
            );
        }

        #[test]
        fn zero_entries_still_call_metal() {
            let mut t = setup();
            t.with_encoder();
            t.d.decode_set_buffers(Hold::NONE, &mut cursor(head(0, 0)))
                .unwrap();
            assert_eq!(
                t.last("setBuffers:offsets:withRange:"),
                vec![
                    Arg::Objs(vec![]),
                    Arg::U64s(vec![]),
                    Arg::val(&NsRange::new(0, 0))
                ]
            );
        }
    }

    mod set_buffers_with_stride {
        use super::*;

        fn head(loc: u32, len: u32) -> Vec<u8> {
            CmdSetBuffersWithStride {
                range_location: loc,
                range_length: len,
            }
            .as_bytes()
            .to_vec()
        }

        fn entry(r: u32, off: u64, stride: u64) -> Vec<u8> {
            CmdSetBufferEntryWithStride {
                buffer_reference: r,
                offset: off,
                attribute_stride: stride,
            }
            .as_bytes()
            .to_vec()
        }

        #[test]
        fn passes_offsets_and_strides() {
            let mut t = setup();
            t.with_encoder();
            let b = t.obj::<kind::Buffer>(2);
            let bytes = cat(&[&head(1, 1), &entry(2, 8, 12)]);
            t.d.decode_set_buffers_with_stride(Hold::NONE, &mut cursor(bytes))
                .unwrap();
            assert_eq!(
                t.last("setBuffers:offsets:attributeStrides:withRange:"),
                vec![
                    Arg::Objs(vec![Some(b.id())]),
                    Arg::U64s(vec![8]),
                    Arg::U64s(vec![12]),
                    Arg::val(&NsRange::new(1, 1))
                ]
            );
        }

        #[test]
        fn the_count_check() {
            let mut t = setup();
            t.with_encoder();
            let e =
                t.d.decode_set_buffers_with_stride(Hold::NONE, &mut cursor(head(31, 1)))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::BufferCount))
            );
            assert_eq!(e.detail(), Some("buffers 31+1 are past the device's 31"));
        }

        /// On a device without dynamic attribute strides the record raises once every entry is
        /// read and looked up, and nothing is sent.
        #[test]
        fn without_dynamic_attribute_stride_raises_after_the_lookups() {
            let mut t = setup();
            t.with_encoder();
            t.set_caps(|c| c.flags.supports_dynamic_attribute_stride = false);
            let bytes = cat(&[&head(1, 1), &entry(0, 0, 4)]);
            let e =
                t.d.decode_set_buffers_with_stride(Hold::NONE, &mut cursor(bytes))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::DynamicAttributeStride))
            );
            assert_eq!(
                e.detail(),
                Some("the device does not support dynamic attribute strides")
            );
            assert_eq!(t.sels(), vec!["get_buffer"]);
        }
    }

    mod set_samplers {
        use super::*;

        fn head(loc: u32, len: u32) -> Vec<u8> {
            CmdSetSamplerStates {
                range_location: loc,
                range_length: len,
            }
            .as_bytes()
            .to_vec()
        }

        #[test]
        fn samplers_by_reference() {
            let mut t = setup();
            t.with_encoder();
            let s = t.obj::<kind::SamplerState>(4);
            let bytes = cat(&[&head(2, 2), &u32s(&[4, 0])]);
            t.d.decode_set_samplers(Hold::NONE, &mut cursor(bytes))
                .unwrap();
            assert_eq!(
                t.sels(),
                vec![
                    "get_sampler_state",
                    "get_sampler_state",
                    "setSamplerStates:withRange:"
                ]
            );
            assert_eq!(
                t.last("setSamplerStates:withRange:"),
                vec![
                    Arg::Objs(vec![Some(s.id()), None]),
                    Arg::val(&NsRange::new(2, 2))
                ]
            );
        }

        #[test]
        fn samplers_count_check() {
            let mut t = setup();
            t.with_encoder();
            let e =
                t.d.decode_set_samplers(Hold::NONE, &mut cursor(head(16, 1)))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::SamplerCount))
            );
            assert_eq!(e.detail(), Some("samplers 16+1 are past the device's 16"));
        }

        #[test]
        fn lod_clamps_pass_bit_exact_floats() {
            let mut t = setup();
            t.with_encoder();
            let s = t.obj::<kind::SamplerState>(4);
            let entry = CmdSetSamplerStatesLODClampEntry {
                sampler_state_reference: 4,
                lod_min_clamp: -0.0,
                lod_max_clamp: f32::from_bits(0x7fc0_0001),
            };
            let bytes = cat(&[
                CmdSetSamplerStatesLODClamp {
                    range_location: 0,
                    range_length: 1,
                }
                .as_bytes(),
                entry.as_bytes(),
            ]);
            t.d.decode_set_samplers_lod_clamp(Hold::NONE, &mut cursor(bytes))
                .unwrap();
            let args = t.last("setSamplerStates:lodMinClamps:lodMaxClamps:withRange:");
            assert_eq!(args[0], Arg::Objs(vec![Some(s.id())]));
            match (&args[1], &args[2]) {
                (Arg::F32s(min), Arg::F32s(max)) => {
                    assert_eq!(min[0].to_bits(), (-0.0f32).to_bits());
                    assert_eq!(max[0].to_bits(), 0x7fc0_0001);
                }
                other => panic!("{other:?}"),
            }
            assert_eq!(args[3], Arg::val(&NsRange::new(0, 1)));
        }

        #[test]
        fn lod_clamp_count_check() {
            let mut t = setup();
            t.with_encoder();
            let bytes = CmdSetSamplerStatesLODClamp {
                range_location: 10,
                range_length: 7,
            }
            .as_bytes()
            .to_vec();
            let e =
                t.d.decode_set_samplers_lod_clamp(Hold::NONE, &mut cursor(bytes))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::SamplerCount))
            );
            assert_eq!(e.detail(), Some("samplers 10+7 are past the device's 16"));
        }
    }

    mod set_textures {
        use super::*;

        fn head(loc: u32, len: u32) -> Vec<u8> {
            CmdSetTextures {
                range_location: loc,
                range_length: len,
            }
            .as_bytes()
            .to_vec()
        }

        #[test]
        fn textures_by_reference() {
            let mut t = setup();
            t.with_encoder();
            let x = t.obj::<kind::Texture>(3);
            let bytes = cat(&[&head(0, 2), &u32s(&[0, 3])]);
            t.d.decode_set_textures(Hold::NONE, &mut cursor(bytes))
                .unwrap();
            assert_eq!(
                t.sels(),
                vec!["get_texture", "get_texture", "setTextures:withRange:"]
            );
            assert_eq!(
                t.last("setTextures:withRange:"),
                vec![
                    Arg::Objs(vec![None, Some(x.id())]),
                    Arg::val(&NsRange::new(0, 2))
                ]
            );
        }

        #[test]
        fn the_count_check() {
            let mut t = setup();
            t.with_encoder();
            let e =
                t.d.decode_set_textures(Hold::NONE, &mut cursor(head(128, 1)))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::TextureCount))
            );
            assert_eq!(e.detail(), Some("textures 128+1 are past the device's 128"));
        }
    }

    mod resource_lists {
        use super::*;

        fn use_resources(count: u32, usage: u32, refs: &[u32]) -> Vec<u8> {
            cat(&[CmdUseResources { count, usage }.as_bytes(), &u32s(refs)])
        }

        #[test]
        fn use_resources_maps_each_kind_and_zero_extends_usage() {
            let mut t = setup();
            t.with_encoder();
            let b = t.obj::<kind::Buffer>(1);
            let x = t.obj::<kind::Texture>(2);
            let i = t.obj::<kind::Icb>(3);
            let bytes = use_resources(4, 0xffff_fffe, &[1, 2, 3, 0]);
            t.d.decode_use_resources(Hold::NONE, &mut cursor(bytes))
                .unwrap();
            assert_eq!(t.p.call_log().count("get_resource"), 4);
            assert_eq!(
                t.last("useResources:count:usage:"),
                vec![
                    Arg::Objs(vec![Some(b.id()), Some(x.id()), Some(i.id()), None]),
                    Arg::U64(0xffff_fffe)
                ]
            );
        }

        #[test]
        fn more_than_256_entries() {
            let mut t = setup();
            t.with_encoder();
            let refs = vec![0u32; 300];
            t.d.decode_use_resources(Hold::NONE, &mut cursor(use_resources(300, 1, &refs)))
                .unwrap();
            match &t.last("useResources:count:usage:")[0] {
                Arg::Objs(v) => assert_eq!(v.len(), 300),
                other => panic!("{other:?}"),
            }
        }

        #[test]
        fn the_remaining_check_comes_before_any_lookup() {
            let mut t = setup();
            t.with_encoder();
            // 2 entries declared, 7 bytes left: 7 < 8.
            let mut bytes = use_resources(2, 0, &[1]);
            bytes.extend_from_slice(&[0; 3]);
            let e =
                t.d.decode_use_resources(Hold::NONE, &mut cursor(bytes))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::ResourceCount))
            );
            assert_eq!(e.detail(), Some("reference count 2 runs past the record"));
            assert!(t.sels().is_empty());
            // A count whose * 4 is past u32 is still compared in u64.
            let e = t
                .d
                .decode_use_resources(Hold::NONE, &mut cursor(use_resources(0x4000_0001, 0, &[])))
                .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::ResourceCount))
            );
        }

        #[test]
        fn a_short_head_throws_inside_the_try() {
            let mut t = setup();
            t.with_encoder();
            let e =
                t.d.decode_use_resources(Hold::NONE, &mut cursor(vec![0; 7]))
                    .unwrap_err();
            assert!(read_failed(&e), "{e}");
        }

        /// The list's frame is a handler: a raise in the body goes on when the frame ends, to the
        /// segment's handler, as the raise's own id.
        #[test]
        fn the_frame_passes_its_raise_on() {
            let mut t = setup();
            t.with_encoder();
            let site =
                t.d.decode_use_resources(Hold::NONE, &mut cursor(use_resources(9, 0, &[])))
                    .unwrap_err()
                    .site();
            assert_eq!(
                site,
                DeathSite {
                    kind: DeathKind::UnhandledRaise,
                    file: ComputeRaise::ResourceCount.name(),
                    line: 0
                }
            );
        }

        #[test]
        fn a_heap_from_the_seam_is_passed_as_nil() {
            let mut t = setup();
            t.with_encoder();
            t.obj::<kind::Heap>(8);
            t.d.decode_use_resources(Hold::NONE, &mut cursor(use_resources(1, 0, &[8])))
                .unwrap();
            assert_eq!(
                t.last("useResources:count:usage:")[0],
                Arg::Objs(vec![None])
            );
        }

        #[test]
        fn use_heaps_remaining_check() {
            let mut t = setup();
            t.with_encoder();
            let bytes = CmdUseHeaps { count: 1 }.as_bytes().to_vec();
            let e =
                t.d.decode_use_heaps(Hold::NONE, &mut cursor(bytes))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::ResourceCount))
            );
            assert_eq!(e.detail(), Some("reference count 1 runs past the record"));
        }

        #[test]
        fn barrier_resources_by_reference() {
            let mut t = setup();
            t.with_encoder();
            let b = t.obj::<kind::Buffer>(7);
            let bytes = cat(&[
                CmdMemoryBarrierResources { count: 1 }.as_bytes(),
                &u32s(&[7]),
            ]);
            t.d.decode_barrier_resources(Hold::NONE, &mut cursor(bytes))
                .unwrap();
            assert_eq!(
                t.sels(),
                vec!["get_resource", "memoryBarrierWithResources:count:"]
            );
            assert_eq!(
                t.last("memoryBarrierWithResources:count:"),
                vec![Arg::Objs(vec![Some(b.id())])]
            );
        }

        #[test]
        fn barrier_resources_remaining_check() {
            let mut t = setup();
            t.with_encoder();
            let bytes = CmdMemoryBarrierResources { count: 3 }.as_bytes().to_vec();
            let e =
                t.d.decode_barrier_resources(Hold::NONE, &mut cursor(bytes))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::ResourceCount))
            );
        }
    }

    mod set_buffer_offset {
        use super::*;

        #[test]
        fn offset_at_index() {
            let mut t = setup();
            t.with_encoder();
            let bytes = CmdSetBufferOffset {
                index: 30,
                offset: 0x100,
            }
            .as_bytes()
            .to_vec();
            t.d.decode_set_buffer_offset(&mut cursor(bytes)).unwrap();
            assert_eq!(
                t.last("setBufferOffset:atIndex:"),
                vec![Arg::U64(0x100), Arg::U64(30)]
            );
        }

        #[test]
        fn index_at_the_limit_throws() {
            let mut t = setup();
            t.with_encoder();
            let bytes = CmdSetBufferOffset {
                index: 31,
                offset: 0,
            }
            .as_bytes()
            .to_vec();
            let e =
                t.d.decode_set_buffer_offset(&mut cursor(bytes))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::BufferIndex))
            );
            assert_eq!(
                e.detail(),
                Some("buffer index 31 is past the device's limit")
            );
        }

        fn with_stride(index: u32) -> Vec<u8> {
            CmdSetBufferOffsetWithStride {
                index,
                offset: 4,
                attribute_stride: 16,
            }
            .as_bytes()
            .to_vec()
        }

        #[test]
        fn with_stride_sends_all_three() {
            let mut t = setup();
            t.with_encoder();
            t.d.decode_set_buffer_offset_with_stride(&mut cursor(with_stride(2)))
                .unwrap();
            assert_eq!(
                t.last("setBufferOffset:attributeStride:atIndex:"),
                vec![Arg::U64(4), Arg::U64(16), Arg::U64(2)]
            );
        }

        /// The index check comes before the support check.
        #[test]
        fn with_stride_checks_index_then_support() {
            let mut t = setup();
            t.with_encoder();
            t.set_caps(|c| c.flags.supports_dynamic_attribute_stride = false);
            let e =
                t.d.decode_set_buffer_offset_with_stride(&mut cursor(with_stride(31)))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::BufferIndex))
            );
            assert!(t.sels().is_empty());
            let e =
                t.d.decode_set_buffer_offset_with_stride(&mut cursor(with_stride(0)))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::DynamicAttributeStride))
            );
            assert_eq!(
                e.detail(),
                Some("the device does not support dynamic attribute strides")
            );
        }
    }

    mod set_pipeline_state {
        use super::*;

        fn rec_of(r: u32) -> Vec<u8> {
            CmdSetRenderPipelineState {
                pipeline_state_reference: r,
            }
            .as_bytes()
            .to_vec()
        }

        #[test]
        fn reference_zero_throws_before_the_lookup() {
            let mut t = setup();
            t.with_encoder();
            let e =
                t.d.decode_set_pipeline_state(Hold::NONE, &mut cursor(rec_of(0)))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::PipelineReference))
            );
            assert_eq!(e.detail(), Some("compute pipeline state reference 0"));
            assert!(t.sels().is_empty());
        }
    }

    mod stage_in_region {
        use super::*;

        #[test]
        fn region_size_first_in_the_record() {
            let mut t = setup();
            t.with_encoder();
            let bytes = CmdSetComputeStageInRegion {
                region_size: size(1, 2, 3),
                region_origin: MtlOriginRecord { x: 4, y: 5, z: 6 },
            }
            .as_bytes()
            .to_vec();
            t.d.decode_set_stage_in_region(&mut cursor(bytes)).unwrap();
            assert_eq!(
                t.last("setStageInRegion:"),
                vec![Arg::val(&MtlRegion::new(
                    MtlOrigin::new(4, 5, 6),
                    MtlSize::new(1, 2, 3)
                ))]
            );
        }
    }

    mod set_threadgroup_memory_length {
        use super::*;

        fn rec_of(index: u32) -> Vec<u8> {
            CmdSetThreadgroupMemoryLength { length: 64, index }
                .as_bytes()
                .to_vec()
        }

        #[test]
        fn index_check_then_call() {
            let mut t = setup();
            t.with_encoder();
            t.set_caps(|c| c.limits.max_compute_local_memory_sizes = 2);
            let e =
                t.d.decode_set_threadgroup_memory_length(&mut cursor(rec_of(2)))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::ThreadgroupMemoryIndex))
            );
            assert_eq!(
                e.detail(),
                Some("threadgroup memory index 2 is past the device's limit")
            );
            t.d.decode_set_threadgroup_memory_length(&mut cursor(rec_of(1)))
                .unwrap();
            assert_eq!(
                t.last("setThreadgroupMemoryLength:atIndex:"),
                vec![Arg::U64(64), Arg::U64(1)]
            );
        }
    }

    mod fences {
        use super::*;

        fn rec_of(r: u32) -> Vec<u8> {
            CmdFenceOp { fence_reference: r }.as_bytes().to_vec()
        }

        #[test]
        fn update_and_wait() {
            let mut t = setup();
            t.with_encoder();
            let f = t.obj::<kind::Fence>(2);
            t.d.decode_update_fence(Hold::NONE, &mut cursor(rec_of(2)))
                .unwrap();
            t.d.decode_wait_for_fence(Hold::NONE, &mut cursor(rec_of(2)))
                .unwrap();
            assert_eq!(
                t.sels(),
                vec!["get_fence", "updateFence:", "get_fence", "waitForFence:"]
            );
            assert_eq!(t.last("updateFence:"), vec![Arg::obj(&f)]);
            assert_eq!(t.last("waitForFence:"), vec![Arg::obj(&f)]);
        }
    }

    mod barrier_scope {
        use super::*;

        #[test]
        fn only_the_low_16_bits() {
            let mut t = setup();
            t.with_encoder();
            let bytes = CmdMemoryBarrierScope { bits: 0xabcd_8001 }
                .as_bytes()
                .to_vec();
            t.d.decode_barrier_scope(&mut cursor(bytes)).unwrap();
            assert_eq!(t.last("memoryBarrierWithScope:"), vec![Arg::U64(0x8001)]);
        }
    }

    mod set_imageblock_width {
        use super::*;

        #[test]
        fn read_then_support_then_call() {
            let mut t = setup();
            t.with_encoder();
            let bytes = CmdSetImageblockWidth {
                width: 32,
                height: 16,
            }
            .as_bytes()
            .to_vec();
            t.d.decode_set_imageblock_width(&mut cursor(bytes.clone()))
                .unwrap();
            assert_eq!(
                t.last("setImageblockWidth:height:"),
                vec![Arg::U64(32), Arg::U64(16)]
            );
            t.set_caps(|c| c.flags.supports_image_blocks = false);
            let e =
                t.d.decode_set_imageblock_width(&mut cursor(bytes))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::ImageBlocks))
            );
            assert_eq!(e.detail(), Some("the device does not support imageblocks"));
            // A short record fails the read before the support check.
            t.p.call_log().clear();
            let e =
                t.d.decode_set_imageblock_width(&mut cursor(vec![0; 4]))
                    .unwrap_err();
            assert!(read_failed(&e), "{e}");
            assert!(t.sels().is_empty());
        }
    }

    mod jumps {
        use super::*;

        fn jump(r: u32) -> Vec<u8> {
            CmdEncodeStartIf {
                buffer_reference: r,
                offset: 0x10,
                comparison: 0xffff_ffff,
                reference_value: 7,
            }
            .as_bytes()
            .to_vec()
        }

        type Handler = fn(&mut ComputeDecoder<FakePlatform>, &mut Cursor) -> RaiseResult<()>;

        fn all() -> Vec<(Handler, bool)> {
            // (handler, reads a record)
            vec![
                (ComputeDecoder::decode_encode_start_do_while, false),
                (|d, c| d.decode_encode_end_do_while(Hold::NONE, c), true),
                (|d, c| d.decode_encode_start_while(Hold::NONE, c), true),
                (ComputeDecoder::decode_encode_end_while, false),
                (|d, c| d.decode_encode_start_if(Hold::NONE, c), true),
                (ComputeDecoder::decode_encode_start_else, false),
                (ComputeDecoder::decode_encode_end_if, false),
            ]
        }

        #[test]
        fn without_jump_support_each_raises_after_its_read() {
            for (h, reads) in all() {
                let mut t = setup();
                t.with_encoder();
                t.set_caps(|c| c.flags.supports_command_buffer_jump = false);
                let e = h(&mut t.d, &mut cursor(jump(1))).unwrap_err();
                assert_eq!(
                    e.id(),
                    RaiseId::from(RaiseKind::Compute(ComputeRaise::CommandBufferJump))
                );
                assert_eq!(
                    e.detail(),
                    Some("the device does not support command buffer jumps")
                );
                // The readers fail a short record first; the others never read.
                let mut t = setup();
                t.with_encoder();
                let r = h(&mut t.d, &mut cursor(vec![0; 3]));
                assert_eq!(r.is_err(), reads);
            }
        }

        #[test]
        fn the_record_sends_zero_extend_the_comparison() {
            let mut t = setup();
            t.with_encoder();
            let b = t.obj::<kind::Buffer>(1);
            t.d.decode_encode_start_while(Hold::NONE, &mut cursor(jump(1)))
                .unwrap();
            t.d.decode_encode_start_if(Hold::NONE, &mut cursor(jump(0)))
                .unwrap();
            t.d.decode_encode_end_do_while(Hold::NONE, &mut cursor(jump(1)))
                .unwrap();
            let expect = |buf: Arg| vec![buf, Arg::U64(0x10), Arg::U64(0xffff_ffff), Arg::U32(7)];
            assert_eq!(
                t.last("encodeStartWhile:offset:comparison:referenceValue:"),
                expect(Arg::obj(&b))
            );
            assert_eq!(
                t.last("encodeStartIf:offset:comparison:referenceValue:"),
                expect(Arg::Obj(None))
            );
            assert_eq!(
                t.last("encodeEndDoWhile:offset:comparison:referenceValue:"),
                expect(Arg::obj(&b))
            );
            assert_eq!(
                t.sels(),
                vec![
                    "get_buffer",
                    "encodeStartWhile:offset:comparison:referenceValue:",
                    "get_buffer",
                    "encodeStartIf:offset:comparison:referenceValue:",
                    "get_buffer",
                    "encodeEndDoWhile:offset:comparison:referenceValue:",
                ]
            );
        }

        #[test]
        fn the_bare_sends() {
            let mut t = setup();
            t.with_encoder();
            for h in [
                ComputeDecoder::decode_encode_start_do_while as Handler,
                ComputeDecoder::decode_encode_end_while,
                ComputeDecoder::decode_encode_start_else,
                ComputeDecoder::decode_encode_end_if,
            ] {
                h(&mut t.d, &mut cursor(vec![])).unwrap();
            }
            assert_eq!(
                t.sels(),
                vec![
                    "encodeStartDoWhile",
                    "encodeEndWhile",
                    "encodeStartElse",
                    "encodeEndIf",
                ]
            );
        }

        /// An End* returning NO only logs.
        #[test]
        fn an_end_returning_no_only_logs() {
            let mut t = setup();
            t.with_encoder();
            for sel in [
                "encodeEndDoWhile:offset:comparison:referenceValue:",
                "encodeEndWhile",
                "encodeEndIf",
            ] {
                t.p.script().push(sel, Step::value(false));
            }
            let logs = crate::testkit::logs::capture();
            t.d.decode_encode_end_do_while(Hold::NONE, &mut cursor(jump(0)))
                .unwrap();
            t.d.decode_encode_end_while(&mut cursor(vec![])).unwrap();
            t.d.decode_encode_end_if(&mut cursor(vec![])).unwrap();
            let target = concat!(env!("CARGO_CRATE_NAME"), "::deserializer::compute");
            let line = |what: &str| {
                (
                    log::Level::Warn,
                    target.to_owned(),
                    format!("the compute encoder refused to end the {what} block"),
                )
            };
            assert_eq!(
                logs.take(),
                vec![line("do-while"), line("while"), line("if")]
            );
            // YES logs nothing.
            t.d.decode_encode_end_if(&mut cursor(vec![])).unwrap();
            assert!(logs.take().is_empty());
        }
    }

    mod insert_flush {
        use super::*;

        #[test]
        fn no_read_no_check() {
            let mut t = setup();
            t.with_encoder();
            t.d.decode_insert_compressed_texture_reinterpretation_flush(&mut cursor(vec![1]))
                .unwrap();
            assert_eq!(
                t.sels(),
                vec!["insertCompressedTextureReinterpretationFlush"]
            );
        }
    }

    mod execute_ranged {}

    mod execute {
        use super::*;

        fn rec_of(icb: u32, range: u32) -> Vec<u8> {
            CmdExecuteCommandsInBuffer {
                indirect_command_buffer_reference: icb,
                indirect_range_buffer_reference: range,
                indirect_buffer_offset: 0x20,
            }
            .as_bytes()
            .to_vec()
        }

        #[test]
        fn sends_both() {
            let mut t = setup();
            t.with_encoder();
            let icb = t.obj::<kind::Icb>(3);
            let b = t.obj::<kind::Buffer>(5);
            t.d.decode_execute_command_in_buffer(Hold::NONE, &mut cursor(rec_of(3, 5)))
                .unwrap();
            assert_eq!(
                t.last("executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:"),
                vec![Arg::obj(&icb), Arg::obj(&b), Arg::U64(0x20)]
            );
        }
    }

    mod compute_pass_descriptor {
        use super::*;

        fn dispatch(ty: u32) -> Vec<u8> {
            rec(
                0xdb,
                CmdSetComputePassDescriptorDispatchType { dispatch_type: ty }.as_bytes(),
            )
        }

        #[test]
        fn empty_iterator_gives_a_bare_descriptor() {
            let t = setup();
            let mut it = cursor(vec![]);
            let d = t.d.decode_compute_pass_descriptor(&mut it).unwrap();
            assert_eq!(d, ComputePassDesc::default());
        }

        #[test]
        fn a_first_0xdb_sets_the_dispatch_type_and_is_consumed() {
            let t = setup();
            let mut it = cursor(cat(&[&dispatch(0xffff_ffff), &rec(0xe3, &[])]));
            let d = t.d.decode_compute_pass_descriptor(&mut it).unwrap();
            assert_eq!(d.dispatch_type, Some(0xffff_ffff));
            assert_eq!(it.remaining(), 8);
        }

        /// Another first record is validated but left in place.
        #[test]
        fn another_first_record_stays() {
            let t = setup();
            let mut it = cursor(rec(0xe3, &[1, 2]));
            let d = t.d.decode_compute_pass_descriptor(&mut it).unwrap();
            assert_eq!(it.remaining(), 10);
            assert_eq!(d, ComputePassDesc::default());
        }

        #[test]
        fn the_peek_throws_where_the_loop_would_not() {
            let t = setup();
            // 1-7 bytes: the header read fails.
            let e =
                t.d.decode_compute_pass_descriptor(&mut cursor(vec![0; 7]))
                    .unwrap_err();
            assert!(read_failed(&e), "{e}");
            // length < 8.
            let mut h = rec(0xe3, &[]);
            h[4] = 7;
            let e =
                t.d.decode_compute_pass_descriptor(&mut cursor(h))
                    .unwrap_err();
            assert_eq!(e.id(), RaiseId::from(RaiseKind::RecordUnderflow));
            assert_eq!(
                e.detail(),
                Some("record length 7 is shorter than its header")
            );
            // A record past the end.
            let mut h = rec(0xe3, &[]);
            h[4] = 9;
            let e =
                t.d.decode_compute_pass_descriptor(&mut cursor(h))
                    .unwrap_err();
            assert_eq!(e.id(), RaiseId::from(RaiseKind::RecordOverrun));
            assert_eq!(
                e.detail().unwrap_or(""),
                "record not readable: a sub-cursor of 0x1 bytes, with 0x0 left"
            );
        }

        #[test]
        fn a_short_0xdb_advances_then_fails_its_read() {
            let t = setup();
            let mut it = cursor(rec(0xdb, &[0, 0]));
            let e = t.d.decode_compute_pass_descriptor(&mut it).unwrap_err();
            assert!(read_failed(&e), "{e}");
            assert_eq!(it.remaining(), 0);
        }

        #[test]
        fn set_dispatch_type_reads_four_bytes() {
            let t = setup();
            let e =
                t.d.decode_compute_pass_descriptor(&mut cursor(rec(0xdb, &[1, 0, 0])))
                    .unwrap_err();
            assert!(read_failed(&e), "{e}");
            let d =
                t.d.decode_compute_pass_descriptor(&mut cursor(rec(0xdb, &[1, 0, 0, 0, 9])))
                    .unwrap();
            assert_eq!(d.dispatch_type, Some(1));
        }
    }

    mod decode {
        use super::*;

        fn pipeline_rec(r: u32) -> Vec<u8> {
            rec(
                0xd0,
                CmdSetRenderPipelineState {
                    pipeline_state_reference: r,
                }
                .as_bytes(),
            )
        }

        /// A segment whose first record is 0xdb 1 looks UseResources' references up with the
        /// lookup that raises for an empty entry, and its continuation too; a serial segment, a
        /// type other than 1 or no 0xdb, with the nullable one.
        #[test]
        fn a_concurrent_pass_looks_resources_up_raising_empty() {
            let dispatch_type = |ty: u32| {
                rec(
                    0xdb,
                    CmdSetComputePassDescriptorDispatchType { dispatch_type: ty }.as_bytes(),
                )
            };
            let uses = rec(
                0x87,
                &cat(&[
                    CmdUseResources { count: 1, usage: 1 }.as_bytes(),
                    &u32s(&[0]),
                ]),
            );
            let mut t = setup();
            let first = cat(&[&dispatch_type(1), &uses]);
            t.d.decode(Hold::NONE, &seg(0, 1), cursor(first)).unwrap();
            t.d.decode(Hold::NONE, &seg(1, 0), cursor(uses.clone()))
                .unwrap();
            assert_eq!(t.p.call_log().count("get_resource_raising_empty"), 2);
            assert_eq!(t.p.call_log().count("get_resource"), 0);
            for first in [
                cat(&[&dispatch_type(0), &uses]),
                cat(&[&dispatch_type(2), &uses]),
                uses.clone(),
            ] {
                let mut t = setup();
                t.d.decode(Hold::NONE, &seg(0, 0), cursor(first)).unwrap();
                assert_eq!(t.p.call_log().count("get_resource"), 1);
                assert_eq!(t.p.call_log().count("get_resource_raising_empty"), 0);
            }
        }

        #[test]
        fn a_nil_command_buffer_throws_after_the_prefix() {
            // The descriptor is decoded, the encoder sent to nil is nil, throws.
            let mut t = setup();
            t.d.command_buffer = None;
            let r = t.d.decode(Hold::NONE, &seg(0, 0), cursor(Vec::new()));
            let e = r.err().unwrap();
            assert_eq!(e.detail(), Some("no compute encoder"));
            assert_eq!(
                t.p.call_log().count("computeCommandEncoderWithDescriptor:"),
                0
            );
            assert!(t.d.encoder.is_none());
        }

        #[test]
        fn continued_by_next_keeps_the_encoder_open_and_the_pipeline() {
            let mut t = setup();
            t.obj::<kind::ComputePipeline>(1);
            t.d.decode(Hold::NONE, &seg(0, 1), cursor(pipeline_rec(1)))
                .unwrap();
            assert_eq!(t.p.call_log().count("endEncoding"), 0);
            assert!(t.d.pipeline.is_some());
            // The continuation reuses the encoder: no new one.
            let dispatch = rec(
                0xca,
                CmdDispatchThreads {
                    threads_per_grid: size(1, 1, 1),
                    threads_per_threadgroup: size(1, 1, 1),
                }
                .as_bytes(),
            );
            t.d.decode(Hold::NONE, &seg(1, 0), cursor(dispatch))
                .unwrap();
            assert_eq!(
                t.p.call_log().count("computeCommandEncoderWithDescriptor:"),
                1
            );
            assert_eq!(
                t.p.call_log()
                    .count("dispatchThreads:threadsPerThreadgroup:"),
                1
            );
            assert_eq!(t.p.call_log().count("endEncoding"), 1);
        }

        #[test]
        fn a_continuation_without_an_encoder_throws() {
            let mut t = setup();
            let e =
                t.d.decode(Hold::NONE, &seg(1, 0), cursor(vec![]))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::NoEncoder))
            );
            assert_eq!(e.detail(), Some("no compute encoder"));
            assert!(t.sels().is_empty(), "the fault messages nil");
        }

        #[test]
        fn a_nil_encoder_throws_after_the_prefix() {
            let mut t = setup();
            t.p.script()
                .push("computeCommandEncoderWithDescriptor:", Step::Nil);
            let e =
                t.d.decode(Hold::NONE, &seg(0, 0), cursor(rec(0xdb, &u32s(&[2]))))
                    .unwrap_err();
            assert_eq!(
                e.id(),
                RaiseId::from(RaiseKind::Compute(ComputeRaise::NoEncoder))
            );
            assert_eq!(t.sels(), vec!["computeCommandEncoderWithDescriptor:"]);
            assert_eq!(
                t.last("computeCommandEncoderWithDescriptor:"),
                vec![Arg::val(&ComputePassDesc {
                    dispatch_type: Some(2)
                })]
            );
        }

        #[test]
        fn the_loop_failures() {
            let cases: Vec<(Vec<u8>, RaiseKind, &str)> = vec![
                (
                    {
                        let mut h = rec(0xe3, &[]);
                        h[4] = 0;
                        cat(&[&rec(0xe3, &[]), &h])
                    },
                    RaiseKind::RecordUnderflow,
                    "record length 0 is shorter than its header",
                ),
                (
                    {
                        let mut h = rec(0xe3, &[]);
                        h[4] = 12;
                        cat(&[&rec(0xe3, &[]), &h])
                    },
                    RaiseKind::RecordOverrun,
                    "record not readable: a sub-cursor of 0x4 bytes, with 0x0 left",
                ),
                // 0xdb past the first record.
                (
                    cat(&[&rec(0xe3, &[]), &rec(0xdb, &u32s(&[1]))]),
                    RaiseKind::UnrecognizedCommand,
                    "unknown command 219",
                ),
                (
                    rec(0x88, &[]),
                    RaiseKind::UnrecognizedCommand,
                    "unknown command 136",
                ),
            ];
            for (stream, kind, reason) in cases {
                let mut t = setup();
                let e =
                    t.d.decode(Hold::NONE, &seg(0, 0), cursor(stream))
                        .unwrap_err();
                assert_eq!(e.id(), RaiseId::from(kind));
                assert_eq!(e.detail(), Some(reason));
                // The fault ended the encoder and dropped it.
                assert_eq!(t.p.call_log().count("endEncoding"), 1);
                assert!(t.d.encoder.is_none());
            }
        }

        #[test]
        fn every_command_id_reaches_its_handler() {
            let with = |id: u32, payload: Vec<u8>, sel: &'static str| (id, payload, sel);
            let mtl = CmdDispatchThreadgroups {
                threadgroups_per_grid: size(1, 1, 1),
                threads_per_threadgroup: size(1, 1, 1),
            }
            .as_bytes()
            .to_vec();
            let jump = vec![0u8; 0x14];
            let cases = vec![
                with(
                    0xc8,
                    mtl.clone(),
                    "dispatchThreadgroups:threadsPerThreadgroup:",
                ),
                with(
                    0xc9,
                    vec![0; 0x24],
                    "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:",
                ),
                with(0xca, mtl, "dispatchThreads:threadsPerThreadgroup:"),
                with(0xcb, vec![0; 8], "setBuffers:offsets:withRange:"),
                with(0xcc, vec![0; 8], "setSamplerStates:withRange:"),
                with(
                    0xcd,
                    vec![0; 8],
                    "setSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
                ),
                with(0xce, vec![0; 8], "setTextures:withRange:"),
                with(0xcf, vec![0; 0xc], "setBufferOffset:atIndex:"),
                with(0xd0, u32s(&[1]), "setComputePipelineState:"),
                with(0xd1, vec![0; 0x30], "setStageInRegion:"),
                with(
                    0xd2,
                    cat(&[&u32s(&[2]), &[0; 8]]),
                    "setStageInRegionWithIndirectBuffer:indirectBufferOffset:",
                ),
                with(0xd3, vec![0; 0xc], "setThreadgroupMemoryLength:atIndex:"),
                with(0xd4, u32s(&[3]), "updateFence:"),
                with(0xd5, u32s(&[3]), "waitForFence:"),
                with(0xd6, vec![0; 4], "memoryBarrierWithResources:count:"),
                with(0xd7, vec![0; 4], "memoryBarrierWithScope:"),
                with(0xd8, vec![0; 8], "setImageblockWidth:height:"),
                with(
                    0xd9,
                    vec![0; 8],
                    "setBuffers:offsets:attributeStrides:withRange:",
                ),
                with(
                    0xda,
                    vec![0; 0x14],
                    "setBufferOffset:attributeStride:atIndex:",
                ),
                with(0xdc, vec![], "encodeStartDoWhile"),
                with(
                    0xdd,
                    jump.clone(),
                    "encodeEndDoWhile:offset:comparison:referenceValue:",
                ),
                with(
                    0xde,
                    jump.clone(),
                    "encodeStartWhile:offset:comparison:referenceValue:",
                ),
                with(0xdf, vec![], "encodeEndWhile"),
                with(
                    0xe0,
                    jump,
                    "encodeStartIf:offset:comparison:referenceValue:",
                ),
                with(0xe1, vec![], "encodeStartElse"),
                with(0xe2, vec![], "encodeEndIf"),
                with(0xe3, vec![], "insertCompressedTextureReinterpretationFlush"),
                with(
                    0xe4,
                    cat(&[&u32s(&[4]), &[0; 16]]),
                    "executeCommandsInBuffer:withRange:",
                ),
                with(
                    0xe5,
                    cat(&[&u32s(&[4, 2]), &[0; 8]]),
                    "executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:",
                ),
                with(
                    0xe6,
                    vec![0; 0xc],
                    "dispatchThreadsWithIndirectBuffer:indirectBufferOffset:",
                ),
                with(0x86, vec![0; 4], "useHeaps:count:"),
                with(0x87, vec![0; 8], "useResources:count:usage:"),
            ];
            for (id, payload, sel) in cases {
                let mut t = setup();
                t.obj::<kind::ComputePipeline>(1);
                t.obj::<kind::Buffer>(2);
                t.obj::<kind::Fence>(3);
                t.obj::<kind::Icb>(4).set_size(1);
                // A pipeline first, so the dispatches pass; continued so the pipeline stays.
                t.d.decode(Hold::NONE, &seg(0, 1), cursor(pipeline_rec(1)))
                    .unwrap();
                t.d.decode(Hold::NONE, &seg(1, 1), cursor(rec(id, &payload)))
                    .unwrap_or_else(|e| panic!("{id:#x}: {e}"));
                assert_eq!(
                    t.p.call_log().count(sel),
                    1 + usize::from(id == 0xd0),
                    "{id:#x}"
                );
            }
        }

        /// A carried decoder reused for a new segment overwrites its encoder, leaking the old one.
        #[test]
        fn a_reopened_segment_leaks_the_old_encoder() {
            let mut t = setup();
            let old = t.with_encoder();
            assert_eq!(old.retain_count(), 2);
            t.d.decode(Hold::NONE, &seg(0, 0), cursor(vec![])).unwrap();
            assert_ne!(t.d.encoder.as_ref().map(Handle::id), Some(old.id()));
            assert_eq!(old.retain_count(), 2, "not released");
            // Metal's own assert on a second open encoder is the fake's crash.
            let mut t = setup();
            t.with_encoder();
            t.p.script()
                .push("computeCommandEncoderWithDescriptor:", Step::crash());
            let site =
                catch_death(|| t.d.decode(Hold::NONE, &seg(0, 0), cursor(vec![]))).unwrap_err();
            assert_eq!(site.kind, DeathKind::Abort);
        }

        #[test]
        fn a_metal_raise_faults_and_propagates() {
            let mut t = setup();
            t.p.script().push(
                "insertCompressedTextureReinterpretationFlush",
                Step::unrecognized("x"),
            );
            let pl = t.with_pipeline();
            let e =
                t.d.decode(Hold::NONE, &seg(0, 0), cursor(rec(0xe3, &[])))
                    .unwrap_err();
            assert_eq!(
                platform_site(&e),
                Some("insertCompressedTextureReinterpretationFlush")
            );
            assert_eq!(t.p.call_log().count("endEncoding"), 1);
            assert!(t.d.encoder.is_none() && t.d.pipeline.is_none());
            assert_eq!(pl.retain_count(), 1);
        }

        /// A raise from `endEncoding` in the handler's fault replaces the exception and leaves the
        /// encoder in place.
        #[test]
        fn a_raise_in_the_handlers_fault_replaces_the_exception() {
            let mut t = setup();
            t.p.script().push(
                "endEncoding",
                Step::raise("NSInternalInconsistencyException", "x"),
            );
            let e =
                t.d.decode(Hold::NONE, &seg(0, 0), cursor(rec(0x01, &[])))
                    .unwrap_err();
            assert_eq!(platform_site(&e), Some("endEncoding"));
            assert_eq!(e.detail(), Some("x"));
            assert!(t.d.encoder.is_some());
        }

        /// A raise at the segment's endEncoding is inside the handler: the fault ends it again.
        #[test]
        fn a_raise_at_the_segment_end_faults() {
            let mut t = setup();
            t.p.script()
                .push("endEncoding", Step::raise("NSGenericException", "end"));
            let e =
                t.d.decode(Hold::NONE, &seg(0, 0), cursor(vec![]))
                    .unwrap_err();
            assert_eq!(e.detail(), Some("end"));
            assert_eq!(t.p.call_log().count("endEncoding"), 2);
            assert!(t.d.encoder.is_none());
        }

        /// Under the walk's handler the raise goes on with its own id, after the fault.
        #[test]
        fn the_rethrow_goes_on_with_the_original_id() {
            let mut t = setup();
            let site =
                t.d.decode(Hold::NONE, &seg(0, 0), cursor(rec(0x01, &[])))
                    .unwrap_err()
                    .site();
            assert_eq!(
                site,
                DeathSite {
                    kind: DeathKind::UnhandledRaise,
                    file: RaiseKind::UnrecognizedCommand.name(),
                    line: 0
                }
            );
            // The fault ran first.
            assert_eq!(t.p.call_log().count("endEncoding"), 1);
        }
    }

    mod fault {
        use super::*;

        #[test]
        fn ends_releases_and_nils_both_then_messages_nil() {
            let mut t = setup();
            let e = t.with_encoder();
            let pl = t.with_pipeline();
            t.d.fault().unwrap();
            assert_eq!(t.sels(), vec!["endEncoding"]);
            assert!(t.d.encoder.is_none() && t.d.pipeline.is_none());
            assert_eq!((e.retain_count(), pl.retain_count()), (1, 1));
            t.d.fault().unwrap();
            assert_eq!(t.sels(), vec!["endEncoding"], "a second fault messages nil");
        }

        #[test]
        fn a_raise_from_end_encoding_stops_the_fault() {
            let mut t = setup();
            t.with_encoder();
            t.with_pipeline();
            t.p.script()
                .push("endEncoding", Step::raise("NSGenericException", "x"));
            t.d.fault().unwrap_err();
            assert!(t.d.encoder.is_some() && t.d.pipeline.is_some());
        }

        /// The raise from `endEncoding` is what leaves the fault, not the raise in flight.
        #[test]
        fn fault_surfaces_the_end_encoding_raise() {
            let mut t = setup();
            t.with_encoder();
            t.with_pipeline();
            t.p.script().push(
                "endEncoding",
                Step::raise("NSInternalInconsistencyException", "x"),
            );
            let e = t.d.fault().unwrap_err();
            assert_eq!(platform_site(&e), Some("endEncoding"));
            assert!(t.d.encoder.is_some() && t.d.pipeline.is_some());
        }
    }
}
