//! Creation of an ICB (object type 20): the object record, the creation command, the Metal
//! descriptor of its serialized one, the placement's resource ID checked and both words written
//! back, and the resource over the new buffer.

use zerocopy::IntoBytes;

use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::memory::cursor::Cursor;
use crate::platform::Handle;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::icb::IndirectCommandBuffer;
use crate::platform::values::IcbDesc;
use crate::raise::RaiseResult;
use crate::raise::{Kept, keep_forever};
use crate::resources::icb::{IndirectCommandBufferFlags, IndirectCommandLayout};
use crate::resources::manager::DeathKind as ManagerDeath;
use crate::resources::manager::records::OperationHeader;

use super::ManagerOps;
use super::records::{
    ICB_OBJECT_TYPE, NEW_ICB_COMMAND, NewIcbCommand, ObjectIcb, SerializedIcbDescriptor,
};

/// The value, or `Ok(Err(e))` from the enclosing creator.
macro_rules! refuse {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Ok(Err(e)),
        }
    };
}

/// The Metal descriptor of the serialized one, with `resource_index` when given.
fn descriptor(s: &SerializedIcbDescriptor, resource_index: Option<u64>) -> IcbDesc {
    IcbDesc {
        command_types: u64::from(s.command_types),
        max_vertex_buffer_bind_count: u64::from(s.max_vertex_buffer_bind_count),
        max_fragment_buffer_bind_count: u64::from(s.max_fragment_buffer_bind_count),
        max_kernel_buffer_bind_count: u64::from(s.max_kernel_buffer_bind_count),
        max_object_buffer_bind_count: u64::from(s.max_object_buffer_bind_count),
        max_mesh_buffer_bind_count: u64::from(s.max_mesh_buffer_bind_count),
        max_kernel_threadgroup_memory_bind_count: u64::from(
            s.max_kernel_threadgroup_memory_bind_count,
        ),
        max_object_threadgroup_memory_bind_count: u64::from(
            s.max_object_threadgroup_memory_bind_count,
        ),
        inherit_pipeline_state: s.bit(0),
        inherit_buffers: s.bit(1),
        support_ray_tracing: s.bit(2),
        support_dynamic_attribute_stride: s.bit(3),
        max_scissor_rect_count: u64::from(s.max_scissor_rect_count),
        max_viewport_count: u64::from(s.max_viewport_count),
        inherit_stencil_reference_values: s.bit(6),
        inherit_depth_test_bounds: s.bit(11),
        inherit_scissor_rects: s.bit(12),
        inherit_viewports: s.bit(13),
        inherit_blend_color: s.bit(14),
        // The u32 zero-extended into the setter's 64-bit argument
        allow_override_render_states: u64::from(s.allow_override_render_states).cast_signed(),
        inherit_depth_stencil_state: s.bit(4),
        inherit_depth_bias: s.bit(5),
        inherit_depth_clip_mode: s.bit(7),
        inherit_cull_mode: s.bit(8),
        inherit_front_facing_winding: s.bit(9),
        inherit_triangle_fill_mode: s.bit(10),
        resource_index,
    }
}

/// A new ICB, its command layout and its flags. `None` is Metal's nil, which is not checked:
/// the resource then holds nil.
type CreatedIcb<P> = (
    Option<<P as Platform>::Icb>,
    IndirectCommandLayout,
    IndirectCommandBufferFlags,
);

/// The ICB the creation command describes, made on the manager's device.
///
/// With a placement, its resource index (at +0) is set on the descriptor when its resource ID
/// (at +8) is nonzero, and when the index is nonzero the new ICB's resource ID must be the
/// placement's. On success the ICB's resource ID and then its resource index are written back.
fn new_icb<P: Platform, M: ManagerOps<P>>(
    m: &M,
    cursor: &mut Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<CreatedIcb<P>, String>> {
    // The command is read whole, 0x50 bytes, its tail included.
    let Ok(command) = cursor.read::<NewIcbCommand>() else {
        return Ok(Err(
            "indirect command buffer command not readable".to_owned()
        ));
    };
    let (resource_index, resource_id) = match placement {
        Some(p) => (
            p.read_u64(0, ManagerDeath::NullPlacement),
            p.read_u64(8, ManagerDeath::NullPlacement),
        ),
        None => (0, 0),
    };
    // The index is set (not forced) when the ID is nonzero, and checked when the index is nonzero.
    let check_resource_id = resource_index != 0;
    let descriptor = descriptor(
        &command.descriptor,
        (resource_id != 0).then_some(resource_index),
    );
    // Metal's nil is not checked
    let buffer = match m.device() {
        Some(device) => device.new_indirect_command_buffer(
            &descriptor,
            u64::from(command.max_command_count),
            u64::from(command.options),
        )?,
        None => None,
    };
    let gpu_resource_id = |b: &Option<P::Icb>| b.as_ref().map_or(0, |b| b.gpu_resource_id());
    if check_resource_id && gpu_resource_id(&buffer) != resource_id {
        // The ID is asked a second time, for the line, whether or not the line is written.
        let got = gpu_resource_id(&buffer);
        log::warn!(
            "indirect command buffer placed at resource ID {got:#x}, not the guest's \
             {resource_id:#x}"
        );
        drop(buffer);
        return Ok(Err(
            "indirect command buffer not placed where the guest asked".to_owned(),
        ));
    }
    if let Some(p) = placement {
        p.write_u64(8, gpu_resource_id(&buffer), ManagerDeath::NullPlacement);
        p.write_u64(
            0,
            buffer.as_ref().map_or(0, |b| b.resource_index()),
            ManagerDeath::NullPlacement,
        );
    }
    let d = &command.descriptor;
    let flags = IndirectCommandBufferFlags {
        bits: u32::from(d.bit(0))
            | u32::from(d.bit(1)).wrapping_shl(1)
            | u32::from(d.bit(3)).wrapping_shl(2),
        command_types: d.command_types,
        max_vertex_buffer_bind_count: u64::from(d.max_vertex_buffer_bind_count),
        max_fragment_buffer_bind_count: u64::from(d.max_fragment_buffer_bind_count),
        max_kernel_buffer_bind_count: u64::from(d.max_kernel_buffer_bind_count),
        max_kernel_threadgroup_memory_bind_count: u64::from(
            d.max_kernel_threadgroup_memory_bind_count,
        ),
    };
    Ok(Ok((buffer, command.command_layout, flags)))
}

/// An ICB resource from the object's record and creation command. A refusal is the error; a
/// raise from Metal's creation goes on.
pub(crate) fn create_indirect_command_buffer<P: Platform, M: ManagerOps<P>>(
    m: &M,
    entry: &ObjectEntry,
    mut cursor: Cursor,
    placement: Option<GuestPtr>,
) -> RaiseResult<Result<M::IcbResource, String>> {
    if entry.object_type() != ICB_OBJECT_TYPE {
        return Ok(Err(format!(
            "object of type {} is not an indirect command buffer",
            entry.object_type()
        )));
    }
    // A short record is the cursor's error, bare
    let record = refuse!(cursor.read::<ObjectIcb>());
    let header = refuse!(cursor.read::<OperationHeader>());
    if header.command_type != NEW_ICB_COMMAND {
        return Ok(Err(format!(
            "unknown command type {} for an indirect command buffer",
            header.command_type
        )));
    }
    let (icb, layout, flags) = refuse!(new_icb(m, &mut cursor, placement)?);
    // Metal's creation reference is never released; the resource takes one of its own, so the ICB
    // (its resource index and GPU address) outlives it.
    if let Some(icb) = &icb {
        keep_forever(icb.clone_ref(), Kept::MetalNumbering);
    }
    match m.new_icb_resource(record.as_bytes(), icb, layout, flags)? {
        Some(resource) => Ok(Ok(resource)),
        // The resource's creation never answers nil; a nil would reach the registration's
        // check
        None => crate::resources::manager::cache::nil_object(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::testing::*;
    use super::*;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{Step, kind};

    const NEW: &str = "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:";

    #[test]
    fn the_descriptor_has_the_records_fields() {
        let s = SerializedIcbDescriptor {
            command_types: 0x21,
            max_vertex_buffer_bind_count: 1,
            max_fragment_buffer_bind_count: 2,
            max_kernel_buffer_bind_count: 3,
            max_object_buffer_bind_count: 4,
            max_mesh_buffer_bind_count: 5,
            max_kernel_threadgroup_memory_bind_count: 6,
            max_object_threadgroup_memory_bind_count: 7,
            max_scissor_rect_count: 8,
            max_viewport_count: 9,
            field_0d: 0xee,
            bits: 0b0101_0101_0101_0101,
            allow_override_render_states: 0xffff_fff0,
        };
        // Bits 0, 2, 4, … set.
        assert_eq!(
            descriptor(&s, Some(4)),
            IcbDesc {
                command_types: 0x21,
                max_vertex_buffer_bind_count: 1,
                max_fragment_buffer_bind_count: 2,
                max_kernel_buffer_bind_count: 3,
                max_object_buffer_bind_count: 4,
                max_mesh_buffer_bind_count: 5,
                max_kernel_threadgroup_memory_bind_count: 6,
                max_object_threadgroup_memory_bind_count: 7,
                inherit_pipeline_state: true,
                inherit_buffers: false,
                support_ray_tracing: true,
                support_dynamic_attribute_stride: false,
                max_scissor_rect_count: 8,
                max_viewport_count: 9,
                inherit_stencil_reference_values: true,
                inherit_depth_test_bounds: false,
                inherit_scissor_rects: true,
                inherit_viewports: false,
                inherit_blend_color: true,
                // A u32 zero-extended into the 64-bit argument.
                allow_override_render_states: 0xffff_fff0,
                inherit_depth_stencil_state: true,
                inherit_depth_bias: false,
                inherit_depth_clip_mode: false,
                inherit_cull_mode: true,
                inherit_front_facing_winding: false,
                inherit_triangle_fill_mode: true,
                resource_index: Some(4),
            }
        );
    }

    /// The creator's value for `icb_command`'s descriptor, with `resource_index`.
    fn sent(resource_index: Option<u64>) -> Arg {
        Arg::val(&descriptor(&icb_command(1, 0).descriptor, resource_index))
    }

    fn create(
        o: &FakeOps,
        ty: u8,
        data: &[u8],
        placement: Option<GuestPtr>,
    ) -> Result<MadeIcb, String> {
        create_indirect_command_buffer(o, &entry(u32::from(ty)), cursor(data), placement).unwrap()
    }

    #[test]
    fn the_object_type_must_be_20() {
        let o = FakeOps::new();
        let e = create(&o, 6, &icb_object(54, &icb_command(1, 0)), None).unwrap_err();
        assert_eq!(e, "object of type 6 is not an indirect command buffer");
        assert!(o.log().is_empty());
    }

    #[test]
    fn short_records_give_the_cursors_error_bare() {
        let o = FakeOps::new();
        let full = icb_object(54, &icb_command(1, 0));
        let e = create(&o, 20, &full[..11], None).unwrap_err();
        assert!(!e.is_empty() && !e.starts_with("Failed"), "{e}");
        let e = create(&o, 20, &full[..12 + 7], None).unwrap_err();
        assert!(!e.is_empty() && !e.starts_with("Failed"), "{e}");
        assert!(o.calls().is_empty());
    }

    #[test]
    fn another_command_type_is_refused() {
        let o = FakeOps::new();
        let e = create(&o, 20, &icb_object(53, &icb_command(1, 0)), None).unwrap_err();
        assert_eq!(e, "unknown command type 53 for an indirect command buffer");
    }

    #[test]
    fn the_command_is_read_whole_including_its_tail_padding() {
        // 0x4e bytes of fields, 0x50 read.
        let o = FakeOps::new();
        let full = icb_object(54, &icb_command(1, 0));
        let e = create(&o, 20, &full[..full.len() - 2], None).unwrap_err();
        assert_eq!(e, "indirect command buffer command not readable");
        assert!(o.log().calls_to(NEW).is_empty());
    }

    #[test]
    fn without_a_placement_the_resource_gets_the_record_layout_and_flags() {
        let o = FakeOps::new();
        let cmd = icb_command(5, 0x21);
        let made = create(&o, 20, &icb_object(54, &cmd), None).unwrap();
        assert_eq!(
            made.descriptor,
            bytes(&ObjectIcb {
                length: 0x1234,
                page: 7
            })
        );
        assert_eq!(made.layout, layout());
        assert_eq!(
            made.flags,
            IndirectCommandBufferFlags {
                // pipeline -> bit 0, buffers -> bit 1, stride (bit 3) -> bit 2
                bits: 0b101,
                command_types: DRAW | DRAW_INDEXED,
                max_vertex_buffer_bind_count: 3,
                max_fragment_buffer_bind_count: 4,
                max_kernel_buffer_bind_count: 5,
                max_kernel_threadgroup_memory_bind_count: 6,
            }
        );
        let icb = made.icb.unwrap();
        assert_eq!(
            last_args(&o, NEW),
            [
                Arg::val(&descriptor(&cmd.descriptor, None)),
                Arg::U64(5),
                Arg::U64(0x21)
            ]
        );
        assert!(o.log().calls().iter().all(|c| c.obj != icb.id()));
        assert_eq!(o.calls(), vec!["new_icb_resource"]);
    }

    #[test]
    fn a_nonzero_id_sets_the_index_unforced_and_a_zero_index_is_not_checked() {
        // The index is set (not forced) when the ID is nonzero
        let o = FakeOps::new();
        let (mem, p) = placement(0, 0x77);
        let made = create(&o, 20, &icb_object(54, &icb_command(1, 0)), Some(p)).unwrap();
        let icb = made.icb.unwrap();
        assert_eq!(last_args(&o, NEW)[0], sent(Some(0)));
        assert_eq!(mem.read_u64(8), Some(icb.gpu_resource_id_value()));
    }

    #[test]
    fn a_nonzero_index_checks_the_id() {
        let o = FakeOps::new();
        o.p.world()
            .on_new::<kind::Icb>(|icb| icb.set_gpu_resource_id(0x77));
        let (mem, p) = placement(3, 0x77);
        let made = create(&o, 20, &icb_object(54, &icb_command(1, 0)), Some(p)).unwrap();
        let icb = made.icb.unwrap();
        assert_eq!(last_args(&o, NEW)[0], sent(Some(3)));
        assert_eq!(mem.read_u64(8), Some(icb.gpu_resource_id_value()));
        assert_eq!(mem.read_u64(0), Some(icb.resource_index_value()));
        assert_eq!(icb.resource_index_value(), 3);
    }

    #[test]
    fn a_mismatched_id_fails_and_logs() {
        let o = FakeOps::new();
        o.p.world()
            .on_new::<kind::Icb>(|icb| icb.set_gpu_resource_id(0x55));
        let (mem, p) = placement(3, 0x77);
        let (r, lines) = logged(|| create(&o, 20, &icb_object(54, &icb_command(1, 0)), Some(p)));
        assert_eq!(
            r.unwrap_err(),
            "indirect command buffer not placed where the guest asked"
        );
        // The line's ID is the ICB's.
        assert_eq!(
            lines,
            vec!["indirect command buffer placed at resource ID 0x55, not the guest's 0x77"]
        );
        // Nothing written back, no resource.
        assert_eq!(mem.read_u64(0), Some(3));
        assert_eq!(mem.read_u64(8), Some(0x77));
        assert!(o.calls().is_empty());
    }

    #[test]
    fn a_zero_id_with_a_nonzero_index_is_checked_but_not_set() {
        // The index is set only when the ID is nonzero
        let o = FakeOps::new();
        let (_mem, p) = placement(3, 0);
        let e = create(&o, 20, &icb_object(54, &icb_command(1, 0)), Some(p)).unwrap_err();
        assert_eq!(
            e,
            "indirect command buffer not placed where the guest asked"
        );
        assert_eq!(last_args(&o, NEW)[0], sent(None));
    }

    #[test]
    fn a_zero_placement_is_written_back_unchecked() {
        let o = FakeOps::new();
        let (mem, p) = placement(0, 0);
        let made = create(&o, 20, &icb_object(54, &icb_command(1, 0)), Some(p)).unwrap();
        let icb = made.icb.unwrap();
        assert_eq!(last_args(&o, NEW)[0], sent(None));
        assert_eq!(mem.read_u64(8), Some(icb.gpu_resource_id_value()));
        assert_eq!(mem.read_u64(0), Some(icb.resource_index_value()));
    }

    #[test]
    fn metals_nil_is_wrapped_and_writes_back_zeros() {
        let o = FakeOps::new();
        o.p.script().push(NEW, Step::Nil);
        let (mem, p) = placement(0, 0x9);
        let made = create(&o, 20, &icb_object(54, &icb_command(1, 0)), Some(p)).unwrap();
        assert!(made.icb.is_none());
        assert_eq!(mem.read_u64(0), Some(0));
        assert_eq!(mem.read_u64(8), Some(0));
    }

    #[test]
    fn a_nil_resource_dies_at_the_registration_check() {
        let mut o = FakeOps::new();
        o.icb_nil = true;
        let data = icb_object(54, &icb_command(1, 0));
        let d = catch_death(|| create(&o, 20, &data, None).is_ok()).unwrap_err();
        assert_eq!(d, ManagerDeath::NilObject.site());
    }

    #[test]
    fn the_creation_reference_leaks() {
        // The resource's handle is a second reference over Metal's creation reference, which is
        // never released, so the buffer outlives every handle.
        let o = FakeOps::new();
        let made = create(&o, 20, &icb_object(54, &icb_command(1, 0)), None).unwrap();
        let icb = made.icb.unwrap();
        let id = icb.id();
        let weak = Arc::downgrade(icb.obj());
        drop(icb);
        assert!(weak.upgrade().is_some(), "{id:?} was released");
    }
}
