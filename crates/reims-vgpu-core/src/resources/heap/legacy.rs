//! The legacy heap: a private Metal heap with a private buffer over all of it, paged by blits
//! through a no-copy transfer buffer over the guest's pages that is bound at the first prepare
//! and kept until a discard.

use std::sync::Arc;

use crate::guest::GuestPtr;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::device::GpuDevice;
use crate::platform::resource::Heap;
use crate::platform::values::{HeapDesc, OptionSet};
use crate::platform::{Handle, Platform};
use crate::raise::{Kept, keep_forever};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::buffer::child::HeapBufferResource;
use crate::resources::heap::HeapResourceDescriptor;
use crate::resources::resource::{
    HAZARD_TRACKING_DEFAULT, HAZARD_TRACKING_TRACKED, HEAP_TYPE_PLACEMENT, PURGEABLE_EMPTY,
    PURGEABLE_NON_VOLATILE, PURGEABLE_VOLATILE, RESOURCE_STORAGE_MODE_PRIVATE, Resource,
    ResourceClass, ResourceKind, STORAGE_MODE_PRIVATE, TaskRef, features_of, platform_for,
    read_descriptor,
};
use crate::resources::{RaiseKind, raise};
use crate::rt::Hold;
use crate::sync::Mutex;

/// A private heap paged by blits through a buffer over all of it.
pub struct LegacyHeapResource<P: Platform> {
    task: TaskRef<P>,
    heap: Option<P::Heap>,
    /// The guest's pages, bound at the first prepare.
    transfer_buffer: Mutex<Option<P::Buffer>>,
    /// A private buffer over the whole heap.
    transfer_heap_buffer: Option<P::Buffer>,
    length: u64,
    page: u32,
}

impl<P: Platform> LegacyHeapResource<P> {
    /// A private heap of the record's size, and the buffer over it. A shared heap is `None`
    /// with no log; a placement is refused (logged); a tracked heap needs hazard tracking
    /// (logged). A raise making the heap buffer leaks the heap.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        task: TaskRef<P>,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let platform = platform_for(&task);
        let d: HeapResourceDescriptor = read_descriptor(descriptor);
        // A shared heap is silently nothing (testing.md 7.1).
        if !d.is_private() {
            return Ok(None);
        }
        if placement.is_some() {
            log::warn!("private heap refused: it takes no placement");
            return Ok(None);
        }
        // The heap is made only inside a platform's scope.
        if platform.is_none() {
            return Ok(None);
        }
        if d.is_tracked() && !features_of(&task).supports_heap_hazard_tracking {
            log::warn!("tracked heap refused: the task's features have no heap hazard tracking");
            return Ok(None);
        }
        let desc = HeapDesc {
            size: d.length(),
            heap_type: HEAP_TYPE_PLACEMENT,
            options: vec![
                OptionSet::StorageMode(STORAGE_MODE_PRIVATE),
                OptionSet::HazardTrackingMode(if d.is_tracked() {
                    HAZARD_TRACKING_TRACKED
                } else {
                    HAZARD_TRACKING_DEFAULT
                }),
            ],
            address_ranges: None,
            pinned_gpu_address: None,
        };
        // A raise leaves nothing made.
        let heap = match task.upgrade() {
            Some(t) => t
                .mtl_device()
                .map_or(Ok(None), |device| device.new_heap(&desc).map(|m| m.heap))?,
            None => None,
        };
        // Nothing made, no log.
        let Some(heap) = heap else {
            return Ok(None);
        };
        let transfer_heap_buffer =
            match heap.new_buffer_of_length(d.length(), RESOURCE_STORAGE_MODE_PRIVATE, 0) {
                Ok(b) => b,
                Err(e) => {
                    // The heap is kept for good, with its GPU VA range.
                    keep_forever(heap, Kept::MetalNumbering);
                    return Err(e);
                }
            };
        // Refused: the heap is released.
        if transfer_heap_buffer.is_none() {
            return Ok(None);
        }
        Ok(Some(Resource::new(ResourceKind::LegacyHeap(
            LegacyHeapResource {
                task,
                heap: Some(heap),
                transfer_buffer: Mutex::new(None),
                transfer_heap_buffer,
                length: d.length(),
                page: d.page(),
            },
        ))))
    }

    /// The task.
    pub fn task(&self) -> &TaskRef<P> {
        &self.task
    }

    /// Makes a discarded heap non-volatile again; if Metal emptied it, the host's copy is
    /// stale.
    fn devolatilize(&self, this: &Resource<P>) -> RaiseResult<()> {
        if this.state.is_volatile() {
            let previous = self
                .heap
                .as_ref()
                .map_or(Ok(0), |h| h.set_purgeable_state(PURGEABLE_NON_VOLATILE))?;
            this.state.set_is_volatile(false);
            // A discarded heap is copied again only if Metal emptied it.
            if previous == PURGEABLE_EMPTY {
                this.state.set_host_valid(false);
            }
        }
        Ok(())
    }

    /// Binds the guest pages unless bound; a torn-down task gives none, which raises.
    fn ensure_physical(&self, hold: Hold<'_>) -> RaiseResult<()> {
        // Kept until a discard.
        if self.transfer_buffer.lock().is_some() {
            return Ok(());
        }
        let tb = match self.task.upgrade() {
            Some(t) => Some(t.new_buffer_for_virtual_page(hold, self.page, self.length)?),
            None => None,
        };
        let Some(tb) = tb else {
            return raise(RaiseKind::NoTransferBuffer);
        };
        // A plain store: a buffer a racing caller stored between the check above and this
        // store is overwritten unreleased, and kept for good with its GPU VA and its guest
        // pages' mapping.
        if let Some(raced) = self.transfer_buffer.lock().replace(tb) {
            keep_forever(raced, Kept::MetalNumbering);
        }
        Ok(())
    }

    /// Encodes one copy between the transfer buffer and the heap's buffer (skipped with no
    /// encoder).
    fn copy(&self, encoder: Option<&P::BlitEncoder>, upload: bool) -> RaiseResult<()> {
        let tb = self.transfer_buffer.lock();
        let (src, dst) = if upload {
            (tb.as_ref(), self.transfer_heap_buffer.as_ref())
        } else {
            (self.transfer_heap_buffer.as_ref(), tb.as_ref())
        };
        encoder.map_or(Ok(()), |e| {
            e.encode(
                OnFault::Return,
                BlitCmd::CopyBufferToBuffer {
                    source_buffer: src,
                    source_offset: 0,
                    destination_buffer: dst,
                    destination_offset: 0,
                    size: self.length,
                },
            )
        })
    }
}

impl<P: Platform> ResourceClass<P> for LegacyHeapResource<P> {
    fn initial_guest_valid(&self) -> bool {
        true
    }

    fn get_heap(&self, this: &Resource<P>) -> RaiseResult<Option<P::Heap>> {
        let _ = this;
        Ok(self.heap.as_ref().map(Handle::clone_ref))
    }

    fn prepare_in_encoder(
        &self,
        hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        self.devolatilize(this)?;
        if this.state.host_valid() {
            return Ok(());
        }
        if !this.state.guest_valid() {
            log::warn!("preparing a heap whose guest copy is not valid");
        }
        self.ensure_physical(hold)?;
        self.copy(encoder, true)?;
        this.state.set_host_valid(true);
        Ok(())
    }

    fn synchronize_in_encoder(
        &self,
        hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        if this.state.guest_valid() {
            return Ok(());
        }
        if !this.state.host_valid() {
            log::warn!("synchronizing a heap whose host copy is not valid");
        }
        self.ensure_physical(hold)?;
        self.copy(encoder, false)?;
        // Host-valid, not guest-valid, so the heap stays guest-invalid and every later synchronize
        // copies again.
        this.state.set_host_valid(true);
        Ok(())
    }

    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let old = self.transfer_buffer.lock().take();
        drop(old);
        // Metal first, then the flag (the private buffer's order is the opposite).
        self.heap
            .as_ref()
            .map_or(Ok(0), |h| h.set_purgeable_state(PURGEABLE_VOLATILE))?;
        this.state.set_is_volatile(true);
        Ok(())
    }

    fn new_child_buffer_resource_with_buffer(
        &self,
        this: &Arc<Resource<P>>,
        buffer: P::Buffer,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        // Not discardable.
        Ok(Some(HeapBufferResource::new(this.clone(), buffer, false)))
    }
}

impl<P: Platform> Drop for LegacyHeapResource<P> {
    fn drop(&mut self) {
        // The heap, the transfer buffer, the heap's buffer.
        drop(self.heap.take());
        drop(self.transfer_buffer.get_mut().take());
        drop(self.transfer_heap_buffer.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::heap::tests::heap_desc;
    use crate::resources::resource::test_support::{Env, gone_task, kind, logged, placement};
    use crate::resources::{DeathKind, Op};
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::{Fk, Model, kind};
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Fp = FakePlatform;
    const COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";

    fn legacy(env: &Env, flags: u8) -> Option<Arc<Resource<Fp>>> {
        env.run(|| LegacyHeapResource::new(env.task(), &heap_desc(6, 0x8000, flags), None))
            .unwrap()
    }

    fn parts(r: &Resource<Fp>) -> &LegacyHeapResource<Fp> {
        match &r.kind {
            ResourceKind::LegacyHeap(k) => k,
            _ => panic!("kind"),
        }
    }

    /// The value the last heap was made from, as the creator logged it.
    fn made_from(env: &Env) -> String {
        match env.p.call_log().calls_to("newHeapWithDescriptor:").last() {
            Some(c) => match &c.args[0] {
                Arg::Val(text) => text.clone(),
                x => panic!("{x:?}"),
            },
            None => panic!("no heap made"),
        }
    }

    fn encoder(env: &Env) -> Fk<kind::BlitEncoder> {
        env.p.world().new_obj::<kind::BlitEncoder>(Model::None)
    }

    /// A transfer buffer a racing caller (another FIFO's synchronize) stored between this
    /// prepare's check and its store is overwritten unreleased: it stays live, with its GPU VA
    /// and its guest pages' mapping, for good
    #[test]
    fn a_transfer_buffer_a_racing_caller_stored_is_kept() {
        let env = Env::new();
        let r = legacy(&env, 1).unwrap();
        let enc = encoder(&env);
        let racer = Arc::clone(&r);
        let raced = Arc::new(Mutex::new(None));
        let seen = Arc::clone(&raced);
        env.rec.during_next("new_buffer_for_virtual_page", move || {
            let k = parts(&racer);
            k.ensure_physical(Hold::NONE).unwrap();
            *seen.lock() = k.transfer_buffer.lock().as_ref().map(Handle::id);
        });
        r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
        let raced = raced.lock().take().expect("the racer stored a buffer");
        let stored = parts(&r).transfer_buffer.lock().as_ref().map(Handle::id);
        assert_ne!(stored, Some(raced));
        assert!(
            env.p.world().find(raced).is_some(),
            "the raced buffer is kept"
        );
    }

    mod init {
        use super::*;

        #[test]
        fn a_private_heap_and_its_whole_heap_buffer() {
            let env = Env::new();
            let r = legacy(&env, 1).unwrap();
            let log = env.p.call_log();
            assert_eq!(
                log.sels(),
                [
                    "mtl_device",
                    "newHeapWithDescriptor:",
                    "newBufferWithLength:options:offset:"
                ]
            );
            // A private placement heap, untracked.
            assert_eq!(
                made_from(&env),
                "HeapDesc { size: 32768, heap_type: 1, options: [StorageMode(2), \
                 HazardTrackingMode(0)], address_ranges: None, pinned_gpu_address: None }"
            );
            assert_eq!(
                log.calls_to("newBufferWithLength:options:offset:")[0].args,
                vec![Arg::U64(0x8000), Arg::U64(0x20), Arg::U64(0)]
            );
            assert!(r.state.guest_valid() && !r.state.host_valid());
            assert!(parts(&r).transfer_buffer.lock().is_none());
            assert_eq!(env.rec.log().count("new_buffer_for_virtual_page"), 0);
        }

        #[test]
        fn a_shared_heap_is_nil_silently() {
            // In-process heaps must be private.
            let env = Env::new();
            let (made, lines) = logged(|| legacy(&env, 0));
            assert!(made.is_none());
            assert!(env.p.call_log().is_empty());
            assert!(lines.is_empty());
        }

        #[test]
        fn placement_is_refused_after_the_private_check() {
            let env = Env::new();
            let (_m, pl) = placement(0);
            let (r, lines) = logged(|| {
                env.run(|| LegacyHeapResource::new(env.task(), &heap_desc(1, 16, 1), Some(pl)))
            });
            assert!(r.unwrap().is_none());
            assert_eq!(lines, vec!["private heap refused: it takes no placement"]);
            let (r, lines) = logged(|| {
                env.run(|| LegacyHeapResource::new(env.task(), &heap_desc(1, 16, 0), Some(pl)))
            });
            assert!(r.unwrap().is_none());
            assert!(lines.is_empty(), "the shared check comes first");
        }

        #[test]
        fn tracked_needs_hazard_tracking() {
            let env = Env::new();
            let (made, lines) = logged(|| legacy(&env, 3));
            assert!(made.is_none());
            assert_eq!(
                lines,
                vec!["tracked heap refused: the task's features have no heap hazard tracking"]
            );
            assert_eq!(env.p.call_log().count("newHeapWithDescriptor:"), 0);
            env.set_features(|f| f.supports_heap_hazard_tracking = true);
            assert!(legacy(&env, 3).is_some());
            assert!(made_from(&env).contains("HazardTrackingMode(2)"));
        }

        #[test]
        fn nil_heap_or_heap_buffer_is_nil() {
            let env = Env::new();
            env.p.script().push("newHeapWithDescriptor:", Step::Nil);
            assert!(legacy(&env, 1).is_none());
            env.p
                .script()
                .push("newBufferWithLength:options:offset:", Step::Nil);
            assert!(legacy(&env, 1).is_none());
        }

        #[test]
        fn a_heap_buffer_raise_propagates() {
            let env = Env::new();
            env.p.script().push(
                "newBufferWithLength:options:offset:",
                Step::raise("NSInvalidArgumentException", "x"),
            );
            assert!(
                env.run(|| LegacyHeapResource::new(env.task(), &heap_desc(6, 16, 1), None))
                    .is_err()
            );
        }

        #[test]
        fn tracked_on_a_gone_task_dies_reading_its_features() {
            let env = Env::new();
            let d = catch_death(|| {
                env.run(|| LegacyHeapResource::<Fp>::new(gone_task(), &heap_desc(1, 16, 3), None))
            });
            assert_eq!(d.err(), Some(DeathKind::GoneTaskFeatures.site()));
        }
    }

    mod prepare {
        use super::*;

        #[test]
        fn transfer_buffer_is_kept_until_discard() {
            let env = Env::new();
            let r = legacy(&env, 1).unwrap();
            let enc = encoder(&env);
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            let thb = parts(&r).transfer_heap_buffer.as_ref().map(|b| b.id());
            let tb = parts(&r).transfer_buffer.lock().as_ref().map(|b| b.id());
            let c = env.p.call_log().calls_to(COPY);
            assert_eq!(
                c[0].args,
                vec![
                    Arg::Obj(tb),
                    Arg::U64(0),
                    Arg::Obj(thb),
                    Arg::U64(0),
                    Arg::U64(0x8000)
                ]
            );
            assert_eq!(
                env.rec.log().calls_to("new_buffer_for_virtual_page")[0].args,
                vec![Arg::U32(6), Arg::U64(0x8000)]
            );
            assert!(r.state.host_valid());
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(env.p.call_log().count(COPY), 1);
            r.set_is_guest_valid(false);
            r.set_is_host_valid(false);
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            // The transfer buffer of the first prepare is kept.
            assert_eq!(env.rec.log().count("new_buffer_for_virtual_page"), 1);
        }

        #[test]
        fn a_gone_task_raises_no_transfer_buffer() {
            let other = LegacyHeapResource::<Fp> {
                task: gone_task(),
                heap: None,
                transfer_buffer: Mutex::new(None),
                transfer_heap_buffer: None,
                length: 0x8000,
                page: 0,
            };
            let hold = Hold::NONE;
            assert_eq!(
                kind(other.ensure_physical(hold)),
                Some(RaiseKind::NoTransferBuffer)
            );
        }

        #[test]
        fn devolatilize_after_discard_recopies_only_when_emptied() {
            // Opcode 63 makes it volatile, and the next prepare copies again only if Metal emptied
            // it.
            let env = Env::new();
            let r = legacy(&env, 1).unwrap();
            let enc = encoder(&env);
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            r.discard().unwrap();
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(env.p.call_log().count(COPY), 1);
            assert!(!r.state.is_volatile());
            r.discard().unwrap();
            env.p
                .script()
                .push("setPurgeableState:", Step::value(PURGEABLE_EMPTY));
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(env.p.call_log().count(COPY), 2);
        }
    }

    mod synchronize {
        use super::*;

        #[test]
        fn marks_host_valid_so_every_synchronize_copies_again() {
            // A synchronize sets host-valid: the heap stays guest-invalid.
            let env = Env::new();
            let r = legacy(&env, 1).unwrap();
            let enc = encoder(&env);
            r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(env.p.call_log().count(COPY), 0, "guest-valid: nothing");
            r.set_is_guest_valid(false);
            r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            let thb = parts(&r).transfer_heap_buffer.as_ref().map(|b| b.id());
            assert_eq!(env.p.call_log().calls_to(COPY)[0].args[0], Arg::Obj(thb));
            assert!(r.state.host_valid() && !r.state.guest_valid());
            assert!(r.needs_sync());
            r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(env.p.call_log().count(COPY), 2);
        }
    }

    mod discard_getters {
        use super::*;

        #[test]
        fn discard_sends_metal_then_marks_volatile() {
            let env = Env::new();
            let r = legacy(&env, 1).unwrap();
            r.prepare_in_encoder(Hold::NONE, None).unwrap();
            let m = env.p.call_log().mark();
            r.discard().unwrap();
            let c = env.p.call_log().since(m);
            assert_eq!(
                (c[0].sel, c[0].args.clone()),
                ("setPurgeableState:", vec![Arg::U64(3)])
            );
            assert!(r.state.is_volatile() && parts(&r).transfer_buffer.lock().is_none());
        }

        #[test]
        fn get_heap_and_no_get_buffer() {
            let env = Env::new();
            let r = legacy(&env, 1).unwrap();
            let h = r.get_heap().unwrap().unwrap();
            assert_eq!(Some(h.id()), parts(&r).heap.as_ref().map(|x| x.id()));
            let hold = Hold::NONE;
            // In-process a heap listed where a buffer is expected faults on this.
            assert_eq!(
                kind(r.get_buffer()),
                Some(RaiseKind::Unrecognized(Op::GetBuffer))
            );
            assert_eq!(
                kind(r.get_resource()),
                Some(RaiseKind::Unrecognized(Op::GetResource))
            );
            assert_eq!(
                kind(r.prepare_with_cpu(hold)),
                Some(RaiseKind::Unrecognized(Op::PrepareCpu))
            );
            assert!(!r.needs_cpu_paging() && r.supports_discard());
        }

        #[test]
        fn children_are_plain_heap_buffers_holding_their_buffer() {
            let env = Env::new();
            let r = legacy(&env, 1).unwrap();
            let h = r.get_heap().unwrap().unwrap();
            let b = h.new_buffer_of_length(16, 0x20, 0).unwrap().unwrap();
            let c = r.new_child_buffer_resource_with_buffer(b).unwrap().unwrap();
            assert!(matches!(&c.kind, ResourceKind::HeapBuffer(k) if !k.discardable()));
            assert!(!c.supports_discard());
            // Deleting the base leaves the child working.
            drop(r);
            assert!(c.get_buffer().unwrap().is_some());
        }

        #[test]
        fn last_release_releases_the_three_handles() {
            let env = Env::new();
            let r = legacy(&env, 1).unwrap();
            r.prepare_in_encoder(Hold::NONE, None).unwrap();
            let k = parts(&r);
            let h = k.heap.as_ref().map(|x| x.clone_ref()).unwrap();
            let tb = k
                .transfer_buffer
                .lock()
                .as_ref()
                .map(|x| x.clone_ref())
                .unwrap();
            let thb = k
                .transfer_heap_buffer
                .as_ref()
                .map(|x| x.clone_ref())
                .unwrap();
            let before = (h.retain_count(), tb.retain_count(), thb.retain_count());
            drop(r);
            assert_eq!(
                (h.retain_count(), tb.retain_count(), thb.retain_count()),
                (before.0 - 1, before.1 - 1, before.2 - 1)
            );
        }
    }
}
