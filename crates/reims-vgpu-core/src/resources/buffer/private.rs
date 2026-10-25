//! The private buffer: a private Metal buffer, paged by blits through a no-copy transfer buffer
//! over the guest's pages that is bound at the first prepare and kept until a discard.

use std::sync::Arc;

use crate::guest::GuestPtr;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::device::GpuDevice;
use crate::platform::resource::GpuResource;
use crate::platform::{Handle, Platform};
use crate::raise::{Kept, OnFault, RaiseResult, keep_forever};
use crate::resources::buffer::BufferResourceDescriptor;
use crate::resources::resource::AnyResource;
use crate::resources::resource::{
    PURGEABLE_EMPTY, PURGEABLE_NON_VOLATILE, PURGEABLE_VOLATILE, RESOURCE_STORAGE_MODE_PRIVATE,
    Resource, ResourceClass, ResourceKind, TaskRef, read_descriptor,
};
use crate::resources::{RaiseKind, raise};
use crate::rt::Hold;
use crate::sync::Mutex;

/// A private buffer paged by blits.
pub struct PrivateBufferResource<P: Platform> {
    task: TaskRef<P>,
    /// The private buffer: never `None` after a successful creation.
    buffer: Option<P::Buffer>,
    /// The guest's pages, bound at the first prepare and kept until a discard.
    transfer_buffer: Mutex<Option<P::Buffer>>,
    length: u64,
    page: u32,
}

impl<P: Platform> PrivateBufferResource<P> {
    /// A private buffer of the record's length. A placement is refused (logged, `None`); Metal's
    /// nil is `None` without a log. A torn-down task makes nothing.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        task: TaskRef<P>,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        if placement.is_some() {
            log::warn!("private buffer refused: it takes no placement");
            return Ok(None);
        }
        let d: BufferResourceDescriptor = read_descriptor(descriptor);
        // A raise leaves nothing made.
        let buffer = match task.upgrade() {
            Some(t) => t.mtl_device().map_or(Ok(None), |device| {
                device.new_buffer_of_length(
                    OnFault::Return,
                    d.length,
                    RESOURCE_STORAGE_MODE_PRIVATE,
                )
            })?,
            None => None,
        };
        // Metal refuses a length of 0 or one too large.
        if buffer.is_none() {
            return Ok(None);
        }
        Ok(Some(Resource::new(ResourceKind::PrivateBuffer(
            PrivateBufferResource {
                task,
                buffer,
                transfer_buffer: Mutex::new(None),
                length: d.length,
                page: d.page,
            },
        ))))
    }

    /// The task.
    pub fn task(&self) -> &TaskRef<P> {
        &self.task
    }

    /// Binds the guest pages unless bound; a torn-down task gives none, which raises.
    fn ensure_physical(&self, hold: Hold<'_>) -> RaiseResult<()> {
        if self.transfer_buffer.lock().is_some() {
            return Ok(());
        }
        let tb = match self.task.upgrade() {
            Some(t) => Some(t.new_buffer_for_virtual_page(hold, self.page, self.length)?),
            None => None,
        };
        let Some(tb) = tb else {
            return raise(RaiseKind::NoGuestBuffer);
        };
        // A plain store: a buffer a racing caller stored between the check above and this
        // store is overwritten unreleased, and kept for good with its GPU VA and its guest
        // pages' mapping.
        if let Some(raced) = self.transfer_buffer.lock().replace(tb) {
            keep_forever(raced, Kept::MetalNumbering);
        }
        Ok(())
    }

    /// Makes a discarded buffer non-volatile again; if Metal emptied it, the host's copy is
    /// stale.
    fn devolatilize(&self, this: &Resource<P>) -> RaiseResult<()> {
        if this.state.is_volatile() {
            let previous = self
                .buffer
                .as_ref()
                .map_or(Ok(0), |b| b.set_purgeable_state(PURGEABLE_NON_VOLATILE))?;
            this.state.set_is_volatile(false);
            // Only an emptied buffer is copied again.
            if previous == PURGEABLE_EMPTY {
                this.state.set_host_valid(false);
            }
        }
        Ok(())
    }

    fn buffer(&self) -> RaiseResult<P::Buffer> {
        match &self.buffer {
            Some(b) => Ok(b.clone_ref()),
            None => raise(RaiseKind::BufferInvalidated),
        }
    }

    /// Encodes one copy between the transfer buffer and the private buffer (skipped with no
    /// encoder).
    fn copy(&self, encoder: Option<&P::BlitEncoder>, upload: bool) -> RaiseResult<()> {
        let tb = self.transfer_buffer.lock();
        let (src, dst) = if upload {
            (tb.as_ref(), self.buffer.as_ref())
        } else {
            (self.buffer.as_ref(), tb.as_ref())
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

impl<P: Platform> Drop for PrivateBufferResource<P> {
    fn drop(&mut self) {
        // The private buffer, then the transfer buffer.
        drop(self.buffer.take());
        drop(self.transfer_buffer.get_mut().take());
    }
}

impl<P: Platform> ResourceClass<P> for PrivateBufferResource<P> {
    fn initial_guest_valid(&self) -> bool {
        true
    }

    fn get_resource(&self, this: &Resource<P>) -> RaiseResult<Option<AnyResource<P>>> {
        let _ = this;
        Ok(Some(AnyResource::Buffer(self.buffer()?)))
    }

    fn get_buffer(&self, this: &Resource<P>) -> RaiseResult<Option<P::Buffer>> {
        let _ = this;
        Ok(Some(self.buffer()?))
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
            log::warn!("preparing a private buffer whose guest copy is not valid");
        }
        self.ensure_physical(hold)?;
        self.copy(encoder, true)?;
        // Encoded, not completed.
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
            log::warn!("synchronizing a private buffer whose host copy is not valid");
        }
        self.ensure_physical(hold)?;
        self.copy(encoder, false)?;
        this.state.set_guest_valid(true);
        Ok(())
    }

    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let old = self.transfer_buffer.lock().take();
        drop(old);
        // The flag first, then Metal (the legacy heap's order is the opposite). The previous
        // state is ignored.
        this.state.set_is_volatile(true);
        self.buffer
            .as_ref()
            .map_or(Ok(0), |b| b.set_purgeable_state(PURGEABLE_VOLATILE))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::Op;
    use crate::resources::resource::test_support::{Env, gone_task, kind, logged, placement};
    use crate::testkit::calllog::Arg;
    use crate::testkit::objects::{Fk, Model, kind};
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;
    use zerocopy::IntoBytes;

    type Fp = FakePlatform;
    const COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";

    fn desc(length: u64, page: u32) -> Vec<u8> {
        BufferResourceDescriptor {
            length,
            page,
            flags: 1,
            unknown0d: [0; 3],
        }
        .as_bytes()
        .to_vec()
    }

    fn private(env: &Env, len: u64) -> Arc<Resource<Fp>> {
        env.run(|| PrivateBufferResource::new(env.task(), &desc(len, 5), None))
            .unwrap()
            .unwrap()
    }

    fn encoder(env: &Env) -> Fk<kind::BlitEncoder> {
        env.p.world().new_obj::<kind::BlitEncoder>(Model::None)
    }

    fn parts(r: &Resource<Fp>) -> &PrivateBufferResource<Fp> {
        match &r.kind {
            ResourceKind::PrivateBuffer(k) => k,
            _ => panic!("kind"),
        }
    }

    mod init {
        use super::*;

        #[test]
        fn makes_a_private_buffer_and_no_transfer_buffer() {
            let env = Env::new();
            let r = private(&env, 0x100);
            let c = env.p.call_log().calls_to("newBufferWithLength:options:");
            assert_eq!(c.len(), 1);
            assert_eq!(c[0].args, vec![Arg::U64(0x100), Arg::U64(0x20)]);
            assert_eq!(env.rec.log().count("new_buffer_for_virtual_page"), 0);
            assert!(r.state.guest_valid() && !r.state.host_valid());
            assert!(parts(&r).transfer_buffer.lock().is_none());
        }

        #[test]
        fn placement_is_refused_before_metal() {
            let env = Env::new();
            let (_m, pl) = placement(0);
            let r = env.run(|| PrivateBufferResource::new(env.task(), &desc(16, 1), Some(pl)));
            assert!(r.unwrap().is_none());
            assert_eq!(env.p.call_log().count("newBufferWithLength:options:"), 0);
        }

        #[test]
        fn a_nil_buffer_is_nil_without_a_log() {
            let env = Env::new();
            env.p
                .script()
                .push("newBufferWithLength:options:", Step::Nil);
            let (r, lines) =
                logged(|| env.run(|| PrivateBufferResource::new(env.task(), &desc(16, 1), None)));
            assert!(r.unwrap().is_none());
            assert!(lines.is_empty());
        }

        #[test]
        fn a_metal_raise_propagates_and_a_gone_task_is_nil() {
            let env = Env::new();
            env.p.script().push(
                "newBufferWithLength:options:",
                Step::raise("NSInvalidArgumentException", "x"),
            );
            assert!(
                env.run(|| PrivateBufferResource::new(env.task(), &desc(16, 1), None))
                    .is_err()
            );
            assert!(
                PrivateBufferResource::<Fp>::new(gone_task(), &desc(16, 1), None)
                    .unwrap()
                    .is_none()
            );
        }
    }

    mod prepare {
        use super::*;

        #[test]
        fn binds_the_guest_pages_once_and_copies_up() {
            let env = Env::new();
            let r = private(&env, 0x40);
            let enc = encoder(&env);
            let buf = r.get_buffer().unwrap().unwrap();
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert!(r.state.host_valid());
            let tb = parts(&r).transfer_buffer.lock().as_ref().map(|b| b.id());
            let c = env.p.call_log().calls_to(COPY);
            assert_eq!(c.len(), 1);
            assert_eq!(
                c[0].args,
                vec![
                    Arg::Obj(tb),
                    Arg::U64(0),
                    Arg::Obj(Some(buf.id())),
                    Arg::U64(0),
                    Arg::U64(0x40)
                ]
            );
            assert_eq!(
                env.rec.log().calls_to("new_buffer_for_virtual_page")[0].args,
                vec![Arg::U32(5), Arg::U64(0x40)]
            );
            // Host-valid: nothing more.
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(env.p.call_log().count(COPY), 1);
            // The transfer buffer of the first prepare is kept.
            r.set_is_host_valid(false);
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(env.p.call_log().count(COPY), 2);
            assert_eq!(env.rec.log().count("new_buffer_for_virtual_page"), 1);
        }

        /// A transfer buffer a racing caller (another FIFO's synchronize) stored between this
        /// prepare's check and its store is overwritten unreleased: it stays live, with its GPU
        /// VA and its guest pages' mapping, for good
        #[test]
        fn a_transfer_buffer_a_racing_caller_stored_is_kept() {
            let env = Env::new();
            let r = private(&env, 0x40);
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

        #[test]
        fn logs_a_guest_invalid_prepare_and_still_copies() {
            let env = Env::new();
            let r = private(&env, 0x40);
            r.set_is_guest_valid(false);
            let (prepared, lines) =
                logged(|| r.prepare_in_encoder(Hold::NONE, Some(&encoder(&env))));
            prepared.unwrap();
            assert_eq!(
                lines,
                vec!["preparing a private buffer whose guest copy is not valid"]
            );
            assert_eq!(env.p.call_log().count(COPY), 1);
        }

        #[test]
        fn a_nil_encoder_still_marks_host_valid() {
            let env = Env::new();
            let r = private(&env, 0x40);
            r.prepare_in_encoder(Hold::NONE, None).unwrap();
            assert!(r.state.host_valid());
            assert_eq!(env.p.call_log().count(COPY), 0);
        }

        #[test]
        fn a_bind_raise_leaves_host_invalid() {
            let env = Env::new();
            let r = private(&env, 0x40);
            env.p.script().push(
                "new_buffer_for_virtual_page",
                Step::raise("NSGenericException", "Failed to get guest pointer"),
            );
            let hold = Hold::NONE;
            assert!(r.prepare_in_encoder(hold, Some(&encoder(&env))).is_err());
            assert!(!r.state.host_valid());
            assert_eq!(env.p.call_log().count(COPY), 0);
        }

        #[test]
        fn a_gone_task_raises_no_guest_buffer() {
            let k = PrivateBufferResource::<Fp> {
                task: gone_task(),
                buffer: None,
                transfer_buffer: Mutex::new(None),
                length: 0x40,
                page: 1,
            };
            let hold = Hold::NONE;
            assert_eq!(
                kind(k.ensure_physical(hold)),
                Some(RaiseKind::NoGuestBuffer)
            );
        }
    }

    mod discard_devolatilize {
        use super::*;

        #[test]
        fn discard_drops_the_transfer_buffer_then_marks_volatile() {
            let env = Env::new();
            let r = private(&env, 0x40);
            r.prepare_in_encoder(Hold::NONE, Some(&encoder(&env)))
                .unwrap();
            let m = env.p.call_log().mark();
            r.discard().unwrap();
            assert!(parts(&r).transfer_buffer.lock().is_none());
            assert!(r.state.is_volatile() && r.state.host_valid());
            let c = env.p.call_log().since(m);
            assert_eq!(c.len(), 1);
            assert_eq!(
                (c[0].sel, c[0].args.clone()),
                ("setPurgeableState:", vec![Arg::U64(3)])
            );
            assert!(r.needs_prepare(), "volatile");
        }

        #[test]
        fn devolatilize_recopies_only_when_emptied() {
            // 63 on a legacy heap / private buffer / texture makes it volatile and the next prepare
            // re-copies only if Metal emptied it (6.19 `_legacy_heap_discard`)
            let env = Env::new();
            let r = private(&env, 0x40);
            let enc = encoder(&env);
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            r.discard().unwrap();
            // Metal kept the contents: previous state Volatile.
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert!(!r.state.is_volatile());
            assert_eq!(env.p.call_log().count(COPY), 1, "no re-copy");
            // Metal emptied it.
            r.discard().unwrap();
            env.p
                .script()
                .push("setPurgeableState:", Step::value(PURGEABLE_EMPTY));
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(env.p.call_log().count(COPY), 2);
            assert_eq!(env.rec.log().count("new_buffer_for_virtual_page"), 2);
            let set = env.p.call_log().calls_to("setPurgeableState:");
            assert_eq!(set.last().map(|c| c.args.clone()), Some(vec![Arg::U64(2)]));
        }
    }

    mod synchronize {
        use super::*;

        #[test]
        fn copies_down_only_when_guest_invalid() {
            let env = Env::new();
            let r = private(&env, 0x40);
            let enc = encoder(&env);
            r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert_eq!(env.p.call_log().count(COPY), 0);
            r.set_is_guest_valid(false);
            r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            let buf = r.get_buffer().unwrap().unwrap();
            let c = env.p.call_log().calls_to(COPY);
            assert_eq!(c.len(), 1);
            assert_eq!(c[0].args[0], Arg::Obj(Some(buf.id())));
            assert!(r.state.guest_valid());
            // Synchronize never devolatilizes.
            r.state.set_is_volatile(true);
            r.set_is_guest_valid(false);
            let m = env.p.call_log().mark();
            r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            assert!(
                !env.p
                    .call_log()
                    .sels_since(m)
                    .contains(&"setPurgeableState:")
            );
        }
    }

    mod getters {
        use super::*;

        #[test]
        fn getters_return_the_private_buffer_and_last_release_releases_both() {
            let env = Env::new();
            let r = private(&env, 0x40);
            let b = r.get_buffer().unwrap().unwrap();
            assert!(
                matches!(r.get_resource().unwrap(), Some(AnyResource::Buffer(x)) if x.id() == b.id())
            );
            r.prepare_in_encoder(Hold::NONE, None).unwrap();
            let tb = parts(&r)
                .transfer_buffer
                .lock()
                .as_ref()
                .map(|x| x.clone_ref())
                .unwrap();
            assert_eq!((b.retain_count(), tb.retain_count()), (2, 2));
            drop(r);
            assert_eq!((b.retain_count(), tb.retain_count()), (1, 1));
        }

        #[test]
        fn a_nil_buffer_raises_at_each_getter() {
            let k = PrivateBufferResource::<Fp> {
                task: gone_task(),
                buffer: None,
                transfer_buffer: Mutex::new(None),
                length: 0,
                page: 0,
            };
            let r = Resource::new(ResourceKind::PrivateBuffer(k));
            assert_eq!(kind(r.get_resource()), Some(RaiseKind::BufferInvalidated));
            assert_eq!(kind(r.get_buffer()), Some(RaiseKind::BufferInvalidated));
        }

        #[test]
        fn inherited_predicates() {
            let env = Env::new();
            let r = private(&env, 0x40);
            assert!(!r.needs_cpu_paging() && !r.needs_compute_paging() && r.supports_discard());
            assert!(r.needs_prepare() && !r.needs_sync());
            let hold = Hold::NONE;
            assert_eq!(
                kind(r.prepare_with_cpu(hold)),
                Some(RaiseKind::Unrecognized(Op::PrepareCpu))
            );
            // What a heap-buffer record naming a private buffer raises.
            assert_eq!(
                kind(r.get_heap()),
                Some(RaiseKind::Unrecognized(Op::GetHeap))
            );
        }
    }
}
