//! The shared buffer: a no-copy Metal buffer over the guest's pages, made at creation and again
//! at the first CPU prepare after a discard.

use std::sync::Arc;

use crate::guest::GuestPtr;
use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::resources::buffer::BufferResourceDescriptor;
use crate::resources::resource::AnyResource;
use crate::resources::resource::{Resource, ResourceClass, ResourceKind, TaskRef, read_descriptor};
use crate::resources::{RaiseKind, raise};
use crate::rt::Hold;
use crate::sync::Mutex;

/// A no-copy buffer over guest pages. Host validity is whether it holds the buffer; the guest's
/// copy is the host's, so it is always guest-valid.
pub struct SharedBufferResource<P: Platform> {
    task: TaskRef<P>,
    /// `None` after a discard, until the next CPU prepare.
    buffer: Mutex<Option<P::Buffer>>,
    length: u64,
    page: u32,
}

impl<P: Platform> SharedBufferResource<P> {
    /// A shared buffer over the record's pages, bound at once. A placement is refused (logged,
    /// `None`). A raise binding the pages releases the half-made resource, which holds nothing.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        hold: Hold<'_>,
        task: TaskRef<P>,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        if placement.is_some() {
            log::warn!("shared buffer refused: it takes no placement");
            return Ok(None);
        }
        let d: BufferResourceDescriptor = read_descriptor(descriptor);
        let r = Resource::new(ResourceKind::SharedBuffer(SharedBufferResource {
            task,
            buffer: Mutex::new(None),
            length: d.length,
            page: d.page,
        }));
        let ResourceKind::SharedBuffer(k) = &r.kind else {
            return Ok(None);
        };
        if let Err(e) = k.ensure_physical(hold) {
            // A raise out of the creation releases the half-made resource, which holds nothing
            // yet (no buffer, a weak task).
            drop(r);
            return Err(e);
        }
        Ok(Some(r))
    }

    /// The task.
    pub fn task(&self) -> &TaskRef<P> {
        &self.task
    }

    /// Binds the guest pages unless bound. The task raises rather than answer nothing; a
    /// torn-down task gives nothing, which raises here.
    fn ensure_physical(&self, hold: Hold<'_>) -> RaiseResult<()> {
        if self.buffer.lock().is_some() {
            return Ok(());
        }
        let buffer = match self.task.upgrade() {
            Some(t) => Some(t.new_buffer_for_virtual_page(hold, self.page, self.length)?),
            None => None,
        };
        let Some(buffer) = buffer else {
            return raise(RaiseKind::NoGuestBuffer);
        };
        // Never raced: after the creation only a CPU prepare binds it, and prepares run one at
        // a time (the in-process task's prepare lock, the service's queue).
        *self.buffer.lock() = Some(buffer);
        Ok(())
    }

    fn buffer(&self) -> RaiseResult<P::Buffer> {
        match self.buffer.lock().as_ref() {
            Some(b) => Ok(b.clone_ref()),
            None => raise(RaiseKind::BufferInvalidated),
        }
    }
}

impl<P: Platform> ResourceClass<P> for SharedBufferResource<P> {
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

    fn is_host_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        // Derived from the handle; the setter writes a dead flag.
        self.buffer.lock().is_some()
    }

    fn is_guest_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    fn prepare_with_cpu(&self, hold: Hold<'_>, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        self.ensure_physical(hold)
    }

    fn synchronize_with_cpu(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        Ok(())
    }

    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        // The next prepare binds what the page table then names. Released outside the lock.
        let old = self.buffer.lock().take();
        drop(old);
        Ok(())
    }

    fn needs_sync(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }

    fn needs_cpu_paging(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::Op;
    use crate::resources::buffer::tests::desc;
    use crate::resources::resource::test_support::{Env, gone_task, kind, placement};
    use crate::testkit::calllog::Arg;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Fp = FakePlatform;

    fn shared(env: &Env, len: u64, page: u32) -> Arc<Resource<Fp>> {
        env.run(|| SharedBufferResource::new(Hold::NONE, env.task(), &desc(len, page, 0), None))
            .unwrap()
            .unwrap()
    }

    /// Creation binds the guest pages at once; it starts guest-valid
    #[test]
    fn binds_the_guest_pages_at_once_and_starts_guest_valid() {
        let env = Env::new();
        let r = shared(&env, 0x8000, 7);
        let calls = env.rec.log().calls_to("new_buffer_for_virtual_page");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].args, vec![Arg::U32(7), Arg::U64(0x8000)]);
        assert!(r.state.guest_valid() && !r.state.host_valid());
        assert!(r.is_host_valid() && r.is_guest_valid());
    }

    /// A placement is refused, and nothing is made
    #[test]
    fn a_placement_is_refused_and_nothing_made() {
        let env = Env::new();
        let (_m, pl) = placement(0);
        let r = env
            .run(|| SharedBufferResource::new(Hold::NONE, env.task(), &desc(16, 1, 0), Some(pl)))
            .unwrap();
        assert!(r.is_none());
        assert_eq!(env.rec.log().count("new_buffer_for_virtual_page"), 0);
    }

    /// The task's raise goes out of the creation, and releases the half-made resource
    #[test]
    fn a_task_raise_propagates_out_of_the_creation() {
        let env = Env::new();
        env.p.script().push(
            "new_buffer_for_virtual_page",
            Step::raise("NSGenericException", "Failed to get guest pointer"),
        );
        let hold = Hold::NONE;
        let e = env
            .run(|| SharedBufferResource::new(hold, env.task(), &desc(16, 1, 0), None))
            .expect_err("raise");
        assert_eq!(
            e.detail().unwrap_or_default(),
            "Failed to get guest pointer"
        );
        // The half-made resource is released: nothing holds the task weakly any more.
        assert_eq!(Arc::weak_count(&env.host), 0);
    }

    /// A torn-down task gives no pages, which raises
    #[test]
    fn a_gone_task_raises_no_guest_buffer() {
        let hold = Hold::NONE;
        assert_eq!(
            kind(SharedBufferResource::<Fp>::new(
                hold,
                gone_task(),
                &desc(16, 1, 0),
                None
            )),
            Some(RaiseKind::NoGuestBuffer)
        );
    }

    /// The getters hand out the buffer until a discard, then raise
    #[test]
    fn getters_hand_out_the_buffer_until_discard_then_raise() {
        let env = Env::new();
        let r = shared(&env, 16, 1);
        let b = r.get_buffer().unwrap().unwrap();
        match r.get_resource().unwrap() {
            Some(AnyResource::Buffer(x)) => assert_eq!(x.id(), b.id()),
            other => panic!("{other:?}"),
        }
        r.discard().unwrap();
        assert!(!r.is_host_valid());
        assert_eq!(kind(r.get_buffer()), Some(RaiseKind::BufferInvalidated));
        assert_eq!(kind(r.get_resource()), Some(RaiseKind::BufferInvalidated));
    }

    /// The last release releases the buffer
    #[test]
    fn last_release_releases_the_buffer() {
        let env = Env::new();
        let r = shared(&env, 16, 1);
        let b = r.get_buffer().unwrap().unwrap();
        assert_eq!(b.retain_count(), 2);
        drop(r);
        assert_eq!(b.retain_count(), 1);
    }

    /// Validity ignores the setters
    #[test]
    fn validity_ignores_the_setters() {
        // Host validity is derived from the handle.
        let env = Env::new();
        let r = shared(&env, 16, 1);
        r.set_is_host_valid(false);
        r.set_is_guest_valid(false);
        assert!(r.is_host_valid() && r.is_guest_valid());
        assert!(!r.state.guest_valid(), "the setter writes the dead flag");
        assert!(!r.needs_sync() && r.needs_cpu_paging() && !r.needs_prepare());
        r.discard().unwrap();
        assert!(r.needs_prepare());
    }

    /// A CPU prepare binds the pages again once after a discard
    #[test]
    fn prepare_with_cpu_rebinds_once_after_discard() {
        let env = Env::new();
        let r = shared(&env, 16, 3);
        r.prepare_with_cpu(Hold::NONE).unwrap();
        assert_eq!(env.rec.log().count("new_buffer_for_virtual_page"), 1);
        r.discard().unwrap();
        r.prepare_with_cpu(Hold::NONE).unwrap();
        r.prepare_with_cpu(Hold::NONE).unwrap();
        assert_eq!(env.rec.log().count("new_buffer_for_virtual_page"), 2);
        let m = env.p.call_log().mark();
        r.synchronize_with_cpu().unwrap();
        assert!(env.p.call_log().since(m).is_empty());
    }

    /// No encoder paging, and no heap
    #[test]
    fn no_encoder_paging() {
        let env = Env::new();
        let r = shared(&env, 16, 3);
        let hold = Hold::NONE;
        assert_eq!(
            kind(r.prepare_in_encoder(hold, None)),
            Some(RaiseKind::Unrecognized(Op::Prepare))
        );
        // What a heap-buffer record naming a buffer raises.
        assert_eq!(
            kind(r.get_heap()),
            Some(RaiseKind::Unrecognized(Op::GetHeap))
        );
    }
}
