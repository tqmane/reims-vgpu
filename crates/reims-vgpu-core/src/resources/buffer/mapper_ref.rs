//! The mapper-ref buffer: a buffer over an IOSurface the mapper names, made at the first CPU
//! prepare and handed back at a discard or the last release.
//!
//! The surface is retained from the task before the buffer is made and released once per made
//! buffer: at once when Metal refuses the buffer, else at the discard or the last release.

use std::sync::Arc;

use parking_lot::Mutex;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::guest::GuestPtr;
use crate::platform::device::GpuDevice;
use crate::platform::resource::Buffer;
use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::raise::{Kept, keep_forever};
use crate::resources::resource::AnyResource;
use crate::resources::resource::{Resource, ResourceClass, ResourceKind, TaskRef, read_descriptor};
use crate::resources::{RaiseKind, raise_with};
use crate::rt::Hold;

/// The mapper-ref buffer record (type 15): 8 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct MapperRefBufferDescriptor {
    /// The mapper's IOSurface mapping.
    pub mapping_id: u64,
}

const _: () = assert!(size_of::<MapperRefBufferDescriptor>() == 8);

/// A buffer over the IOSurface the mapper names, made lazily.
pub struct MapperRefBufferResource<P: Platform> {
    /// Asked afresh at each use: a torn-down task answers nothing.
    task: TaskRef<P>,
    buffer: Mutex<Option<P::Buffer>>,
    mapping_id: u64,
}

impl<P: Platform> MapperRefBufferResource<P> {
    /// A mapper-ref buffer for the record's mapping. The placement is ignored, and nothing is
    /// made until the first prepare.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        task: TaskRef<P>,
        descriptor: &[u8],
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let _ = placement;
        let d: MapperRefBufferDescriptor = read_descriptor(descriptor);
        // Neither validity flag is set.
        Ok(Some(Resource::new(ResourceKind::MapperRefBuffer(
            MapperRefBufferResource {
                task,
                buffer: Mutex::new(None),
                mapping_id: d.mapping_id,
            },
        ))))
    }

    /// The mapper's mapping.
    pub fn mapping_id(&self) -> u64 {
        self.mapping_id
    }

    /// Makes the buffer over the surface unless made. An unknown mapping raises; Metal's
    /// refusal hands the surface back and leaves the buffer to the next prepare.
    fn ensure_physical(&self) -> RaiseResult<()> {
        if self.buffer.lock().is_some() {
            return Ok(());
        }
        // +1, or nothing (also from a torn-down task).
        let surface = match self.task.upgrade() {
            Some(t) => t.retain_io_surface_with_mapping_id(self.mapping_id)?,
            None => None,
        };
        let Some(surface) = surface else {
            return raise_with(
                RaiseKind::MapperSurfaceMissing,
                format_args!("no IOSurface for mapping {}", self.mapping_id),
            );
        };
        // The task is asked again.
        let made = match self.task.upgrade() {
            Some(t) => t.mtl_device().map_or(Ok(None), |device| {
                device.new_buffer_on_surface(Some(&surface))
            }),
            None => Ok(None),
        };
        let buffer = match made {
            Ok(b) => b,
            Err(e) => {
                // The raise skips the surface's release: the retain is kept.
                keep_forever(surface, Kept::SurfaceUseCount);
                return Err(e);
            }
        };
        match buffer {
            Some(b) => {
                // Never raced: only a CPU prepare makes it, and prepares run one at a time (the
                // in-process task's prepare lock, the service's queue).
                // The buffer holds the surface now.
                *self.buffer.lock() = Some(b);
                drop(surface);
            }
            None => {
                // No raise; the buffer stays unmade and the next prepare tries again.
                match self.task.upgrade() {
                    Some(t) => t.release_io_surface_with_mapping_id(self.mapping_id, Some(surface)),
                    // Nobody to hand it back to: the retain is kept.
                    None => keep_forever(surface, Kept::SurfaceUseCount),
                }
            }
        }
        Ok(())
    }

    /// Releases the buffer and hands its surface back to the task.
    fn discard_buffer(&self) {
        let buffer = self.buffer.lock().take();
        // The buffer's surface, retained; none for no buffer.
        let surface = buffer.as_ref().and_then(Buffer::iosurface);
        drop(buffer);
        // Nothing while the task is being torn down.
        let task = self.task.upgrade();
        match (task, surface) {
            // The task takes the retain.
            (Some(t), Some(s)) => t.release_io_surface_with_mapping_id(self.mapping_id, Some(s)),
            // With the task gone the retain is kept.
            (None, Some(s)) => keep_forever(s, Kept::SurfaceUseCount),
            (_, None) => {}
        }
    }
}

impl<P: Platform> ResourceClass<P> for MapperRefBufferResource<P> {
    fn get_buffer(&self, this: &Resource<P>) -> RaiseResult<Option<P::Buffer>> {
        let _ = this;
        // Nothing before the first prepare, no raise.
        Ok(self.buffer.lock().as_ref().map(Handle::clone_ref))
    }

    fn get_resource(&self, this: &Resource<P>) -> RaiseResult<Option<AnyResource<P>>> {
        let _ = this;
        Ok(self
            .buffer
            .lock()
            .as_ref()
            .map(|b| AnyResource::Buffer(b.clone_ref())))
    }

    fn needs_cpu_paging(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    fn prepare_with_cpu(&self, _hold: Hold<'_>, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        self.ensure_physical()
    }

    fn synchronize_with_cpu(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        Ok(())
    }

    fn supports_discard(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    fn is_host_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        self.buffer.lock().is_some()
    }

    fn is_guest_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        self.discard_buffer();
        Ok(())
    }

    fn needs_sync(&self, this: &Resource<P>) -> bool {
        let _ = this;
        false
    }
}

impl<P: Platform> Drop for MapperRefBufferResource<P> {
    /// The last release discards: the surface goes back to the task.
    fn drop(&mut self) {
        self.discard_buffer();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::os::surface::SurfaceSpec;
    use crate::resources::Op;
    use crate::resources::resource::test_support::{Env, gone_task, kind, placement};
    use crate::testkit::calllog::Arg;
    use crate::testkit::objects::{Fk, kind};
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Fp = FakePlatform;

    fn mapper(env: &Env, id: u64) -> Arc<Resource<Fp>> {
        MapperRefBufferResource::new(env.task(), &id.to_le_bytes(), None)
            .unwrap()
            .unwrap()
    }

    fn surface(env: &Env, id: u64) -> Fk<kind::Surface> {
        let s = env
            .p
            .surfaces()
            .create(&SurfaceSpec {
                alloc_size: 0x4000,
                width: 16,
                height: 16,
                bytes_per_element: 4,
                bytes_per_row: 64,
                ..SurfaceSpec::default()
            })
            .expect("surface");
        env.rec.provide(id, &s);
        s
    }

    mod init {
        use super::*;

        #[test]
        fn keeps_the_mapping_id_and_makes_nothing() {
            let env = Env::new();
            let (_m, pl) = placement(0);
            let r = MapperRefBufferResource::new(env.task(), &42u64.to_le_bytes(), Some(pl))
                .unwrap()
                .unwrap();
            assert!(env.p.call_log().is_empty());
            let ResourceKind::MapperRefBuffer(k) = &r.kind else {
                panic!()
            };
            assert_eq!(k.mapping_id(), 42);
            assert!(!r.state.guest_valid() && !r.state.host_valid());
        }
    }

    mod prepare {
        use super::*;

        #[test]
        fn makes_the_buffer_over_the_surface_and_releases_its_retain() {
            let env = Env::new();
            let s = surface(&env, 7);
            let base = s.retain_count();
            let r = mapper(&env, 7);
            r.prepare_with_cpu(Hold::NONE).unwrap();
            assert!(r.is_host_valid());
            let c = env.p.call_log().calls_to("newBufferWithIOSurface:");
            assert_eq!(c[0].args, vec![Arg::obj(&s)]);
            // One more count: the buffer's.
            assert_eq!(s.retain_count(), base + 1);
            // Twice in one exec: the second returns early (`mutgap_tex_*_mapper_buffer_twice`).
            r.prepare_with_cpu(Hold::NONE).unwrap();
            assert_eq!(env.rec.log().count("retain_io_surface_with_mapping_id"), 1);
            let m = env.p.call_log().mark();
            r.synchronize_with_cpu().unwrap();
            assert!(env.p.call_log().since(m).is_empty());
        }

        #[test]
        fn an_unknown_mapping_raises_with_its_id() {
            let env = Env::new();
            let r = mapper(&env, 99);
            let hold = Hold::NONE;
            let e = r.prepare_with_cpu(hold).unwrap_err();
            assert_eq!(
                crate::resources::raised_kind(&e),
                Some(RaiseKind::MapperSurfaceMissing)
            );
            assert_eq!(
                e.detail().unwrap_or_default(),
                "no IOSurface for mapping 99"
            );
            assert_eq!(env.p.call_log().count("newBufferWithIOSurface:"), 0);
        }

        #[test]
        fn a_failed_make_hands_the_surface_back_without_raising() {
            // The buffer stays unmade, with no raise, until a prepare makes it.
            let env = Env::new();
            let s = surface(&env, 3);
            let base = s.retain_count();
            env.p.script().push("newBufferWithIOSurface:", Step::Nil);
            let r = mapper(&env, 3);
            r.prepare_with_cpu(Hold::NONE).unwrap();
            assert!(!r.is_host_valid() && r.get_buffer().unwrap().is_none());
            let rel = env.rec.log().calls_to("release_io_surface_with_mapping_id");
            assert_eq!(rel.len(), 1);
            assert_eq!(rel[0].args, vec![Arg::U64(3), Arg::obj(&s)]);
            assert_eq!(s.retain_count(), base);
            // The next prepare tries again.
            r.prepare_with_cpu(Hold::NONE).unwrap();
            assert!(r.is_host_valid());
        }

        #[test]
        fn a_make_raise_leaks_the_surface_retain() {
            let env = Env::new();
            let s = surface(&env, 4);
            let base = s.retain_count();
            env.p.script().push(
                "newBufferWithIOSurface:",
                Step::raise("NSInvalidArgumentException", "x"),
            );
            let r = mapper(&env, 4);
            let hold = Hold::NONE;
            assert!(r.prepare_with_cpu(hold).is_err());
            assert_eq!(s.retain_count(), base + 1);
            assert_eq!(env.rec.log().count("release_io_surface_with_mapping_id"), 0);
        }

        #[test]
        fn a_gone_task_raises_the_missing_surface() {
            let r = MapperRefBufferResource::<Fp>::new(gone_task(), &5u64.to_le_bytes(), None)
                .unwrap()
                .unwrap();
            let hold = Hold::NONE;
            assert_eq!(
                kind(r.prepare_with_cpu(hold)),
                Some(RaiseKind::MapperSurfaceMissing)
            );
        }
    }

    mod discard {
        use super::*;

        #[test]
        fn discard_releases_the_surface_once() {
            let env = Env::new();
            let s = surface(&env, 7);
            let base = s.retain_count();
            let r = mapper(&env, 7);
            r.prepare_with_cpu(Hold::NONE).unwrap();
            r.discard().unwrap();
            assert!(!r.is_host_valid());
            let rel = env.rec.log().calls_to("release_io_surface_with_mapping_id");
            assert_eq!(rel.len(), 1);
            assert_eq!(rel[0].args, vec![Arg::U64(7), Arg::obj(&s)]);
            // retain (+1), buffer (+1), release (-1); getter retain (+1), buffer release (-1),
            // the task's release (-1).
            assert_eq!(s.retain_count(), base);
            // Discarding again (and the last release) with no buffer does nothing.
            r.discard().unwrap();
            drop(r);
            assert_eq!(env.rec.log().count("release_io_surface_with_mapping_id"), 1);
        }

        #[test]
        fn last_release_discards() {
            let env = Env::new();
            let _s = surface(&env, 7);
            let r = mapper(&env, 7);
            r.prepare_with_cpu(Hold::NONE).unwrap();
            drop(r);
            assert_eq!(env.rec.log().count("release_io_surface_with_mapping_id"), 1);
        }

        #[test]
        fn a_gone_task_leaks_the_extra_retain() {
            // A mapper resource still alive when its task is torn down never hands its surface back
            // (the service's cache count is never decremented, the device never told).
            let env = Env::new();
            let s = surface(&env, 7);
            let base = s.retain_count();
            let r = mapper(&env, 7);
            r.prepare_with_cpu(Hold::NONE).unwrap();
            let Env { rec, host, p } = env;
            drop(host);
            let log = rec.log().clone();
            drop(rec);
            drop(r);
            assert_eq!(log.count("release_io_surface_with_mapping_id"), 0);
            // The buffer's count went, the getter's retain stayed.
            assert_eq!(s.retain_count(), base);
            drop(p);
        }
    }

    mod accessors {
        use super::*;

        #[test]
        fn nil_before_the_first_prepare() {
            // The service registers address 0 for it.
            let env = Env::new();
            let _s = surface(&env, 1);
            let r = mapper(&env, 1);
            assert!(r.get_buffer().unwrap().is_none());
            assert!(r.get_resource().unwrap().is_none());
            assert!(r.is_guest_valid() && !r.is_host_valid() && r.needs_prepare());
            assert!(r.needs_cpu_paging() && r.supports_discard() && !r.needs_sync());
            r.prepare_with_cpu(Hold::NONE).unwrap();
            let b = r.get_buffer().unwrap().unwrap();
            assert!(
                matches!(r.get_resource().unwrap(), Some(AnyResource::Buffer(x)) if x.id() == b.id())
            );
            assert!(!r.needs_prepare());
            r.set_is_guest_valid(false);
            assert!(r.is_guest_valid(), "the setter writes a dead flag");
        }

        /// No encoder paging, and no heap
        #[test]
        fn encoder_and_heap_operations_are_not_answered() {
            let env = Env::new();
            let _s = surface(&env, 1);
            let r = mapper(&env, 1);
            let hold = Hold::NONE;
            assert_eq!(
                kind(r.prepare_in_encoder(hold, None)),
                Some(RaiseKind::Unrecognized(Op::Prepare))
            );
            assert_eq!(
                kind(r.get_heap()),
                Some(RaiseKind::Unrecognized(Op::GetHeap))
            );
        }
    }
}
