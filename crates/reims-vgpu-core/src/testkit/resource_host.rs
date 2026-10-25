//! Fake: a task as its resources, its resource manager and its object-list managers see it.
//!
//! [`FakeResourceHost`] implements [`ResourceHost`], the one task seam (`LocalTask` and `GpuTask`
//! both implement it), for `FakePlatform`, so a resource's tests need no task. Each call is
//! logged with class `"ResourceHost"` and the Rust method name as its selector, then the script
//! decides, then a default answers:
//! - `RaiseResult` methods: `Step::Raise` returns `Err(Raise)`, the task's raise for its caller to
//!   catch ([`Policy::Caught`]), and `Step::Die` dies;
//! - value returns: `Step::value(v)` of the return type, else the default;
//! - handles the fake cannot clone out of a script (surfaces, address ranges, shared texture
//!   handles, the XR10 pipelines at keys 0 and 1) come from what the test
//!   [`provide`](FakeResourceHost::provide)s.

use crate::testkit::script::Policy;
use std::sync::Arc;

use crate::sync::Mutex;

use super::calllog::{Arg, CallLog};
use super::objects::{AnyHandle, Fk, Kind, Model, kind};
use super::platform::FakePlatform;
use super::script::{Script, Step};
use crate::deserializer::Deserializer;
use crate::guest::records::{FeaturesRef, SharedFeatures};
use crate::host::HostPtr;
use crate::memory::cursor::{Cursor, WritableCursor};
use crate::memory::mapping::Mapping;
use crate::platform::{Handle, ObjId};
use crate::raise::RaiseResult;
use crate::resources::host::{CallBorrow, MappedAddress, ResourceHost};
use crate::resources::manager::ResourceManager;
use crate::resources::resource::Resource;
use crate::rt::Hold;

type P = FakePlatform;

const CLASS: &str = "ResourceHost";

/// A hook [`FakeResourceHost::during_next`] runs once, inside the next call of its selector.
type During = (&'static str, Box<dyn FnOnce() + Send>);

struct Shared {
    log: CallLog,
    script: Script,
    platform: Arc<FakePlatform>,
    features: Mutex<Option<FeaturesRef>>,
    provided: Mutex<Vec<(u64, AnyHandle)>>,
    paging_queue: Mutex<Option<Fk<kind::CommandQueue>>>,
    /// What runs inside the next call of a selector, once (a racing caller's whole call).
    during: Mutex<Vec<During>>,
}

/// A scripted task. Siblings share one log, script and state.
pub struct FakeResourceHost {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for FakeResourceHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeResourceHost").finish_non_exhaustive()
    }
}

fn ptr_of<T>(p: *const T) -> Arg {
    Arg::Ptr(p.addr())
}

impl FakeResourceHost {
    /// A task logging into `platform`'s call log and scripted by its script.
    pub fn new(platform: Arc<FakePlatform>) -> Arc<FakeResourceHost> {
        Arc::new(FakeResourceHost {
            shared: Arc::new(Shared {
                log: platform.call_log().clone(),
                script: platform.script().clone(),
                platform,
                features: Mutex::new(None),
                provided: Mutex::new(Vec::new()),
                paging_queue: Mutex::new(None),
                during: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Another handle on the same state: a second task object whose last release a test
    /// controls on its own.
    pub fn sibling(&self) -> Arc<FakeResourceHost> {
        Arc::new(FakeResourceHost {
            shared: self.shared.clone(),
        })
    }

    /// The call log.
    pub fn log(&self) -> &CallLog {
        &self.shared.log
    }

    /// The platform.
    pub fn platform(&self) -> &Arc<FakePlatform> {
        &self.shared.platform
    }

    /// The features every `features()` returns (default: lazily `SharedFeatures::new()`).
    pub fn set_features(&self, f: FeaturesRef) {
        *self.shared.features.lock() = Some(f);
    }

    /// Provides `h` under `key`: a mapping ID's surface, a page's address ranges, a shared
    /// texture handle's identifier, or the XR10 pipelines (0 in, 1 out).
    pub fn provide<K: Kind>(&self, key: u64, h: &Fk<K>) {
        let mut g = self.shared.provided.lock();
        g.retain(|(k, x)| !(*k == key && x.obj().class() == K::CLASS));
        g.push((key, h.erase()));
    }

    fn provided<K: Kind>(&self, key: u64) -> Option<Fk<K>> {
        self.shared
            .provided
            .lock()
            .iter()
            .find(|(k, h)| *k == key && h.obj().class() == K::CLASS)
            .and_then(|(_, h)| h.cast::<K>())
    }

    /// Runs `f` inside the next call of `sel`, once, after it is logged and before it answers:
    /// another caller's whole call between this one's start and its answer, as when two threads
    /// race through it.
    pub fn during_next(&self, sel: &'static str, f: impl FnOnce() + Send + 'static) {
        self.shared.during.lock().push((sel, Box::new(f)));
    }

    fn step(&self, sel: &'static str, args: Vec<Arg>) -> Step {
        self.shared.log.push(ObjId(0), CLASS, sel, args);
        let during = {
            let mut g = self.shared.during.lock();
            g.iter().position(|(s, _)| *s == sel).map(|i| g.remove(i).1)
        };
        if let Some(f) = during {
            f();
        }
        self.shared.script.next(ObjId(0), sel)
    }

    fn ret<T: Clone + 'static>(
        &self,
        sel: &'static str,
        args: Vec<Arg>,
        d: impl FnOnce() -> T,
    ) -> T {
        self.step(sel, args).getter_value(sel, d)
    }

    fn thr<T: Clone + 'static>(
        &self,
        policy: Policy,
        sel: &'static str,
        args: Vec<Arg>,
        d: impl FnOnce() -> T,
    ) -> RaiseResult<T> {
        self.step(sel, args).throws_value(policy, sel, d)
    }

    fn act(&self, policy: Policy, sel: &'static str, args: Vec<Arg>) -> RaiseResult<()> {
        self.step(sel, args).throws(policy, sel)
    }

    fn device(&self) -> Fk<kind::Device> {
        self.shared.platform.device()
    }

    fn features_ref(&self) -> FeaturesRef {
        self.shared
            .features
            .lock()
            .get_or_insert_with(|| Arc::new(SharedFeatures::new()))
            .clone()
    }
}

impl ResourceHost<P> for FakeResourceHost {
    fn new_buffer_for_virtual_page(
        &self,
        _hold: Hold<'_>,
        page: u32,
        length: u64,
    ) -> RaiseResult<Fk<kind::Buffer>> {
        let step = self.step(
            "new_buffer_for_virtual_page",
            vec![Arg::U32(page), Arg::U64(length)],
        );
        step.throws(Policy::Caught, "new_buffer_for_virtual_page")?;
        let w = self.shared.platform.world();
        Ok(
            super::impls::new_buffer(w, length, 0, super::impls::Backing::Fresh, None)
                .unwrap_or_else(|| w.new_obj(Model::None)),
        )
    }

    fn prepare_backing(
        &self,
        _hold: Hold<'_>,
        backing: &Resource<P>,
        encoder: Option<&Fk<kind::BlitEncoder>>,
    ) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "prepare_backing",
            vec![ptr_of(std::ptr::from_ref(backing)), Arg::opt(encoder)],
        )
    }

    fn synchronize_backing(
        &self,
        _hold: Hold<'_>,
        backing: &Resource<P>,
        encoder: Option<&Fk<kind::BlitEncoder>>,
    ) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "synchronize_backing",
            vec![ptr_of(std::ptr::from_ref(backing)), Arg::opt(encoder)],
        )
    }

    fn retain_io_surface_with_mapping_id(
        &self,
        identifier: u64,
    ) -> RaiseResult<Option<Fk<kind::Surface>>> {
        let step = self.step(
            "retain_io_surface_with_mapping_id",
            vec![Arg::U64(identifier)],
        );
        step.throws(Policy::Caught, "retain_io_surface_with_mapping_id")?;
        if step.is_nil() {
            return Ok(None);
        }
        Ok(self.provided(identifier))
    }

    fn release_io_surface_with_mapping_id(
        &self,
        identifier: u64,
        surface: Option<Fk<kind::Surface>>,
    ) {
        self.ret(
            "release_io_surface_with_mapping_id",
            vec![Arg::U64(identifier), Arg::opt(surface.as_ref())],
            || (),
        );
        drop(surface);
    }

    fn address_ranges_for_virtual_page(
        &self,
        _hold: Hold<'_>,
        page: u32,
        length: u64,
        read_only: &mut bool,
    ) -> RaiseResult<Option<Fk<kind::AddressRangeArray>>> {
        let step = self.step(
            "address_ranges_for_virtual_page",
            vec![Arg::U32(page), Arg::U64(length), Arg::Bool(*read_only)],
        );
        step.throws(Policy::Caught, "address_ranges_for_virtual_page")?;
        Ok(self.provided(u64::from(page)))
    }

    fn new_shared_texture_handle_for_id(
        &self,
        _hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<Fk<kind::SharedTextureHandle>>> {
        let step = self.step(
            "new_shared_texture_handle_for_id",
            vec![Arg::U32(identifier)],
        );
        step.throws(Policy::Caught, "new_shared_texture_handle_for_id")?;
        Ok(self.provided(u64::from(identifier)))
    }

    fn data_from_virtual_offset(
        &self,
        _hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Vec<u8>>> {
        self.thr(
            Policy::Caught,
            "data_from_virtual_offset",
            vec![Arg::U64(offset), Arg::U64(length)],
            || None,
        )
    }

    fn backing_for_id(
        &self,
        _hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        self.thr(
            Policy::Caught,
            "backing_for_id",
            vec![Arg::U32(identifier)],
            || None,
        )
    }

    fn cursor_from_virtual_offset(
        &self,
        _hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Cursor>> {
        self.thr(
            Policy::Caught,
            "cursor_from_virtual_offset",
            vec![Arg::U64(offset), Arg::U64(length)],
            || None,
        )
    }

    fn writable_cursor_from_virtual_offset(
        &self,
        _hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<WritableCursor>> {
        self.thr(
            Policy::Caught,
            "writable_cursor_from_virtual_offset",
            vec![Arg::U64(offset), Arg::U64(length)],
            || None,
        )
    }

    fn deserializer(&self) -> Option<CallBorrow<'_, Deserializer<P>>> {
        self.ret("deserializer", vec![], || None::<Arc<Deserializer<P>>>)
            .map(CallBorrow::new)
    }

    fn mtl_device(&self) -> Option<Fk<kind::Device>> {
        self.step("mtl_device", vec![]).getter("mtl_device");
        Some(self.device())
    }

    fn features(&self) -> FeaturesRef {
        self.step("features", vec![]).getter("features");
        self.features_ref()
    }

    fn resource_manager(&self) -> Option<CallBorrow<'_, ResourceManager<P>>> {
        self.ret(
            "resource_manager",
            vec![],
            || None::<Arc<ResourceManager<P>>>,
        )
        .map(CallBorrow::new)
    }

    fn blit_in_rgb_2p_xr10_a8_pipeline(&self) -> Option<Fk<kind::ComputePipeline>> {
        self.step("blit_in_rgb_2p_xr10_a8_pipeline", vec![])
            .getter("blit_in_rgb_2p_xr10_a8_pipeline");
        self.provided(0)
    }

    fn blit_out_rgb_2p_xr10_a8_pipeline(&self) -> Option<Fk<kind::ComputePipeline>> {
        self.step("blit_out_rgb_2p_xr10_a8_pipeline", vec![])
            .getter("blit_out_rgb_2p_xr10_a8_pipeline");
        self.provided(1)
    }

    fn supports_buffer_with_address_ranges(&self) -> bool {
        self.ret("supports_buffer_with_address_ranges", vec![], || true)
    }

    fn supports_heap_with_address_ranges(&self) -> bool {
        self.ret("supports_heap_with_address_ranges", vec![], || true)
    }

    fn supports_resource_detach_backing(&self) -> bool {
        self.ret("supports_resource_detach_backing", vec![], || true)
    }

    fn paging_queue(&self) -> Option<Fk<kind::CommandQueue>> {
        let step = self.step("paging_queue", vec![]);
        step.getter("paging_queue");
        if step.is_nil() {
            return None;
        }
        let mut g = self.shared.paging_queue.lock();
        let q = g.get_or_insert_with(|| {
            self.shared
                .platform
                .world()
                .new_obj::<kind::CommandQueue>(Model::None)
        });
        Some(q.clone_ref())
    }

    fn new_virtual_mapping(
        &self,
        _hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<Option<Arc<Mapping<P>>>> {
        self.thr(
            Policy::Caught,
            "new_virtual_mapping",
            vec![Arg::U64(offset), Arg::U64(length), Arg::Bool(need_writable)],
            || None,
        )
    }

    fn mapped_address_for_offset(
        &self,
        _hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<MappedAddress<P>> {
        self.thr(
            Policy::Caught,
            "mapped_address_for_offset",
            vec![Arg::U64(offset), Arg::U64(length), Arg::Bool(need_writable)],
            || HostPtr::NULL,
        )
        .map(MappedAddress::unmapped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calls are logged with class `ResourceHost`; defaults answer; surfaces come from provide
    #[test]
    fn logs_defaults_and_provided_handles() {
        let p = Arc::new(FakePlatform::new());
        let r = FakeResourceHost::new(p.clone());
        let h: &dyn ResourceHost<P> = &*r;
        let s = p.fake_surfaces().make_simple(1, 1, 0, 4);
        r.provide(9, &s);
        let got = h.retain_io_surface_with_mapping_id(9).unwrap().unwrap();
        assert_eq!(got.id(), s.id());
        h.release_io_surface_with_mapping_id(9, Some(got));
        let buf = h.new_buffer_for_virtual_page(Hold::NONE, 1, 64).unwrap();
        assert_eq!(crate::platform::resource::Buffer::length(&buf), 64);
        assert_eq!(
            h.paging_queue().unwrap().id(),
            h.paging_queue().unwrap().id()
        );
        let c = &r.log().calls_to("retain_io_surface_with_mapping_id")[0];
        assert_eq!((c.class, c.args.clone()), (CLASS, vec![Arg::U64(9)]));
    }
}
