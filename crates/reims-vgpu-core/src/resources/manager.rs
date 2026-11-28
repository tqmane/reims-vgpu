//! The resource manager: one per task, owning the task's per-kind object caches and handing
//! the guest's objects out by reference.
//!
//! [`ResourceManager`] holds eight caches ([`cache`]) keyed by the guest's object reference: the
//! resources (buffers, heaps, ICBs, textures) and the Metal states. A lookup of a reference not
//! yet cached reads the guest's object entry and record and makes the object ([`create`] for the
//! buffers, heaps and Metal states, whose kinds are [`states`]; [`texture`] for the textures;
//! the ICB manager for the ICBs). [`flush`] writes dirty resources
//! back before a suspend, [`reserve`] reserves the guest's argument-buffer indices, [`lookup`]
//! is the deserializer's object lookup, [`resolver`] the function resolver Metal calls while it
//! deserializes a pipeline descriptor, and [`icb`] the indirect command buffers' creation and
//! CPU paging.
//!
//! # Contract
//!
//! What callers and the guest observe, each held by the tests named (under each module's
//! `tests`):
//! - **One cache per kind**, released in a fixed order when the manager goes: samplers,
//!   depth-stencil states, render and compute pipelines, fences, rate maps, resources, then (after
//!   the lookup) the functions; each cache in its list order
//!   (`tests::maps_release_in_order_functions_after_the_lookup`).
//! - **Reference 0** is nil for every kind and flows on
//!   (`cache::tests::get_object::reference_zero_is_nil_without_touching_the_list`).
//! - **A handed-out object stays alive** while the caller holds it, past a delete
//!   (`cache::tests::get_object::a_handed_out_object_outlives_its_delete_while_held`).
//! - **Creation holds the recursive lock**; a creation that looks up another reference re-enters
//!   it, and one that looks up its own recurses without detection
//!   (`cache::tests::new_object::a_nested_creation_reenters_the_lock`,
//!   `cache::tests::new_object::a_self_reference_recurses_without_detection`). Cached lookups never
//!   wait for a creation on another thread (`cache::lock_scope_tests`).
//! - **A delete that clears** zeroes the placement slot with one 16-byte store, whether or not
//!   anything was cached (`cache::tests::clear_placement::the_clear_is_one_16_byte_store`).
//! - **Flush order**: CPU evictions first, then one command buffer with the blit evictions, then
//!   the compute evictions, each in cache order; an eviction's raise is logged and the walk
//!   goes on (`flush::tests`).
//! - **Reservations**: textures, samplers, then ICBs, each only if any; pinned buffers are made
//!   and discarded first; a raise skips the reservations, and what the walk held goes on with
//!   it (`reserve::tests::a_raise_in_the_discards_reserves_nothing`).
//! - **The address and pipeline registries** keep the newest registration and are never
//!   unregistered (`cache::tests::registries`).
//! - **A lookup failure** logs and raises [`RaiseKind::Lookup`], except the untyped resource
//!   lookup, which answers nil; its raising variant raises for an empty entry
//!   (`lookup::tests`).

pub mod cache;
pub mod create;
#[cfg(test)]
pub(crate) mod fixture;
pub mod flush;
pub mod icb;
pub mod lookup;
pub mod records;
pub mod reserve;
pub mod resolver;
pub mod states;
pub mod texture;

use std::fmt;
use std::sync::{Arc, Weak};

use parking_lot::{Mutex, ReentrantMutex};

use crate::platform::Platform;
use crate::raise::death::DeathId;
use crate::raise::death::DeathSite;
use crate::resources::host::ResourceHost;
use crate::seams::links::ObjectLookupRef;
use crate::tasks::object_lists::ObjectLists;

pub use cache::{Cache, Caches, ManagedKind};
pub use lookup::ManagerLookup;
pub use states::{
    ComputePipelineStates, DepthStencilStates, Fences, Functions, RasterizationRateMaps,
    RenderPipelineStates, SamplerStates,
};

/// The kinds of object the deserializer looks up by reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectKind {
    /// A shader function.
    Function,
    /// A compute pipeline state.
    ComputePipeline,
    /// A render pipeline state.
    RenderPipeline,
    /// A buffer.
    Buffer,
    /// A texture.
    Texture,
    /// A sampler state.
    Sampler,
    /// A depth-stencil state.
    DepthStencil,
    /// A fence.
    Fence,
    /// A heap.
    Heap,
    /// A rasterization rate map.
    RateMap,
    /// An indirect command buffer.
    Icb,
    /// Whichever resource a reference names (the untyped lookup's raising variant).
    Resource,
}

impl ObjectKind {
    /// The kind's name, as the lookup's failure line gives it.
    pub const fn log_name(self) -> &'static str {
        match self {
            ObjectKind::Function => "function",
            ObjectKind::ComputePipeline => "compute pipeline",
            ObjectKind::RenderPipeline => "render pipeline",
            ObjectKind::Buffer => "buffer",
            ObjectKind::Texture => "texture",
            ObjectKind::Sampler => "sampler state",
            ObjectKind::DepthStencil => "depth-stencil state",
            ObjectKind::Fence => "fence",
            ObjectKind::Heap => "heap",
            ObjectKind::RateMap => "rasterization rate map",
            ObjectKind::Icb => "indirect command buffer",
            ObjectKind::Resource => "resource",
        }
    }

    /// The raise's detail when the lookup fails (diagnostics).
    pub const fn failure(self) -> &'static str {
        match self {
            ObjectKind::Function => "function lookup failed",
            ObjectKind::ComputePipeline => "compute pipeline lookup failed",
            ObjectKind::RenderPipeline => "render pipeline lookup failed",
            ObjectKind::Buffer => "buffer lookup failed",
            ObjectKind::Texture => "texture lookup failed",
            ObjectKind::Sampler => "sampler state lookup failed",
            ObjectKind::DepthStencil => "depth-stencil state lookup failed",
            ObjectKind::Fence => "fence lookup failed",
            ObjectKind::Heap => "heap lookup failed",
            ObjectKind::RateMap => "rasterization rate map lookup failed",
            ObjectKind::Icb => "indirect command buffer lookup failed",
            ObjectKind::Resource => "resource lookup failed",
        }
    }
}

/// What the resource manager raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// The deserializer's lookup of an object of this kind failed.
    Lookup(ObjectKind),
}

impl RaiseKind {
    /// Its dotted name.
    pub const fn name(self) -> &'static str {
        match self {
            RaiseKind::Lookup(k) => match k {
                ObjectKind::Function => "manager.lookup.function",
                ObjectKind::ComputePipeline => "manager.lookup.compute_pipeline",
                ObjectKind::RenderPipeline => "manager.lookup.render_pipeline",
                ObjectKind::Buffer => "manager.lookup.buffer",
                ObjectKind::Texture => "manager.lookup.texture",
                ObjectKind::Sampler => "manager.lookup.sampler",
                ObjectKind::DepthStencil => "manager.lookup.depth_stencil",
                ObjectKind::Fence => "manager.lookup.fence",
                ObjectKind::Heap => "manager.lookup.heap",
                ObjectKind::RateMap => "manager.lookup.rate_map",
                ObjectKind::Icb => "manager.lookup.icb",
                ObjectKind::Resource => "manager.lookup.resource",
            },
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<RaiseKind> for crate::raise::RaiseId {
    fn from(k: RaiseKind) -> Self {
        crate::raise::RaiseId::Manager(k)
    }
}

/// How the resource manager dies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// A creator succeeded with no object (an assert).
    NilObject,
    /// A NULL placement slot is read or written.
    NullPlacement,
    /// The lookup is used after its manager is gone.
    ManagerGone,
    /// A heap texture's size query built no texture descriptor (an assert).
    NoHeapTextureDescriptor,
    /// A creator runs after its task is gone.
    TaskGone,
    /// A rotated IOSurface texture's descriptor did not keep the command's rotation (an
    /// assert).
    RotationMismatch,
    /// Metal made no library from a function's IR and gave no error: the refusal's message
    /// formats the missing error's description (never seen).
    NilLibraryWithoutError,
    /// Metal made no pipeline state and gave no error: the refusal's message formats the
    /// missing error's description (never seen).
    NilPipelineWithoutError,
    /// A rate map descriptor's layer count is not the record's after its layers were set (an
    /// assert).
    RateMapLayerCount,
    /// A copy into a rate map layer's sample storage faults (never: the storage is checked
    /// non-NULL first).
    SampleStorage,
    /// The ICB paging's ([`icb::DeathKind`]).
    Icb(icb::DeathKind),
}

impl DeathKind {
    /// Its dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::NilObject => "manager.create.nil_object",
            DeathKind::NullPlacement => "manager.placement.null",
            DeathKind::ManagerGone => "manager.lookup.manager_gone",
            DeathKind::NoHeapTextureDescriptor => "manager.heap_texture.no_descriptor",
            DeathKind::TaskGone => "manager.create.task_gone",
            DeathKind::RotationMismatch => "manager.texture.rotation_mismatch",
            DeathKind::NilLibraryWithoutError => "manager.function.nil_library_without_error",
            DeathKind::NilPipelineWithoutError => "manager.pipeline.nil_state_without_error",
            DeathKind::RateMapLayerCount => "manager.rate_map.layer_count",
            DeathKind::SampleStorage => "manager.rate_map.sample_storage",
            DeathKind::Icb(k) => k.id(),
        }
    }

    /// The site `testkit::capture` records and the stderr line names: the kind of death, and the
    /// dotted id.
    pub const fn site(self) -> DeathSite {
        let kind = match self {
            DeathKind::NullPlacement
            | DeathKind::ManagerGone
            | DeathKind::TaskGone
            | DeathKind::NilLibraryWithoutError
            | DeathKind::NilPipelineWithoutError
            | DeathKind::SampleStorage => crate::raise::death::DeathKind::NullDeref,
            DeathKind::NilObject
            | DeathKind::NoHeapTextureDescriptor
            | DeathKind::RotationMismatch
            | DeathKind::RateMapLayerCount => crate::raise::death::DeathKind::Assert,
            DeathKind::Icb(k) => k.kind(),
        };
        DeathSite {
            kind,
            file: self.id(),
            line: 0,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<DeathKind> for DeathId {
    fn from(k: DeathKind) -> Self {
        DeathId::Manager(k)
    }
}

/// The resource manager of one task, which owns it and holds its task unretained.
///
/// `creation` is the recursive creation lock: held across a creation (a creator may look up
/// other references, and Metal's function resolver calls back into a lookup on the same
/// thread), by a delete and by the writable-cursor lookup, and nowhere else. The caches sit
/// under their own leaf lock, taken only for a short closure that never calls out, so a cached
/// lookup never waits for a creation on another thread.
pub struct ResourceManager<P: Platform> {
    platform: Arc<P>,
    /// The task's own device handle, shared without a reference of the manager's own: the
    /// manager neither retains nor releases it. `None` is a task with no device; every send to it
    /// answers zero.
    device: Option<Arc<P::Device>>,
    task: Weak<dyn ResourceHost<P>>,
    object_lists: Arc<ObjectLists<P>>,
    register_gpu_addresses: bool,
    creation: ReentrantMutex<()>,
    /// The caches' leaf lock: never held across a callout, never taken around `creation`.
    caches: Mutex<Caches<P>>,
    lookup: Mutex<Option<ObjectLookupRef<P>>>,
}

impl<P: Platform> ResourceManager<P> {
    /// A manager over `device` for `task`, with its object lists.
    ///
    /// `register_gpu_addresses` says whether buffers and heaps register their GPU addresses
    /// (the service's tasks do when the device supports indirect command buffers; in-process
    /// tasks never do). The lookup is made here, holding the manager unretained.
    ///
    /// `device` is the task's own handle, which the manager keeps without a reference of its
    /// own. An owned handle (`P::Device`) is accepted too; the manager is then its only holder
    /// and releases it when it goes.
    pub fn new<D: Into<Arc<P::Device>>>(
        platform: Arc<P>,
        device: Option<D>,
        task: Weak<dyn ResourceHost<P>>,
        object_lists: Arc<ObjectLists<P>>,
        register_gpu_addresses: bool,
    ) -> Arc<Self> {
        Arc::new_cyclic(|this| {
            let lookup: ObjectLookupRef<P> = ManagerLookup::new(this.clone());
            ResourceManager {
                platform,
                device: device.map(Into::into),
                task,
                object_lists,
                register_gpu_addresses,
                creation: ReentrantMutex::new(()),
                caches: Mutex::new(Caches::new()),
                lookup: Mutex::new(Some(lookup)),
            }
        })
    }

    /// The device; `None` is nil.
    pub fn device(&self) -> Option<&P::Device> {
        self.device.as_deref()
    }

    /// The task, if it is still there.
    pub fn task(&self) -> Option<Arc<dyn ResourceHost<P>>> {
        self.task.upgrade()
    }

    /// Whether buffers and heaps register their GPU addresses.
    pub fn register_gpu_addresses(&self) -> bool {
        // In-process tasks pass false, so no address is ever registered there.
        self.register_gpu_addresses
    }

    /// The process's platform: what the creators make objects through.
    pub fn platform(&self) -> &Arc<P> {
        &self.platform
    }

    /// The object lists (shared with the task).
    pub fn object_list_manager(&self) -> &Arc<ObjectLists<P>> {
        &self.object_lists
    }

    /// The manager's object lookup (the deserializer's).
    pub fn object_delegate(self: &Arc<Self>) -> ObjectLookupRef<P> {
        let mut slot = self.lookup.lock();
        // The one the constructor made; made again only once the manager's teardown dropped it
        let lookup = slot.get_or_insert_with(|| {
            let l: ObjectLookupRef<P> = ManagerLookup::new(Arc::downgrade(self));
            l
        });
        Arc::clone(lookup)
    }
}

impl<P: Platform> Drop for ResourceManager<P> {
    /// The caches release in a fixed order, each in its list order: every cache but the
    /// functions, then the lookup, then the functions.
    fn drop(&mut self) {
        let caches = self.caches.get_mut();
        drop(caches.sampler_states.drain());
        drop(caches.depth_stencil_states.drain());
        drop(caches.render_pipeline_states.drain());
        drop(caches.compute_pipeline_states.drain());
        drop(caches.fences.drain());
        drop(caches.rasterization_rate_maps.drain());
        drop(caches.resources.drain());
        drop(self.lookup.get_mut().take());
        caches.clear_registries();
        drop(caches.functions.drain());
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;
    use crate::platform::Handle;
    use crate::testkit::kind;

    fn entries3() -> Vec<(u8, u32, u64)> {
        vec![(0, 0, 0), (6, 0x20, REC), (6, 0x10, REC + 0x100)]
    }

    #[test]
    fn stores_its_arguments_and_makes_one_lookup() {
        let fx = Fx::new(&entries3(), Placement::Off);
        assert_eq!(
            fx.m.device().map(|d| d.id()),
            Some(fx.platform.device().id())
        );
        assert!(Arc::ptr_eq(&fx.m.task().unwrap(), &fx.task_dyn));
        assert!(Arc::ptr_eq(fx.m.object_list_manager(), &fx.olm));
        assert!(fx.m.register_gpu_addresses());
        // The in-process task's `false` is kept as given.
        let off = ResourceManager::new(
            fx.platform.clone(),
            Some(fx.platform.device()),
            Arc::downgrade(&fx.task_dyn),
            fx.olm.clone(),
            false,
        );
        assert!(!off.register_gpu_addresses());
        let a = fx.m.object_delegate();
        let b = fx.m.object_delegate();
        assert!(Arc::ptr_eq(&a, &b));
    }

    /// Over the task's own reference, the device reads the same retain count while the manager
    /// lives and after it goes, and the manager sends it nothing.
    #[test]
    fn the_device_is_held_unretained_and_never_released() {
        let fx = Fx::new(&entries3(), Placement::Off);
        let task_device = Arc::new(fx.platform.device());
        let count = task_device.retain_count();
        let mark = fx.log().calls().len();
        let m = ResourceManager::new(
            fx.platform.clone(),
            Some(task_device.clone()),
            Arc::downgrade(&fx.task_dyn),
            fx.olm.clone(),
            false,
        );
        assert_eq!(task_device.retain_count(), count);
        assert!(std::ptr::eq(m.device().unwrap(), &*task_device));
        drop(m);
        assert_eq!(task_device.retain_count(), count);
        assert!(fx.sels_since(mark).is_empty());
    }

    #[test]
    fn the_platform_is_the_callers() {
        let fx = Fx::new(&entries3(), Placement::Off);
        assert!(Arc::ptr_eq(fx.m.platform(), &fx.platform));
    }

    #[test]
    fn a_gone_task_is_none() {
        let fx = Fx::new(&entries3(), Placement::Off);
        let Fx {
            m, task, task_dyn, ..
        } = fx;
        drop(task);
        drop(task_dyn);
        assert!(m.task().is_none());
    }

    #[test]
    fn maps_release_in_order_functions_after_the_lookup() {
        let fx = Fx::new(&entries3(), Placement::Off);
        let log = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let m = fx.m.clone();
        assert!(m.register_object::<Functions>(1, fx.tracked::<kind::Function>("function", &log)));
        assert!(m.register_object::<Fences>(1, fx.tracked::<kind::Fence>("fence", &log)));
        assert!(m.register_object::<SamplerStates>(
            1,
            fx.tracked::<kind::SamplerState>("sampler", &log)
        ));
        assert!(m.register_object::<RasterizationRateMaps>(
            1,
            fx.tracked::<kind::RateMap>("ratemap", &log)
        ));
        assert!(m.register_object::<DepthStencilStates>(
            1,
            fx.tracked::<kind::DepthStencilState>("depth", &log)
        ));
        assert!(m.register_object::<RenderPipelineStates>(
            1,
            fx.tracked::<kind::RenderPipeline>("render", &log)
        ));
        assert!(m.register_object::<ComputePipelineStates>(
            1,
            fx.tracked::<kind::ComputePipeline>("compute", &log)
        ));
        // Within one cache, the cache's own order.
        assert!(m.register_object::<SamplerStates>(
            2,
            fx.tracked::<kind::SamplerState>("sampler2", &log)
        ));
        let lookup = Arc::downgrade(&m.object_delegate());
        drop(m);
        let Fx { m, .. } = fx;
        drop(m);
        assert!(lookup.upgrade().is_none());
        let log = log.lock();
        let (samplers, rest) = log.split_at(2);
        let mut samplers = samplers.to_vec();
        samplers.sort_unstable();
        assert_eq!(samplers, ["sampler", "sampler2"]);
        assert_eq!(
            rest,
            ["depth", "render", "compute", "fence", "ratemap", "function"]
        );
    }

    #[test]
    fn ids_and_sites() {
        assert_eq!(
            RaiseKind::Lookup(ObjectKind::Heap).to_string(),
            "manager.lookup.heap"
        );
        assert_eq!(
            DeathKind::NullPlacement.site().kind,
            crate::raise::death::DeathKind::NullDeref
        );
        assert_eq!(
            DeathKind::NilObject.site().kind,
            crate::raise::death::DeathKind::Assert
        );
    }
}
