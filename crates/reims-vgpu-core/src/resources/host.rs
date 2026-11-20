//! What a resource, the resource manager and the object-list managers ask of their task:
//! [`ResourceHost`], and [`CallBorrow`], the loan of a task part for one call.
//!
//! One trait, because every caller holds the one task. `LocalTask` and `GpuTask` implement it;
//! resources, the manager and the object-list managers hold it as a `Weak`, whose failed
//! upgrade is a torn-down task.

use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use crate::deserializer::Deserializer;
use crate::guest::records::FeaturesRef;
use crate::host::HostPtr;
use crate::memory::cursor::{Cursor, WritableCursor};
use crate::memory::map::GuestPages;
use crate::memory::mapping::Mapping;
use crate::platform::Platform;
use crate::raise::RaiseResult;
use crate::resources::manager::ResourceManager;
use crate::resources::resource::Resource;
use crate::rt::Hold;

/// One of a task's parts (its deserializer, its resource manager) lent for the length of one
/// call.
///
/// The task's teardown destroys its parts at that moment, releasing their Metal objects. Which
/// Metal objects are alive decides the resource indices and GPU VAs Metal hands out next (a
/// released texture's index is reused), and the guest sees both (D40), so nothing may keep a part
/// alive past the call that borrowed it. A `CallBorrow` cannot be cloned, cannot give up its `Arc`,
/// and cannot outlive the `&self` of the `ResourceHost` call that made it, so a holder (an ICB
/// resource, a decoder) cannot store it. It holds a strong reference while it lives, so a
/// `teardown` on another thread during the call frees the part when the call ends instead of under
/// it.
pub struct CallBorrow<'a, T> {
    part: Arc<T>,
    _call: PhantomData<&'a ()>,
}

impl<T> CallBorrow<'_, T> {
    /// Lends `part` for one call. Only a `ResourceHost` implementation makes one, from the
    /// task's own `Arc` (`BaseTask::parts`).
    pub fn new(part: Arc<T>) -> Self {
        CallBorrow {
            part,
            _call: PhantomData,
        }
    }
}

impl<T> Deref for CallBorrow<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.part
    }
}

impl<T> std::fmt::Debug for CallBorrow<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CallBorrow")
    }
}

/// A task as its resources, its resource manager and its object-list managers see it.
pub trait ResourceHost<P: Platform>: Send + Sync {
    // ---- What a resource asks.

    /// A +1 no-copy buffer over the guest's pages. Raises when the pages or the buffer cannot
    /// be had; never `None` from a live task.
    fn new_buffer_for_virtual_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
    ) -> RaiseResult<P::Buffer>;

    /// The guest's pages from `page`, `length` bytes, read and written through the memory map
    /// with no mapping made ([`GuestPages`]); `None` where the task has no memory map, or (the
    /// default) cannot answer them: a caller then makes a buffer over the pages.
    fn guest_pages_for_virtual_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
    ) -> RaiseResult<Option<GuestPages<P>>> {
        let _ = (hold, page, length);
        Ok(None)
    }

    /// Prepares an IOSurface backing. `encoder` is `None` when Metal gave no encoder, which
    /// nothing checks: the backing's validity writes still happen.
    fn prepare_backing(
        &self,
        hold: Hold<'_>,
        backing: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()>;

    /// Synchronizes an IOSurface backing (`encoder` as for a prepare).
    fn synchronize_backing(
        &self,
        hold: Hold<'_>,
        backing: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()>;

    /// The mapper's surface for `identifier`, +1, or `None`. Under isolation a host round trip
    /// that blocks the service queue.
    fn retain_io_surface_with_mapping_id(&self, identifier: u64)
    -> RaiseResult<Option<P::Surface>>;

    /// Hands back the reference [`Self::retain_io_surface_with_mapping_id`] gave. The service
    /// dies on an unknown or mismatched release.
    fn release_io_surface_with_mapping_id(&self, identifier: u64, surface: Option<P::Surface>);

    /// The guest's address ranges for a page run: the service's only (the in-process task dies).
    /// `read_only` is an out-parameter, possibly written before a raise.
    fn address_ranges_for_virtual_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
        read_only: &mut bool,
    ) -> RaiseResult<Option<P::AddressRangeArray>>;

    /// A +1 shared texture handle, or `None`. Under isolation a host round trip.
    fn new_shared_texture_handle_for_id(
        &self,
        hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<P::SharedTextureHandle>>;

    /// A copy of guest bytes, or `None`.
    fn data_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Vec<u8>>>;

    /// Task 0's backing of `identifier` (the service's dies).
    fn backing_for_id(
        &self,
        hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>>;

    /// A cursor over guest memory, `None` on failure.
    fn cursor_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Cursor>>;

    /// A writable cursor over guest memory; in-process this dies.
    fn writable_cursor_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<WritableCursor>>;

    // ---- The task's parts.

    /// The task's deserializer, `None` after its teardown. Lent for this call only
    /// ([`CallBorrow`]): never stored.
    fn deserializer(&self) -> Option<CallBorrow<'_, Deserializer<P>>>;

    /// The Metal device, retained; `None` for a GPU task whose handshake found no device, where
    /// every question to it answers zero.
    fn mtl_device(&self) -> Option<P::Device>;

    /// The guest's features, read at each use.
    fn features(&self) -> FeaturesRef;

    /// The task's resource manager; `None` after its teardown (the caller dies there). Lent for
    /// this call only ([`CallBorrow`]): the task is its only owner, and the teardown's destruction
    /// must not be delayed.
    fn resource_manager(&self) -> Option<CallBorrow<'_, ResourceManager<P>>>;

    /// The XR10 two-plane upload pipeline, retained.
    fn blit_in_rgb_2p_xr10_a8_pipeline(&self) -> Option<P::ComputePipeline>;

    /// The XR10 two-plane download pipeline, retained.
    fn blit_out_rgb_2p_xr10_a8_pipeline(&self) -> Option<P::ComputePipeline>;

    /// Whether a buffer can be backed by address ranges.
    fn supports_buffer_with_address_ranges(&self) -> bool;

    /// Whether a heap can be backed by address ranges.
    fn supports_heap_with_address_ranges(&self) -> bool;

    /// Whether resources detach their backings (the service's tasks).
    fn supports_resource_detach_backing(&self) -> bool;

    /// The paging queue, retained; `None` when Metal gave none (the device's blit-down queue
    /// in-process, a new queue in the service; neither is checked).
    fn paging_queue(&self) -> Option<P::CommandQueue>;

    // ---- What an object-list manager asks.

    /// A +1 mapping of a task range, or `None`.
    fn new_virtual_mapping(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<Option<Arc<Mapping<P>>>>;

    /// A pointer into task memory with the mapping behind it, which the caller holds while it
    /// uses the pointer. NULL on failure.
    fn mapped_address_for_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<MappedAddress<P>>;
}

/// A host address into task memory and the mapping that backs it: the address is valid while
/// the mapping lives. The mapping is `None` where the host's own task maps the memory (the
/// blocks setup) and for NULL.
pub struct MappedAddress<P: Platform> {
    /// The address.
    pub address: HostPtr,
    /// The mapping that backs it.
    pub mapping: Option<Arc<Mapping<P>>>,
}

impl<P: Platform> MappedAddress<P> {
    /// An address no mapping of ours backs.
    pub fn unmapped(address: HostPtr) -> Self {
        MappedAddress {
            address,
            mapping: None,
        }
    }
}
