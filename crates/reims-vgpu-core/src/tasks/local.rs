//! The local task: a guest task run in this process (task 0 everywhere, and every task in the
//! blocks and map setups). [`LocalTask`] answers the task commands (through
//! [`Task`](crate::tasks::Task)), the kernel task's backing and memory lookups, and is its
//! resources' host (`ResourceHost`).
//!
//! Task memory is reached one of two ways ([`Backend`]), both through the guest's page table,
//! which the task root page names:
//! - **the blocks setup**: the host maps the guest pages into a host task of its own
//!   (`createTask`, `mapMemory`, `unmapMemory`) on first use, and a range tracker keeps what is
//!   mapped;
//! - **the map setup**: each access makes a fresh mapping of the pages' physical ranges from the
//!   device's memory map.
//!
//! The parts:
//! - [`new`]: making a task: its parts, its host task, the task root and the page-table walker;
//! - [`memory`]: task memory: bounds checks, host mappings, no-copy buffers, copies, cursors;
//! - [`exec`]: exec commands: resource preparation, decode, commit and completion;
//! - [`info`]: the info command, the compute-info and heap-size replies, and the kernel task's
//!   backing lookups.
//!
//! Every entry point runs with the task's platform as the resources' platform (`with_platform`).
//!
//! Ordering contracts (the suite's blocks and map setups exercise them; no unit test here drives
//! a whole local task):
//! - **Init**: the Metal device, the memory-map test, the host task (blocks setup), the task
//!   root's mapping, the paging queue, the paging event, then the root's list fields go to the
//!   object lists and the IOSurface-buffer test comes last. A refused init drops the partial
//!   task, which destroys its host task; a raise out of the init leaks the half-built task with
//!   its host task and parts, the deserializer included.
//! - **Task root**: the page-table walker's weak device reference is formed after the root is
//!   mapped and before its depth is read, and ends the process once the device's teardown has
//!   begun; the root's task length is written last.
//! - **Lazy host mapping** (blocks setup): an access maps each run the tracker reports
//!   unmapped; a run grows only from its first page and stops at two pages; the tracker takes
//!   the range only once every run is mapped (a raising host map leaves it untracked), and a
//!   mapped range is never asked for again.
//! - **Unmap**: the object list's unmap, the tracker, then the host (a raising host unmap goes
//!   on after the tracker).
//! - **Exec**: the preparation (the paging buffer committed even when a prepare raised, then CPU
//!   paging), the exec buffer's wait, a copy of every command buffer, the decode, the completion
//!   handler, the FIFO value's increment and signal, the commit, then the synchronize buffer; a
//!   resource the host invalidated is marked guest-valid. A raise reports 2 once, and twice when
//!   the synchronize buffer was already committed.
//! - **Info**: waits 5 s for the previous exec, decodes without a commit, then advances and
//!   signals the FIFO value on the CPU.
//! - **Drop**: the task root's mapping first, then the paging event and the walker, then the host
//!   task's `destroyTask` (blocks setup), then the core's parts.
//!
//! The commands answer `RaiseResult`: this area's raises are [`RaiseKind`]s through
//! `tasks::raise`, and its handlers are the frames that take their `Err`.

mod exec;
mod info;
mod memory;
mod new;
#[cfg(test)]
mod tests;

use std::fmt;
use std::mem::offset_of;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::deserializer::Deserializer;
use crate::device::DeviceCore;
use crate::fifo::wire::Records;
use crate::guest::GuestPtr;
use crate::guest::records::{
    APV_PAGE_SHIFT, APV_PAGE_SIZE, CmdExecIndirectResource, CmdExecIndirectResource3, FeaturesRef,
    TaskRoot2,
};
use crate::memory::cursor::{Cursor, WritableCursor};
use crate::memory::mapping::Mapping;
use crate::memory::page_table::PageTableWalker;
use crate::memory::tracker::RangeTracker;
use crate::num;
use crate::platform::Platform;
use crate::raise::OnFault;
use crate::raise::RaiseResult;
use crate::raise::death::DeathKind as Fault;
use crate::rt::Hold;

use crate::resources::host::{CallBorrow, MappedAddress, ResourceHost};
use crate::resources::manager::ResourceManager;
use crate::resources::resource::{Resource, with_platform};
use crate::seams::links::{KernelTaskRef, TaskDeviceLink};
use crate::tasks::object_lists::ObjectLists;
use crate::tasks::resource_list::ResourceList;
use crate::tasks::shared::TaskCore;
use crate::tasks::{
    Completion, ExecArgs, ExecStatus, ResourceInvalidation, ResourceReference, TaskInfo,
    Unsupported,
};

/// The guest page: 16 KiB (4 KiB on x86-64).
const PAGE: u64 = APV_PAGE_SIZE;
/// The offset of an address within its page.
const PAGE_MASK: u64 = PAGE - 1;

/// The task root's page-table root (a guest page number).
const ROOT_PAGE_TABLE_ROOT: u64 = num::const_u64(offset_of!(TaskRoot2, page_table_root));
/// The task root's page-table depth.
const ROOT_PAGE_TABLE_DEPTH: u64 = num::const_u64(offset_of!(TaskRoot2, page_table_depth));
/// The task root's object-list offset.
const ROOT_OBJECT_LIST_OFFSET: u64 = num::const_u64(offset_of!(TaskRoot2, object_list_offset));
/// The task root's object-list length.
const ROOT_OBJECT_LIST_LENGTH: u64 = num::const_u64(offset_of!(TaskRoot2, object_list_length));
/// The task root's task length, which the task writes.
const ROOT_TASK_LENGTH: u64 = num::const_u64(offset_of!(TaskRoot2, task_length));

/// The host address of guest page `page`'s first byte in a physical or task address space.
fn page_address(page: u32) -> u64 {
    u64::from(page).wrapping_shl(APV_PAGE_SHIFT)
}

/// The local task's raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// A range past the task's end, or one whose end wraps.
    Range,
    /// A copy's range reaching the task's end.
    CopyRange,
    /// A map of task memory in a task with no page table (task root 0).
    NoPageTable,
    /// A map at an offset that is not page-aligned.
    UnalignedOffset,
    /// A map of a length that is not page-aligned.
    UnalignedLength,
    /// A map of length zero.
    ZeroLength,
    /// The host refused to map a run of pages into its task.
    HostMap,
    /// The host refused to unmap a range of its task.
    HostUnmap,
    /// No mapping could be made over the task's pages.
    Mapping,
    /// The mapping made is read-only where a writable one is needed.
    ReadOnlyMapping,
    /// Task memory's host address is NULL.
    NullAddress,
    /// Metal refused the no-copy buffer over task memory.
    Buffer,
    /// The object lists could not be mapped.
    ObjectLists,
    /// The object an identifier names is not a backing (or not a shared-texture backing).
    Backing,
    /// The compute pipeline a reference names could not be had.
    Pipeline,
    /// A reply length outside 16 bytes to a page.
    ReplyLength,
}

impl RaiseKind {
    /// The raise's dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            RaiseKind::Range => "tasks.local.range",
            RaiseKind::CopyRange => "tasks.local.copy_range",
            RaiseKind::NoPageTable => "tasks.local.no_page_table",
            RaiseKind::UnalignedOffset => "tasks.local.unaligned_offset",
            RaiseKind::UnalignedLength => "tasks.local.unaligned_length",
            RaiseKind::ZeroLength => "tasks.local.zero_length",
            RaiseKind::HostMap => "tasks.local.host_map",
            RaiseKind::HostUnmap => "tasks.local.host_unmap",
            RaiseKind::Mapping => "tasks.local.mapping",
            RaiseKind::ReadOnlyMapping => "tasks.local.read_only_mapping",
            RaiseKind::NullAddress => "tasks.local.null_address",
            RaiseKind::Buffer => "tasks.local.buffer",
            RaiseKind::ObjectLists => "tasks.local.object_lists",
            RaiseKind::Backing => "tasks.local.backing",
            RaiseKind::Pipeline => "tasks.local.pipeline",
            RaiseKind::ReplyLength => "tasks.local.reply_length",
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// The local task's deaths (the tasks area's own cover the rest: a torn-down part, a NULL task
/// root, an unsupported operation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// A reply is written through a NULL host address.
    NullReply,
    /// The device is gone where the task reads its features.
    DeviceGone,
    /// The page-table walker's weak reference to the device is formed once the device's
    /// teardown has begun (the runtime's abort).
    DeviceDeallocating,
    /// A paging buffer was made with no paging event (the task's `newEvent` answered nil).
    NoPagingEvent,
    /// Resources complete with no FIFO event.
    NoFifoEvent,
    /// A kernel-task lookup reached a user task.
    NotKernelTask,
    /// A backing that pages through a compute encoder is synchronized in a blit encoder.
    ComputePagedBacking,
    /// An IOSurface is retained by mapping id without the mapper feature.
    NoMapperSurfaces,
}

impl DeathKind {
    /// The death's dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::NullReply => "tasks.local.null_reply",
            DeathKind::DeviceGone => "tasks.local.device_gone",
            DeathKind::DeviceDeallocating => "tasks.local.device_deallocating",
            DeathKind::NoPagingEvent => "tasks.local.no_paging_event",
            DeathKind::NoFifoEvent => "tasks.local.no_fifo_event",
            DeathKind::NotKernelTask => "tasks.local.not_kernel_task",
            DeathKind::ComputePagedBacking => "tasks.local.compute_paged_backing",
            DeathKind::NoMapperSurfaces => "tasks.local.no_mapper_surfaces",
        }
    }

    /// The kind of fault there.
    pub const fn fault(self) -> Fault {
        match self {
            DeathKind::NullReply | DeathKind::DeviceGone => Fault::NullDeref,
            DeathKind::DeviceDeallocating => Fault::Abort,
            DeathKind::NoPagingEvent
            | DeathKind::NoFifoEvent
            | DeathKind::NotKernelTask
            | DeathKind::ComputePagedBacking
            | DeathKind::NoMapperSurfaces => Fault::Assert,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<RaiseKind> for crate::tasks::RaiseKind {
    fn from(k: RaiseKind) -> crate::tasks::RaiseKind {
        crate::tasks::RaiseKind::Local(k)
    }
}

impl From<DeathKind> for crate::tasks::DeathKind {
    fn from(k: DeathKind) -> crate::tasks::DeathKind {
        crate::tasks::DeathKind::Local(k)
    }
}

impl From<DeathKind> for crate::raise::DeathId {
    fn from(k: DeathKind) -> crate::raise::DeathId {
        crate::raise::DeathId::Tasks(k.into())
    }
}

/// Raises `kind` with `detail` on the area's inner channel, in a function reached both ways
/// (`tasks::raise_on`).
#[track_caller]
fn raise_on<T>(on: OnFault, kind: RaiseKind, detail: impl fmt::Display) -> RaiseResult<T> {
    crate::tasks::raise_on(on, kind.into(), detail)
}

/// Raises `kind` with `detail` at a site no path reaches without a handler
/// (`tasks::raise_caught`).
#[track_caller]
fn raise_caught<T>(kind: RaiseKind, detail: impl fmt::Display) -> RaiseResult<T> {
    crate::tasks::raise_caught(kind.into(), detail)
}

/// Raises `kind` with `detail` carrying `v` outward (`tasks::raise_carrying`), at a site no path
/// reaches without a handler.
#[track_caller]
fn raise_carrying<T>(
    kind: RaiseKind,
    detail: impl fmt::Display,
    v: impl std::any::Any + Send,
) -> RaiseResult<T> {
    crate::tasks::raise_carrying(kind.into(), detail, v)
}

/// The kernel task, as a user task reaches it through its device (a backing lookup, a
/// shared-texture handle, a guest invalidation), always under a handler. The hold's holder keeps a
/// reference to its end ([`Hold::keep`]: the FIFO command's or a landing frame's inside it, an API
/// entry's on a host thread, or with none the thread's last-resort holder), so a kernel task
/// deleted meanwhile on another FIFO goes (with its `destroyTask` and root-page unmap) after the
/// rest of that work, never inside it.
fn kernel_task<P: Platform>(
    hold: Hold<'_>,
    device: &DeviceCore<P>,
) -> RaiseResult<KernelTaskRef<P>> {
    let task = device.get_kernel_task(OnFault::Return)?;
    hold.keep(Arc::clone(&task));
    Ok(task)
}

/// How a local task reaches its memory.
#[expect(clippy::large_enum_variant, reason = "one per task, never moved")]
enum Backend {
    /// The blocks setup: the ranges mapped into the host's task so far.
    Blocks {
        /// What is mapped, in task offsets.
        mapped: RangeTracker,
    },
    /// The map setup: a fresh mapping of physical ranges for each access.
    MemoryMap,
}

/// A guest task in this process. It holds its device unretained ([`TaskDeviceLink`]).
pub struct LocalTask<P: Platform> {
    /// The core both task kinds share.
    pub base: TaskCore<P>,
    /// The platform the task logs to, and its resources run on.
    platform: Arc<P>,
    /// The device.
    device: TaskDeviceLink<P>,
    /// The task's length (the core's stays 0 in process).
    length: u64,
    /// The task's id (0 is the kernel task).
    task_id: u32,
    /// The host's task (blocks setup; zero in the map setup, and until `createTask` answers).
    task_info: TaskInfo,
    /// How the task reaches its memory.
    backend: Backend,
    /// The task root page's mapping, released first when the task goes.
    task_root_mapping: Option<Arc<Mapping<P>>>,
    /// The mapped task root page: NULL when the task has none.
    task_root: GuestPtr,
    /// The walker over the guest's page table, when the task root set one up.
    walker: Option<PageTableWalker>,
    /// Whether buffers may be made over IOSurfaces.
    io_surface_buffers_allowed: bool,
    /// Serializes the resource preparations of concurrent execs.
    prepare_mutex: Mutex<()>,
    /// The object list's offset and length as the object lists last stored them (zeros after a
    /// refusal), under the task's lock: what an unmap tests its range against. The placement
    /// list of a local task is always empty.
    object_list_range: Mutex<(u64, u32)>,
}

impl<P: Platform> LocalTask<P> {
    /// The task's id.
    pub fn task_id(&self) -> u32 {
        self.task_id
    }

    /// Whether buffers may be made over IOSurfaces: the device supports them and the guest
    /// asked for them, when the task was made.
    pub fn io_surface_buffers_allowed(&self) -> bool {
        self.io_surface_buffers_allowed
    }

    /// The task's parts for this call; a use after `teardown` dies here.
    #[expect(clippy::type_complexity, reason = "the three parts")]
    fn parts(
        &self,
    ) -> (
        Arc<Deserializer<P>>,
        Arc<ResourceManager<P>>,
        Arc<ObjectLists<P>>,
    ) {
        match self.base.parts() {
            Some(parts) => parts,
            None => crate::fatal!(crate::tasks::DeathKind::TornDown),
        }
    }

    /// The object lists mapped for this call, or the raise that they could not be.
    fn map_object_lists(&self, hold: Hold<'_>, lists: &ObjectLists<P>) -> RaiseResult<()> {
        if lists.ensure_mapped(hold) {
            Ok(())
        } else {
            raise_caught(
                RaiseKind::ObjectLists,
                "the task's object lists could not be mapped",
            )
        }
    }

    /// Runs `f` with the task's platform as the resources' platform.
    fn on_platform<T>(&self, f: impl FnOnce() -> T) -> T {
        with_platform(&self.platform, f)
    }

    /// A command: `f` on the task's platform, its raise handed on as the device and the FIFO
    /// take it.
    fn command<T>(&self, f: impl FnOnce() -> RaiseResult<T>) -> RaiseResult<T> {
        self.on_platform(f)
    }

    /// The device's features. The device is never gone while a resource asks.
    fn features(&self) -> FeaturesRef {
        match self.device.upgrade() {
            Some(d) => d.features(),
            None => crate::fatal!(DeathKind::DeviceGone),
        }
    }
}

impl<P: Platform> Drop for LocalTask<P> {
    /// The task root's mapping goes first (in the blocks setup its release unmaps the root page),
    /// then the paging event and the page-table walker, then, in the blocks setup, the host's
    /// task (not with the device gone). The core's parts follow in its own drop.
    fn drop(&mut self) {
        drop(self.task_root_mapping.take());
        self.task_root = GuestPtr::default();
        drop(self.base.paging_event.take());
        drop(self.walker.take());
        if let Backend::Blocks { .. } = self.backend
            && let Some(device) = self.device.upgrade()
        {
            device.destroy_task(&self.task_info);
        }
    }
}

/// The task commands ([`Task`](crate::tasks::Task)'s, where each is documented).
impl<P: Platform> LocalTask<P> {
    /// Tears the task down.
    pub fn teardown(&self) {
        self.on_platform(|| self.base.teardown());
    }

    /// Deletes the object a serialized delete names.
    pub fn delete_object_with_serialized_data(
        &self,
        hold: Hold<'_>,
        data: &[u8],
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| {
            self.base.delete_object_with_serialized_data(hold, data)?;
            done(ExecStatus::OK, None);
            Ok(())
        })
    }

    /// Deletes a resource.
    pub fn delete_resource(
        &self,
        hold: Hold<'_>,
        resource: u32,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| {
            self.base.delete_resource(hold, resource)?;
            done(ExecStatus::OK, None);
            Ok(())
        })
    }

    /// Discards resources.
    pub fn discard_resources(
        &self,
        hold: Hold<'_>,
        resources: Records<'_, ResourceReference>,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| {
            let deleted = self.base.discard_resources(hold, resources)?;
            done(ExecStatus::OK, None);
            drop(deleted);
            Ok(())
        })
    }

    /// A FIFO was deleted: completes at once.
    pub fn fifo_deleted(&self, _stamp_index: u32, done: Completion) -> RaiseResult<()> {
        done(ExecStatus::OK, None);
        Ok(())
    }

    /// Flushes the resources into guest memory.
    pub fn flush_resources(&self, hold: Hold<'_>) -> RaiseResult<()> {
        self.command(|| self.base.flush_resources(hold))
    }

    /// Applies validity flags to resources.
    pub fn invalidate_resources(
        &self,
        resources: Records<'_, ResourceInvalidation>,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| {
            self.base.invalidate_resources(resources)?;
            done(ExecStatus::OK, None);
            Ok(())
        })
    }

    /// Maps task memory.
    pub fn map_memory_at_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| self.map_memory(hold, offset, length, done))
    }

    /// Discards the resource over replaced physical memory.
    pub fn replace_physical(&self, physical: u32, done: Completion) -> RaiseResult<()> {
        self.command(|| {
            self.base.replace_physical(physical)?;
            done(ExecStatus::OK, None);
            Ok(())
        })
    }

    /// Sets the object and placement lists; a refused list completes with 2.
    pub fn set_object_list_offset(
        &self,
        hold: Hold<'_>,
        object_list_offset: u64,
        object_list_length: u32,
        placement_list_offset: u64,
        placement_list_length: u32,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| {
            let _lock = self.base.mutex.write();
            let (_, _, lists) = self.parts();
            let stored = lists.set_object_list(
                hold,
                object_list_offset,
                object_list_length,
                placement_list_offset,
                placement_list_length,
                self.task_root,
            );
            *self.object_list_range.lock() = if stored {
                (object_list_offset, object_list_length)
            } else {
                (0, 0)
            };
            let status = if stored {
                ExecStatus::OK
            } else {
                ExecStatus::FAIL
            };
            done(status, None);
            Ok(())
        })
    }

    /// Synchronizes resources into guest memory, and with `discard` discards them.
    pub fn synchronize_resources(
        &self,
        hold: Hold<'_>,
        resources: Records<'_, ResourceReference>,
        discard: bool,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| {
            self.base
                .synchronize_resources(hold, resources, discard, done)
        })
    }

    /// Unmaps task memory.
    pub fn unmap_memory_at_offset(
        &self,
        offset: u64,
        length: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| self.unmap_memory(offset, length, done))
    }

    /// Executes command buffers with 24-byte resource records.
    pub fn exec_indirect2(
        &self,
        hold: Hold<'_>,
        args: ExecArgs<'_, P, CmdExecIndirectResource>,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| {
            self.exec_indirect(
                hold,
                args.cmd_buffers,
                ResourceList::V2(args.resources),
                args.fifo_event,
                args.fifo_event_value,
                done,
            )
        })
    }

    /// Executes command buffers with 8-byte resource records.
    pub fn exec_indirect3(
        &self,
        hold: Hold<'_>,
        args: ExecArgs<'_, P, CmdExecIndirectResource3>,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| {
            self.exec_indirect(
                hold,
                args.cmd_buffers,
                ResourceList::V3(args.resources),
                args.fifo_event,
                args.fifo_event_value,
                done,
            )
        })
    }

    /// Writes a compute pipeline's info into task memory.
    pub fn get_compute_info(
        &self,
        hold: Hold<'_>,
        pipeline_state_reference: u32,
        max_key: u32,
        count: u32,
        offset: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| {
            self.compute_info(hold, pipeline_state_reference, max_key, count, offset, done)
        })
    }

    /// Writes a heap texture's size and alignment into task memory.
    pub fn heap_texture_size_and_align(
        &self,
        serializer_payload: &[u8],
        reply_virtual_offset: u64,
        reply_length: u64,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| {
            self.heap_texture_reply(serializer_payload, reply_virtual_offset, reply_length, done)
        })
    }

    /// Writes the info of command buffers into task memory.
    pub fn info_indirect(
        &self,
        hold: Hold<'_>,
        args: ExecArgs<'_, P, CmdExecIndirectResource3>,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| self.info(hold, args, done))
    }

    /// Resets a rasterization rate map.
    pub fn reset_rasterization_rate_map(&self, map: u32, done: Completion) -> RaiseResult<()> {
        self.command(|| self.reset_rate_map(map, &done))
    }
}

/// The kernel task's (task 0's) lookups: the display's and the user tasks' reads of its
/// memory and backings.
impl<P: Platform> LocalTask<P> {
    /// The backing resource `identifier` names.
    pub fn backing_for_id(
        &self,
        hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        self.command(|| self.backing(hold, identifier))
    }

    /// Copies task memory into `dst` (the display's gamma table and cursor glyph reads, always
    /// under a handler).
    pub fn copy_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        dst: &mut [u8],
    ) -> RaiseResult<()> {
        self.command(|| self.copy(hold, OnFault::Return, offset, dst))
    }

    /// A fresh buffer of task memory (the display's ICC profile read). Its raises are as `on`
    /// says: the read is reached under a FIFO command's handler and from API entries with none.
    pub fn data_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        on: OnFault,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Vec<u8>>> {
        self.command(|| self.data(hold, on, offset, length))
    }

    /// Deletes a backing (opcode 54).
    pub fn delete_backing(
        &self,
        hold: Hold<'_>,
        backing: u32,
        done: Completion,
    ) -> RaiseResult<()> {
        self.command(|| self.remove_backing(hold, backing, done))
    }

    /// Marks shared texture backings guest-invalid: from a user task's resource preparation
    /// and the remote task's exec reply.
    pub fn invalidate_guest_for_shared_texture_backing(
        &self,
        hold: Hold<'_>,
        backings: Records<'_, u32>,
    ) -> RaiseResult<()> {
        self.command(|| self.invalidate_guest(hold, backings))
    }

    /// A new shared texture handle for the backing `identifier` names (the caller owns it).
    pub fn new_shared_texture_handle_for_id(
        &self,
        hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<P::SharedTextureHandle>> {
        self.command(|| self.shared_texture_handle(hold, identifier))
    }

    /// The prepared backing `identifier` names: the display's present path.
    pub fn prepared_backing_for_id(
        &self,
        hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        self.command(|| self.prepared_backing(hold, identifier))
    }
}

impl<P: Platform> ResourceHost<P> for LocalTask<P> {
    fn new_buffer_for_virtual_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
    ) -> RaiseResult<P::Buffer> {
        self.on_platform(|| self.buffer_for_page(hold, page, length))
    }

    fn guest_pages_for_virtual_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
    ) -> RaiseResult<Option<crate::memory::map::GuestPages<P>>> {
        if let Backend::Blocks { .. } = self.backend {
            return Ok(None);
        }
        self.on_platform(|| self.guest_pages(hold, page_address(page), length))
    }

    fn prepare_backing(
        &self,
        hold: Hold<'_>,
        backing: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        self.on_platform(|| {
            let _lock = self.base.mutex.read();
            backing.prepare_in_encoder(hold, encoder)
        })
    }

    fn synchronize_backing(
        &self,
        hold: Hold<'_>,
        backing: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        self.on_platform(|| {
            let _lock = self.base.mutex.read();
            crate::fatal_unless!(
                !backing.needs_compute_paging(),
                DeathKind::ComputePagedBacking
            );
            backing.synchronize_in_encoder(hold, encoder)
        })
    }

    fn retain_io_surface_with_mapping_id(
        &self,
        identifier: u64,
    ) -> RaiseResult<Option<P::Surface>> {
        // A guest below version 10 that uses type 11 or 15 ends the process here (testing.md
        // 7.4).
        crate::fatal_unless!(
            self.features().load().mapper_io_surfaces,
            DeathKind::NoMapperSurfaces
        );
        // None without a mapper service
        Ok(self
            .device
            .upgrade()
            .and_then(|d| d.mapper_service())
            .and_then(|m| m.retain_surface_with_mapping_id(identifier)))
    }

    fn release_io_surface_with_mapping_id(&self, _identifier: u64, surface: Option<P::Surface>) {
        drop(surface);
    }

    fn address_ranges_for_virtual_page(
        &self,
        _hold: Hold<'_>,
        _page: u32,
        _length: u64,
        _read_only: &mut bool,
    ) -> RaiseResult<Option<P::AddressRangeArray>> {
        // Unreachable: a local task reports no address-range support.
        self.base.unsupported(Unsupported::AddressRanges)
    }

    fn new_shared_texture_handle_for_id(
        &self,
        hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<P::SharedTextureHandle>> {
        self.on_platform(|| self.shared_texture_handle(hold, identifier))
    }

    fn data_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Vec<u8>>> {
        // The resource manager's reads, always under a handler.
        self.on_platform(|| self.data(hold, OnFault::Return, offset, length))
    }

    fn backing_for_id(
        &self,
        hold: Hold<'_>,
        identifier: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        self.on_platform(|| self.backing(hold, identifier))
    }

    fn cursor_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Cursor>> {
        Ok(self.on_platform(|| self.cursor(hold, offset, length)))
    }

    fn writable_cursor_from_virtual_offset(
        &self,
        _hold: Hold<'_>,
        _offset: u64,
        _length: u64,
    ) -> RaiseResult<Option<WritableCursor>> {
        // A local task has no writable cursor; asking ends the process.
        self.base.unsupported(Unsupported::WritableCursor)
    }

    fn deserializer(&self) -> Option<CallBorrow<'_, Deserializer<P>>> {
        self.base.deserializer()
    }

    fn mtl_device(&self) -> Option<P::Device> {
        self.base.mtl_device()
    }

    fn features(&self) -> FeaturesRef {
        LocalTask::features(self)
    }

    fn resource_manager(&self) -> Option<CallBorrow<'_, ResourceManager<P>>> {
        self.base.resource_manager()
    }

    fn blit_in_rgb_2p_xr10_a8_pipeline(&self) -> Option<P::ComputePipeline> {
        self.device
            .upgrade()
            .and_then(|d| d.blit_in_rgb_2p_xr10_a8_pipeline())
    }

    fn blit_out_rgb_2p_xr10_a8_pipeline(&self) -> Option<P::ComputePipeline> {
        self.device
            .upgrade()
            .and_then(|d| d.blit_out_rgb_2p_xr10_a8_pipeline())
    }

    fn supports_buffer_with_address_ranges(&self) -> bool {
        self.base.supports_buffer_with_address_ranges()
    }

    fn supports_heap_with_address_ranges(&self) -> bool {
        self.base.supports_heap_with_address_ranges()
    }

    fn supports_resource_detach_backing(&self) -> bool {
        self.base.supports_resource_detach_backing()
    }

    fn paging_queue(&self) -> Option<P::CommandQueue> {
        self.base.paging_queue()
    }

    fn new_virtual_mapping(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<Option<Arc<Mapping<P>>>> {
        self.on_platform(|| self.virtual_mapping(hold, offset, length, need_writable))
    }

    fn mapped_address_for_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<MappedAddress<P>> {
        self.on_platform(|| self.mapped_address(hold, offset, length, need_writable))
    }
}
