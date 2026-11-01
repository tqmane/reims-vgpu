//! What the platform tells the runtime of host pages that change under an address.
//!
//! The runtime lets a no-copy buffer made over exactly the bytes a live one was made over lie in
//! that one's import, which holds the pages that were there when that one was made. Metal wires
//! the pages at the address when it makes each buffer. So a host address that keeps its value
//! while its pages are replaced (an unmap and a map at the same address) would give a buffer
//! made after the change the old pages, unless the runtime is told:
//! `rt::Device::host_range_remapped`, which [`crate::gpu::device::host_range_remapped`] calls on
//! every device of the process.
//!
//! A host address takes new pages through one of two hands, and each reports what it changed,
//! with the call's own arguments and answer passed through unchanged. A call that maps reports
//! after it, and one that unmaps before it, at the unmap: a mapping that reports nothing (an
//! IOSurface's, `malloc`'s) could take the freed address between the unmap and a later report,
//! while a live buffer still lies in an import of the old pages there:
//! - **Core's Mach VM calls** ([`VulkanVm`], the platform's `HostVm`): a memory map's mappings
//!   (`remap_shared_overwrite`), a mapping task's unmaps (`allocate_fixed_overwrite`), every
//!   mapping's teardown, core's detach temporaries among them (`deallocate`, before the call:
//!   one Mach refuses reports pages that did not change, which only has a later buffer over
//!   them import them anew), and fresh ranges, which may lie where others were freed
//!   (`allocate_anywhere`). Each reports the pages the call changes, from the page of its first
//!   byte to the page past its last. `protect_max` changes no pages.
//! - **The host's task and memory blocks** in a device without a memory map ([`reporting`],
//!   `Platform::host_blocks`), where the host does the mapping work: `createTask` reports its
//!   task's whole range, as the host may hand back addresses it freed before; `mapMemory` the
//!   ranges' summed length from the task's base plus the offset, as a map over mapped pages
//!   replaces them; `unmapMemory` its range, before the host's block; `destroyTask` the task's
//!   whole range, before the host's block, the task taken out of the wrapper's table first: once
//!   the host freed it, a `createTask` on another thread may answer the same handle, whose entry is
//!   the new task's. One wrapper covers the root, local and EFI tasks and every caller, a no-copy
//!   buffer's deallocator among them.
//!
//! The platform's own Mach VM calls in [`crate::gpu::backing`] report nothing: an alias's
//! mapping (`gather`), the block of zeros, and an alias's release. Each range is a fresh one of
//! the platform's own, which backs a heap, a rebind, or the no-copy buffers over a surface at an
//! offset into its allocation or whose GPU writes never reach its memory
//! (`buffer_over_surface_at` in [`crate::gpu::device`]). The runtime lets no buffer lie in the
//! import of a heap's bytes or of the memory a rebound buffer takes, and a surface's alias is
//! released by the keep-alive of the last buffer over it, which the runtime drops after it freed
//! that buffer's import, so no buffer made later at the address could lie in one.
//!
//! **Not reported**: the pages of the host's own RAM a memory map names, which core reads at the
//! host's addresses with no mapping of its own. The host changes those without the library: no call
//! of the C API tells it, and Metal needs none. The only no-copy buffers at those addresses are
//! those under a surface whose base address lies in the guest's pages it was made over
//! ([`crate::os::base_in_pages`]: one over a single page; one over several pages, a range each, has
//! a mapping of its own there, measured), at no offset into its allocation and with GPU writes that
//! reach its memory (the others lie over an alias): one made after the host replaced the pages lies
//! in a live one's import, over the old pages, and what Metal's texture on such a surface reads
//! then is not measured. An isolated task's detached buffers and heaps lie in aliases of their own,
//! fresh at each attach ([`crate::gpu::backing`]), never there.
//!
//! **Core's mappings of the guest's pages** ([`GuestMappings`]): every remap through [`VulkanVm`]
//! is one of core's memory map's guest pages, which it maps afresh for a no-copy buffer over a
//! task's pages (the transfer buffers of paged textures and buffers, a legacy heap's, the XR10
//! plane views' and a task's shared buffers without detach backing). Heaps and buffers over
//! address ranges and surfaces over the guest's pages name those pages by the memory map's own
//! host addresses, where a mapping names them by its own. [`VulkanVm`] keeps which of its ranges
//! show which pages, from the remap to the unmap, so [`crate::gpu::over_pages`] can name a
//! buffer's bytes in such a mapping by the memory map's addresses ([`as_guests`]), and keep a
//! command's uses of them ([`shows_guest_pages`]).
//!
//! **Unmaps an import waits for** ([`UnmapHold`]). The runtime runs a no-copy buffer's
//! deallocator, given with a keep-alive, as the buffer is released, Metal's moment, and then
//! frees the device's import of its pages (`vkFreeMemory`), dropping the keep-alive after that.
//! Vulkan requires imported pages to stay mapped while the import lives
//! (`VK_EXT_external_memory_host`). A deallocator of core's unmaps core's mapping of the guest's
//! pages through [`VulkanVm`] (a memory map's `Mapping`, core's detach temporaries), so each
//! no-copy buffer this platform makes over memory it was given gives the runtime an
//! [`UnmapHold`] on the pages in its keep-alive ([`crate::gpu::backing::keep_alive`]). While the
//! hold lives, [`VulkanVm`]'s `deallocate` of pages it meets reports them, as every unmap does,
//! answers success and leaves them mapped; the hold's drop, after the free, unmaps them, whole,
//! once no other hold meets them. So the pages go after the import, as they did when the runtime
//! ran the deallocator after the free, and the deallocator, which is the guest's and the host's
//! (core's mapping's release, and with it what the host sees), stays at Metal's moment.
//!
//! - **Nothing waits.** `deallocate` never waits for a hold: it records the unmap under a leaf
//!   lock and returns. The hold's drop takes the same lock, takes the unmaps no hold meets any
//!   more, lets the lock go, and unmaps them (`munmap` or `mach_vm_deallocate`), calling nothing
//!   else. So neither side waits for another thread, and no thread can wait on a hold whose drop
//!   needs it, or a thread it blocks: not core's FIFO threads, which release buffers and so may
//!   run core's deallocators, nor a vCPU thread of QEMU's that holds its BQL inside a call of
//!   the C API. The calls of `deallocate` are core's: a mapping's last release on the thread that
//!   lets it go (a FIFO thread, an isolated task's thread, the host's thread inside a call, or the
//!   runtime's thread that runs a buffer's deallocator, the releasing thread or one of its
//!   handler threads as a command buffer completes), a detach temporary's, a failed map's, and
//!   a mapping task's whole range at its release. The holds go on the runtime's thread that frees
//!   the import: the releasing thread, or a handler thread once the last command buffer that held
//!   the import completed, or the creating thread where the runtime refused the call.
//! - **What it keeps.** The pages stay mapped, at their address, until the hold goes: the kernel
//!   gives no range over them meanwhile. Core deallocates only whole ranges it made and never
//!   reaches again, so nothing maps over a waiting unmap but a fixed overwrite, which takes those
//!   pages out of it (`Holds::replaced`).
//! - **What is not ours to unmap**: memory a no-copy buffer lies in whose unmap is the host's or
//!   the system's, which no hold defers: the host's own mapping in a device without a memory map
//!   (`unmapMemory`, the blocks setup; QEMU passes a memory map, so a guest under QEMU never
//!   reaches it, and only the suite's runner makes the blocks setup), an IOSurface's memory that
//!   a buffer over it at no offset imports as it is (core's buffer over a guest's IOSurface,
//!   `MapperRef`, whichever guest maps one), and on Linux the surface stand-in's own memory and
//!   views (`os::linux::surface`). There the deallocator (the host's unmap, the surface's
//!   release) may unmap the pages before the free, in the same release, or, where a command
//!   buffer submitted before a texture over them was released still holds the import, until it
//!   completes. Ground: MoltenVK's no-copy buffer keeps the
//!   pages it was made over, which Metal wires as it makes the buffer (`tests/no_copy_remap.rs`
//!   measures a remap under one), and NVIDIA's host-pointer import pins its
//!   pages as it imports them (Linux drivers pin imported pages, the runtime's `host_memory`), so
//!   on this platform's devices the GPU still reads the pages it imported, as Metal's reads those
//!   it wired. A driver that validates such pages again at each submission (ANV's and i915's
//!   userptr) would fail a submission that lists the import in that window.
//!
//! What a report costs, measured on the M3 Max (release build, 100,000 calls, three runs): with
//! no no-copy buffer live, about 11 ns per wrapped block call, the task lookup and the runtime's
//! early return, and nothing a 270 ns `mach_vm_deallocate` shows above its noise; with 1,000
//! live no-copy buffers the runtime imported, 40 to 62 ns per `deallocate` and 57 to 70 ns per
//! wrapped `mapMemory` or `unmapMemory`, beside the host's own call.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;

use reims_vgpu_core::host::{
    CreateTask, DestroyTask, HostBlocks, HostPtr, HostTask, MapMemory, UnmapMemory,
};
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use reims_vgpu_core::sync::Mutex;

use crate::os::OsVm;

mod holds;

use holds::Holds;

/// What a report tells: the pages under the given host bytes were replaced.
pub(crate) type Report = Arc<dyn Fn(HostPtr, u64) + Send + Sync>;

/// The pages a Mach VM call over `len` bytes at `addr` changes: from the page of `addr` to the
/// page past its last byte, as the kernel takes the range.
fn pages(addr: HostAddr, len: u64, page: u64) -> (HostPtr, u64) {
    let mask = page.wrapping_sub(1);
    let start = addr.addr().to_u64();
    let first = start & !mask;
    let end = start.saturating_add(len).saturating_add(mask) & !mask;
    (
        HostPtr::from_addr(first.to_usize()),
        end.saturating_sub(first),
    )
}

/// Core's mappings of the guest's pages (module docs): each run of a mapping's host bytes, keyed
/// by its first byte, with its end and the first byte of the pages it shows, as the memory map
/// names them. The runs share no byte. A leaf lock: nothing is called under it.
#[derive(Debug, Default)]
pub(crate) struct GuestMappings(BTreeMap<u64, (u64, u64)>);

/// The process's [`GuestMappings`], which [`VulkanVm`] keeps.
static GUEST_MAPPINGS: Mutex<GuestMappings> = Mutex::new(GuestMappings::new());

impl GuestMappings {
    /// None.
    const fn new() -> GuestMappings {
        GuestMappings(BTreeMap::new())
    }

    /// The run that holds the byte `at`: its first byte, end and source.
    fn holding(&self, at: u64) -> Option<(u64, u64, u64)> {
        (self.0.range(..=at).next_back())
            .map(|(&first, &(end, source))| (first, end, source))
            .filter(|&(_, end, _)| at < end)
    }

    /// `first..end` shows the pages from `source` from now on, which a mapping of a mapping
    /// names as that one does where it lies in one run of it.
    fn mapped(&mut self, first: u64, end: u64, source: u64) {
        let length = end - first;
        let source = match self.holding(source) {
            Some((at, run_end, from)) if source.saturating_add(length) <= run_end => {
                from + (source - at)
            }
            _ => source,
        };
        self.unmapped(first, end);
        if first < end {
            self.0.insert(first, (end, source));
        }
    }

    /// `first..end` shows none of the guest's pages from now on: what of a run lies in it goes.
    fn unmapped(&mut self, first: u64, end: u64) {
        let meeting: Vec<(u64, u64, u64)> = (self.0.range(..end).rev())
            .map(|(&at, &(run_end, source))| (at, run_end, source))
            .take_while(|&(_, run_end, _)| run_end > first)
            .collect();
        for (at, run_end, source) in meeting {
            self.0.remove(&at);
            if at < first {
                self.0.insert(at, (first, source));
            }
            if end < run_end {
                self.0.insert(end, (run_end, source + (end - at)));
            }
        }
    }

    /// `memory`'s ranges named by the memory map's addresses where they lie in a mapping, each
    /// cut where a run starts or ends; any other byte as it is.
    fn named(&self, memory: Vec<Range<u64>>) -> Vec<Range<u64>> {
        if self.0.is_empty() {
            return memory;
        }
        let mut named = Vec::with_capacity(memory.len());
        for range in memory {
            let mut at = range.start;
            while at < range.end {
                match self.holding(at) {
                    Some((first, end, source)) => {
                        let to = end.min(range.end);
                        named.push(source + (at - first)..source + (to - first));
                        at = to;
                    }
                    None => {
                        let next = self.0.range(at..).next().map(|(&first, _)| first);
                        let to = next.unwrap_or(u64::MAX).min(range.end);
                        named.push(at..to);
                        at = to;
                    }
                }
            }
        }
        named
    }

    /// Whether a byte of `memory` lies in a mapping.
    fn shows(&self, memory: &Range<u64>) -> bool {
        self.holding(memory.start).is_some()
            || (self.0.range(memory.start..).next()).is_some_and(|(&first, _)| first < memory.end)
    }
}

/// `memory`'s ranges named as the memory map names the guest's pages, where they lie in one of
/// core's mappings of them, so that they meet the ranges of a heap or buffer over the same pages
/// (module docs); any other byte as it is.
pub(crate) fn as_guests(memory: Vec<Range<u64>>) -> Vec<Range<u64>> {
    GUEST_MAPPINGS.lock().named(memory)
}

/// Whether a byte of the `length` bytes at `first` lies in one of core's mappings of the
/// guest's pages.
pub(crate) fn shows_guest_pages(first: HostPtr, length: u64) -> bool {
    let first = first.addr().to_u64();
    let memory = first..first.saturating_add(length);
    !memory.is_empty() && GUEST_MAPPINGS.lock().shows(&memory)
}

/// The process's holds on pages runtime imports cover, and the unmaps that wait for them
/// ([`UnmapHold`]). A leaf lock: nothing is called under it.
static HOLDS: Mutex<Holds> = Mutex::new(Holds::new());

/// A hold on the host pages a runtime import covers, which the runtime is given in a no-copy
/// buffer's keep-alive: while it lives, [`VulkanVm`]'s `deallocate` of pages it meets leaves them
/// mapped, and its drop, which the runtime makes once it freed the import, unmaps them (module
/// docs, "Unmaps an import waits for").
#[derive(Debug)]
pub(crate) struct UnmapHold {
    /// The first byte of its pages.
    first: u64,
    /// Its number among the holds.
    id: u64,
}

impl UnmapHold {
    /// A hold on the pages the `length` bytes at `bytes` lie in (one page for a length of 0).
    pub(crate) fn new(bytes: NonNull<u8>, length: u64) -> UnmapHold {
        let at = HostPtr::from_ptr(bytes.as_ptr());
        let (at, length) = pages(at, length.max(1), OsVm::new().page_size());
        let first = at.addr().to_u64();
        let id = HOLDS.lock().hold(first..first.saturating_add(length));
        UnmapHold { first, id }
    }
}

impl Drop for UnmapHold {
    /// Unmaps what waited for this hold and no other: under no lock, calling nothing else.
    fn drop(&mut self) {
        let free = HOLDS.lock().release(self.first, self.id);
        if free.is_empty() {
            return;
        }
        let vm = OsVm::new();
        for pages in free {
            let at = HostPtr::from_addr(pages.start.to_usize());
            let _ = vm.deallocate(at, pages.end - pages.start);
        }
    }
}

/// Whether an unmap of a page of the `length` bytes at `at` waits for a hold, for tests.
#[cfg(test)]
pub(crate) fn unmap_waits(at: HostPtr, length: u64) -> bool {
    let (at, length) = pages(at, length, OsVm::new().page_size());
    let first = at.addr().to_u64();
    HOLDS.lock().waits(&(first..first.saturating_add(length)))
}

/// The platform's `HostVm`: the OS's ([`OsVm`]: Mach VM on macOS, its calls over `mmap` on
/// Linux), each call that changes pages then reported (module docs), and an unmap of pages a
/// runtime import holds made once its hold goes ([`UnmapHold`]).
#[derive(Debug)]
pub struct VulkanVm {
    /// The calls themselves.
    vm: OsVm,
    /// Where the changed pages go: every device's runtime, but in this module's tests.
    report: fn(HostPtr, u64),
}

impl VulkanVm {
    /// The process's `HostVm`, reporting to every device made in the process.
    pub fn new() -> VulkanVm {
        VulkanVm {
            vm: OsVm::new(),
            report: crate::gpu::device::host_range_remapped,
        }
    }

    /// Reports the pages a call over `len` bytes at `addr` changed, which show none of the
    /// guest's pages from now on unless a remap says so after ([`GuestMappings`]): their range.
    fn changed(&self, addr: HostAddr, len: u64) -> Range<u64> {
        let (at, length) = pages(addr, len, self.vm.page_size());
        let first = at.addr().to_u64();
        GUEST_MAPPINGS
            .lock()
            .unmapped(first, first.saturating_add(length));
        (self.report)(at, length);
        first..first.saturating_add(length)
    }
}

impl Default for VulkanVm {
    fn default() -> VulkanVm {
        VulkanVm::new()
    }
}

impl HostVm for VulkanVm {
    fn page_size(&self) -> u64 {
        self.vm.page_size()
    }

    fn page_shift(&self) -> u32 {
        self.vm.page_shift()
    }

    fn aliases_guest_in_place(&self) -> bool {
        self.vm.aliases_guest_in_place()
    }

    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let at = self.vm.allocate_anywhere(len)?;
        self.changed(at, len);
        Ok(at)
    }

    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn> {
        self.vm.allocate_fixed_overwrite(addr, len)?;
        let pages = self.changed(addr, len);
        HOLDS.lock().replaced(&pages);
        Ok(())
    }

    fn remap_shared_overwrite(
        &self,
        dst: HostAddr,
        src: HostAddr,
        len: u64,
    ) -> Result<HostAddr, KernReturn> {
        let at = self.vm.remap_shared_overwrite(dst, src, len)?;
        // The kernel maps the source's pages, from the page of its first byte: as many bytes
        // past `dst` as the source's offset in its page, and `len` more.
        let mask = self.vm.page_size().wrapping_sub(1);
        let offset = src.addr().to_u64() & mask;
        let replaced = self.changed(dst, len.saturating_add(offset));
        HOLDS.lock().replaced(&replaced);
        // Those pages are one of core's mappings of the guest's (module docs).
        let (first, length) = pages(dst, len.saturating_add(offset), self.vm.page_size());
        let first = first.addr().to_u64();
        let source = src.addr().to_u64() & !mask;
        GUEST_MAPPINGS
            .lock()
            .mapped(first, first.saturating_add(length), source);
        Ok(at)
    }

    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn> {
        self.vm.protect_max(addr, len, prot)
    }

    fn deallocate(&self, addr: HostAddr, len: u64) -> KernReturn {
        // At the unmap (module docs).
        let pages = self.changed(addr, len);
        // Pages a runtime import holds are unmapped once its hold goes (module docs).
        if HOLDS.lock().unmap(pages) {
            return KernReturn::SUCCESS;
        }
        self.vm.deallocate(addr, len)
    }
}

/// The base and size of each task the host's `createTask` made, until its `destroyTask`. A leaf
/// lock: nothing is called under it.
#[derive(Default)]
struct Tasks(Mutex<HashMap<HostTask, (HostPtr, u64)>>);

impl Tasks {
    /// The base of `task`, or `None` for a task `createTask` did not answer.
    fn base(&self, task: HostTask) -> Option<HostPtr> {
        self.0.lock().get(&task).map(|&(base, _)| base)
    }
}

/// The host's task and memory blocks, each that changes pages then reporting to `report`
/// (module docs). Each wrapper calls the host's block once per call, with the same arguments,
/// answers its answer, and holds no other block, so the blocks release as the host's own do; a
/// nil block stays nil, and the other blocks are the host's own.
pub(crate) fn reporting(blocks: HostBlocks, report: Report) -> HostBlocks {
    let tasks = Arc::new(Tasks::default());
    let HostBlocks {
        raise_interrupt,
        create_task,
        destroy_task,
        map_memory,
        unmap_memory,
        read_memory,
        add_trace_range,
        remove_trace_range,
    } = blocks;
    let create_task = create_task.map(|host| -> CreateTask {
        let (tasks, report) = (Arc::clone(&tasks), Arc::clone(&report));
        Arc::new(move |size, base: &mut HostPtr| {
            let task = host(size, base);
            if !task.is_null() {
                tasks.0.lock().insert(task, (*base, size));
                report(*base, size);
            }
            task
        })
    });
    let destroy_task = destroy_task.map(|host| -> DestroyTask {
        let (tasks, report) = (Arc::clone(&tasks), Arc::clone(&report));
        Arc::new(move |task| {
            // Out of the table before the host frees the task, and reported at the unmap
            // (module docs).
            let gone = tasks.0.lock().remove(&task);
            if let Some((base, size)) = gone {
                report(base, size);
            }
            host(task);
        })
    });
    let map_memory = map_memory.map(|host| -> MapMemory {
        let (tasks, report) = (Arc::clone(&tasks), Arc::clone(&report));
        Arc::new(move |task, count, offset, read_only, ranges| {
            let mapped = host(task, count, offset, read_only, ranges);
            if let Some(base) = tasks.base(task) {
                let length = ranges
                    .iter()
                    .take(count.to_usize())
                    .fold(0u64, |sum, r| sum.saturating_add(r.physical_length));
                report(base.wrapping_add(offset), length);
            }
            mapped
        })
    });
    let unmap_memory = unmap_memory.map(|host| -> UnmapMemory {
        let (tasks, report) = (Arc::clone(&tasks), Arc::clone(&report));
        Arc::new(move |task, offset, length| {
            if let Some(base) = tasks.base(task) {
                report(base.wrapping_add(offset), length);
            }
            host(task, offset, length)
        })
    });
    HostBlocks {
        raise_interrupt,
        create_task,
        destroy_task,
        map_memory,
        unmap_memory,
        read_memory,
        add_trace_range,
        remove_trace_range,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use reims_vgpu_core::host::{HostTraceRange, PhysicalMemoryRange};

    use super::*;

    thread_local! {
        /// What [`seen`] was told on this thread.
        static SEEN: RefCell<Vec<(HostPtr, u64, bool)>> = const { RefCell::new(Vec::new()) };
    }

    /// A report kept on this thread, with whether its first page was mapped as it came.
    fn seen(at: HostPtr, length: u64) {
        SEEN.with(|s| s.borrow_mut().push((at, length, crate::os::mapped(at))));
    }

    /// What this thread was told since the last take.
    fn take() -> Vec<(HostPtr, u64, bool)> {
        SEEN.with(|s| std::mem::take(&mut *s.borrow_mut()))
    }

    /// Each Mach VM call that changes pages reports them, from the page of its first byte to
    /// the page past its last: one that maps after it, `deallocate` at the unmap, its pages still
    /// mapped; `protect_max` and a refused remap report nothing
    #[test]
    fn the_vm_reports_the_pages_its_calls_change() {
        let vm = VulkanVm {
            vm: OsVm::new(),
            report: seen,
        };
        let page = vm.page_size();
        let at = vm.allocate_anywhere(page + 0x100).unwrap();
        assert_eq!(take(), [(at, 2 * page, true)]);
        vm.allocate_fixed_overwrite(at.wrapping_add(page + 0x80), 0x100)
            .unwrap();
        assert_eq!(take(), [(at.wrapping_add(page), page, true)]);
        let src = crate::os::allocate_aliasable(2 * page).unwrap();
        vm.remap_shared_overwrite(at, src.wrapping_add(0x200), page)
            .unwrap();
        assert_eq!(take(), [(at, 2 * page, true)]);
        vm.protect_max(at, page, VmProt::READ | VmProt::WRITE)
            .unwrap();
        assert_eq!(take(), []);
        assert!(vm.deallocate(at, 2 * page).is_success());
        assert_eq!(take(), [(at, 2 * page, true)]);
        assert!(!crate::os::mapped(at));
        // The source is no longer mapped: the remap fails, and changes nothing.
        assert!(vm.remap_shared_overwrite(src, at, page).is_err());
        assert_eq!(take(), []);
        assert!(OsVm::new().deallocate(src, 2 * page).is_success());
    }

    /// A mapping's runs name its bytes by the pages they show, cut where a run starts or ends,
    /// and any other byte as it is; an unmap of part of a run keeps the rest, and a map over a
    /// run replaces it; a mapping of a mapping names the first's pages
    #[test]
    #[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
    fn guest_mappings_name_the_pages_they_show() {
        let mut m = GuestMappings::new();
        assert_eq!(m.named(vec![0x1000..0x2000]), [0x1000..0x2000]);
        m.mapped(0x10_0000, 0x10_8000, 0x90_0000);
        assert_eq!(m.named(vec![0x10_1000..0x10_1010]), [0x90_1000..0x90_1010]);
        assert_eq!(
            m.named(vec![0xf_c000..0x10_4000, 0x10_6000..0x10_a000]),
            [
                0xf_c000..0x10_0000,
                0x90_0000..0x90_4000,
                0x90_6000..0x90_8000,
                0x10_8000..0x10_a000
            ]
        );
        assert!(m.shows(&(0xf_c000..0x10_0001)));
        assert!(!m.shows(&(0xf_c000..0x10_0000)));
        assert!(!m.shows(&(0x10_8000..0x10_9000)));
        m.unmapped(0x10_2000, 0x10_4000);
        assert_eq!(
            m.named(vec![0x10_1000..0x10_5000]),
            [
                0x90_1000..0x90_2000,
                0x10_2000..0x10_4000,
                0x90_4000..0x90_5000
            ]
        );
        m.mapped(0x10_4000, 0x10_6000, 0x70_0000);
        assert_eq!(
            m.named(vec![0x10_4000..0x10_7000]),
            [0x70_0000..0x70_2000, 0x90_6000..0x90_7000]
        );
        m.mapped(0x20_0000, 0x20_1000, 0x10_1000);
        assert_eq!(m.named(vec![0x20_0000..0x20_0010]), [0x90_1000..0x90_1010]);
        m.unmapped(0, u64::MAX);
        assert_eq!(m.named(vec![0x10_1000..0x10_1010]), [0x10_1000..0x10_1010]);
        assert!(m.0.is_empty());
    }

    /// The platform's remap names its range by the source's pages from the remap, and a
    /// deallocate of the range ends it: a buffer over it then lies over memory of its own
    #[test]
    #[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
    fn a_remap_names_its_range_by_the_sources_pages() {
        let vm = VulkanVm {
            vm: OsVm::new(),
            report: seen,
        };
        let page = vm.page_size();
        let source = crate::os::allocate_aliasable(2 * page).unwrap();
        let first = source.addr().to_u64();
        let at = vm.allocate_anywhere(page).unwrap();
        let mapped = at.addr().to_u64();
        assert_eq!(as_guests(vec![mapped..mapped + 16]), [mapped..mapped + 16]);
        assert!(!shows_guest_pages(at, 16));
        vm.remap_shared_overwrite(at, source.wrapping_add(page), page)
            .unwrap();
        let named = first + page;
        assert_eq!(as_guests(vec![mapped..mapped + 16]), [named..named + 16]);
        assert!(shows_guest_pages(at.wrapping_add(8), 1));
        assert!(vm.deallocate(at, page).is_success());
        assert_eq!(as_guests(vec![mapped..mapped + 16]), [mapped..mapped + 16]);
        assert!(!shows_guest_pages(at, 16));
        assert!(OsVm::new().deallocate(source, 2 * page).is_success());
        take();
    }

    /// What the host's fake blocks were called with, in order.
    type Calls = Arc<Mutex<Vec<String>>>;

    /// Fake host blocks: `createTask` answers task 7 at `0x1_0000_0000` for a size of 0x100000
    /// and NULL otherwise, `mapMemory` true and `unmapMemory` false; each records its call.
    fn host(calls: &Calls) -> HostBlocks {
        let c = Arc::clone(calls);
        let create: CreateTask = Arc::new(move |size, base| {
            c.lock().push(format!("create {size:#x}"));
            if size != 0x10_0000 {
                return HostTask::NULL;
            }
            *base = HostPtr::from_addr(0x1_0000_0000);
            HostTask::from_addr(7)
        });
        let c = Arc::clone(calls);
        let destroy: DestroyTask = Arc::new(move |task| {
            c.lock().push(format!("destroy {}", task.addr()));
        });
        let c = Arc::clone(calls);
        let map: MapMemory = Arc::new(move |task, count, offset, read_only, ranges| {
            c.lock().push(format!(
                "map {} {count} {offset:#x} {read_only} {}",
                task.addr(),
                ranges.len()
            ));
            true
        });
        let c = Arc::clone(calls);
        let unmap: UnmapMemory = Arc::new(move |task, offset, length| {
            c.lock()
                .push(format!("unmap {} {offset:#x} {length:#x}", task.addr()));
            false
        });
        HostBlocks {
            create_task: Some(create),
            destroy_task: Some(destroy),
            map_memory: Some(map),
            unmap_memory: Some(unmap),
            ..HostBlocks::default()
        }
    }

    /// The wrapped blocks call the host's with the same arguments and answer its answers, and
    /// report what each changed: a task's whole range after its creation and before its
    /// destruction, a map's summed length and an unmap's range from the task's base plus the
    /// offset, after them; nothing for a task the host did not make, or one destroyed
    #[test]
    fn the_blocks_report_what_the_host_changed() {
        let calls = Calls::default();
        let c = Arc::clone(&calls);
        let b = reporting(
            host(&calls),
            Arc::new(move |at, length| {
                c.lock()
                    .push(format!("report {:#x} {length:#x}", at.addr()));
            }),
        );
        let (create, destroy) = (b.create_task.unwrap(), b.destroy_task.unwrap());
        let (map, unmap) = (b.map_memory.unwrap(), b.unmap_memory.unwrap());
        let mut at = HostPtr::NULL;
        assert!(create(0x2000, &mut at).is_null());
        let task = create(0x10_0000, &mut at);
        assert_eq!(
            (task, at),
            (HostTask::from_addr(7), HostPtr::from_addr(0x1_0000_0000))
        );
        let range = |length| PhysicalMemoryRange {
            physical_address: 0x8000_0000,
            physical_length: length,
        };
        let ranges = [range(0x4000), range(0x8000), range(0x10000)];
        assert!(map(task, 2, 0x8000, true, &ranges));
        assert!(!unmap(task, 0xc000, 0x4000));
        let stranger = HostTask::from_addr(9);
        assert!(map(stranger, 1, 0, false, &ranges));
        assert!(!unmap(stranger, 0, 0x4000));
        destroy(task);
        assert!(!unmap(task, 0, 0x4000));
        assert_eq!(
            *calls.lock(),
            [
                "create 0x2000",
                "create 0x100000",
                "report 0x100000000 0x100000",
                "map 7 2 0x8000 true 3",
                "report 0x100008000 0xc000",
                "report 0x10000c000 0x4000",
                "unmap 7 0xc000 0x4000",
                "map 9 1 0x0 false 3",
                "unmap 9 0x0 0x4000",
                "report 0x100000000 0x100000",
                "destroy 7",
                "unmap 7 0x0 0x4000",
            ]
        );
    }

    /// A handle the host answers again once it freed the task, to a `createTask` on another
    /// thread while the destroy is still in the host's block, is the new task's: the destroy
    /// reports the old task's range alone, and the new task's maps report from its own base
    #[test]
    fn a_handle_answered_again_during_its_destroy_is_the_new_tasks() {
        let calls = Calls::default();
        let wrapped: Arc<std::sync::OnceLock<CreateTask>> = Arc::default();
        let bases = Arc::new(Mutex::new(vec![0x2_0000_0000_usize, 0x1_0000_0000]));
        let c = Arc::clone(&calls);
        let create: CreateTask = Arc::new(move |size, base| {
            *base = HostPtr::from_addr(bases.lock().pop().unwrap());
            c.lock().push(format!("create {size:#x}"));
            HostTask::from_addr(7)
        });
        let (c, w) = (Arc::clone(&calls), Arc::clone(&wrapped));
        let destroy: DestroyTask = Arc::new(move |task| {
            c.lock().push(format!("destroy {}", task.addr()));
            // Another thread's createTask, in the host's destroy once it freed the task.
            let mut base = HostPtr::NULL;
            assert_eq!(w.get().unwrap()(0x4000, &mut base), task);
        });
        let map: MapMemory = Arc::new(|_, _, _, _, _| true);
        let c = Arc::clone(&calls);
        let b = reporting(
            HostBlocks {
                create_task: Some(create),
                destroy_task: Some(destroy),
                map_memory: Some(map),
                ..HostBlocks::default()
            },
            Arc::new(move |at, length| {
                c.lock()
                    .push(format!("report {:#x} {length:#x}", at.addr()));
            }),
        );
        let (create, destroy) = (b.create_task.unwrap(), b.destroy_task.unwrap());
        assert!(wrapped.set(Arc::clone(&create)).is_ok());
        let mut base = HostPtr::NULL;
        let task = create(0x8000, &mut base);
        destroy(task);
        let range = PhysicalMemoryRange {
            physical_address: 0x8000_0000,
            physical_length: 0x4000,
        };
        assert!(b.map_memory.unwrap()(task, 1, 0x100, false, &[range]));
        assert_eq!(
            *calls.lock(),
            [
                "create 0x8000",
                "report 0x100000000 0x8000",
                "report 0x100000000 0x8000",
                "destroy 7",
                "create 0x4000",
                "report 0x200000000 0x4000",
                "report 0x200000100 0x4000",
            ]
        );
    }

    /// A `deallocate` of pages a hold meets reports them at the call and answers success,
    /// leaving them mapped until the hold goes, which unmaps them; one of pages no hold meets
    /// unmaps at once. Unmapped pages may be mapped again at once by another test's thread, so
    /// an unmap shows as pages unmapped or no longer holding the bytes written there
    #[test]
    fn an_unmap_of_held_pages_waits_for_the_hold() {
        let vm = VulkanVm {
            vm: OsVm::new(),
            report: seen,
        };
        let page = vm.page_size();
        let at = vm.allocate_anywhere(2 * page).unwrap();
        let other = vm.allocate_anywhere(page).unwrap();
        for (p, length) in [(at, 2 * page), (other, page)] {
            // SAFETY: pages this test just mapped, readable and writable.
            unsafe { std::ptr::write_bytes(p.as_ptr::<u8>(), 0xa5, length.to_usize()) };
        }
        let _ = take();
        let gone = |p: HostPtr| {
            // SAFETY: a read of a page that is mapped, the first test's.
            !crate::os::mapped(p) || unsafe { p.as_ptr::<u8>().read_volatile() } != 0xa5
        };
        let first = NonNull::new(at.wrapping_add(0x10).as_ptr::<u8>()).unwrap();
        let hold = UnmapHold::new(first, page);
        assert!(vm.deallocate(other, page).is_success());
        assert!(!unmap_waits(other, page), "no hold meets it");
        assert!(gone(other), "unmapped at once");
        assert!(vm.deallocate(at, 2 * page).is_success());
        assert_eq!(
            take(),
            [(other, page, true), (at, 2 * page, true)],
            "each reported at the call"
        );
        assert!(unmap_waits(at, 2 * page));
        assert!(
            !gone(at) && !gone(at.wrapping_add(page)),
            "still mapped, whole"
        );
        drop(hold);
        assert!(!unmap_waits(at, 2 * page));
        assert!(
            gone(at) && gone(at.wrapping_add(page)),
            "unmapped as the hold went"
        );
    }

    /// A nil block stays nil, the blocks that change no pages are the host's own, and each
    /// wrapper holds its own block alone: the host's blocks' last references stay where they
    /// were
    #[test]
    fn the_blocks_hold_only_their_own() {
        let calls = Calls::default();
        let mut given = host(&calls);
        given.unmap_memory = None;
        given.read_memory = Some(Arc::new(|_, _, _| true));
        given.raise_interrupt = Some(Arc::new(|_| ()));
        given.add_trace_range = Some(Arc::new(|_, _| HostTraceRange::NULL));
        given.remove_trace_range = Some(Arc::new(|_| ()));
        let b = reporting(given.clone(), Arc::new(|_, _| ()));
        assert!(b.unmap_memory.is_none());
        fn same<T: ?Sized>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
            matches!((a, b), (Some(a), Some(b)) if Arc::ptr_eq(a, b))
        }
        assert!(same(&b.read_memory, &given.read_memory));
        assert!(same(&b.raise_interrupt, &given.raise_interrupt));
        assert!(same(&b.add_trace_range, &given.add_trace_range));
        assert!(same(&b.remove_trace_range, &given.remove_trace_range));
        let count = |given: &HostBlocks| {
            [
                given.create_task.as_ref().map(Arc::strong_count),
                given.destroy_task.as_ref().map(Arc::strong_count),
                given.map_memory.as_ref().map(Arc::strong_count),
            ]
        };
        assert_eq!(count(&given), [Some(2); 3]);
        drop(b);
        assert_eq!(count(&given), [Some(1); 3]);
    }
}
