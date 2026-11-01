//! Shared section views over Windows placeholders. A remap aliases pages; it never copies them.

use super::api::*;
use crate::os::backing::{self as files, Backing, Piece, Registration};
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal};
use reims_vgpu_core::sync::Mutex;
use std::collections::BTreeMap;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::sync::Arc;

const INVALID_ADDRESS: KernReturn = KernReturn(1);
const PROTECTION_FAILURE: KernReturn = KernReturn(2);
const NO_SPACE: KernReturn = KernReturn(3);
const INVALID_ARGUMENT: KernReturn = KernReturn(4);

#[derive(Clone, Debug)]
struct View {
    length: u64,
    backing: Backing,
    prot: i32,
}
static VIEWS: Mutex<BTreeMap<u64, View>> = Mutex::new(BTreeMap::new());

pub(super) fn page_size() -> u64 {
    static PAGE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PAGE.get_or_init(|| {
        let mut info = SystemInfo::default();
        // SAFETY: a live SYSTEM_INFO output.
        unsafe { GetSystemInfo(&mut info) };
        u64::from(info.page_size)
    })
}

fn pages(address: HostAddr, length: u64) -> Option<(u64, u64)> {
    let page = page_size();
    let first = address.addr().to_u64() & !(page - 1);
    let end = address
        .addr()
        .to_u64()
        .checked_add(length)?
        .checked_next_multiple_of(page)?;
    Some((
        first,
        if length == 0 {
            0
        } else {
            end.checked_sub(first)?
        },
    ))
}

fn protection(prot: i32) -> u32 {
    if prot & 2 != 0 {
        PAGE_READWRITE
    } else if prot & 1 != 0 {
        PAGE_READONLY
    } else {
        PAGE_NOACCESS
    }
}
fn ptr(at: u64) -> *mut std::ffi::c_void {
    std::ptr::with_exposed_provenance_mut(at.to_usize())
}
fn error(call: &str, kr: KernReturn) -> KernReturn {
    log::error!(target: "reims_vgpu_vulkan::os", "{call}: {}", std::io::Error::last_os_error());
    kr
}
fn cannot_restore(call: &'static str) -> ! {
    error(call, INVALID_ADDRESS);
    fatal(DeathId::Platform {
        kind: DeathKind::Abort,
        call,
    })
}

fn section(length: u64) -> Result<Arc<OwnedHandle>, KernReturn> {
    let high = u32::try_from(length >> 32).map_err(|_| NO_SPACE)?;
    let low = u32::try_from(length & u64::from(u32::MAX)).map_err(|_| NO_SPACE)?;
    // SAFETY: an unnamed, pagefile-backed section; no user pointer is dereferenced.
    let handle = unsafe {
        CreateFileMappingW(
            std::ptr::with_exposed_provenance_mut(usize::MAX),
            std::ptr::null(),
            PAGE_READWRITE,
            high,
            low,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        return Err(error("CreateFileMappingW", NO_SPACE));
    }
    // SAFETY: this call owns the new section handle.
    Ok(Arc::new(unsafe { OwnedHandle::from_raw_handle(handle) }))
}

fn reserve(length: u64) -> Result<u64, KernReturn> {
    // SAFETY: reserves fresh address space, without committing physical memory.
    let base = unsafe {
        VirtualAlloc2(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            length.to_usize(),
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            std::ptr::null(),
            0,
        )
    };
    if base.is_null() {
        Err(error("VirtualAlloc2", NO_SPACE))
    } else {
        Ok(base.addr().to_u64())
    }
}

fn map(first: u64, length: u64, backing: &Backing, prot: i32) -> Result<(), KernReturn> {
    // SAFETY: the exact placeholder is replaced with a view of a retained section. This flag
    // permits page-aligned offsets and addresses, including the guest's 4 KiB pages.
    let mapped = unsafe {
        MapViewOfFile3(
            backing.file.as_raw_handle(),
            GetCurrentProcess(),
            ptr(first),
            backing.offset,
            length.to_usize(),
            MEM_REPLACE_PLACEHOLDER,
            protection(prot),
            std::ptr::null(),
            0,
        )
    };
    if mapped != ptr(first) {
        Err(error("MapViewOfFile3", NO_SPACE))
    } else {
        Ok(())
    }
}
fn unmap(first: u64) {
    // SAFETY: a view owned by this table, restored to its placeholder.
    if unsafe { UnmapViewOfFile2(GetCurrentProcess(), ptr(first), MEM_PRESERVE_PLACEHOLDER) } == 0 {
        cannot_restore("UnmapViewOfFile2");
    }
}
fn split_placeholder(first: u64, length: u64) -> Result<(), KernReturn> {
    // SAFETY: splits an existing placeholder; it does not free the address range.
    if unsafe {
        VirtualFree(
            ptr(first),
            length.to_usize(),
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        )
    } == 0
    {
        Err(error("VirtualFree split placeholder", INVALID_ADDRESS))
    } else {
        Ok(())
    }
}
fn coalesce(first: u64, length: u64) {
    // SAFETY: adjacent placeholders created while replacing this owned range.
    if unsafe {
        VirtualFree(
            ptr(first),
            length.to_usize(),
            MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
        )
    } == 0
    {
        cannot_restore("VirtualFree coalesce placeholders");
    }
}

/// Installs pieces in one placeholder. On failure restores one empty placeholder.
fn install(first: u64, length: u64, pieces: &[Piece]) -> Result<(), KernReturn> {
    let mut mapped = 0usize;
    let mut splits = 0usize;
    let result = (|| {
        for (i, piece) in pieces.iter().enumerate() {
            if i + 1 < pieces.len() {
                split_placeholder(first + piece.at, piece.length)?;
                splits += 1;
            }
            map(first + piece.at, piece.length, &piece.backing, piece.prot)?;
            mapped += 1;
        }
        Ok(())
    })();
    if result.is_err() {
        for piece in pieces.iter().take(mapped) {
            unmap(first + piece.at);
        }
        if splits != 0 {
            coalesce(first, length);
        }
    }
    result
}

fn pieces(first: u64, views: &[(u64, View)]) -> Vec<Piece> {
    views
        .iter()
        .map(|(at, view)| Piece {
            at: at - first,
            length: view.length,
            backing: view.backing.clone(),
            prot: view.prot,
        })
        .collect()
}

/// Splits a live view without changing any of its bytes or protections.
fn split(views: &mut BTreeMap<u64, View>, at: u64) -> Result<(), KernReturn> {
    let Some((&first, view)) = views.range(..=at).next_back() else {
        return Ok(());
    };
    if at == first || at >= first + view.length {
        return Ok(());
    }
    let old = view.clone();
    let head = View {
        length: at - first,
        ..old.clone()
    };
    let tail = View {
        length: first + old.length - at,
        backing: Backing {
            file: Arc::clone(&old.backing.file),
            offset: old.backing.offset + at - first,
        },
        prot: old.prot,
    };
    unmap(first);
    let cut = pieces(first, &[(first, head.clone()), (at, tail.clone())]);
    if let Err(kr) = install(first, old.length, &cut) {
        if install(first, old.length, &pieces(first, &[(first, old)])).is_err() {
            cannot_restore("restore split section view");
        }
        return Err(kr);
    }
    views.insert(first, head);
    views.insert(at, tail);
    Ok(())
}

/// Diagnostics of remaps (`remap_shared_overwrite`): their phases' times, summed over a second and
/// written as a warning where they took 100 ms or more of it, so a FIFO thread that maps a guest
/// texture page by page shows where its time went.
mod remap_stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::{Duration, Instant};

    pub(super) static CALLS: AtomicU64 = AtomicU64::new(0);
    pub(super) static PIECES: AtomicU64 = AtomicU64::new(0);
    pub(super) static BYTES: AtomicU64 = AtomicU64::new(0);
    pub(super) static TOTAL: AtomicU64 = AtomicU64::new(0);
    pub(super) static LOCK: AtomicU64 = AtomicU64::new(0);
    pub(super) static SPLIT: AtomicU64 = AtomicU64::new(0);
    pub(super) static UNMAP: AtomicU64 = AtomicU64::new(0);
    pub(super) static INSTALL: AtomicU64 = AtomicU64::new(0);
    pub(super) static SLOWEST: AtomicU64 = AtomicU64::new(0);
    static WINDOW: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);

    pub(super) fn add(counter: &AtomicU64, since: Instant) {
        counter.fetch_add(since.elapsed().as_nanos() as u64, Relaxed);
    }

    /// Ends one remap that began at `started`, writing the second's sums once it is over.
    pub(super) fn finish(started: Instant, pieces: usize, bytes: u64, views: usize) {
        let took = started.elapsed().as_nanos() as u64;
        CALLS.fetch_add(1, Relaxed);
        PIECES.fetch_add(pieces as u64, Relaxed);
        BYTES.fetch_add(bytes, Relaxed);
        TOTAL.fetch_add(took, Relaxed);
        SLOWEST.fetch_max(took, Relaxed);
        let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
        let begun = *window.get_or_insert_with(Instant::now);
        if begun.elapsed() < Duration::from_secs(1) {
            return;
        }
        *window = Some(Instant::now());
        let ms = |c: &AtomicU64| c.swap(0, Relaxed) as f64 / 1e6;
        let (calls, pieces, bytes) = (CALLS.swap(0, Relaxed), PIECES.swap(0, Relaxed), BYTES.swap(0, Relaxed));
        let (total, lock, split, unmap, install, slowest) =
            (ms(&TOTAL), ms(&LOCK), ms(&SPLIT), ms(&UNMAP), ms(&INSTALL), ms(&SLOWEST));
        if total >= 100.0 {
            log::warn!(
                "remaps over {:.2} s: {calls} calls, {pieces} pieces, {} KiB, {total:.1} ms (lock                  wait {lock:.1}, splits {split:.1}, unmaps {unmap:.1}, installs {install:.1};                  slowest call {slowest:.1}); {views} views",
                begun.elapsed().as_secs_f64(),
                bytes >> 10
            );
        }
    }
}

fn replace(first: u64, length: u64, new: &[Piece]) -> Result<(), KernReturn> {
    let end = first.checked_add(length).ok_or(INVALID_ARGUMENT)?;
    let phase = std::time::Instant::now();
    let mut views = VIEWS.lock();
    remap_stats::add(&remap_stats::LOCK, phase);
    let mut at = first;
    while at < end {
        let Some((&base, view)) = views.range(..=at).next_back() else {
            return Err(INVALID_ADDRESS);
        };
        if at >= base + view.length {
            return Err(INVALID_ADDRESS);
        }
        at = (base + view.length).min(end);
    }
    let phase = std::time::Instant::now();
    split(&mut views, first)?;
    split(&mut views, end)?;
    remap_stats::add(&remap_stats::SPLIT, phase);
    let old: Vec<_> = views
        .range(first..end)
        .map(|(&at, v)| (at, v.clone()))
        .collect();
    let phase = std::time::Instant::now();
    for (at, _) in &old {
        unmap(*at);
    }
    if old.len() > 1 {
        coalesce(first, length);
    }
    remap_stats::add(&remap_stats::UNMAP, phase);
    let phase = std::time::Instant::now();
    let installed = install(first, length, new);
    remap_stats::add(&remap_stats::INSTALL, phase);
    if let Err(kr) = installed {
        if install(first, length, &pieces(first, &old)).is_err() {
            cannot_restore("restore overwritten section views");
        }
        return Err(kr);
    }
    for (at, _) in old {
        views.remove(&at);
    }
    for piece in new {
        views.insert(
            first + piece.at,
            View {
                length: piece.length,
                backing: piece.backing.clone(),
                prot: piece.prot,
            },
        );
    }
    Ok(())
}

/// Maps the retained sections and records their views, including writable import aliases.
pub(super) fn map_parts(first: u64, length: u64, parts: &[Piece]) -> Result<(), KernReturn> {
    replace(first, length, parts)?;
    for piece in parts {
        files::record(
            first + piece.at,
            piece.length,
            piece.backing.clone(),
            piece.prot,
        );
    }
    Ok(())
}

fn allocate(length: u64) -> Result<HostAddr, KernReturn> {
    let backing = Backing {
        file: section(length)?,
        offset: 0,
    };
    let first = reserve(length)?;
    if let Err(kr) = map(first, length, &backing, 3) {
        // SAFETY: the placeholder was never mapped.
        unsafe { VirtualFree(ptr(first), 0, MEM_RELEASE) };
        return Err(kr);
    }
    VIEWS.lock().insert(
        first,
        View {
            length,
            backing: backing.clone(),
            prot: 3,
        },
    );
    files::record(first, length, backing, 3);
    Ok(HostPtr::from_addr(first.to_usize()))
}

/// A zero-filled range whose pages can be shared in other views.
#[cfg(test)]
pub(crate) fn allocate_aliasable(length: u64) -> Result<HostAddr, KernReturn> {
    WindowsVm::new().allocate_anywhere(length)
}

/// Memory owned by an IOSurface, which core cannot deallocate.
#[derive(Debug)]
pub(crate) struct OwnMemory {
    first: HostAddr,
    length: u64,
    registration: Option<Registration>,
}
impl OwnMemory {
    pub(crate) fn new(length: u64) -> Option<Self> {
        let (_, length) = pages(HostAddr::NULL, length)?;
        if length == 0 {
            return None;
        }
        let first = allocate(length).ok()?;
        let backing = files::pieces(first.addr().to_u64(), length)?
            .first()?
            .backing
            .clone();
        Some(Self {
            first,
            length,
            registration: Some(Registration::new(first.addr().to_u64(), length, backing)),
        })
    }
    pub(crate) fn first(&self) -> u64 {
        self.first.addr().to_u64()
    }
}
impl Drop for OwnMemory {
    fn drop(&mut self) {
        drop(self.registration.take());
        let _ = WindowsVm.deallocate(self.first, self.length);
    }
}

/// Mach VM's shared-view operations on Windows.
#[derive(Debug, Default)]
pub struct WindowsVm;
impl WindowsVm {
    /// The process's virtual memory service.
    pub fn new() -> Self {
        Self
    }
    /// Whether each byte is committed and readable, without dereferencing guest pointers.
    pub fn readable(&self, address: HostAddr, length: u64) -> bool {
        // The table's answer stands where it knows every byte: its views and the host's
        // registered memory (QEMU's guest memory, one view of gigabytes, whose `VirtualQuery`
        // walks the page tables of the whole view under the address space lock).
        match files::readable(address.addr().to_u64(), length) {
            Some(answer) => return answer,
            None => {}
        }
        let Some(end) = address.addr().to_u64().checked_add(length) else {
            return false;
        };
        let mut at = address.addr().to_u64();
        while at < end {
            let mut info = MemoryInfo::default();
            // SAFETY: queries an address without reading it, into a live output.
            if unsafe { VirtualQuery(ptr(at), &mut info, size_of::<MemoryInfo>()) } == 0
                || info.state != MEM_COMMIT
                || info.protection & (PAGE_NOACCESS | 0x100) != 0
            {
                return false;
            }
            let next = info
                .base
                .addr()
                .to_u64()
                .saturating_add(info.region_size.to_u64());
            if next <= at {
                return false;
            }
            at = next;
        }
        true
    }
}

impl HostVm for WindowsVm {
    fn page_size(&self) -> u64 {
        page_size()
    }
    fn page_shift(&self) -> u32 {
        page_size().trailing_zeros()
    }
    fn aliases_guest_in_place(&self) -> bool {
        true
    }
    fn allocate_anywhere(&self, length: u64) -> Result<HostAddr, KernReturn> {
        let (_, length) = pages(HostAddr::NULL, length).ok_or(NO_SPACE)?;
        if length == 0 {
            Ok(HostAddr::NULL)
        } else {
            allocate(length)
        }
    }
    fn allocate_fixed_overwrite(&self, address: HostAddr, length: u64) -> Result<(), KernReturn> {
        let (first, length) = pages(address, length).ok_or(INVALID_ARGUMENT)?;
        if length == 0 {
            return Ok(());
        }
        let backing = Backing {
            file: section(length)?,
            offset: 0,
        };
        replace(
            first,
            length,
            &[Piece {
                at: 0,
                length,
                backing: backing.clone(),
                prot: 3,
            }],
        )?;
        files::record(first, length, backing, 3);
        Ok(())
    }
    fn remap_shared_overwrite(
        &self,
        destination: HostAddr,
        source: HostAddr,
        length: u64,
    ) -> Result<HostAddr, KernReturn> {
        if length == 0 {
            return Ok(destination);
        }
        let offset = source.addr().to_u64() & (page_size() - 1);
        let (first, length) = pages(
            destination,
            length.checked_add(offset).ok_or(INVALID_ARGUMENT)?,
        )
        .ok_or(INVALID_ARGUMENT)?;
        let started = std::time::Instant::now();
        let (source, _) = pages(source, 1).ok_or(INVALID_ADDRESS)?;
        let parts = files::pieces(source, length).ok_or(INVALID_ADDRESS)?;
        map_parts(first, length, &parts)?;
        let views = VIEWS.lock().len();
        remap_stats::finish(started, parts.len(), length, views);
        Ok(destination)
    }
    fn protect_max(&self, address: HostAddr, length: u64, prot: VmProt) -> Result<(), KernReturn> {
        let (first, length) = pages(address, length).ok_or(INVALID_ARGUMENT)?;
        if length == 0 {
            return Ok(());
        }
        let end = first.checked_add(length).ok_or(INVALID_ARGUMENT)?;
        let mut views = VIEWS.lock();
        split(&mut views, first)?;
        split(&mut views, end)?;
        let mut at = first;
        for (&base, view) in views.range(first..end) {
            if base != at || prot.raw() & !view.prot != 0 {
                return Err(PROTECTION_FAILURE);
            }
            at += view.length;
        }
        if at != end {
            return Err(INVALID_ADDRESS);
        }
        for (&base, view) in views.range_mut(first..end) {
            let mut old = 0;
            // SAFETY: each view is owned and no access is made; only lowers its maximum.
            if unsafe {
                VirtualProtect(
                    ptr(base),
                    view.length.to_usize(),
                    protection(prot.raw()),
                    &mut old,
                )
            } == 0
            {
                return Err(error("VirtualProtect", PROTECTION_FAILURE));
            }
            view.prot = prot.raw();
        }
        files::protect(first, length, prot.raw());
        Ok(())
    }
    fn deallocate(&self, address: HostAddr, length: u64) -> KernReturn {
        let Some((first, length)) = pages(address, length) else {
            return INVALID_ARGUMENT;
        };
        if length == 0 {
            return KernReturn::SUCCESS;
        }
        if files::meets_registration(first, length) {
            return INVALID_ARGUMENT;
        }
        let end = first + length;
        let mut views = VIEWS.lock();
        if split(&mut views, first).is_err() || split(&mut views, end).is_err() {
            return INVALID_ADDRESS;
        }
        let bases: Vec<_> = views.range(first..end).map(|(&base, _)| base).collect();
        if bases.is_empty() {
            return INVALID_ADDRESS;
        }
        for base in bases {
            unmap(base);
            // SAFETY: releases the exact owned placeholder after its view is gone.
            if unsafe { VirtualFree(ptr(base), 0, MEM_RELEASE) } == 0 {
                cannot_restore("VirtualFree release placeholder");
            }
            views.remove(&base);
        }
        files::forget(first, length);
        KernReturn::SUCCESS
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) const PAGE: u64 = 4096;
    pub(crate) fn mapped(at: HostAddr) -> bool {
        WindowsVm.readable(at, 1)
    }
    pub(crate) fn writable(at: HostAddr, length: u64) -> bool {
        WindowsVm
            .protect_max(at, length, VmProt::READ | VmProt::WRITE)
            .is_ok()
    }

    #[test]
    fn aliases_are_shared_page_aligned_and_outlive_the_source() {
        let vm = WindowsVm;
        let source = vm.allocate_anywhere(3 * PAGE).unwrap();
        let target = vm.allocate_anywhere(3 * PAGE).unwrap();
        let middle = HostAddr::from_addr(source.addr() + PAGE.to_usize());
        // SAFETY: live writable allocations.
        unsafe { middle.as_ptr::<u8>().write(7) };
        vm.remap_shared_overwrite(target, middle, PAGE).unwrap();
        // SAFETY: the target shares the source's middle page.
        unsafe {
            assert_eq!(target.as_ptr::<u8>().read(), 7);
            target.as_ptr::<u8>().write(9);
            assert_eq!(middle.as_ptr::<u8>().read(), 9);
        }
        assert!(vm.deallocate(source, 3 * PAGE).is_success());
        // SAFETY: the alias independently keeps the section's page alive.
        unsafe {
            assert_eq!(target.as_ptr::<u8>().read(), 9);
        }
        assert!(vm.deallocate(target, 3 * PAGE).is_success());
        assert!(!mapped(target));
    }
    /// How long core's mapping of a texture over scattered guest pages takes: a target the size
    /// of a 2774x1708 BGRA8 surface, each page remapped from another page of a guest region, as
    /// `MemoryMap::map_physical_range` remaps them one physical range at a time.
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn times_a_texture_mapped_page_by_page() {
        let vm = WindowsVm;
        let guest_pages = 65536u64;
        let guest = vm.allocate_anywhere(guest_pages * PAGE).unwrap();
        let pages = (2774u64 * 1708 * 4).div_ceil(PAGE);
        // Rounds 2 and 3 beside 8 threads that keep writing their own memory, as QEMU's vCPUs
        // keep running in the process while a FIFO thread maps.
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut busy = Vec::new();
        for round in 0..4 {
            if round == 2 {
                for _ in 0..8 {
                    let stop = Arc::clone(&stop);
                    busy.push(std::thread::spawn(move || {
                        let mut memory = vec![0u8; 64 << 20];
                        let mut i = 0usize;
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            let at = i % memory.len();
                            memory[at] = memory[at].wrapping_add(1);
                            i = i.wrapping_add(4096 + 64);
                        }
                    }));
                }
            }
            let target = vm.allocate_anywhere(pages * PAGE).unwrap();
            let started = std::time::Instant::now();
            for i in 0..pages {
                let from = (i * 7919 + round * 13) % guest_pages;
                let source = HostAddr::from_addr(guest.addr() + (from * PAGE).to_usize());
                let to = HostAddr::from_addr(target.addr() + (i * PAGE).to_usize());
                vm.remap_shared_overwrite(to, source, PAGE).unwrap();
            }
            let elapsed = started.elapsed();
            println!(
                "round {round}: {pages} pages in {:.1} ms, {:.1} us a page",
                elapsed.as_secs_f64() * 1e3,
                elapsed.as_secs_f64() * 1e6 / pages as f64
            );
            assert!(vm.deallocate(target, pages * PAGE).is_success());
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for thread in busy {
            thread.join().unwrap();
        }
        assert!(vm.deallocate(guest, guest_pages * PAGE).is_success());
    }
    /// How a remap's time grows with the views the process already holds: QEMU's had 246000
    /// single-page views of guest memory when each remap took about 10 ms.
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn times_remaps_beside_many_views() {
        let vm = WindowsVm;
        let guest_pages: u64 = std::env::var("REIMS_TEST_GUEST_PAGES").ok().and_then(|v| v.parse().ok()).unwrap_or(65536);
        let guest = vm.allocate_anywhere(guest_pages * PAGE).unwrap();
        // Resident, as QEMU's guest memory is.
        for page in 0..guest_pages.min(262_144) {
            // SAFETY: a live writable allocation.
            unsafe { guest.as_ptr::<u8>().add((page * PAGE).to_usize()).write(1) };
        }
        // And mapped into a WHPX partition as guest physical memory, as QEMU's is.
        #[link(name = "WinHvPlatform")]
        unsafe extern "system" {
            fn WHvCreatePartition(partition: *mut *mut core::ffi::c_void) -> i32;
            fn WHvSetPartitionProperty(
                partition: *mut core::ffi::c_void,
                code: u32,
                buffer: *const core::ffi::c_void,
                size: u32,
            ) -> i32;
            fn WHvSetupPartition(partition: *mut core::ffi::c_void) -> i32;
            fn WHvMapGpaRange(
                partition: *mut core::ffi::c_void,
                source: *mut core::ffi::c_void,
                guest_address: u64,
                size: u64,
                flags: u32,
            ) -> i32;
            fn WHvDeletePartition(partition: *mut core::ffi::c_void) -> i32;
        }
        let whpx = std::env::var_os("REIMS_TEST_WHPX").is_some();
        let mut partition = std::ptr::null_mut();
        if whpx {
            // SAFETY: WHPX calls on a partition this test owns; the guest range stays mapped
            // in this process until the partition is deleted below.
            unsafe {
                assert_eq!(WHvCreatePartition(&mut partition), 0);
                let processors: u32 = 1;
                // WHvPartitionPropertyCodeProcessorCount
                assert_eq!(WHvSetPartitionProperty(partition, 0x1fff, (&raw const processors).cast(), 4), 0);
                assert_eq!(WHvSetupPartition(partition), 0);
                // Read | Write | Execute
                assert_eq!(WHvMapGpaRange(partition, guest.as_ptr(), 0, guest_pages * PAGE, 7), 0);
            }
            println!("guest memory mapped into a WHPX partition");
        }
        let held_pages = 200_000u64;
        let held = vm.allocate_anywhere(held_pages * PAGE).unwrap();
        let mut made = 0u64;
        for goal in [0u64, 10_000, 50_000, 100_000, 200_000] {
            while made < goal {
                let from = (made * 7919) % guest_pages;
                let source = HostAddr::from_addr(guest.addr() + (from * PAGE).to_usize());
                let to = HostAddr::from_addr(held.addr() + (made * PAGE).to_usize());
                vm.remap_shared_overwrite(to, source, PAGE).unwrap();
                made += 1;
            }
            let target = vm.allocate_anywhere(256 * PAGE).unwrap();
            let started = std::time::Instant::now();
            for i in 0..256u64 {
                let from = (i * 104_729) % guest_pages;
                let source = HostAddr::from_addr(guest.addr() + (from * PAGE).to_usize());
                let to = HostAddr::from_addr(target.addr() + (i * PAGE).to_usize());
                vm.remap_shared_overwrite(to, source, PAGE).unwrap();
            }
            println!(
                "{} views: {:.1} us a remap",
                VIEWS.lock().len(),
                started.elapsed().as_secs_f64() * 1e6 / 256.0
            );
            assert!(vm.deallocate(target, 256 * PAGE).is_success());
        }
        assert!(vm.deallocate(held, held_pages * PAGE).is_success());
        if whpx {
            // SAFETY: the partition made above.
            unsafe { WHvDeletePartition(partition) };
        }
        assert!(vm.deallocate(guest, guest_pages * PAGE).is_success());
    }
    /// A remap's time while WHPX vCPUs run in the process, as QEMU's do: `REIMS_TEST_VCPUS`
    /// processors spin in real mode on `jmp $` (0xEB 0xFE at the reset vector) while the remaps
    /// map pages of a guest region the partition maps as guest memory.
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn times_remaps_beside_running_vcpus() {
        type Partition = *mut core::ffi::c_void;
        #[link(name = "WinHvPlatform")]
        unsafe extern "system" {
            fn WHvCreatePartition(partition: *mut Partition) -> i32;
            fn WHvSetPartitionProperty(p: Partition, code: u32, b: *const core::ffi::c_void, s: u32) -> i32;
            fn WHvSetupPartition(p: Partition) -> i32;
            fn WHvMapGpaRange(p: Partition, src: *mut core::ffi::c_void, gpa: u64, size: u64, flags: u32) -> i32;
            fn WHvCreateVirtualProcessor(p: Partition, index: u32, flags: u32) -> i32;
            fn WHvRunVirtualProcessor(p: Partition, index: u32, exit: *mut core::ffi::c_void, size: u32) -> i32;
            fn WHvCancelRunVirtualProcessor(p: Partition, index: u32, flags: u32) -> i32;
            fn WHvDeleteVirtualProcessor(p: Partition, index: u32) -> i32;
            fn WHvDeletePartition(p: Partition) -> i32;
        }
        struct Shared(Partition);
        // SAFETY: WHPX partition handles are used from any thread.
        unsafe impl Send for Shared {}
        unsafe impl Sync for Shared {}
        let vcpus: u32 = std::env::var("REIMS_TEST_VCPUS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let vm = WindowsVm;
        let guest_pages = 65536u64;
        let guest = vm.allocate_anywhere(guest_pages * PAGE).unwrap();
        for page in 0..guest_pages {
            // SAFETY: a live writable allocation.
            unsafe { guest.as_ptr::<u8>().add((page * PAGE).to_usize()).write(1) };
        }
        let reset = vm.allocate_anywhere(16 * PAGE).unwrap();
        // The reset vector jumps to F000:0000, whose loop writes a byte of every 4 KiB page of
        // guest memory 0..0xA0000, again and again, as a guest keeps its pages in use.
        const WRITER: [u8; 24] = [
            0x31, 0xc0, 0x8e, 0xd8, 0x31, 0xdb, 0xfe, 0x07, 0x81, 0xc3, 0x00, 0x10, 0x75, 0xf8,
            0x05, 0x00, 0x10, 0x3d, 0x00, 0xa0, 0x72, 0xec, 0xeb, 0xe8,
        ];
        // SAFETY: live writable allocations: the guest's 0xf0000 and the reset vector page.
        unsafe {
            for (i, byte) in WRITER.iter().enumerate() {
                guest.as_ptr::<u8>().add(0xf0000 + i).write(*byte);
            }
            for (i, byte) in [0xea, 0x00, 0x00, 0x00, 0xf0].iter().enumerate() {
                reset.as_ptr::<u8>().add(0xfff0 + i).write(*byte);
                // WHPX starts a processor at CS base 0, RIP 0xfff0.
                guest.as_ptr::<u8>().add(0xfff0 + i).write(*byte);
            }
        }
        let mut partition: Partition = std::ptr::null_mut();
        // SAFETY: WHPX calls on a partition this test owns.
        unsafe {
            assert_eq!(WHvCreatePartition(&mut partition), 0);
            assert_eq!(WHvSetPartitionProperty(partition, 0x1fff, (&raw const vcpus).cast(), 4), 0);
            assert_eq!(WHvSetupPartition(partition), 0);
            assert_eq!(WHvMapGpaRange(partition, guest.as_ptr(), 0, guest_pages * PAGE, 7), 0);
            assert_eq!(WHvMapGpaRange(partition, reset.as_ptr(), 0xffff_0000, 16 * PAGE, 7), 0);
            for i in 0..vcpus {
                assert_eq!(WHvCreateVirtualProcessor(partition, i, 0), 0);
            }
        }
        let shared = Arc::new(Shared(partition));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let measure = |label: &str| {
            let target = vm.allocate_anywhere(256 * PAGE).unwrap();
            let started = std::time::Instant::now();
            let mut done = 0u64;
            for i in 0..256u64 {
                if started.elapsed() > std::time::Duration::from_secs(5) {
                    break;
                }
                done += 1;
                // A page the vCPUs keep writing.
                let from = (i * 7) % 0xa0;
                let source = HostAddr::from_addr(guest.addr() + (from * PAGE).to_usize());
                let to = HostAddr::from_addr(target.addr() + (i * PAGE).to_usize());
                vm.remap_shared_overwrite(to, source, PAGE).unwrap();
            }
            let mapped = started.elapsed();
            let started = std::time::Instant::now();
            assert!(vm.deallocate(target, 256 * PAGE).is_success());
            println!(
                "{label}: {done} remaps, {:.1} us a remap; unmapping {:.1} ms",
                mapped.as_secs_f64() * 1e6 / done.max(1) as f64,
                started.elapsed().as_secs_f64() * 1e3
            );
        };
        measure("no vCPU running");
        let mut runners = Vec::new();
        for i in 0..vcpus {
            let (shared, stop) = (Arc::clone(&shared), Arc::clone(&stop));
            runners.push(std::thread::spawn(move || {
                let mut exit = vec![0u8; 4096];
                let mut first = true;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    // SAFETY: the partition lives until every runner joined.
                    let hr = unsafe { WHvRunVirtualProcessor(shared.0, i, exit.as_mut_ptr().cast(), 4096) };
                    if first && i == 0 {
                        first = false;
                        let reason = u32::from_le_bytes(exit[0..4].try_into().unwrap());
                        println!("vCPU 0 first exit: hr {hr:#x}, reason {reason:#x}, rip bytes {:x?}", &exit[0..48]);
                    }
                }
            }));
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
        measure(&format!("{vcpus} vCPUs running"));
        measure(&format!("{vcpus} vCPUs running, again"));
        // Beside a thread that keeps emptying the process's working set, as Windows trims a
        // process under memory pressure: the vCPUs' next writes fault their pages back in.
        {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn SetProcessWorkingSetSize(process: *mut core::ffi::c_void, min: usize, max: usize) -> i32;
            }
            let trimming = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let trimmer = {
                let trimming = Arc::clone(&trimming);
                std::thread::spawn(move || {
                    let mut trims = 0u64;
                    while !trimming.load(std::sync::atomic::Ordering::Relaxed) {
                        // SAFETY: this process's own working set; usize::MAX twice empties it.
                        unsafe { SetProcessWorkingSetSize(GetCurrentProcess(), usize::MAX, usize::MAX) };
                        trims += 1;
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    trims
                })
            };
            std::thread::sleep(std::time::Duration::from_millis(100));
            measure("vCPUs running, working set trimmed");
            measure("vCPUs running, working set trimmed, again");
            trimming.store(true, std::sync::atomic::Ordering::Relaxed);
            println!("trims: {}", trimmer.join().unwrap());
        }
        // Beside a thread that keeps locking and unlocking 64 MiB of the guest's pages, as a
        // driver's import of host memory probes and locks them.
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn VirtualLock(address: *mut core::ffi::c_void, size: usize) -> i32;
            fn VirtualUnlock(address: *mut core::ffi::c_void, size: usize) -> i32;
            fn SetProcessWorkingSetSize(process: *mut core::ffi::c_void, min: usize, max: usize) -> i32;
        }
        // SAFETY: this process's own working set.
        unsafe { SetProcessWorkingSetSize(GetCurrentProcess(), 512 << 20, 1024 << 20) };
        let locking = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let locker = {
            let locking = Arc::clone(&locking);
            let at = guest.addr() + (16 << 20);
            let lock_bytes: usize = std::env::var("REIMS_TEST_LOCK_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(8) << 20;
            std::thread::spawn(move || {
                let mut locks = 0u64;
                while !locking.load(std::sync::atomic::Ordering::Relaxed) {
                    // SAFETY: a live range of the guest allocation.
                    unsafe {
                        if VirtualLock(std::ptr::with_exposed_provenance_mut(at), lock_bytes) != 0 {
                            locks += 1;
                        }
                        VirtualUnlock(std::ptr::with_exposed_provenance_mut(at), lock_bytes);
                    }
                }
                locks
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(100));
        measure("vCPUs and a locker running");
        locking.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("locks taken: {}", locker.join().unwrap());
        // SAFETY: a live allocation the vCPUs write.
        let counts = unsafe { [guest.as_ptr::<u8>().add(0x1_0000).read(), guest.as_ptr::<u8>().add(0x9_f000).read()] };
        println!("bytes the vCPUs count at 0x10000 and 0x9f000: {counts:?}");
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for i in 0..vcpus {
            // SAFETY: the partition's processors made above.
            unsafe { WHvCancelRunVirtualProcessor(partition, i, 0) };
        }
        for runner in runners {
            runner.join().unwrap();
        }
        // SAFETY: the partition and processors made above, none running.
        unsafe {
            for i in 0..vcpus {
                WHvDeleteVirtualProcessor(partition, i);
            }
            WHvDeletePartition(partition);
        }
        assert!(vm.deallocate(guest, guest_pages * PAGE).is_success());
    }
    /// A remap's time beside a thread that keeps importing the guest's pages into Vulkan buffers
    /// and freeing them, as the library's no-copy buffers over guest pages do: the driver locks
    /// a host range's pages when it imports them.
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn times_remaps_beside_imports() {
        use metal2vulkan_runtime as rt;
        let vm = WindowsVm;
        let guest_pages = 65536u64;
        let guest = vm.allocate_anywhere(guest_pages * PAGE).unwrap();
        for page in 0..guest_pages {
            // SAFETY: a live writable allocation.
            unsafe { guest.as_ptr::<u8>().add((page * PAGE).to_usize()).write(1) };
        }
        let device = rt::Device::system_default().unwrap();
        let measure = |label: &str| {
            let target = vm.allocate_anywhere(256 * PAGE).unwrap();
            let started = std::time::Instant::now();
            let mut done = 0u64;
            for i in 0..256u64 {
                if started.elapsed() > std::time::Duration::from_secs(5) {
                    break;
                }
                done += 1;
                let from = (i * 104_729) % guest_pages;
                let source = HostAddr::from_addr(guest.addr() + (from * PAGE).to_usize());
                let to = HostAddr::from_addr(target.addr() + (i * PAGE).to_usize());
                vm.remap_shared_overwrite(to, source, PAGE).unwrap();
            }
            let mapped = started.elapsed();
            assert!(vm.deallocate(target, 256 * PAGE).is_success());
            println!("{label}: {done} remaps, {:.1} us a remap", mapped.as_secs_f64() * 1e6 / done.max(1) as f64);
        };
        measure("no imports");
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mib: u64 = std::env::var("REIMS_TEST_IMPORT_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let importer = {
            let (stop, device) = (Arc::clone(&stop), device.clone());
            let base = guest.addr();
            std::thread::spawn(move || {
                let mut imports = 0u64;
                let mut at = 0u64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let offset = (at * (mib << 20)) % ((guest_pages * PAGE) - (mib << 20));
                    at += 1;
                    let bytes = std::ptr::NonNull::new(std::ptr::with_exposed_provenance_mut::<u8>(base + offset.to_usize())).unwrap();
                    // SAFETY: a live writable range of the guest allocation, kept mapped until
                    // the buffer drops below.
                    let buffer = unsafe {
                        device.new_buffer_with_bytes_no_copy(bytes, mib << 20, rt::ResourceOptions::shared(), None)
                    };
                    if buffer.is_ok() {
                        imports += 1;
                    }
                    drop(buffer);
                }
                imports
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(200));
        measure(&format!("beside {mib} MiB imports"));
        measure(&format!("beside {mib} MiB imports, again"));
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("imports made: {}", importer.join().unwrap());
        assert!(vm.deallocate(guest, guest_pages * PAGE).is_success());
    }
    /// A remap's time beside a thread that keeps reading a file into the guest's pages, as
    /// QEMU's block layer reads a disk image into guest memory: `REIMS_TEST_READ_FILE` names the
    /// file, `REIMS_TEST_UNBUFFERED` reads past the cache (`FILE_FLAG_NO_BUFFERING`).
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn times_remaps_beside_file_reads() {
        use std::os::windows::fs::OpenOptionsExt;
        use std::os::windows::io::AsRawHandle;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn ReadFile(file: *mut core::ffi::c_void, buffer: *mut core::ffi::c_void, length: u32, read: *mut u32, overlapped: *mut core::ffi::c_void) -> i32;
            fn SetFilePointerEx(file: *mut core::ffi::c_void, distance: i64, new: *mut i64, method: u32) -> i32;
        }
        let path = std::env::var("REIMS_TEST_READ_FILE").expect("REIMS_TEST_READ_FILE");
        let unbuffered = std::env::var_os("REIMS_TEST_UNBUFFERED").is_some();
        let vm = WindowsVm;
        let guest_pages = 65536u64;
        let guest = vm.allocate_anywhere(guest_pages * PAGE).unwrap();
        for page in 0..guest_pages {
            // SAFETY: a live writable allocation.
            unsafe { guest.as_ptr::<u8>().add((page * PAGE).to_usize()).write(1) };
        }
        let measure = |label: &str| {
            let target = vm.allocate_anywhere(256 * PAGE).unwrap();
            let started = std::time::Instant::now();
            let mut done = 0u64;
            for i in 0..256u64 {
                if started.elapsed() > std::time::Duration::from_secs(5) {
                    break;
                }
                done += 1;
                let from = (i * 104_729) % guest_pages;
                let source = HostAddr::from_addr(guest.addr() + (from * PAGE).to_usize());
                let to = HostAddr::from_addr(target.addr() + (i * PAGE).to_usize());
                vm.remap_shared_overwrite(to, source, PAGE).unwrap();
            }
            let mapped = started.elapsed();
            assert!(vm.deallocate(target, 256 * PAGE).is_success());
            println!("{label}: {done} remaps, {:.1} us a remap", mapped.as_secs_f64() * 1e6 / done.max(1) as f64);
        };
        measure("no reads");
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let stop = Arc::clone(&stop);
            let base = guest.addr();
            std::thread::spawn(move || {
                let mut options = std::fs::OpenOptions::new();
                options.read(true);
                if unbuffered {
                    options.custom_flags(0x2000_0000); // FILE_FLAG_NO_BUFFERING
                }
                let file = options.open(&path).unwrap();
                let (mut reads, mut at) = (0u64, 0usize);
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let mut read = 0u32;
                    let into = base + ((at * 65536) % (200 << 20));
                    at += 1;
                    // SAFETY: a live writable 64 KiB of the guest allocation, page aligned.
                    let ok = unsafe { ReadFile(file.as_raw_handle(), std::ptr::with_exposed_provenance_mut(into), 65536, &mut read, std::ptr::null_mut()) };
                    if ok == 0 || read < 65536 {
                        // SAFETY: the file opened above.
                        unsafe { SetFilePointerEx(file.as_raw_handle(), 0, std::ptr::null_mut(), 0) };
                    }
                    reads += 1;
                }
                reads
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(200));
        measure(if unbuffered { "beside unbuffered reads" } else { "beside cached reads" });
        measure("again");
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("reads: {}", reader.join().unwrap());
        assert!(vm.deallocate(guest, guest_pages * PAGE).is_success());
    }
    /// How long `VirtualQuery` takes in one large committed, resident view, as QEMU's guest
    /// memory is: the runtime's writability check before each import queries it.
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn times_virtual_query_in_a_large_view() {
        #[repr(C)]
        struct Mbi { base: usize, allocation_base: usize, allocation_protect: u32, partition: u16, region_size: usize, state: u32, protect: u32, kind: u32 }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn VirtualQuery(at: *const core::ffi::c_void, info: *mut Mbi, length: usize) -> usize;
        }
        let vm = WindowsVm;
        for gib in [1u64, 4] {
            let pages = gib << 18;
            let guest = vm.allocate_anywhere(pages * PAGE).unwrap();
            for page in 0..pages {
                // SAFETY: a live writable allocation.
                unsafe { guest.as_ptr::<u8>().add((page * PAGE).to_usize()).write(1) };
            }
            for at in [0u64, pages / 2, pages - 2] {
                let started = std::time::Instant::now();
                let mut info = std::mem::MaybeUninit::<Mbi>::zeroed();
                // SAFETY: a query of a live allocation into a sized output.
                unsafe { VirtualQuery(std::ptr::with_exposed_provenance(guest.addr() + (at * PAGE).to_usize()), info.as_mut_ptr(), std::mem::size_of::<Mbi>()) };
                // SAFETY: filled above.
                let size = unsafe { info.assume_init().region_size };
                println!("{gib} GiB view, query at page {at}: {:.1} us, region {} MiB", started.elapsed().as_secs_f64() * 1e6, size >> 20);
                #[repr(C)]
                struct Region { allocation_base: usize, allocation_protect: u32, region_type: u32, region_size: usize, commit_size: usize, partition: usize, node: usize }
                #[link(name = "ntdll")]
                unsafe extern "system" {
                    fn NtQueryVirtualMemory(process: *mut core::ffi::c_void, at: *const core::ffi::c_void, class: u32, info: *mut Region, length: usize, returned: *mut usize) -> i32;
                }
                let started = std::time::Instant::now();
                let mut region = std::mem::MaybeUninit::<Region>::zeroed();
                // SAFETY: a query of a live allocation into a sized output.
                let mut status = 0;
                for (class, length) in [(2u32, 48usize), (2, 40), (2, 32), (7, 48), (7, 40)] {
                    // SAFETY: as above.
                    status = unsafe { NtQueryVirtualMemory(std::ptr::with_exposed_provenance_mut(usize::MAX), std::ptr::with_exposed_provenance(guest.addr() + (at * PAGE).to_usize()), class, region.as_mut_ptr(), length, std::ptr::null_mut()) };
                    if status == 0 {
                        println!("   class {class} length {length} works");
                        break;
                    }
                }
                // SAFETY: filled above where the status is success.
                let region = unsafe { region.assume_init() };
                println!("   region information: status {status:#x}, {:.1} us, protect {:#x}, type {:#x}, size {} MiB, commit {} MiB", started.elapsed().as_secs_f64() * 1e6, region.allocation_protect, region.region_type, region.region_size >> 20, region.commit_size >> 20);
            }
            assert!(vm.deallocate(guest, pages * PAGE).is_success());
        }
    }
    #[test]
    fn failed_remap_preserves_bytes_and_overwrite_only_zeros_its_page() {
        let vm = WindowsVm;
        let at = vm.allocate_anywhere(3 * PAGE).unwrap();
        // SAFETY: writable live allocation.
        unsafe {
            at.as_ptr::<u8>().write(4);
            at.as_ptr::<u8>().add(PAGE.to_usize()).write(5);
        }
        assert!(vm.remap_shared_overwrite(at, HostAddr::NULL, PAGE).is_err());
        vm.allocate_fixed_overwrite(HostAddr::from_addr(at.addr() + PAGE.to_usize()), PAGE)
            .unwrap();
        // SAFETY: same allocation, with a fresh middle page.
        unsafe {
            assert_eq!(at.as_ptr::<u8>().read(), 4);
            assert_eq!(at.as_ptr::<u8>().add(PAGE.to_usize()).read(), 0);
        }
        assert!(vm.deallocate(at, 3 * PAGE).is_success());
    }
}
