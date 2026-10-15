//! `HostVm` on Linux: Mach VM's calls, as core makes them, over `mmap`.
//!
//! Linux cannot remap an address range's pages to another address as `mach_vm_remap` does, so
//! a view of guest pages ("virtual memory layouts") maps their files instead: the address space
//! is reserved first (core's `allocate_anywhere`), then each run of the source's bytes is mapped
//! over it with `mmap(MAP_SHARED | MAP_FIXED)` from the file and offset the backing table names
//! for it ([`super::files`]), and the view is recorded there as runs of its own. A view of a view
//! maps the file too, so it keeps showing the pages after the first view goes, as Mach's aliases
//! do; runs whose file offsets follow on are one VMA to the kernel.
//!
//! - **`allocate_anywhere`** is private anonymous memory (`MAP_NORESERVE`), zero-filled and in no
//!   file, so no view can be made of it: it serves core's reservations, its detach temporaries
//!   and the address space views are mapped over. A range a view must be made of comes from
//!   [`allocate_aliasable`], a shared mapping of a file of its own.
//! - **`remap_shared_overwrite`** fails with `KERN_INVALID_ADDRESS` where a byte of the source
//!   has no file behind it, as Mach fails for an unmapped source, before it maps anything. A
//!   piece the kernel refuses fails the call with nothing changed, as Mach's remap is all or
//!   nothing: the pieces mapped before it are put back as the target was ([`put_back`]).
//! - **`protect_max`** is `mprotect`: Linux has no maximum protection, and the current one
//!   stands in for it, so a lowered protection can be raised again, which Mach refuses. A
//!   forbidden write is a `SIGSEGV`, where Mach's is a `SIGBUS`: a death either way.
//! - Each call takes whole host pages (`sysconf(_SC_PAGESIZE)`), from the page of its first byte
//!   to the page past its last, as Mach's do.
//!
//! **Mappings.** Each run of a view is a VMA, so a view of scattered pages costs one per run: a
//! 1080p surface over 16 KiB granules about 507. The views share a budget short of
//! `vm.max_map_count` ([`super::budget`]): a remap, a range a view can be made of or a stand-in's
//! memory that would pass it ends the process before anything is mapped, so the rest of the
//! process keeps its mappings. A map the kernel refuses for want of mappings, the process at
//! `vm.max_map_count`, ends it the same way, naming how many the process has and the limit
//! ([`no_space`]): macOS has no such limit, so `KERN_NO_SPACE` there would be an answer Mach never
//! gives. One refused for want of memory or address space is `KERN_NO_SPACE`, as Mach's is. A
//! file the kernel gives none of for the library's memory ([`memfd`]) ends the process too,
//! naming the open-file limit.

use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::sync::Arc;

use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};

use super::budget;
use super::files::{self, Backing};

/// `KERN_INVALID_ADDRESS`.
const INVALID_ADDRESS: KernReturn = KernReturn(1);
/// `KERN_PROTECTION_FAILURE`.
const PROTECTION_FAILURE: KernReturn = KernReturn(2);
/// `KERN_NO_SPACE`.
const NO_SPACE: KernReturn = KernReturn(3);
/// `KERN_INVALID_ARGUMENT`.
const INVALID_ARGUMENT: KernReturn = KernReturn(4);
/// `KERN_FAILURE`.
const FAILURE: KernReturn = KernReturn(5);

/// The host's page size.
pub(super) fn page_size() -> u64 {
    // SAFETY: a sysconf query, which cannot fail for this name.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(size).unwrap_or(4096)
}

/// The whole pages `len` bytes at `addr` lie in: the first byte of the first, and their length;
/// `None` where they pass the end of the address space.
fn pages(addr: HostAddr, len: u64) -> Option<(u64, u64)> {
    let mask = page_size() - 1;
    let start = addr.addr().to_u64();
    let first = start & !mask;
    let end = start.checked_add(len)?.checked_add(mask)? & !mask;
    Some((first, end - first))
}

/// A Mach answer for the `errno` of a refused call, which `what` names: `ENOMEM` is
/// `KERN_NO_SPACE` for a map ([`no_space`]), `KERN_INVALID_ADDRESS` for a range not mapped.
fn refused(what: &str, unmapped_is_enomem: bool) -> KernReturn {
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    match errno {
        libc::ENOMEM if unmapped_is_enomem => INVALID_ADDRESS,
        libc::ENOMEM => no_space(what),
        libc::EINVAL => INVALID_ARGUMENT,
        libc::EACCES | libc::EPERM => PROTECTION_FAILURE,
        _ => FAILURE,
    }
}

/// `KERN_NO_SPACE` for a map the kernel refused for want of memory or address space, logged with
/// the process's mappings and their limit (`vm.max_map_count`); one it refused for want of
/// mappings, the process at the limit ([`at_the_limit`]), ends the process there, naming them
/// (module docs).
fn no_space(what: &str) -> KernReturn {
    let (mappings, limit) = map_census();
    if at_the_limit(&mappings, &limit) {
        fatal_with_detail(
            DeathId::Platform {
                kind: DeathKind::Abort,
                call: "mmap",
            },
            &format!(
                "the process ran out of kernel mappings: the kernel refused {what} (ENOMEM) with \
                 {mappings} of vm.max_map_count {limit} held, {} of them the library's views of \
                 memory; macOS has no such limit, and `sysctl -w vm.max_map_count=` a higher \
                 value gives room",
                budget::held()
            ),
        )
    }
    log::error!(
        target: "reims_vgpu_vulkan::os",
        "{what}: the kernel refused the map (ENOMEM): the process has {mappings} mappings, \
         vm.max_map_count is {limit}"
    );
    NO_SPACE
}

/// Whether a process holding `mappings` mappings is at `vm.max_map_count` `limit`, each as
/// [`map_census`] read it: within the 2 a map may add, splitting a mapping in two. `false` where
/// either does not read.
fn at_the_limit(mappings: &str, limit: &str) -> bool {
    match (mappings.parse::<u64>(), limit.parse::<u64>()) {
        (Ok(mappings), Ok(limit)) => mappings.saturating_add(2) >= limit,
        _ => false,
    }
}

/// The process's mappings (`/proc/self/maps`' lines) and `vm.max_map_count`, each as read, or
/// `"?"` where it does not read.
pub(super) fn map_census() -> (String, String) {
    let mappings = std::fs::read_to_string("/proc/self/maps")
        .map_or_else(|_| "?".to_owned(), |m| m.lines().count().to_string());
    let limit = std::fs::read_to_string("/proc/sys/vm/max_map_count")
        .map_or_else(|_| "?".to_owned(), |l| l.trim().to_owned());
    (mappings, limit)
}

/// `VmProt` as `PROT_*`: the same bits.
fn prot_bits(prot: VmProt) -> i32 {
    prot.raw() & (libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC)
}

/// A pointer to `at` in the process's address space.
pub(super) fn ptr(at: u64) -> *mut libc::c_void {
    std::ptr::with_exposed_provenance_mut(at.to_usize())
}

/// Maps `length` bytes of anonymous zero-filled memory, read and write, at `at` (`MAP_FIXED`)
/// or anywhere: its first byte, or the `errno`'s answer.
pub(super) fn anonymous(at: Option<u64>, length: u64, what: &str) -> Result<u64, KernReturn> {
    let fixed = if at.is_some() { libc::MAP_FIXED } else { 0 };
    // SAFETY: a fresh private mapping; at a fixed address, replacing that range is the caller's
    // decision (core's reservation of its own).
    let mapped = unsafe {
        libc::mmap(
            ptr(at.unwrap_or(0)),
            length.to_usize(),
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | fixed,
            -1,
            0,
        )
    };
    if mapped == libc::MAP_FAILED {
        return Err(refused(what, false));
    }
    Ok(mapped.addr().to_u64())
}

/// Maps `length` bytes of `backing`'s file, shared, with protection `prot`, at `at`
/// (`MAP_FIXED`) or anywhere: its first byte, or the `errno`'s answer.
pub(super) fn file(
    at: Option<u64>,
    length: u64,
    backing: &Backing,
    prot: i32,
) -> Result<u64, KernReturn> {
    let fixed = if at.is_some() { libc::MAP_FIXED } else { 0 };
    let offset = libc::off_t::try_from(backing.offset).map_err(|_| INVALID_ARGUMENT)?;
    // SAFETY: a shared mapping of an open file; at a fixed address, replacing that range is the
    // caller's decision (a view over core's reservation).
    let mapped = unsafe {
        libc::mmap(
            ptr(at.unwrap_or(0)),
            length.to_usize(),
            prot,
            libc::MAP_SHARED | fixed,
            backing.file.as_raw_fd(),
            offset,
        )
    };
    if mapped == libc::MAP_FAILED {
        return Err(refused("a view of guest pages", false));
    }
    Ok(mapped.addr().to_u64())
}

/// A fresh range of `len` bytes, rounded up to whole pages, of zeros, that a view can be made of:
/// a shared mapping of a file of its own, recorded in the backing table. Freed by `deallocate`,
/// as any other range. The block of zeros a detached object reads, and the tests' stand-in for
/// guest memory, are such ranges.
pub(crate) fn allocate_aliasable(len: u64) -> Result<HostAddr, KernReturn> {
    let Some((_, length)) = pages(HostAddr::NULL, len) else {
        return Err(NO_SPACE);
    };
    if length == 0 {
        return Ok(HostAddr::NULL);
    }
    budget::admit(1, "a range a view can be made of");
    let fd = memfd(length);
    let backing = Backing {
        file: fd,
        offset: 0,
    };
    let prot = libc::PROT_READ | libc::PROT_WRITE;
    let at = file(None, length, &backing, prot)?;
    files::record(at, length, backing, prot);
    Ok(HostAddr::from_addr(at.to_usize()))
}

/// Memory of the library's own that a view can be made of and that only its holder frees: a
/// shared mapping of a new file, read and write, registered in the backing table
/// ([`files::Registration`]), so that `deallocate` refuses it. An IOSurface stand-in's memory.
#[derive(Debug)]
pub(crate) struct OwnMemory {
    first: u64,
    length: u64,
    registration: Option<files::Registration>,
}

impl OwnMemory {
    /// `len` bytes, rounded up to whole pages, of zeros; `None` for none, or where the host
    /// gives no file or no address space.
    pub(crate) fn new(len: u64) -> Option<OwnMemory> {
        let (_, length) = pages(HostAddr::NULL, len)?;
        if length == 0 {
            return None;
        }
        budget::admit(1, "an IOSurface stand-in's memory");
        let backing = Backing {
            file: memfd(length),
            offset: 0,
        };
        let first = file(None, length, &backing, libc::PROT_READ | libc::PROT_WRITE).ok()?;
        budget::own_hold(1, true);
        Some(OwnMemory {
            first,
            length,
            registration: Some(files::Registration::new(first, length, backing)),
        })
    }

    /// The first byte.
    pub(crate) fn first(&self) -> u64 {
        self.first
    }
}

impl Drop for OwnMemory {
    /// Drops the registration, then the mapping.
    fn drop(&mut self) {
        drop(self.registration.take());
        // SAFETY: the mapping this made, which nothing uses once its holder lets go of it.
        let _ = unsafe { libc::munmap(ptr(self.first), self.length.to_usize()) };
        budget::own_hold(1, false);
    }
}

/// A new file of `length` bytes of zeros in memory (`memfd_create`), sealed against growing and
/// shrinking, as `/dev/udmabuf` takes only a memfd sealed against shrinking
/// ([`super::dma_buf`]). Where the kernel gives none, the process ends there, naming the error and
/// the open-file limit (module docs).
pub(crate) fn memfd(length: u64) -> Arc<OwnedFd> {
    let name = c"reims-vgpu";
    // SAFETY: a NUL-terminated name.
    let raw =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if raw < 0 {
        no_file("memfd_create", length);
    }
    // SAFETY: a descriptor just made, owned from here on.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let Ok(size) = libc::off_t::try_from(length) else {
        no_file("ftruncate", length);
    };
    // SAFETY: an open memfd.
    if unsafe { libc::ftruncate(fd.as_raw_fd(), size) } != 0 {
        no_file("ftruncate", length);
    }
    let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK;
    // SAFETY: an open memfd made to take seals.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals) } != 0 {
        no_file("fcntl(F_ADD_SEALS)", length);
    }
    Arc::new(fd)
}

/// The death at `call` of a file of `length` bytes the kernel gave none of ([`memfd`]), with the
/// error just set and the open-file limit.
fn no_file(call: &'static str, length: u64) -> ! {
    let error = std::io::Error::last_os_error();
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` is a valid rlimit to fill.
    let read = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } == 0;
    let limit = if read {
        limit.rlim_cur.to_string()
    } else {
        "?".to_owned()
    };
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call,
        },
        &format!(
            "the kernel gave no file of {length:#x} bytes for the library's memory ({error}); \
             the open-file limit is {limit}; IOSurface's and Mach's calls need no file on macOS"
        ),
    )
}

/// Maps `length` bytes from `at`, a piece of a remap that failed after mapping it, back to what
/// they showed before it, so the failed remap changes nothing: the files of the table's runs and
/// registrations there, at the protection each was set to (the remap records nothing until every
/// piece is mapped), and fresh zeros with no access over bytes of no file.
///
/// Bytes of no file in a remap's target are a reservation: a mapping task's, which has no access,
/// or a fresh range its caller frees after the failure (core's range mappings, an alias, a
/// surface's view), so the protection of those zeros never shows. No import of the runtime's
/// lies over them, as the runtime imports no page without read access and nothing imports a
/// fresh range before its remaps succeed, so putting zeros there is reported to no one
/// ([`crate::remap`]); the table's files are mapped back as they were, the same pages. A
/// put-back the kernel refuses too (`vm.max_map_count`) leaves the range forgotten by the table,
/// logged.
fn put_back(at: u64, length: u64) {
    let page = page_size();
    let has_file = |first: u64| files::pieces(first, page).is_some();
    let end = at + length;
    let mut from = at;
    while from < end {
        // The pages from `from` that all have a file, or all have none.
        let filed = has_file(from);
        let mut to = from + page;
        while to < end && has_file(to) == filed {
            to += page;
        }
        let restored = if filed {
            files::pieces(from, to - from).is_some_and(|pieces| {
                pieces
                    .iter()
                    .all(|p| file(Some(from + p.at), p.length, &p.backing, p.prot).is_ok())
            })
        } else {
            anonymous(Some(from), to - from, "a failed remap's put-back").is_ok()
                // SAFETY: the zeros just mapped, a reservation of the remap's caller (above).
                && unsafe { libc::mprotect(ptr(from), (to - from).to_usize(), libc::PROT_NONE) } == 0
        };
        if !restored {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "a failed remap's put-back of {:#x} bytes at {from:#x} was refused: {}; the \
                 range is forgotten",
                to - from,
                std::io::Error::last_os_error()
            );
            files::forget(from, to - from);
        }
        from = to;
    }
}

/// The most mappings a view of `pieces` adds: one for each run of them the kernel holds as one
/// (each piece following the one before in the same file at the same protection), and one more
/// where the view cuts a run of its target in two.
fn mappings_added(pieces: &[files::Piece]) -> u64 {
    let follows = |a: &files::Piece, b: &files::Piece| {
        Arc::ptr_eq(&a.backing.file, &b.backing.file)
            && a.backing.offset + a.length == b.backing.offset
            && a.prot == b.prot
    };
    let apart = pieces.windows(2).filter(|w| !follows(&w[0], &w[1])).count();
    apart.to_u64() + 2
}

/// Mach VM's calls on Linux (module docs).
#[derive(Debug, Default)]
pub struct LinuxVm;

impl LinuxVm {
    /// The process's HostVm.
    pub fn new() -> LinuxVm {
        LinuxVm
    }

    /// Whether every page of the `len` bytes at `addr` is mapped with read access: by the
    /// backing table's protections where it knows every byte, else by reading a byte of each
    /// page through the kernel (`process_vm_readv`, which refuses a page unmapped or mapped
    /// without read access). A `len` of 0 is `true`.
    pub fn readable(&self, addr: HostAddr, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        let Some((first, length)) = pages(addr, len) else {
            return false;
        };
        files::readable(first, length).unwrap_or_else(|| read_probe(first, length))
    }
}

/// Whether a byte of each page of `length` bytes from `first` reads, through the kernel.
fn read_probe(first: u64, length: u64) -> bool {
    let page = page_size();
    let count = (length / page).to_usize();
    let mut sink = vec![0u8; count.min(1024)];
    let mut at = first;
    let mut left = count;
    while left > 0 {
        let now = left.min(sink.len());
        let local = libc::iovec {
            iov_base: sink.as_mut_ptr().cast(),
            iov_len: now,
        };
        let remote: Vec<libc::iovec> = (0..now.to_u64())
            .map(|i| libc::iovec {
                iov_base: ptr(at + i * page),
                iov_len: 1,
            })
            .collect();
        let Ok(count) = libc::c_ulong::try_from(remote.len()) else {
            return false;
        };
        // SAFETY: the local buffer holds `now` bytes; the kernel reads the remote ones and
        // refuses those it cannot, writing only into the local buffer.
        let read =
            unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, remote.as_ptr(), count, 0) };
        if usize::try_from(read).ok() != Some(now) {
            return false;
        }
        at += now.to_u64() * page;
        left -= now;
    }
    true
}

impl HostVm for LinuxVm {
    fn page_size(&self) -> u64 {
        page_size()
    }

    fn page_shift(&self) -> u32 {
        page_size().trailing_zeros()
    }

    /// Anonymous memory anywhere; a length of 0 answers address 0, as Mach's does.
    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let Some((_, length)) = pages(HostAddr::NULL, len) else {
            return Err(NO_SPACE);
        };
        if length == 0 {
            return Ok(HostAddr::NULL);
        }
        let at = anonymous(None, length, "allocate_anywhere")?;
        files::forget(at, length);
        Ok(HostAddr::from_addr(at.to_usize()))
    }

    /// Fresh anonymous memory over the pages of the range, which then shows no file.
    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn> {
        let (first, length) = pages(addr, len).ok_or(NO_SPACE)?;
        if length == 0 {
            return Ok(());
        }
        anonymous(Some(first), length, "allocate_fixed_overwrite")?;
        files::forget(first, length);
        Ok(())
    }

    /// The source's pages, from the page of its first byte, mapped from their files over `dst`'s
    /// pages, each with the protection the source has there (module docs).
    fn remap_shared_overwrite(
        &self,
        dst: HostAddr,
        src: HostAddr,
        len: u64,
    ) -> Result<HostAddr, KernReturn> {
        let mask = page_size() - 1;
        let into = src.addr().to_u64() & mask;
        let (target, length) = pages(dst, len.saturating_add(into)).ok_or(INVALID_ARGUMENT)?;
        let (source, _) = pages(src, len).ok_or(INVALID_ADDRESS)?;
        if length == 0 {
            return Ok(dst);
        }
        let Some(pieces) = files::pieces(source, length) else {
            log::debug!(
                target: "reims_vgpu_vulkan::os",
                "remap of {length:#x} bytes at {source:#x}: no file behind them"
            );
            return Err(INVALID_ADDRESS);
        };
        budget::admit(mappings_added(&pieces), "a view of guest pages");
        for (done, piece) in pieces.iter().enumerate() {
            if let Err(kr) = file(
                Some(target + piece.at),
                piece.length,
                &piece.backing,
                piece.prot,
            ) {
                for mapped in pieces.iter().take(done) {
                    put_back(target + mapped.at, mapped.length);
                }
                return Err(kr);
            }
        }
        for piece in pieces {
            files::record(target + piece.at, piece.length, piece.backing, piece.prot);
        }
        Ok(dst)
    }

    /// `mprotect` of the range's pages (module docs).
    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn> {
        let (first, length) = pages(addr, len).ok_or(INVALID_ARGUMENT)?;
        if length == 0 {
            return Ok(());
        }
        let bits = prot_bits(prot);
        // SAFETY: lowering or raising a range's protection is the caller's decision; a later
        // access it forbids faults (a process death).
        if unsafe { libc::mprotect(ptr(first), length.to_usize(), bits) } != 0 {
            return Err(refused("protect_max", true));
        }
        files::protect(first, length, bits);
        Ok(())
    }

    /// `munmap` of the range's pages, which show no file from then on. A range of memory the
    /// library registered and does not own (the guest's RAM) is refused, unmapped by nothing.
    fn deallocate(&self, addr: HostAddr, len: u64) -> KernReturn {
        let Some((first, length)) = pages(addr, len) else {
            return INVALID_ARGUMENT;
        };
        if length == 0 {
            return KernReturn::SUCCESS;
        }
        if files::meets_registration(first, length) {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "deallocate of {length:#x} bytes at {first:#x}: registered memory, not unmapped"
            );
            return INVALID_ARGUMENT;
        }
        files::forget(first, length);
        // SAFETY: the range is one core allocated and no longer uses (its owner's drop).
        if unsafe { libc::munmap(ptr(first), length.to_usize()) } != 0 {
            return refused("deallocate", true);
        }
        KernReturn::SUCCESS
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn vm() -> LinuxVm {
        LinuxVm::new()
    }

    fn page() -> u64 {
        page_size()
    }

    fn read(a: HostAddr, off: u64) -> u8 {
        // SAFETY: test memory this test mapped readable.
        unsafe { a.wrapping_add(off).as_ptr::<u8>().read_volatile() }
    }

    fn write(a: HostAddr, off: u64, v: u8) {
        // SAFETY: test memory this test mapped writable.
        unsafe { a.wrapping_add(off).as_ptr::<u8>().write_volatile(v) }
    }

    /// The host's page, which the tests lay their pages out by: x86-64's 4 KiB.
    pub(crate) const PAGE: u64 = 0x1000;

    /// Whether every page of `len` bytes from `at` is writable now, as `/proc/self/maps` lists
    /// them: Linux has no maximum protection, so the current one is what refuses a write.
    pub(crate) fn writable(at: HostAddr, len: u64) -> bool {
        let first = at.addr().to_u64();
        let end = first.saturating_add(len);
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let mut next = first;
        // The kernel lists the mappings in address order.
        for line in maps.lines() {
            let mut fields = line.split_whitespace();
            let (Some(range), Some(perms)) = (fields.next(), fields.next()) else {
                continue;
            };
            let (low, high) = range.split_once('-').unwrap();
            let low = u64::from_str_radix(low, 16).unwrap();
            let high = u64::from_str_radix(high, 16).unwrap();
            if (low..high).contains(&next) {
                if !perms.contains('w') {
                    return false;
                }
                next = high;
                if next >= end {
                    return true;
                }
            }
        }
        false
    }

    /// Whether the page at `at` is mapped at all (`mincore` refuses an unmapped one).
    pub(crate) fn mapped(at: HostAddr) -> bool {
        let mut resident = 0u8;
        let first = at.addr().to_u64() & !(page() - 1);
        // SAFETY: a query; the kernel writes one byte for one page.
        unsafe { libc::mincore(ptr(first), page().to_usize(), &mut resident) == 0 }
    }

    /// The mappings `/proc/self/maps` lists inside `first..end`.
    pub(crate) fn kernel_mappings(first: u64, end: u64) -> u64 {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let inside = maps.lines().filter(|line| {
            let range = line.split_whitespace().next().unwrap();
            let (low, high) = range.split_once('-').unwrap();
            let low = u64::from_str_radix(low, 16).unwrap();
            let high = u64::from_str_radix(high, 16).unwrap();
            first <= low && high <= end
        });
        inside.count().to_u64()
    }

    /// The table's count of a view's mappings is the kernel's: over a view of pages in and out of
    /// order, a protect inside it, the protection set back and an unmap of part of it, the
    /// mappings `/proc/self/maps` lists in the view's range are as many as the table counts there
    #[test]
    fn the_table_counts_a_views_mappings_as_the_kernel_maps_them() {
        let v = vm();
        let p = page();
        let src = allocate_aliasable(8 * p).unwrap();
        let at = v.allocate_anywhere(8 * p).unwrap();
        // Pages 0-3 in order, then 6 and 7, then 4 and 5: three runs of the file.
        for (to, from) in [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 6),
            (5, 7),
            (6, 4),
            (7, 5),
        ] {
            v.remap_shared_overwrite(at.wrapping_add(to * p), src.wrapping_add(from * p), p)
                .unwrap();
        }
        let (first, end) = (at.addr().to_u64(), at.addr().to_u64() + 8 * p);
        let both = || {
            (
                kernel_mappings(first, end),
                files::mappings_between(first, end),
            )
        };
        assert_eq!(both(), (3, 3), "the view");
        v.protect_max(at.wrapping_add(p), p, VmProt::READ).unwrap();
        assert_eq!(both(), (5, 5), "its second page read-only");
        v.protect_max(at.wrapping_add(p), p, VmProt::READ | VmProt::WRITE)
            .unwrap();
        assert_eq!(both(), (3, 3), "read and write again");
        assert!(v.deallocate(at.wrapping_add(2 * p), p).is_success());
        assert_eq!(both(), (4, 4), "its third page unmapped");
        assert!(v.deallocate(at, 8 * p).is_success());
        assert!(v.deallocate(src, 8 * p).is_success());
    }

    /// The host page is 4 KiB on x86-64 Linux, its shift 12
    #[test]
    fn the_page_is_the_hosts() {
        let v = vm();
        assert_eq!(v.page_size(), 1 << v.page_shift());
        #[cfg(target_arch = "x86_64")]
        assert_eq!((v.page_size(), v.page_shift()), (4096, 12));
    }

    /// An allocation anywhere is page-aligned and zero-filled, and no view can be made of it;
    /// one of 0 bytes is address 0, as Mach's is; one the address space cannot hold is refused
    #[test]
    fn an_allocation_is_zeros_no_view_is_made_of() {
        let v = vm();
        let len = 2 * page();
        let a = v.allocate_anywhere(len).unwrap();
        assert_eq!(a.addr().to_u64() % page(), 0);
        assert_eq!((read(a, 0), read(a, len - 1)), (0, 0));
        write(a, 7, 0xa5);
        let dst = v.allocate_anywhere(len).unwrap();
        assert_eq!(v.remap_shared_overwrite(dst, a, len), Err(INVALID_ADDRESS));
        assert_eq!(v.allocate_anywhere(0), Ok(HostAddr::NULL));
        assert!(
            !v.allocate_anywhere(u64::MAX - page())
                .unwrap_err()
                .is_success()
        );
        assert!(v.deallocate(a, len).is_success());
        assert!(v.deallocate(dst, len).is_success());
        assert!(!mapped(a));
    }

    /// A view shows its source's pages both ways, from the page of the source's first byte; a
    /// view of the view shows them too, and keeps showing them after the first view is unmapped
    #[test]
    fn a_view_of_a_view_outlasts_the_first() {
        let v = vm();
        let len = 4 * page();
        let src = allocate_aliasable(len).unwrap();
        let first = v.allocate_anywhere(len).unwrap();
        let second = v.allocate_anywhere(2 * page()).unwrap();
        write(src, page() + 5, 0x11);
        assert_eq!(
            v.remap_shared_overwrite(first, src.wrapping_add(0x10), len - 0x10),
            Ok(first)
        );
        assert_eq!(read(first, page() + 5), 0x11);
        write(first, 9, 0x33);
        assert_eq!(read(src, 9), 0x33);
        v.remap_shared_overwrite(second, first.wrapping_add(page()), 2 * page())
            .unwrap();
        assert_eq!(read(second, 5), 0x11);
        assert!(v.deallocate(first, len).is_success());
        write(src, page() + 6, 0x22);
        assert_eq!(read(second, 6), 0x22);
        write(second, page(), 0x44);
        assert_eq!(read(src, 2 * page()), 0x44);
        assert!(v.deallocate(second, 2 * page()).is_success());
        assert!(v.deallocate(src, len).is_success());
    }

    /// A view over another view replaces it; an overwrite or an unmap of part of a view leaves
    /// the rest showing its pages; a source with a byte of no file maps nothing
    #[test]
    fn views_are_overwritten_and_unmapped_in_part() {
        let v = vm();
        let (a, b) = (
            allocate_aliasable(2 * page()).unwrap(),
            allocate_aliasable(2 * page()).unwrap(),
        );
        write(a, 0, 1);
        write(a, page(), 2);
        write(b, 0, 3);
        let view = v.allocate_anywhere(2 * page()).unwrap();
        v.remap_shared_overwrite(view, a, 2 * page()).unwrap();
        v.remap_shared_overwrite(view, b, page()).unwrap();
        assert_eq!((read(view, 0), read(view, page())), (3, 2));
        v.allocate_fixed_overwrite(view.wrapping_add(page() + 1), 1)
            .unwrap();
        assert_eq!((read(view, 0), read(view, page())), (3, 0));
        let copy = v.allocate_anywhere(2 * page()).unwrap();
        assert_eq!(
            v.remap_shared_overwrite(copy, view, 2 * page()),
            Err(INVALID_ADDRESS)
        );
        assert_eq!(read(copy, 0), 0, "nothing was mapped");
        v.remap_shared_overwrite(copy, view, page()).unwrap();
        assert_eq!(read(copy, 0), 3);
        for (at, len) in [(view, 2), (copy, 2), (a, 2), (b, 2)] {
            assert!(v.deallocate(at, len * page()).is_success());
        }
    }

    /// A file of one page whose bytes are `byte`, its descriptor read-only (opened again through
    /// `/proc/self/fd`) when `read_only`, so a shared writable map of it is refused (`EACCES`).
    fn page_file(byte: u8, read_only: bool) -> Arc<OwnedFd> {
        let fd = memfd(page());
        let at = file(
            None,
            page(),
            &Backing {
                file: Arc::clone(&fd),
                offset: 0,
            },
            libc::PROT_READ | libc::PROT_WRITE,
        )
        .unwrap();
        // SAFETY: the page just mapped writable, unmapped right after.
        unsafe {
            ptr(at).cast::<u8>().write_bytes(byte, page().to_usize());
            libc::munmap(ptr(at), page().to_usize());
        }
        if !read_only {
            return fd;
        }
        let path = std::ffi::CString::new(format!("/proc/self/fd/{}", fd.as_raw_fd())).unwrap();
        // SAFETY: a NUL-terminated path; the descriptor is owned from here on.
        let raw = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        assert!(raw >= 0);
        // SAFETY: a descriptor just opened, owned once.
        Arc::new(unsafe { OwnedFd::from_raw_fd(raw) })
    }

    /// A remap whose second piece the kernel refuses changes nothing, as Mach's: its first piece
    /// is put back as the target showed it, a view's file there, and zeros with no access over
    /// a reservation's bytes of no file; the table answers as before
    #[test]
    fn a_remap_that_fails_part_way_changes_nothing() {
        let v = vm();
        let p = page();
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        // The source: a run of a file of 0x11s, then a registration of a file of 0x22s whose
        // descriptor is read-only, which the kernel refuses to map shared and writable.
        let src = v.allocate_anywhere(2 * p).unwrap();
        let first = src.addr().to_u64();
        let ones = Backing {
            file: page_file(0x11, false),
            offset: 0,
        };
        file(Some(first), p, &ones, rw).unwrap();
        files::record(first, p, ones, rw);
        let twos = Backing {
            file: page_file(0x22, true),
            offset: 0,
        };
        file(Some(first + p), p, &twos, libc::PROT_READ).unwrap();
        let held = files::Registration::new(first + p, p, twos);
        assert_eq!(files::pieces(first, 2 * p).map(|ps| ps.len()), Some(2));

        // A target whose first page is a view of a file of 0x33s.
        let view = v.allocate_anywhere(2 * p).unwrap();
        let threes = allocate_aliasable(p).unwrap();
        write(threes, 0, 0x33);
        v.remap_shared_overwrite(view, threes, p).unwrap();
        assert_eq!(
            v.remap_shared_overwrite(view, src, 2 * p),
            Err(PROTECTION_FAILURE)
        );
        assert_eq!(read(view, 0), 0x33, "the view's file is back");
        write(view, 1, 0x34);
        assert_eq!(read(threes, 1), 0x34, "the same pages");
        let shown = files::pieces(view.addr().to_u64(), p).unwrap();
        assert_eq!((shown.len(), shown[0].prot), (1, rw));
        assert!(files::pieces(view.addr().to_u64() + p, p).is_none());

        // A target of a reservation's bytes of no file.
        let reserved = v.allocate_anywhere(2 * p).unwrap();
        assert_eq!(
            v.remap_shared_overwrite(reserved, src, 2 * p),
            Err(PROTECTION_FAILURE)
        );
        assert!(files::pieces(reserved.addr().to_u64(), p).is_none());
        assert!(
            mapped(reserved) && !v.readable(reserved, p),
            "zeros, no access"
        );
        assert!(v.readable(reserved.wrapping_add(p), p), "untouched");

        drop(held);
        files::forget(first, p);
        for (at, len) in [(src, 2), (view, 2), (threes, 1), (reserved, 2)] {
            assert!(v.deallocate(at, len * p).is_success());
        }
    }

    /// The protection set is the one the table answers for reads; a range of no access is not
    /// readable, by the table or by the kernel
    #[test]
    fn readable_follows_the_protection() {
        let v = vm();
        let p = page();
        let a = allocate_aliasable(2 * p).unwrap();
        let anon = v.allocate_anywhere(2 * p).unwrap();
        assert!(v.readable(a, 2 * p) && v.readable(anon, 2 * p) && v.readable(a, 0));
        v.protect_max(a.wrapping_add(p), p, VmProt::NONE).unwrap();
        v.protect_max(anon.wrapping_add(p), p, VmProt::NONE)
            .unwrap();
        assert!(v.readable(a, p) && !v.readable(a, 2 * p));
        assert!(v.readable(anon, p) && !v.readable(anon, 2 * p));
        v.protect_max(a, 2 * p, VmProt::READ).unwrap();
        assert!(v.readable(a, 2 * p));
        assert!(v.deallocate(anon, 2 * p).is_success());
        assert!(!v.readable(anon, 1));
        assert_eq!(v.protect_max(anon, p, VmProt::READ), Err(INVALID_ADDRESS));
        assert!(v.deallocate(a, 2 * p).is_success());
    }

    /// Deallocating nothing succeeds, a wrapping range is an invalid argument, and registered
    /// memory is never unmapped
    #[test]
    fn deallocate_answers_as_mach_and_keeps_registered_memory() {
        let v = vm();
        let a = v.allocate_anywhere(page()).unwrap();
        assert_eq!(v.deallocate(a, 0), KernReturn::SUCCESS);
        assert_eq!(
            v.deallocate(HostAddr::from_addr(usize::MAX & !0xfff), 2 * page()),
            INVALID_ARGUMENT
        );
        let fd = memfd(page());
        let at = file(
            None,
            page(),
            &Backing {
                file: Arc::clone(&fd),
                offset: 0,
            },
            3,
        )
        .unwrap();
        let held = files::Registration::new(at, page(), Backing { fd, offset: 0 });
        let ram = HostAddr::from_addr(at.to_usize());
        assert_eq!(v.deallocate(ram, page()), INVALID_ARGUMENT);
        assert!(mapped(ram));
        drop(held);
        assert!(v.deallocate(ram, page()).is_success());
        assert!(v.deallocate(a, page()).is_success());
    }

    /// A map the kernel refuses for want of memory is `KERN_NO_SPACE`, logged with the process's
    /// mappings and their limit; in a child, whose address space is capped
    #[test]
    fn a_refused_map_is_no_space_with_the_mapping_census() {
        const CHILD: &str = "RVG_LINUX_VM_NO_SPACE_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let cap = libc::rlimit {
                rlim_cur: 1 << 36,
                rlim_max: 1 << 36,
            };
            // SAFETY: lowering this child's own address-space limit.
            assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_AS, &cap) }, 0);
            assert_eq!(vm().allocate_anywhere(1 << 37), Err(NO_SPACE));
            let (mappings, limit) = map_census();
            assert!(mappings.parse::<u64>().unwrap() > 0, "{mappings}");
            assert!(limit.parse::<u64>().unwrap() > 0, "{limit}");
            return;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "os::linux::vm::tests::a_refused_map_is_no_space_with_the_mapping_census",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        assert!(
            String::from_utf8_lossy(&out.stdout).contains(" 1 passed"),
            "{out:?}"
        );
    }

    /// A refusal is for want of mappings where the process holds the limit, or the 2 a map may
    /// add short of it; not below, nor where either count does not read
    #[test]
    fn at_the_limit_is_within_a_maps_two() {
        assert!(at_the_limit("65530", "65530"));
        assert!(at_the_limit("65528", "65530"));
        assert!(!at_the_limit("65527", "65530"));
        assert!(!at_the_limit("?", "65530"));
        assert!(!at_the_limit("65530", "?"));
    }
}
