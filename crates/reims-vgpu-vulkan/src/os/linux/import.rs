//! What the runtime imports for host memory on Linux: the memory itself where the process may
//! write all of it, and otherwise a writable twin of it.
//!
//! Linux drivers pin imported pages for writing, so the runtime refuses read-only memory (typed,
//! `HostMemory`), where Metal reads it: the guest's ranges core maps read-only (the EFI
//! framebuffer among them), aliases of a read-only replace or heap, and surfaces the GPU only
//! reads. For those the runtime is given a **twin** instead: a second view of the same file
//! pages, read and write, at an address of its own ([`import_view`]). The GPU then reads what
//! Metal's reads, but its writes reach those pages, where Metal drops them
//! ([`ImportView::writes_kept`]): every command that may write through an object over such an
//! import dies where it is encoded, naming it (`crate::gpu::writes::kept_write`).
//!
//! The block of zeros a read-only alias maps past its ranges' pages is a file of zeros all such
//! aliases share ([`allocate_zeros`]). Its twin is private memory of zeros instead, so the GPU's
//! writes there stay in that twin and never reach the block, though they are kept for the
//! alias's life, where Metal drops them, which the same deaths stop. A detached object lies in
//! no memory (the runtime's detach, `crate::gpu::backing`), so nothing of it is imported.
//!
//! A buffer made over a twin answers the address it was made over as its contents
//! ([`shown_at`]): the runtime's contents are the twin's. The twin goes once the runtime frees the
//! import (its keep-alive, [`ImportView::release`]).
//!
//! Every import also carries a dma-buf source ([`host_memory`]): a device that refuses the pages
//! as host memory, as RADV refuses any page with a file behind it, imports a dma-buf of the same
//! pages instead (`super::dma_buf`). The runtime asks for it only where it would refuse the
//! import otherwise, so every device that imports host memory imports as it did.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::os::fd::OwnedFd;
use std::ptr::NonNull;
use std::sync::Arc;

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, KernReturn};
use reims_vgpu_core::sync::Mutex;

use super::budget;
use super::files;
use super::vm;

/// Whether the runtime imports read-only memory through a writable twin of its files: on Linux,
/// so read-only memory the library makes has a file behind every byte (module docs).
pub(crate) const TWINS_READ_ONLY: bool = true;

/// Every live twin, by its first byte: its length and the first byte of the memory it twins.
static TWINS: Mutex<BTreeMap<u64, (u64, u64)>> = Mutex::new(BTreeMap::new());

/// The files of zeros ([`allocate_zeros`]), held for the process.
static ZERO_FILES: Mutex<Vec<Arc<OwnedFd>>> = Mutex::new(Vec::new());

/// A fresh range of `len` bytes of zeros that a view can be made of, as
/// [`vm::allocate_aliasable`] makes, whose file is kept for the process: an import's twin of it
/// is private zeros, never the file (module docs).
pub(crate) fn allocate_zeros(len: u64) -> Result<HostAddr, KernReturn> {
    let at = vm::allocate_aliasable(len)?;
    if let Some(piece) = files::pieces(at.addr().to_u64(), 1).and_then(|p| p.into_iter().next()) {
        ZERO_FILES.lock().push(piece.backing.file);
    }
    Ok(at)
}

/// Whether `fd` is a file of zeros.
pub(super) fn zeros(fd: &Arc<OwnedFd>) -> bool {
    ZERO_FILES.lock().iter().any(|z| Arc::ptr_eq(z, fd))
}

/// The memory the runtime imports for `length` bytes at `at` (module docs).
#[derive(Debug)]
pub(crate) struct ImportView {
    at: NonNull<u8>,
    twin: Option<Twin>,
}

/// A writable twin's address range, unmapped when it drops.
#[derive(Debug)]
struct Twin {
    first: u64,
    length: u64,
    /// The kernel's mappings it takes, counted in the budget once it is whole.
    mappings: u64,
}

impl Drop for Twin {
    fn drop(&mut self) {
        TWINS.lock().remove(&self.first);
        // SAFETY: the twin's own mapping, which the runtime let go of (it freed the import and
        // dropped its keep-alive, or the call that would import it was refused).
        unsafe { libc::munmap(vm::ptr(self.first), self.length.to_usize()) };
        budget::twins_hold(self.mappings, false);
    }
}

impl ImportView {
    /// The first byte the runtime imports.
    pub(crate) fn at(&self) -> NonNull<u8> {
        self.at
    }

    /// Whether the GPU's writes through the import are kept where Metal drops them: it is a twin
    /// of memory mapped read-only, the guest's pages or zeros (module docs).
    pub(crate) fn writes_kept(&self) -> bool {
        self.twin.is_some()
    }

    /// The runtime's deallocator of the import, and what keeps the import's memory mapped beside
    /// the bytes themselves. With a twin: `inner`, given `original`, the address the bytes were
    /// given at, and the twin, which the runtime drops once it freed the import, so the twin is
    /// unmapped after it. Without one, `inner` as it is, and nothing.
    pub(crate) fn release(
        self,
        original: NonNull<u8>,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        let Some(twin) = self.twin else {
            return (inner, None);
        };
        let original = SendPtr(original);
        let inner = inner.map(|inner| -> rt::Deallocator {
            Box::new(move |_, length| inner(original.get(), length))
        });
        (inner, Some(Box::new(twin)))
    }
}

/// The runtime's host memory at `at`, the first byte of an import ([`ImportView::at`]), with a
/// source of dma-bufs of its pages for a device that refuses them as host memory (module docs).
/// The runtime keeps the source while the import lives, and calls it with no lock of its own
/// held, with a window of the import's pages and the granule it binds chunks at: the whole
/// import's when it makes it, and a texture's own pages, at a granule of their length, when it
/// makes a texture over it. The source answers dma-bufs that are the window's pages in order, each
/// within udmabuf's limits (`super::dma_buf`), which the runtime imports. A twin's window is the
/// twinned memory's pages ([`shown_at`]), which a dma-buf takes whatever their protection. A call
/// that makes none keeps its refusal on its thread for the death there ([`dma_buf_refusal`]), as
/// the runtime then refuses the import as it would have, or makes the texture as it does where
/// it cannot alias the memory.
///
/// # Safety
///
/// `rt::HostMemory::new`'s contract, for the memory at `at`.
pub(crate) unsafe fn host_memory(at: NonNull<u8>) -> rt::HostMemory {
    // An earlier import's refusal is not this one's.
    REFUSAL.with_borrow_mut(|kept| *kept = None);
    let source: rt::DmaBufSource = Arc::new(|request: rt::DmaBufRequest| {
        let first = shown_at(request.start).as_ptr().addr().to_u64();
        let (length, granule) = (request.length, request.granule);
        match super::dma_buf::of_pages(first, length, granule) {
            Ok(chunks) => {
                log::debug!(
                    "{length:#x} bytes at {first:#x} as {} dma-bufs at a granule of {granule:#x}",
                    chunks.len()
                );
                Ok(chunks)
            }
            Err(refusal) => {
                let line = format!("no dma-buf of its {length:#x} bytes at {first:#x}: {refusal}");
                log::debug!("{line}");
                REFUSAL.with_borrow_mut(|kept| *kept = Some(line));
                Err(std::io::Error::other(refusal.to_string()))
            }
        }
    });
    // SAFETY: the caller's.
    unsafe { rt::HostMemory::with_dma_buf(at, source) }
}

thread_local! {
    /// The last refusal of a dma-buf source on this thread ([`host_memory`]).
    static REFUSAL: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Why this thread's last dma-buf source made none, taken, for the death of an import the runtime
/// refused after it ([`host_memory`]).
pub(crate) fn dma_buf_refusal() -> Option<String> {
    REFUSAL.with_borrow_mut(Option::take)
}

/// An address moved into a deallocator, which may run on any thread.
struct SendPtr(NonNull<u8>);

// SAFETY: an address only, handed back to the deallocator it was given for.
unsafe impl Send for SendPtr {}

impl SendPtr {
    /// The address.
    fn get(self) -> NonNull<u8> {
        self.0
    }
}

/// The memory the runtime imports for `length` bytes at `at`: those bytes where every page of
/// them is mapped writable or one has no file behind it (which the runtime then refuses as it
/// refuses any import there), and otherwise a writable twin of their pages, at the same offset in
/// its first page (module docs).
pub(crate) fn import_view(at: NonNull<u8>, length: u64) -> ImportView {
    let plain = ImportView { at, twin: None };
    let start = at.as_ptr().addr().to_u64();
    let page = vm::page_size();
    let first = start & !(page - 1);
    let Some(end) = start
        .checked_add(length.max(1))
        .and_then(|e| e.checked_next_multiple_of(page))
    else {
        return plain;
    };
    let Some(pieces) = files::pieces(first, end - first) else {
        return plain;
    };
    log::debug!(
        "an import of {length:#x} bytes at {start:#x}: {} runs of files",
        twin_mappings(&pieces)
    );
    if pieces.iter().all(|p| p.prot & libc::PROT_WRITE != 0) {
        return plain;
    }
    let Some(twin) = twin(first, end - first, &pieces) else {
        return plain;
    };
    let Some(twin_at) = address(twin.first + (start - first)) else {
        return plain;
    };
    ImportView {
        at: twin_at,
        twin: Some(twin),
    }
}

/// A writable twin of the `length` bytes from `first` that `pieces` show: each piece's file
/// pages, shared, read and write, but zeros' (private zeros); `None` where the host maps none.
fn twin(first: u64, length: u64, pieces: &[files::Piece]) -> Option<Twin> {
    let mappings = twin_mappings(pieces);
    budget::admit(mappings, "an import's twin");
    let base = vm::anonymous(None, length, "an import's twin").ok()?;
    let mut twin = Twin {
        first: base,
        length,
        mappings: 0,
    };
    let rw = libc::PROT_READ | libc::PROT_WRITE;
    for piece in pieces {
        let at = Some(base + piece.at);
        if zeros(&piece.backing.file) {
            // The reservation is private zeros already: mapped again, so no earlier bytes stay.
            vm::anonymous(at, piece.length, "an import's twin").ok()?;
        } else {
            vm::file(at, piece.length, &piece.backing, rw).ok()?;
        }
    }
    budget::twins_hold(mappings, true);
    twin.mappings = mappings;
    TWINS.lock().insert(base, (length, first));
    Some(twin)
}

/// The kernel's mappings a twin of `pieces` takes: one for each run of them that maps as one,
/// pieces of zeros next to each other (private zeros, as the twin's reservation is) or each piece
/// following the one before in the same file (all read and write in the twin).
fn twin_mappings(pieces: &[files::Piece]) -> u64 {
    let zeros: Vec<bool> = pieces.iter().map(|p| zeros(&p.backing.file)).collect();
    let apart = (1..pieces.len())
        .filter(|&i| {
            let (a, b) = (&pieces[i - 1], &pieces[i]);
            let together = if zeros[i - 1] || zeros[i] {
                zeros[i - 1] && zeros[i]
            } else {
                Arc::ptr_eq(&a.backing.file, &b.backing.file)
                    && a.backing.offset + a.length == b.backing.offset
            };
            !together
        })
        .count();
    apart.to_u64() + 1
}

/// The address whose bytes the runtime's memory at `at` shows: `at`, but for an address in a
/// writable twin, the twinned memory's at the same offset (module docs).
pub(crate) fn shown_at(at: NonNull<u8>) -> NonNull<u8> {
    let addr = at.as_ptr().addr().to_u64();
    let twins = TWINS.lock();
    let Some((&twin, &(length, first))) = twins.range(..=addr).next_back() else {
        return at;
    };
    if addr >= twin + length {
        return at;
    }
    address(first + (addr - twin)).unwrap_or(at)
}

/// A pointer to `at` in the process's address space; `None` for 0.
fn address(at: u64) -> Option<NonNull<u8>> {
    NonNull::new(vm::ptr(at).cast::<u8>())
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::os::vm::{HostVm as _, VmProt};

    use super::super::LinuxVm;
    use super::*;

    const PAGE: u64 = 0x1000;

    fn read(at: u64) -> u8 {
        // SAFETY: test memory this test mapped readable.
        unsafe { address(at).unwrap().as_ptr().read_volatile() }
    }

    fn write(at: u64, v: u8) {
        // SAFETY: test memory this test mapped writable.
        unsafe { address(at).unwrap().as_ptr().write_volatile(v) }
    }

    fn addr(p: NonNull<u8>) -> u64 {
        p.as_ptr().addr().to_u64()
    }

    /// Writable memory is imported as it is; read-only memory through a writable twin of its
    /// pages, at the same offset, whose writes reach them, so the import keeps writes, and whose
    /// addresses answer the memory's own until the runtime drops its keep-alive; the deallocator
    /// is given the memory's own address, and the twin outlives it
    #[test]
    fn read_only_memory_is_imported_through_a_writable_twin() {
        let vm = LinuxVm::new();
        let at = vm::allocate_aliasable(2 * PAGE).unwrap();
        let first = at.addr().to_u64();
        let p = address(first + 0x40).unwrap();
        let view = import_view(p, PAGE);
        assert_eq!(view.at(), p, "writable: the memory itself");
        assert!(!view.writes_kept());
        let (run, kept) = view.release(p, None);
        assert!(run.is_none() && kept.is_none());
        assert!(vm.protect_max(at, 2 * PAGE, VmProt::READ).is_ok());
        let view = import_view(p, PAGE);
        assert!(view.writes_kept());
        let twin = addr(view.at());
        assert_ne!(twin, addr(p));
        assert_eq!(twin & (PAGE - 1), 0x40, "the offset in its page");
        write(twin + PAGE, 0x5a);
        assert_eq!(
            read(first + 0x40 + PAGE),
            0x5a,
            "the twin's writes reach the pages"
        );
        assert_eq!(shown_at(view.at()), p);
        let inside = address(twin + 0x10).unwrap();
        assert_eq!(addr(shown_at(inside)), addr(p) + 0x10);
        let given = Arc::new(Mutex::new(None));
        let seen = Arc::clone(&given);
        let inner: rt::Deallocator = Box::new(move |at, length| {
            *seen.lock() = Some((at.as_ptr().addr().to_u64(), length));
        });
        let (free, kept) = view.release(p, Some(inner));
        free.unwrap()(address(twin).unwrap(), PAGE);
        assert_eq!(*given.lock(), Some((addr(p), PAGE)));
        assert_eq!(
            addr(shown_at(inside)),
            addr(p) + 0x10,
            "the twin outlives it"
        );
        drop(kept);
        assert_eq!(addr(shown_at(inside)), twin + 0x10, "the twin is gone");
        assert!(vm.deallocate(at, 2 * PAGE).is_success());
    }

    /// A twin takes a mapping for each run of its pieces that maps as one, following pieces of
    /// one file or neighbouring pieces of zeros, and the kernel maps it so
    #[test]
    fn a_twin_counts_its_mappings_as_the_kernel_maps_them() {
        let f = vm::memfd(4 * PAGE);
        let zero_at = allocate_zeros(PAGE).unwrap();
        let z = files::pieces(zero_at.addr().to_u64(), 1).unwrap()[0]
            .backing
            .file
            .clone();
        let piece = |at: u64, fd: &Arc<OwnedFd>, offset: u64| files::Piece {
            at: at * PAGE,
            length: PAGE,
            backing: files::Backing {
                file: Arc::clone(fd),
                offset: offset * PAGE,
            },
            prot: libc::PROT_READ,
        };
        assert_eq!(twin_mappings(&[piece(0, &f, 0), piece(1, &f, 1)]), 1);
        assert_eq!(twin_mappings(&[piece(0, &f, 1), piece(1, &f, 0)]), 2);
        assert_eq!(twin_mappings(&[piece(0, &z, 0), piece(1, &z, 5)]), 1);
        let mixed = [piece(0, &f, 0), piece(1, &z, 0), piece(2, &f, 2)];
        assert_eq!(twin_mappings(&mixed), 3);
        // A read-only view of the file's pages 2, 3, 0 and 1: its twin maps as two.
        let vm = LinuxVm::new();
        let source = vm::allocate_aliasable(4 * PAGE).unwrap();
        let view = vm.allocate_anywhere(4 * PAGE).unwrap();
        for (to, from) in [(0, 2), (1, 3), (2, 0), (3, 1)] {
            vm.remap_shared_overwrite(
                view.wrapping_add(to * PAGE),
                source.wrapping_add(from * PAGE),
                PAGE,
            )
            .unwrap();
        }
        assert!(vm.protect_max(view, 4 * PAGE, VmProt::READ).is_ok());
        let p = address(view.addr().to_u64()).unwrap();
        let import = import_view(p, 4 * PAGE);
        let twin = addr(import.at());
        assert_ne!(twin, addr(p));
        assert_eq!(vm::tests::kernel_mappings(twin, twin + 4 * PAGE), 2);
        drop(import);
        assert!(vm.deallocate(view, 4 * PAGE).is_success());
        assert!(vm.deallocate(source, 4 * PAGE).is_success());
        assert!(vm.deallocate(zero_at, PAGE).is_success());
    }

    /// A twin of zeros is private zeros: its writes never reach the shared file
    #[test]
    fn a_twin_of_zeros_keeps_its_writes_to_itself() {
        let vm = LinuxVm::new();
        let at = allocate_zeros(PAGE).unwrap();
        let first = at.addr().to_u64();
        assert!(zeros(&files::pieces(first, 1).unwrap()[0].backing.file));
        assert!(vm.protect_max(at, PAGE, VmProt::READ).is_ok());
        let p = address(first).unwrap();
        let view = import_view(p, PAGE);
        let twin = addr(view.at());
        assert_ne!(twin, first);
        assert_eq!(read(twin), 0);
        write(twin, 7);
        assert_eq!((read(twin), read(first)), (7, 0));
        drop(view);
        assert!(vm.deallocate(at, PAGE).is_success());
    }

    /// A read-only alias of guest pages shorter than its size, a heap's or a replace's, is
    /// imported through a writable twin: the pages' bytes, then zeros of the twin's own past
    /// them, as its zeros map the block of zeros
    #[test]
    fn a_read_only_alias_with_zeros_past_its_pages_is_twinned() {
        use reims_vgpu_core::platform::types::ResourceAddressRange;

        let vm = LinuxVm::new();
        let guest = vm::allocate_aliasable(PAGE).unwrap();
        let first = guest.addr().to_u64();
        write(first + 5, 0x6b);
        let ranges = [ResourceAddressRange {
            address: first,
            length: PAGE,
        }];
        let alias = crate::gpu::backing::gather("test", &ranges, 3 * PAGE, true).unwrap();
        let at = alias.first();
        let view = import_view(at, 3 * PAGE);
        let twin = addr(view.at());
        assert_ne!(twin, addr(at), "a twin");
        assert_eq!(read(twin + 5), 0x6b, "the guest's page");
        assert_eq!((read(twin + PAGE), read(twin + 3 * PAGE - 1)), (0, 0));
        write(twin + 2 * PAGE, 9);
        assert_eq!(
            read(addr(at) + 2 * PAGE),
            0,
            "the alias's zeros keep none of it"
        );
        drop(view);
        drop(alias);
        assert!(vm.deallocate(guest, PAGE).is_success());
    }

    /// Readable pages with no file behind them, which no view is made of, end the process at the
    /// call that gathers them into an alias, where Metal takes them
    #[test]
    fn a_gather_of_pages_without_a_file_dies() {
        use reims_vgpu_core::platform::types::ResourceAddressRange;
        use reims_vgpu_core::raise::death::DeathKind;
        use reims_vgpu_core::testkit::capture::catch_death;

        let vm = LinuxVm::new();
        let at = vm.allocate_anywhere(PAGE).unwrap();
        let ranges = [ResourceAddressRange {
            address: at.addr().to_u64(),
            length: PAGE,
        }];
        let gathered = || crate::gpu::backing::gather("test", &ranges, PAGE, false).is_some();
        let site = catch_death(gathered).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "test"));
        assert!(vm.deallocate(at, PAGE).is_success());
    }

    /// Memory with no file behind it is given as it is, for the runtime to refuse
    #[test]
    fn memory_without_a_file_is_imported_as_it_is() {
        let vm = LinuxVm::new();
        let at = vm.allocate_anywhere(PAGE).unwrap();
        assert!(vm.protect_max(at, PAGE, VmProt::READ).is_ok());
        let p = NonNull::new(at.as_ptr::<u8>()).unwrap();
        assert_eq!(import_view(p, PAGE).at(), p);
        assert!(vm.deallocate(at, PAGE).is_success());
    }
}
