//! Virtual allocations: a heap block cannot be replaced by a Windows file view.

use crate::num::{ToU64, Truncate};
use std::alloc::Layout;
use std::ffi::c_void;
use std::os::windows::fs::OpenOptionsExt as _;
use std::os::windows::io::AsRawHandle as _;
use std::sync::atomic::{AtomicUsize, Ordering};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualAlloc(addr: *mut c_void, len: usize, kind: u32, protection: u32) -> *mut c_void;
    fn VirtualFree(addr: *mut c_void, len: usize, kind: u32) -> i32;
    fn UnmapViewOfFile(addr: *const c_void) -> i32;
    fn DeviceIoControl(
        handle: *mut c_void,
        code: u32,
        input: *const c_void,
        input_len: u32,
        output: *mut c_void,
        output_len: u32,
        returned: *mut u32,
        overlapped: *const c_void,
    ) -> i32;
    fn CreateFileMappingW(
        file: *mut c_void,
        attributes: *const c_void,
        protection: u32,
        high: u32,
        low: u32,
        name: *const u16,
    ) -> *mut c_void;
    fn MapViewOfFile(
        section: *mut c_void,
        access: u32,
        high: u32,
        low: u32,
        length: usize,
    ) -> *mut c_void;
    fn CloseHandle(handle: *mut c_void) -> i32;
}

/// Large fake buffers need readable zero pages, not a pagefile commitment of their whole size.
fn sparse(len: usize) -> Option<*mut u8> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "reims-test-region-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .custom_flags(0x0400_0100)
        .open(path)
        .ok()?; // temporary, delete on last close
    let mut returned = 0;
    // SAFETY: FSCTL_SET_SPARSE has no input/output buffers; the file is newly owned here.
    if unsafe {
        DeviceIoControl(
            file.as_raw_handle(),
            0x0009_00c4,
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            &mut returned,
            std::ptr::null(),
        )
    } == 0
    {
        return None;
    }
    file.set_len(len.to_u64()).ok()?;
    let length = len.to_u64();
    // SAFETY: a private temporary file of the exact size, with no user memory replaced.
    let section = unsafe {
        CreateFileMappingW(
            file.as_raw_handle(),
            std::ptr::null(),
            4,
            (length >> 32).low_u32(),
            length.low_u32(),
            std::ptr::null(),
        )
    };
    if section.is_null() {
        return None;
    }
    // SAFETY: the complete section is mapped read/write; the view keeps its file alive.
    let view = unsafe { MapViewOfFile(section, 2, 0, 0, len) };
    // SAFETY: only the creation handle is closed, and any mapped view retains the section.
    unsafe { CloseHandle(section) };
    (!view.is_null()).then_some(view.cast())
}

/// Windows reserves addresses at 64 KiB boundaries; commit whole blocks for an in-place remap.
pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
    let Some(len) = layout.size().checked_add(0xffff).map(|n| n & !0xffff) else {
        return std::ptr::null_mut();
    };
    if len >= 1 << 30 {
        return sparse(len).unwrap_or(std::ptr::null_mut());
    }
    // SAFETY: a fresh, zero-filled reservation; no caller memory is replaced.
    unsafe { VirtualAlloc(std::ptr::null_mut(), len, 0x3000, 4) }.cast()
}

/// Releases the allocation, or the file view the death harness put at the same address.
pub unsafe fn dealloc(ptr: *mut u8, _layout: Layout) {
    // SAFETY: the region owns this reservation or remapped view, and no other reference remains.
    if unsafe { VirtualFree(ptr.cast(), 0, 0x8000) } == 0 {
        // SAFETY: when VirtualFree refuses it, this region owns a mapped file view instead.
        assert_ne!(unsafe { UnmapViewOfFile(ptr.cast()) }, 0);
    }
}

// The core unit binary tests a real 32-GiB Vec boundary too, not only RawRegion.
// File-backed sparse pages keep that exact addressable span without pagefile commitment.
#[cfg(test)]
struct LargeTestAlloc;

#[cfg(test)]
fn large_test_layout(layout: Layout) -> bool {
    layout.size() >= 1 << 30 && layout.align() <= 0x1_0000
}

#[cfg(test)]
// SAFETY: large allocations have Windows' 64-KiB alignment and are returned to their matching
// view allocator; every other layout is passed unchanged to System. Only the unit binary uses it.
unsafe impl std::alloc::GlobalAlloc for LargeTestAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if large_test_layout(layout) {
            // SAFETY: a new zero-filled view satisfies this layout's size and alignment.
            unsafe { alloc_zeroed(layout) }
        } else {
            // SAFETY: delegated with the caller's valid layout.
            unsafe { std::alloc::System.alloc(layout) }
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if large_test_layout(layout) {
            // SAFETY: sparse file holes read as zeros throughout the mapped range.
            unsafe { alloc_zeroed(layout) }
        } else {
            // SAFETY: delegated with the caller's valid layout.
            unsafe { std::alloc::System.alloc_zeroed(layout) }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if large_test_layout(layout) {
            // SAFETY: the allocation came from alloc_zeroed for this same layout.
            unsafe { dealloc(ptr, layout) }
        } else {
            // SAFETY: this layout was delegated to System when allocated.
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: LargeTestAlloc = LargeTestAlloc;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_32_gib_region_is_zero_filled_without_committing_32_gib_of_pagefile() {
        let layout = Layout::from_size_align(32usize << 30, 4096).unwrap();
        // SAFETY: this test owns the allocation and touches only its first and last bytes.
        let at = unsafe { alloc_zeroed(layout) };
        assert!(!at.is_null());
        // SAFETY: both addresses lie in the newly allocated readable/writable range.
        unsafe {
            assert_eq!(at.read(), 0);
            assert_eq!(at.add(layout.size() - 1).read(), 0);
            at.add(layout.size() - 1).write(7);
            assert_eq!(at.add(layout.size() - 1).read(), 7);
            dealloc(at, layout);
        }
    }
}
