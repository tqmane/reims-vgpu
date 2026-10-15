//! Windows OS services and native Vulkan presentation. Guest views share section pages.

mod api;
mod clock;
mod import;
mod threads;
mod vm;

use ash::vk;
use metal2vulkan_runtime as rt;
use std::ffi::CStr;

pub use super::backing::{GuestMemoryLease, lease_guest_memory};
pub use super::standin::{StandInSurface as WindowsSurface, StandInSurfaces as WindowsSurfaces};
pub(crate) use super::standin::{plane_at, plane_shape, recorded_mut, ycbcr_matrix};
pub use clock::WindowsClock;
pub(crate) use import::{TWINS_READ_ONLY, allocate_zeros, import_view, shown_at};
pub use threads::WindowsThreads;
pub(crate) use vm::OwnMemory;

/// Host pages imported through Windows' validated pointer path, without dma-bufs.
///
/// # Safety
///
/// The pointer meets `rt::HostMemory::new`'s lifetime and accessibility contract.
pub(crate) unsafe fn host_memory(at: std::ptr::NonNull<u8>) -> rt::HostMemory {
    // SAFETY: the caller keeps the checked Windows view alive for the import.
    unsafe { rt::HostMemory::new(at) }
}

/// Windows imports no dma-buf source, so it has no dma-buf refusal.
pub(crate) fn dma_buf_refusal() -> Option<String> {
    None
}
pub use vm::WindowsVm;
#[cfg(test)]
pub(crate) use vm::allocate_aliasable;
#[cfg(test)]
pub(crate) use vm::tests::{PAGE, mapped, writable};

/// The clocks.
pub type OsClock = WindowsClock;
/// The threads.
pub type OsThreads = WindowsThreads;
/// Shared virtual memory views.
pub type OsVm = WindowsVm;
/// An IOSurface stand-in.
pub type OsSurface = WindowsSurface;
/// IOSurface creation.
pub type OsSurfaces = WindowsSurfaces;

pub(crate) const WINDOW_SURFACE_EXTENSIONS: [&CStr; 1] = [ash::khr::win32_surface::NAME];
pub(crate) const WINDOW_DEVICE_EXTENSIONS: [&CStr; 0] = [];
pub(crate) fn open_loader() -> Result<ash::Entry, ash::LoadingError> {
    // SAFETY: ash opens vulkan-1.dll, retained for the entry's life.
    unsafe { ash::Entry::load() }
}

/// No drawable prefetch is required by the Win32 swapchain.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Drawables {}
impl Drawables {
    pub(crate) fn load(_: &rt::Device) -> Option<Self> {
        None
    }
    pub(crate) unsafe fn fetch(&self, _: vk::Image) -> bool {
        match *self {}
    }
}

/// The measured Metal release used by the platform's answers, as on Linux.
pub const METAL_LEVEL: Option<rt::MetalLevel> = Some(rt::MetalLevel::MacOS26_7);
/// No native Metal registry ID exists on Windows.
pub fn default_registry_id() -> Option<u64> {
    None
}
/// No native Metal core count exists on Windows.
pub fn default_gpu_core_count() -> Option<u64> {
    None
}
/// Windows needs no Objective-C service or Unix descriptor-limit change.
pub fn install_process_services() {}

pub(crate) fn physical_memory() -> u64 {
    let mut status = api::MemoryStatus {
        length: u32::try_from(size_of::<api::MemoryStatus>()).unwrap_or(0),
        ..Default::default()
    };
    // SAFETY: a correctly sized, live MEMORYSTATUSEX output.
    if unsafe { api::GlobalMemoryStatusEx(&mut status) } == 0 {
        0
    } else {
        status.total_physical
    }
}
