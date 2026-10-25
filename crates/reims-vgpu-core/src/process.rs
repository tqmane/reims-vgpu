//! [`ProcessState`]: what the library keeps for the life of the process.
//!
//! One per platform instance (`Platform::globals`), so the state is process-wide where the
//! product's is, and a fresh platform in a test is a fresh process. An isolated task's own
//! once-per-task state is its `gpu_task::GpuTask`'s.
//!
//! The state is process-wide, not per device: one serial-number database (a
//! second device's display cannot reuse a serial number in use, `tests::serial_numbers`), the
//! last IOSurface host device created wins (`iosurface`'s
//! `defaults_and_the_last_device_wins`), and one report budget (the first nine resumes that
//! ask get it, `tests::the_report_budget_is_nine_resumes`).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::iosurface::{HostDeviceCore, IoSurfaceHostDevice};
use crate::platform::Platform;

/// The resumes that still report their stamp wait timeouts: the count after the increment must
/// stay below this.
const REPORT_BUDGET: u32 = 10;

/// The process's state, for one platform instance.
pub struct ProcessState<P: Platform> {
    /// The resumes that asked for the report budget, wrapping as a `u32`.
    report_requests: AtomicU32,
    /// The display serial numbers in use. Membership only: never iterated.
    serial_nums: Mutex<Vec<u32>>,
    /// The last IOSurface host device created, and its state, both unretained.
    iosurface_device: Mutex<Option<LatestIoSurfaceDevice<P>>>,
}

/// The last IOSurface host device created, and its state, both unretained.
struct LatestIoSurfaceDevice<P: Platform> {
    device: Weak<IoSurfaceHostDevice<P>>,
    core: Weak<HostDeviceCore<P>>,
}

impl<P: Platform> std::fmt::Debug for ProcessState<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessState")
            .field(
                "report_requests",
                &self.report_requests.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl<P: Platform> Default for ProcessState<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Platform> ProcessState<P> {
    /// A fresh process: no budget spent, no serial number in use, no IOSurface device.
    pub fn new() -> Self {
        ProcessState {
            report_requests: AtomicU32::new(0),
            serial_nums: Mutex::new(Vec::new()),
            iosurface_device: Mutex::new(None),
        }
    }

    /// Whether a resumed child FIFO still reports its stamp wait timeouts: counts this request,
    /// and answers whether the count is still below ten. The count wraps as a `u32`, so the
    /// 2^32nd request after a wrap gets the budget again.
    pub fn take_report_budget(&self) -> bool {
        let count = self
            .report_requests
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1);
        count < REPORT_BUDGET
    }

    /// The serial-number database, behind its mutex: callers hold it across their test and
    /// insert (or remove).
    pub fn serial_num_db(&self) -> &Mutex<Vec<u32>> {
        &self.serial_nums
    }

    /// Records `device`, with `core` its state, as the process's last IOSurface host device
    /// created.
    pub(crate) fn register_iosurface_device(
        &self,
        device: Weak<IoSurfaceHostDevice<P>>,
        core: Weak<HostDeviceCore<P>>,
    ) {
        *self.iosurface_device.lock() = Some(LatestIoSurfaceDevice { device, core });
    }

    /// The process's last IOSurface host device created, retained; `None` once its teardown has
    /// begun.
    pub(crate) fn latest_iosurface_device(&self) -> Option<Arc<IoSurfaceHostDevice<P>>> {
        self.iosurface_device
            .lock()
            .as_ref()
            .and_then(|latest| latest.device.upgrade())
    }

    /// The state of the process's last IOSurface host device created, unretained, while the
    /// device is alive; `None` once its teardown has begun. The device itself is never
    /// retained, so its teardown runs only on the thread that lets go of its last reference.
    pub(crate) fn latest_iosurface_core(&self) -> Option<Weak<HostDeviceCore<P>>> {
        let latest = self.iosurface_device.lock();
        let latest = latest.as_ref()?;
        (latest.device.strong_count() > 0).then(|| latest.core.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::FakePlatform;

    type State = ProcessState<FakePlatform>;

    /// The first nine requests get the budget, the tenth and later do not
    #[test]
    fn the_report_budget_is_nine_resumes() {
        let s = State::new();
        let v: Vec<bool> = (0..11).map(|_| s.take_report_budget()).collect();
        assert_eq!(v.iter().filter(|b| **b).count(), 9);
        assert_eq!(v.get(8), Some(&true));
        assert_eq!(v.get(9), Some(&false));
        assert_eq!(v.get(10), Some(&false));
    }

    /// The count wraps as a `u32`: the request after a wrap gets the budget again
    #[test]
    fn the_report_budget_count_wraps() {
        let s = State::new();
        s.report_requests.store(u32::MAX - 1, Ordering::Relaxed);
        assert!(!s.take_report_budget());
        assert!(s.take_report_budget());
    }

    /// One database per process: what one caller inserts, the next sees
    #[test]
    fn serial_numbers() {
        let s = State::new();
        s.serial_num_db().lock().push(7);
        assert!(s.serial_num_db().lock().contains(&7));
        assert!(!State::new().serial_num_db().lock().contains(&7));
    }
}
