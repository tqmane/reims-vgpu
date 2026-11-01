//! A host range over guest memory, and what its last release does.

use std::sync::Arc;

use crate::host::{HostPtr, HostTask};
use crate::memory::DeathKind;
use crate::platform::Platform;
use crate::seams::links::MappingDeviceLink;

/// What a mapping's last release does.
pub enum Teardown<P: Platform> {
    /// Nothing: the range belongs to someone else.
    None,
    /// `mach_vm_deallocate(virtual_address, length)` through `P`'s `HostVm`: the map made the
    /// region.
    Vm {
        /// The platform whose `vm()` deallocates.
        platform: Arc<P>,
    },
    /// The device unmaps the range from the root task. The link is unretained: a mapping dropped
    /// while the device's fields live unmaps, and they live until the device's teardown ends, so
    /// one dropped during the teardown still unmaps; one dropped after skips it.
    RootTask {
        /// The device.
        device: MappingDeviceLink<P>,
    },
    /// The device's `unmapMemory` of the range in a host task.
    Task {
        /// The device, unretained as for `RootTask`.
        device: MappingDeviceLink<P>,
        /// The range's offset in the task.
        virtual_offset: u64,
        /// The host task.
        task: HostTask,
    },
}

/// A host range over guest memory. Shared as `Arc<Mapping<P>>`; the last drop runs the
/// teardown, so where the last reference goes decides when the host sees the unmap.
pub struct Mapping<P: Platform> {
    virtual_address: HostPtr,
    length: u64,
    writable: bool,
    teardown: Teardown<P>,
}

impl<P: Platform> std::fmt::Debug for Mapping<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapping")
            .field("virtual_address", &self.virtual_address)
            .field("length", &self.length)
            .field("writable", &self.writable)
            .finish_non_exhaustive()
    }
}

impl<P: Platform> Mapping<P> {
    /// A mapping of `length` bytes at `virtual_address`. A root-task or task mapping is always
    /// writable, whatever `writable` says, and a root-task mapping dies at a NULL address, a
    /// length of 0 or a gone device, in that order. A task mapping's `virtual_address` is the
    /// task's base plus its offset (u64, wrapping), which the caller sums.
    pub fn new(
        virtual_address: HostPtr,
        length: u64,
        writable: bool,
        teardown: Teardown<P>,
    ) -> Arc<Self> {
        let writable = match &teardown {
            Teardown::RootTask { device } => {
                crate::fatal_unless!(
                    !virtual_address.is_null(),
                    DeathKind::RootMappingNullAddress
                );
                crate::fatal_unless!(length != 0, DeathKind::RootMappingZeroLength);
                crate::fatal_unless!(device.strong_count() != 0, DeathKind::RootMappingNoDevice);
                true
            }
            Teardown::Task { .. } => true,
            Teardown::None | Teardown::Vm { .. } => writable,
        };
        Arc::new(Mapping {
            virtual_address,
            length,
            writable,
            teardown,
        })
    }

    /// Where the range is mapped.
    pub fn virtual_address(&self) -> HostPtr {
        self.virtual_address
    }

    /// The range's length.
    pub fn length(&self) -> u64 {
        self.length
    }

    /// Whether the range may be written.
    pub fn writable(&self) -> bool {
        self.writable
    }

    /// What tears the range down when the mapping drops.
    pub(crate) fn teardown(&self) -> &Teardown<P> {
        &self.teardown
    }
}

impl<P: Platform> Drop for Mapping<P> {
    fn drop(&mut self) {
        match &self.teardown {
            Teardown::None => {}
            Teardown::Vm { platform } => {
                // The result is ignored.
                let _ = platform.vm().deallocate(self.virtual_address, self.length);
            }
            Teardown::RootTask { device } => {
                // A gone device is skipped. A host `unmapMemory` that raises ends the process
                // at the call.
                if let Some(d) = device.upgrade() {
                    d.unmap_range(self.virtual_address, self.length);
                }
            }
            Teardown::Task {
                device,
                virtual_offset,
                task,
            } => {
                // The host's answer is ignored.
                if let Some(d) = device.upgrade() {
                    let _ = d.unmap_memory_in_task(*task, *virtual_offset, self.length);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::root::rig::{Rig, rig};
    use crate::raise::DeathId;

    /// A host `unmapMemory` that raises ends the process at the call.
    const HOST_UNMAP_RAISED: DeathId = DeathId::HostRaised {
        call: "unmapMemory",
    };
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::platform::FakePlatform;

    type P = FakePlatform;

    /// The device, unretained, as a mapping holds it.
    fn weak_dev(r: &Rig) -> MappingDeviceLink<P> {
        r.device.mapping_link()
    }

    fn gone_dev() -> MappingDeviceLink<P> {
        let r = rig();
        weak_dev(&r)
    }

    /// A device with its root task: the root page is guest page 1.
    fn rooted() -> Rig {
        let r = rig();
        r.device.set_root_page(crate::rt::Hold::NONE, 1);
        r
    }

    /// The root task's handle and base (the host's only task).
    fn root_task(r: &Rig) -> (HostTask, HostPtr) {
        let tasks = r.host.tasks();
        assert_eq!(tasks.len(), 1);
        (tasks[0].task, tasks[0].base)
    }

    mod plain {
        use super::*;

        #[test]
        fn keeps_address_length_and_flag_and_tears_down_nothing() {
            let p = Arc::new(FakePlatform::new());
            for w in [false, true] {
                let m =
                    Mapping::<FakePlatform>::new(HostPtr::from_addr(0x1000), 7, w, Teardown::None);
                assert_eq!(m.virtual_address(), HostPtr::from_addr(0x1000));
                assert_eq!(m.length(), 7);
                assert_eq!(m.writable(), w);
                drop(m);
            }
            assert!(p.call_log().is_empty());
        }

        #[test]
        fn plain_mapping_allows_null_and_zero() {
            let m = Mapping::<FakePlatform>::new(HostPtr::NULL, 0, true, Teardown::None);
            assert!(m.virtual_address().is_null());
            assert_eq!(m.length(), 0);
        }
    }

    mod vm {
        use super::*;

        #[test]
        fn deallocates_at_last_release_only() {
            let p = Arc::new(FakePlatform::new());
            let m = Mapping::new(
                HostPtr::from_addr(0x8000),
                0x4000,
                false,
                Teardown::Vm {
                    platform: p.clone(),
                },
            );
            assert!(!m.writable());
            let m2 = m.clone();
            drop(m);
            assert!(p.call_log().is_empty());
            drop(m2);
            let calls = p.call_log().calls();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].sel, "mach_vm_deallocate");
            assert_eq!(calls[0].args, vec![Arg::Ptr(0x8000), Arg::U64(0x4000)]);
        }
    }

    mod root_task {
        use super::*;

        #[test]
        fn always_writable_and_unmaps_through_the_device() {
            let r = rooted();
            let (task, base) = root_task(&r);
            let m = Mapping::new(
                base.wrapping_add(0x4000),
                0x8000,
                false,
                Teardown::RootTask {
                    device: weak_dev(&r),
                },
            );
            assert!(m.writable());
            let mark = r.mark();
            assert!(r.host_calls(mark).is_empty());
            drop(m);
            let c = r.host_calls(mark);
            assert_eq!(c.len(), 1);
            assert_eq!(c[0].sel, "unmapMemory");
            assert_eq!(
                c[0].args,
                vec![Arg::Ptr(task.addr()), Arg::U64(0x4000), Arg::U64(0x8000)]
            );
        }

        #[test]
        fn a_gone_device_skips_the_unmap() {
            let r = rooted();
            let (_, base) = root_task(&r);
            let d = weak_dev(&r);
            let m = Mapping::new(
                base.wrapping_add(0x4000),
                0x4000,
                true,
                Teardown::RootTask { device: d.clone() },
            );
            let platform = r.platform.clone();
            drop(r);
            assert_eq!(d.strong_count(), 0);
            let mark = platform.call_log().mark();
            drop(m);
            assert!(
                platform
                    .call_log()
                    .since(mark)
                    .iter()
                    .all(|c| c.sel != "unmapMemory")
            );
        }

        #[test]
        fn asserts_in_order_address_length_device() {
            let r = rooted();
            let (_, base) = root_task(&r);
            let d = weak_dev(&r);
            let e = catch_death(|| {
                Mapping::new(
                    HostPtr::NULL,
                    0,
                    true,
                    Teardown::RootTask { device: gone_dev() },
                )
            });
            assert_eq!(e.unwrap_err(), DeathKind::RootMappingNullAddress.site());
            let e = catch_death(|| {
                Mapping::new(
                    HostPtr::from_addr(0x4000),
                    0,
                    true,
                    Teardown::RootTask { device: gone_dev() },
                )
            });
            assert_eq!(e.unwrap_err(), DeathKind::RootMappingZeroLength.site());
            let e = catch_death(|| {
                Mapping::new(
                    HostPtr::from_addr(0x4000),
                    1,
                    true,
                    Teardown::RootTask { device: gone_dev() },
                )
            });
            assert_eq!(e.unwrap_err(), DeathKind::RootMappingNoDevice.site());
            let ok = catch_death(|| {
                Mapping::new(
                    base.wrapping_add(0x4000),
                    1,
                    true,
                    Teardown::RootTask { device: d },
                )
            });
            assert!(ok.is_ok());
        }
    }

    mod task {
        use super::*;

        #[test]
        fn always_writable_and_unmaps_in_the_task() {
            let r = rig();
            let m = Mapping::new(
                HostPtr::from_addr(0x14000),
                0x4000,
                false,
                Teardown::Task {
                    device: weak_dev(&r),
                    virtual_offset: 0x4000,
                    task: HostTask::from_addr(0x99),
                },
            );
            assert!(m.writable());
            let mark = r.mark();
            drop(m);
            let c = r.host_calls(mark);
            assert_eq!(c.len(), 1);
            assert_eq!(c[0].sel, "unmapMemory");
            assert_eq!(
                c[0].args,
                vec![Arg::Ptr(0x99), Arg::U64(0x4000), Arg::U64(0x4000)]
            );
        }

        /// The host's `unmapMemory` block raising ends the process at the release, for both
        /// teardowns
        #[test]
        fn a_raising_unmap_dies_at_the_release() {
            let r = rooted();
            let (_, base) = root_task(&r);
            let task = Mapping::new(
                HostPtr::from_addr(0x14000),
                0x4000,
                false,
                Teardown::Task {
                    device: weak_dev(&r),
                    virtual_offset: 0x4000,
                    task: HostTask::from_addr(0x99),
                },
            );
            r.platform.script().push(
                "unmapMemory",
                crate::testkit::Step::raise("HostRefused", "task unmap"),
            );
            let d = catch_death(move || drop(task)).unwrap_err();
            assert_eq!(d, HOST_UNMAP_RAISED.site());
            let root = Mapping::new(
                base.wrapping_add(0x4000),
                0x4000,
                true,
                Teardown::RootTask {
                    device: weak_dev(&r),
                },
            );
            r.platform.script().push(
                "unmapMemory",
                crate::testkit::Step::raise("HostRefused", "root unmap"),
            );
            let d = catch_death(move || drop(root)).unwrap_err();
            assert_eq!(d, HOST_UNMAP_RAISED.site());
        }

        #[test]
        fn no_asserts_and_a_gone_device_skips_the_unmap() {
            let m = Mapping::<FakePlatform>::new(
                HostPtr::NULL,
                0,
                true,
                Teardown::Task {
                    device: gone_dev(),
                    virtual_offset: 0,
                    task: HostTask::NULL,
                },
            );
            drop(m);
        }
    }
}
