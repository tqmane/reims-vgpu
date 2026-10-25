//! The surfaces: the descriptor table and its cached pages, one slot per mapping id, and the
//! three commands that act on a slot, with the teardown of every slot.
//!
//! A mapping id `n` names descriptor `n % per_page` of descriptor page `n / per_page`, where
//! `per_page` is the VM page size over the form's descriptor size. The page's guest address is
//! the descriptor table's entry for it, read and mapped the first time a map reaches it and
//! cached until the table is replaced.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::num::NonZeroU64;

use zerocopy::IntoBytes;

use crate::guest::GuestPtr;
use crate::iosurface::DeathKind;
use crate::iosurface::descriptor::{
    Form, IoSurfaceDescriptorPageTableEntry as Entry, IoSurfaceDeviceDescriptor,
    IoSurfaceDeviceDescriptor2, surface_spec, upgrade_descriptor,
};
use crate::iosurface::device::{HostDeviceCore, QueueState};
use crate::iosurface::ring::IoReturn;
use crate::num::{self, ToU64, ToUsize, Truncate};
use crate::platform::os::surface::{IoSurfaces, Surface, SurfaceSpec};
use crate::platform::{Handle, Platform};
use crate::raise::{Kept, keep_forever};

/// Mapping ids are `1..MAPPING_ID_COUNT`.
pub(super) const MAPPING_ID_COUNT: u64 = 8192;
/// Descriptor pages the device can cache.
pub(super) const PAGE_SLOTS: usize = 4096;
/// The most host-only memory the device makes, in bytes.
pub(super) const HOST_ONLY_LIMIT: u64 = 0x4000_0000;
/// One entry of the descriptor table or of a page table.
pub(super) const TABLE_ENTRY_SIZE: u64 = 4;

/// One mapping id's host state.
pub(super) struct SurfaceSlot<P: Platform> {
    /// The host address of each guest page of the surface; empty for a host-only surface, and
    /// after a failed map or an unmap.
    pub(super) pages: Vec<u64>,
    /// The surface's page count, kept apart from the pages: a host-only surface has a count and
    /// no pages.
    pub(super) page_count: u64,
    /// The descriptor's offset of the surface in its allocation.
    pub(super) base_offset: u64,
    /// The IOSurface: the device's one reference.
    pub(super) surface: Option<P::Surface>,
    /// Made read-only for its clients.
    pub(super) read_only: bool,
    /// Allocated by the host, with no guest pages.
    pub(super) host_only: bool,
}

impl<P: Platform> SurfaceSlot<P> {
    /// A slot that never held a surface.
    pub(super) fn empty() -> Self {
        SurfaceSlot {
            pages: Vec::new(),
            page_count: 0,
            base_offset: 0,
            surface: None,
            read_only: false,
            host_only: false,
        }
    }
}

/// The debug line of a surface `IOSurfaceCreate` refused, describing its properties through
/// `surfaces` only when the line is logged.
fn log_refused_spec<P: Platform>(surfaces: &dyn IoSurfaces<P>, spec: &SurfaceSpec) {
    log::debug!("refused surface properties: {}", surfaces.describe(spec));
}

impl<P: Platform> HostDeviceCore<P> {
    /// Maps the surface at `mapping_id`: Success at once when its slot already holds one;
    /// otherwise the descriptor is read (upgraded from the first form), the surface made, and
    /// the guest descriptor's mapped bit set.
    pub(super) fn map(&self, st: &mut QueueState<P>, mapping_id: u32) -> IoReturn {
        log::debug!("map surface {mapping_id}");
        if mapping_id == 0 || u64::from(mapping_id) >= MAPPING_ID_COUNT {
            log::warn!("map: mapping id {mapping_id} is not in 1..{MAPPING_ID_COUNT}");
            return IoReturn::BadArgument;
        }
        let index = mapping_id.to_usize();
        if st.slots.get(index).is_some_and(|s| s.surface.is_some()) {
            return IoReturn::Success;
        }
        if let Some(s) = st.slots.get_mut(index) {
            *s = SurfaceSlot::empty();
        }
        let id = u64::from(mapping_id);
        let page_index = num::div0(id, st.descriptors_per_page);
        let within = num::rem0(id, st.descriptors_per_page);
        // The table has a page's worth of 4-byte entries. On a 16 KiB page an id below 8192 is
        // on page 511 at most, inside its 4096; on a 4 KiB page the first form's ids stay
        // inside its 1024 (page 1023 at most), and the second form's from 4096 on are past it.
        if page_index >= st.table_len {
            log::warn!(
                "map: surface {mapping_id}'s descriptor page {page_index} is past the table's {} \
                 entries",
                st.table_len
            );
            return IoReturn::BadArgument;
        }
        let page = match self.descriptor_page(st, page_index, mapping_id) {
            Ok(page) => page,
            Err(status) => return status,
        };
        let mut descriptor = IoSurfaceDeviceDescriptor2::default();
        match self.form() {
            Form::Second => {
                // A snapshot of the guest's descriptor.
                let at = page.offset(within.wrapping_mul(Form::Second.size().get()));
                let len = Form::Second.size().get();
                self.mapped_access(at, len, DeathKind::DescriptorRead, || {
                    at.copy_to(0, descriptor.as_mut_bytes(), DeathKind::DescriptorRead);
                });
            }
            Form::First => {
                let mut original = IoSurfaceDeviceDescriptor::default();
                let at = page.offset(within.wrapping_mul(Form::First.size().get()));
                let len = Form::First.size().get();
                self.mapped_access(at, len, DeathKind::DescriptorRead, || {
                    at.copy_to(0, original.as_mut_bytes(), DeathKind::DescriptorRead);
                });
                upgrade_descriptor(&mut descriptor, &original);
            }
        }
        let mut slot = SurfaceSlot::<P>::empty();
        let status = self.make_surface(st, mapping_id, &descriptor, &mut slot);
        let form = self.form();
        let made = status == IoReturn::Success;
        if made && slot.host_only {
            st.host_only_bytes = st
                .host_only_bytes
                .wrapping_add(slot.page_count.wrapping_mul(self.page_size()));
        }
        if !made {
            slot.pages = Vec::new();
        }
        if let Some(s) = st.slots.get_mut(index) {
            *s = slot;
        }
        if made {
            // Read-modify-write of the guest descriptor's first word, in the current form.
            let at = page.offset(within.wrapping_mul(form.size().get()));
            self.mapped_access(at, 4, DeathKind::MappedBitSet, || {
                let site = DeathKind::MappedBitSet;
                let word = at.read_u32(0, site);
                at.write_u32(0, word | Entry::MAPPED, site);
            });
        }
        status
    }

    /// The descriptor page `page_index`: cached, or read from the descriptor table and mapped.
    /// NoMemory when its entry is not valid or the map refuses the page.
    fn descriptor_page(
        &self,
        st: &mut QueueState<P>,
        page_index: u64,
        mapping_id: u32,
    ) -> Result<GuestPtr, IoReturn> {
        let index = page_index.to_usize();
        let Some(&cached) = st.pages.get(index) else {
            // Past the page slots: a VM page above 16 KiB.
            crate::fatal!(DeathKind::PageIndex);
        };
        if let Some(page) = cached {
            return Ok(page);
        }
        // The callers checked the table is mapped.
        let at = st
            .table
            .unwrap_or_default()
            .offset(page_index.wrapping_mul(TABLE_ENTRY_SIZE));
        let entry = Entry(
            self.mapped_access(at, TABLE_ENTRY_SIZE, DeathKind::TableRead, || {
                at.read_u32(0, DeathKind::TableRead)
            }),
        );
        if !entry.valid() {
            log::warn!(
                "map: surface {mapping_id}'s descriptor table entry {page_index} ({:#x}) is not \
                 valid",
                entry.0
            );
            return Err(IoReturn::NoMemory);
        }
        let address = entry.address(self.page_shift());
        let mut maps = self.maps.lock();
        let Some(mapping) = maps.new_contiguous_mapping(address, self.page_size()) else {
            drop(maps);
            log::warn!(
                "map: surface {mapping_id}'s descriptor page at {address:#x} could not be mapped"
            );
            return Err(IoReturn::NoMemory);
        };
        let page = GuestPtr::new(mapping.virtual_address());
        // The slot is empty (a page is cached with its mapping, and dropped with it). A mapping
        // found there is kept for good, still mapped for a reader of it.
        if let Some(old) = maps.pages.get_mut(index).and_then(|s| s.replace(mapping)) {
            keep_forever(old, Kept::StaleReader);
        }
        drop(maps);
        if let Some(s) = st.pages.get_mut(index) {
            *s = Some(page);
        }
        Ok(page)
    }

    /// The surface a descriptor describes, into `slot`: its checks, its pages, its properties
    /// and `IOSurfaceCreate`.
    fn make_surface(
        &self,
        st: &QueueState<P>,
        mapping_id: u32,
        descriptor: &IoSurfaceDeviceDescriptor2,
        slot: &mut SurfaceSlot<P>,
    ) -> IoReturn {
        let page = self.page_size();
        let shift = self.page_shift();
        let base = &descriptor.base;
        if !Entry(base.word0).valid() {
            log::warn!("map: surface {mapping_id}'s descriptor is not valid");
            return IoReturn::BadArgument;
        }
        // Whole pages, wrapping in 64 bits.
        let page_count = num::align_up_wrapping(base.alloc_size, page).wrapping_shr(shift);
        // An upgraded first-form descriptor is never host-only.
        if base.host_only() {
            slot.pages = Vec::new();
            slot.page_count = page_count;
            slot.read_only = false;
            slot.host_only = true;
            let host_only = st
                .host_only_bytes
                .wrapping_add(page_count.wrapping_mul(page));
            if host_only > HOST_ONLY_LIMIT {
                log::warn!(
                    "map: surface {mapping_id} would bring host-only surfaces to {host_only:#x} \
                     bytes, past the limit of {HOST_ONLY_LIMIT:#x}"
                );
                return IoReturn::NoMemory;
            }
        } else if let Err(status) =
            self.walk(mapping_id, descriptor, page_count, (page, shift), slot)
        {
            return status;
        }
        // The cause of a refusal has its own line.
        let Some(spec) = surface_spec(&*self.platform, descriptor, slot) else {
            log::debug!("map: surface {mapping_id}'s properties were refused");
            return IoReturn::NoMemory;
        };
        slot.surface = self.platform.surfaces().create(&spec);
        let Some(surface) = slot.surface.as_ref() else {
            log_refused_spec(self.platform.surfaces(), &spec);
            log::warn!("map: the platform refused to make surface {mapping_id}");
            return IoReturn::BadArgument;
        };
        log::debug!(
            "map: surface {mapping_id} made, IOSurface {:#x}",
            Handle::id(surface).0
        );
        IoReturn::Success
    }

    /// Unmaps the surface at `mapping_id`: Success for an empty slot; Busy while anything else
    /// holds the IOSurface; otherwise the guest descriptor's mapped bit is cleared and the
    /// surface released.
    pub(super) fn unmap(&self, st: &mut QueueState<P>, mapping_id: u32) -> IoReturn {
        log::debug!("unmap surface {mapping_id}");
        if mapping_id == 0 || u64::from(mapping_id) >= MAPPING_ID_COUNT {
            log::warn!("unmap: mapping id {mapping_id} is not in 1..{MAPPING_ID_COUNT}");
            return IoReturn::BadArgument;
        }
        let index = mapping_id.to_usize();
        let Some(retain_count) = st
            .slots
            .get(index)
            .and_then(|s| s.surface.as_ref())
            .map(Surface::retain_count)
        else {
            log::debug!("unmap: surface {mapping_id} is not mapped");
            return IoReturn::Success;
        };
        // Busy while anything else holds the IOSurface.
        if retain_count != 1 {
            log::warn!(
                "unmap: surface {mapping_id} is busy, with {retain_count} references to its \
                 IOSurface"
            );
            return IoReturn::Busy;
        }
        let id = u64::from(mapping_id);
        let page_index = num::div0(id, st.descriptors_per_page);
        // As in `map`: reached only by the second form's ids from 4096 on, on a 4 KiB page.
        if page_index >= st.table_len {
            log::warn!(
                "unmap: surface {mapping_id}'s descriptor page {page_index} is past the table's \
                 {} entries",
                st.table_len
            );
            return IoReturn::BadArgument;
        }
        let Some(&page) = st.pages.get(page_index.to_usize()) else {
            crate::fatal!(DeathKind::PageIndex);
        };
        let Some(page) = page else {
            log::warn!("unmap: surface {mapping_id}'s descriptor page {page_index} is not mapped");
            return IoReturn::BadArgument;
        };
        let within = num::rem0(id, st.descriptors_per_page);
        // Read-modify-write, in the current form.
        let at = page.offset(within.wrapping_mul(self.form().size().get()));
        self.mapped_access(at, 4, DeathKind::MappedBitClear, || {
            let site = DeathKind::MappedBitClear;
            let word = at.read_u32(0, site);
            at.write_u32(0, word & !Entry::MAPPED, site);
        });
        let page_size = self.page_size();
        if let Some(slot) = st.slots.get_mut(index) {
            slot.surface = None;
            if slot.host_only {
                st.host_only_bytes = st
                    .host_only_bytes
                    .wrapping_sub(slot.page_count.wrapping_mul(page_size));
            }
            slot.pages = Vec::new();
        }
        IoReturn::Success
    }

    /// Writes each 8-byte word of the surface's allocation with its own offset, from byte 0 of
    /// the allocation (the base address less the descriptor's offset), under the IOSurface's lock
    /// (whose results are ignored).
    pub(super) fn test_pattern(&self, st: &QueueState<P>, mapping_id: u32) -> IoReturn {
        if mapping_id == 0 || u64::from(mapping_id) >= MAPPING_ID_COUNT {
            log::warn!("test: mapping id {mapping_id} is not in 1..{MAPPING_ID_COUNT}");
            return IoReturn::BadArgument;
        }
        let Some((surface, base_offset)) = st
            .slots
            .get(mapping_id.to_usize())
            .and_then(|s| s.surface.as_ref().map(|surface| (surface, s.base_offset)))
        else {
            log::warn!("test: surface {mapping_id} is not mapped");
            return IoReturn::BadArgument;
        };
        let _ = surface.lock(0);
        let size = surface.alloc_size();
        let base = GuestPtr::new(surface.base_address());
        // The base less the offset, wrapping.
        let start = base.offset(0_u64.wrapping_sub(base_offset));
        for i in 0..size / const { NonZeroU64::new(8).unwrap() } {
            let at = i.wrapping_mul(8);
            start.write_u64(at, at, DeathKind::PatternWrite);
        }
        let _ = surface.unlock(0);
        IoReturn::Success
    }

    /// Releases every surface. The guest descriptors' mapped bits stay set.
    pub(super) fn teardown_surfaces(&self, st: &mut QueueState<P>) {
        log::debug!("releasing every surface");
        let page = self.page_size();
        let mut count: u32 = 0;
        let mut host_only = st.host_only_bytes;
        for (i, slot) in st.slots.iter_mut().enumerate() {
            let Some(surface) = slot.surface.take() else {
                continue;
            };
            if surface.retain_count() != 1 {
                // A second retain count, read only when the debug line is enabled.
                log::debug!(
                    "surface {} is still held elsewhere, with {} references to its IOSurface",
                    i.to_u64().low_u32(),
                    surface.retain_count()
                );
            }
            drop(surface);
            if slot.host_only {
                host_only = host_only.wrapping_sub(slot.page_count.wrapping_mul(page));
            }
            count = count.wrapping_add(1);
            slot.pages = Vec::new();
        }
        st.host_only_bytes = host_only;
        log::debug!("released {count} surfaces");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::host::PhysicalMemoryRange;
    use crate::iosurface::descriptor::element_dimension;
    use crate::iosurface::testing::*;
    use crate::platform::os::surface::{SurfaceComponentSpec, SurfacePlaneSpec};
    use crate::testkit::calllog::Arg;
    use crate::testkit::logs;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    mod map {
        use super::*;

        /// A map's debug lines name the surface, and the IOSurface made for it by its identity in
        /// hex.
        #[test]
        fn mapped_surface_logs_its_iosurface() {
            let r = rig();
            r.setup(true);
            r.root(1);
            r.put_v2(3, &v2(1, ROOT));
            let logs = logs::capture();
            assert_eq!(r.map(3), IoReturn::Success);
            let id = r.surface(3).unwrap().0;
            assert_eq!(
                lines_from(&logs, module_path!()),
                [
                    (log::Level::Debug, "map surface 3".to_owned()),
                    (
                        log::Level::Debug,
                        format!("map: surface 3 made, IOSurface {id:#x}")
                    ),
                ]
            );
        }

        /// Ids 0 and 8192 and above are BadArgument before anything is read.
        #[test]
        fn out_of_range() {
            let r = rig();
            for id in [0, 8192, u32::MAX] {
                assert_eq!(r.map(id), IoReturn::BadArgument);
            }
            assert!(r.created().is_empty());
        }

        /// A guest-backed second-form surface: the properties, the slot, the mapped bit; a second
        /// map is Success without reading the descriptor.
        #[test]
        fn guest_backed_v2() {
            let r = rig();
            r.mapped_v2(3, 2);
            let d = v2(2, ROOT);
            let ranges = vec![
                PhysicalMemoryRange {
                    physical_address: r.host(SURF),
                    physical_length: PAGE,
                },
                PhysicalMemoryRange {
                    physical_address: r.host(SURF + PAGE),
                    physical_length: PAGE,
                },
            ];
            assert_eq!(r.created(), vec![Arg::val(&spec_for(&d, Some(ranges)))]);
            assert_eq!(r.word0_v2(3), d.base.word0 | 2);
            {
                let st = r.dev.state.lock();
                let h = &st.slots[3];
                assert_eq!(h.pages, vec![r.host(SURF), r.host(SURF + PAGE)]);
                assert_eq!(h.page_count, 2);
                assert!(!h.read_only && !h.host_only);
                assert_eq!(st.host_only_bytes, 0);
            }
            r.ram.write_u32(DPAGE + 3 * 0x400, 0);
            r.sync();
            assert_eq!(r.map(3), IoReturn::Success);
            assert_eq!(r.created().len(), 1);
        }

        /// The first form is upgraded: planes, components, widened fields, readOnly as the client
        /// direction, and the mapped bit set in the first form's word.
        #[test]
        fn first_form_upgraded_with_planes() {
            let r = rig();
            r.setup(false);
            r.root(1);
            let mut d = IoSurfaceDeviceDescriptor::default();
            d.base.word0 = pte(ROOT, false);
            d.base.pixel_format = 0x3432_3076;
            d.base.offset = 0;
            d.base.alloc_size = (PAGE / 4 * 3) as u32;
            d.base.word14 = (640 << 8) | 0x82;
            d.base.word18 = (480 << 8) | 1;
            d.base.bytes_per_row = 1280;
            d.base.bytes_per_element = 2;
            d.base.plane_count = 2;
            d.base.bits27 = 0xff;
            d.base.extended_pixels_on_left = 1;
            d.base.field_2c = 0xabcd;
            d.planes[0].base = 0x10;
            d.planes[0].offset = 0x20;
            d.planes[0].size = 0x1000;
            d.planes[0].word14 = (640 << 8) | 1;
            d.planes[0].word18 = (480 << 8) | 1;
            d.planes[0].bytes_per_row = 640;
            d.planes[0].bytes_per_element = 1;
            d.planes[0].component_count = 1;
            d.planes[0].component_bit_depths = [8, 9, 9, 9];
            d.planes[0].extended_pixels_on_bottom = 4;
            d.planes[1].component_count = 2;
            d.planes[1].component_names = [1, 2, 3, 4];
            d.planes[1].compression_footprint = 5;
            r.put_v1(4, &d);
            assert_eq!(r.map(4), IoReturn::Success);
            let created = r.created();
            let mut up = IoSurfaceDeviceDescriptor2::default();
            upgrade_descriptor(&mut up, &d);
            let mut expected = spec_for(
                &up,
                Some(vec![PhysicalMemoryRange {
                    physical_address: r.host(SURF),
                    physical_length: PAGE,
                }]),
            );
            expected.client_direction = Some(2);
            expected.planes = Some(vec![
                SurfacePlaneSpec {
                    base: 0x10,
                    offset: 0x20,
                    width: 640,
                    height: 480,
                    size: 0x1000,
                    bytes_per_row: 640,
                    bytes_per_element: 1,
                    element_width: 1,
                    element_height: 1,
                    extended_pixels_on_bottom: 4,
                    components: vec![SurfaceComponentSpec {
                        bit_depth: 8,
                        ..SurfaceComponentSpec::default()
                    }],
                    ..SurfacePlaneSpec::default()
                },
                SurfacePlaneSpec {
                    compression_footprint: 5,
                    components: vec![
                        SurfaceComponentSpec {
                            name: 1,
                            ..SurfaceComponentSpec::default()
                        },
                        SurfaceComponentSpec {
                            name: 2,
                            ..SurfaceComponentSpec::default()
                        },
                    ],
                    ..SurfacePlaneSpec::default()
                },
            ]);
            assert_eq!(expected.element_width, 4);
            assert_eq!((expected.width, expected.height), (640, 480));
            assert_eq!(created, vec![Arg::val(&expected)]);
            assert_eq!(r.word0_v1(4), pte(ROOT, false) | 2);
            assert!(r.dev.state.lock().slots[4].read_only);
            assert_eq!(element_dimension(0x82), 4);
        }

        /// A table entry that is not valid, or names a page the map refuses: NoMemory.
        #[test]
        fn descriptor_page_refusals() {
            let r = rig();
            r.setup(true);
            r.ram.write_u32(TABLE, 0);
            r.sync();
            assert_eq!(r.map(1), IoReturn::NoMemory);
            r.ram.write_u32(TABLE, pte(HOLE, false));
            r.sync();
            assert_eq!(r.map(1), IoReturn::NoMemory);
            assert!(r.dev.state.lock().pages[0].is_none());
        }

        /// A descriptor without `valid`: BadArgument, and the page stays cached.
        #[test]
        fn invalid_descriptor() {
            let r = rig();
            r.setup(true);
            let mut d = v2(1, ROOT);
            d.base.word0 = 0;
            r.put_v2(2, &d);
            assert_eq!(r.map(2), IoReturn::BadArgument);
            assert!(r.dev.state.lock().pages[0].is_some());
            assert!(r.created().is_empty());
        }

        /// Host-only: no page walk, no ranges, the host-only total grows; past 1 GiB NoMemory.
        #[test]
        fn host_only() {
            let r = rig();
            r.setup(true);
            let mut d = v2(3, HOLE);
            d.base.bits41 = 3; // readOnly is ignored for host-only
            r.put_v2(5, &d);
            assert_eq!(r.map(5), IoReturn::Success);
            assert_eq!(r.created(), vec![Arg::val(&spec_for(&d, None))]);
            assert_eq!(r.dev.state.lock().host_only_bytes, 3 * PAGE);
            let st = r.dev.state.lock();
            let slot = &st.slots[5];
            assert!(slot.host_only && !slot.read_only && slot.pages.is_empty());
            drop(st);
            r.dev.state.lock().host_only_bytes = HOST_ONLY_LIMIT - 3 * PAGE + 1;
            r.put_v2(6, &d);
            assert_eq!(r.map(6), IoReturn::NoMemory);
            assert_eq!(r.created().len(), 1);
        }

        /// `IOSurfaceCreate` refusing: BadArgument, pages freed, no mapped bit.
        #[test]
        fn create_refused() {
            let r = rig();
            r.setup(true);
            r.root(1);
            r.put_v2(1, &v2(1, ROOT));
            r.p.script().push("IOSurfaceCreate", Step::Nil);
            assert_eq!(r.map(1), IoReturn::BadArgument);
            assert_eq!(r.word0_v2(1) & 2, 0);
            assert!(r.surface(1).is_none());
            assert_eq!(r.created().len(), 1);
            assert!(r.dev.state.lock().slots[1].pages.is_empty());
        }

        /// The properties refused (8 planes, or a plane of 5 components): NoMemory; 7 planes and
        /// 4 components pass.
        #[test]
        fn properties_refused() {
            let r = rig();
            r.setup(true);
            r.root(1);
            let mut d = v2(1, ROOT);
            d.base.plane_count = 8;
            r.put_v2(1, &d);
            assert_eq!(r.map(1), IoReturn::NoMemory);
            d.base.plane_count = 7;
            d.planes[6].component_count = 5;
            r.put_v2(1, &d);
            assert_eq!(r.map(1), IoReturn::NoMemory);
            assert!(r.created().is_empty());
            d.planes[6].component_count = 4;
            r.put_v2(1, &d);
            assert_eq!(r.map(1), IoReturn::Success);
        }
    }

    /// The refused properties are described through the platform only for a debug line written:
    /// never while debug lines are off, once for each line written.
    #[test]
    fn refused_spec_is_described_in_its_line() {
        struct Counting(AtomicUsize);
        impl IoSurfaces<FakePlatform> for Counting {
            fn create(&self, _: &SurfaceSpec) -> Option<crate::testkit::surfaces::FakeSurface> {
                None
            }
            fn create_backing(
                &self,
                _: &crate::platform::os::surface::BackingSurfaceSpec,
            ) -> Option<crate::testkit::surfaces::FakeSurface> {
                None
            }
            fn describe(&self, spec: &SurfaceSpec) -> String {
                self.0.fetch_add(1, Ordering::SeqCst);
                format!("<{}x{}>", spec.width, spec.height)
            }
        }
        let surfaces = Counting(AtomicUsize::new(0));
        let spec = SurfaceSpec {
            width: 3,
            height: 4,
            ..SurfaceSpec::default()
        };
        log_refused_spec(&surfaces, &spec);
        assert_eq!(surfaces.0.load(Ordering::SeqCst), 0);
        let logs = logs::capture();
        log_refused_spec(&surfaces, &spec);
        assert_eq!(surfaces.0.load(Ordering::SeqCst), 1);
        assert_eq!(
            lines_from(&logs, module_path!()),
            [(
                log::Level::Debug,
                "refused surface properties: <3x4>".to_owned()
            )]
        );
    }

    mod unmap {
        use super::*;

        /// Out of range is BadArgument; an unmapped id is Success.
        #[test]
        fn range_and_unmapped() {
            let r = rig();
            assert_eq!(r.unmap(0), IoReturn::BadArgument);
            assert_eq!(r.unmap(8192), IoReturn::BadArgument);
            assert_eq!(r.unmap(9), IoReturn::Success);
        }

        /// Busy while another reference exists; then the mapped bit is cleared and the surface
        /// released.
        #[test]
        fn busy_then_released() {
            let r = rig();
            r.mapped_v2(2, 1);
            let held = r.held(2);
            assert_eq!(r.unmap(2), IoReturn::Busy);
            assert!(r.surface(2).is_some());
            assert_eq!(r.word0_v2(2) & 2, 2);
            drop(held);
            let mark = r.p.call_log().mark();
            assert_eq!(r.unmap(2), IoReturn::Success);
            assert_eq!(
                r.p.call_log().sels_since(mark),
                vec!["CFGetRetainCount", "CFRelease"]
            );
            assert_eq!(r.word0_v2(2) & 2, 0);
            assert!(r.surface(2).is_none());
            assert!(r.dev.state.lock().slots[2].pages.is_empty());
        }

        /// A host-only surface gives its pages back; the first form clears its own word.
        #[test]
        fn host_only_and_first_form() {
            let r = rig();
            r.setup(true);
            let mut d = v2(2, HOLE);
            d.base.bits41 = 2;
            r.put_v2(1, &d);
            assert_eq!(r.map(1), IoReturn::Success);
            assert_eq!(r.dev.state.lock().host_only_bytes, 2 * PAGE);
            assert_eq!(r.unmap(1), IoReturn::Success);
            assert_eq!(r.dev.state.lock().host_only_bytes, 0);

            let r = rig();
            r.setup(false);
            r.root(1);
            let mut v1 = IoSurfaceDeviceDescriptor::default();
            v1.base.word0 = pte(ROOT, false);
            v1.base.alloc_size = 0x100;
            r.put_v1(7, &v1);
            assert_eq!(r.map(7), IoReturn::Success);
            assert_eq!(r.word0_v1(7) & 2, 2);
            assert_eq!(r.unmap(7), IoReturn::Success);
            assert_eq!(r.word0_v1(7), pte(ROOT, false));
        }

        /// A descriptor page no longer cached: BadArgument, the surface kept.
        #[test]
        fn page_not_cached() {
            let r = rig();
            r.mapped_v2(1, 1);
            r.dev.state.lock().pages[0] = None;
            assert_eq!(r.unmap(1), IoReturn::BadArgument);
            assert!(r.surface(1).is_some());
        }
    }

    /// The test pattern: each 8-byte word of the allocation holds its own offset, under the
    /// IOSurface's lock; out of range and empty slots are BadArgument.
    #[test]
    fn test_pattern_fills_the_allocation() {
        let r = rig();
        assert_eq!(r.test(0), IoReturn::BadArgument);
        assert_eq!(r.test(8192), IoReturn::BadArgument);
        assert_eq!(r.test(1), IoReturn::BadArgument);
        r.mapped_v2(1, 1);
        let surface = r.held(1);
        let mark = r.p.call_log().mark();
        assert_eq!(r.test(1), IoReturn::Success);
        assert_eq!(
            r.p.call_log().sels_since(mark),
            vec![
                "IOSurfaceLock",
                "IOSurfaceGetAllocSize",
                "IOSurfaceGetBaseAddress",
                "IOSurfaceUnlock"
            ]
        );
        let mem = surface.memory().unwrap();
        for i in [0u64, 1, 100, PAGE / 8 - 1] {
            assert_eq!(mem.read_u64(i * 8), Some(i * 8));
        }
        assert_eq!(surface.lock_count(), 0);
    }

    /// A rig with surface 1 guest-backed and held elsewhere too, and surface 2 host-only: the
    /// rig, and the other holder of surface 1.
    fn two_surfaces_one_held() -> (Rig, <FakePlatform as Platform>::Surface) {
        let r = rig();
        r.mapped_v2(1, 1);
        let mut d = v2(1, HOLE);
        d.base.bits41 = 2;
        r.put_v2(2, &d);
        assert_eq!(r.map(2), IoReturn::Success);
        let held = r.held(1);
        (r, held)
    }

    /// Every surface is released, a held one's retain count read once; the host-only total goes
    /// back; the guest descriptors' mapped bits stay set.
    #[test]
    fn teardown_releases_every_surface() {
        let (r, held) = two_surfaces_one_held();
        let mark = r.p.call_log().mark();
        r.teardown();
        assert_eq!(
            r.p.call_log().sels_since(mark),
            vec![
                "CFGetRetainCount",
                "CFRelease",
                "CFGetRetainCount",
                "CFRelease"
            ]
        );
        assert!(r.surface(1).is_none() && r.surface(2).is_none());
        assert_eq!(r.dev.state.lock().host_only_bytes, 0);
        assert_eq!(r.word0_v2(1) & 2, 2);
        assert_eq!(r.word0_v2(2) & 2, 2);
        // The other holder keeps the IOSurface.
        assert_eq!(Surface::retain_count(&held), 1);
    }

    /// With debug lines on, the teardown names a surface still held elsewhere, reading its retain
    /// count again for the line, and counts the surfaces it released.
    #[test]
    fn teardown_logs_a_held_surface() {
        let (r, _held) = two_surfaces_one_held();
        let logs = logs::capture();
        let mark = r.p.call_log().mark();
        r.teardown();
        assert_eq!(
            r.p.call_log().sels_since(mark),
            vec![
                "CFGetRetainCount",
                "CFGetRetainCount",
                "CFRelease",
                "CFGetRetainCount",
                "CFRelease"
            ]
        );
        assert_eq!(
            lines_from(&logs, module_path!()),
            [
                (log::Level::Debug, "releasing every surface".to_owned()),
                (
                    log::Level::Debug,
                    "surface 1 is still held elsewhere, with 2 references to its IOSurface"
                        .to_owned()
                ),
                (log::Level::Debug, "released 2 surfaces".to_owned()),
            ]
        );
    }
}
