//! The memory map: the guest's physical ranges, validated and sorted at init, and the
//! translations and mappings made from them.

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::guest::records::APV_PAGE_SIZE;
use crate::host::{HostPtr, PhysRange};
use crate::memory::DeathKind;
use crate::memory::descriptor::MemoryMapDescriptor;
use crate::memory::mapping::{Mapping, Teardown};
use crate::num::{ToU64, ToUsize};
use crate::platform::Platform;
use crate::platform::os::vm::VmProt;

/// One range of the map: `{start, start + length, length, host pointer}`. The field names are
/// ours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MemoryMapRange {
    /// `physicalAddress`: the first guest physical address.
    pub physical_address: u64,
    /// `physicalEnd`: one past the last.
    pub physical_end: u64,
    /// `physicalLength`.
    pub physical_length: u64,
    /// `virtualAddress`: where `physical_address` is mapped here.
    pub virtual_address: HostPtr,
}

/// The guest's physical ranges, sorted at init by start and then length, and looked up first match
/// in stored order. Shared as `Arc`; [`MemoryMap::copy`] gives a new map with the same ranges.
/// Mappings it makes tear down through `P`'s `HostVm`.
pub struct MemoryMap<P: Platform> {
    platform: Arc<P>,
    ranges: Vec<MemoryMapRange>,
    /// Aliases kept for reuse where the platform's cost more than they give
    /// (`HostVm::aliases_guest_in_place`): [`AliasCache`].
    aliases: crate::sync::Mutex<AliasCache<P>>,
}

impl<P: Platform> std::fmt::Debug for MemoryMap<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryMap")
            .field("ranges", &self.ranges)
            .finish_non_exhaustive()
    }
}

/// An address as an integer.
fn addr_u64(p: HostPtr) -> u64 {
    p.addr().to_u64()
}

/// An integer as an address.
fn ptr_u64(x: u64) -> HostPtr {
    HostPtr::from_addr(x.to_usize())
}

impl<P: Platform> MemoryMap<P> {
    /// `mach_vm_remap`s `piece_length` bytes of `map_range`'s host memory at `piece_address` to
    /// `target` (shared, overwrite); only `KERN_SUCCESS` is tested.
    fn map_physical_range(
        &self,
        map_range: &MemoryMapRange,
        piece_address: u64,
        piece_length: u64,
        target: u64,
    ) -> bool {
        // u64 arithmetic, wrapping.
        let source = addr_u64(map_range.virtual_address)
            .wrapping_add(piece_address)
            .wrapping_sub(map_range.physical_address);
        self.platform
            .vm()
            .remap_shared_overwrite(ptr_u64(target), ptr_u64(source), piece_length)
            .is_ok()
    }

    /// The host's map: the checked adds, the sort, the validation and its logs. `None` refuses
    /// it ("Failed to create memory map" is the device's to log). No platform call.
    pub fn new(platform: Arc<P>, descriptor: &MemoryMapDescriptor) -> Option<Arc<Self>> {
        // The descriptor's ranges, in the order the host added them.
        let mut ranges = Vec::with_capacity(descriptor.ranges().len());
        for range in descriptor.ranges() {
            let Some(end) = range.physical_address.checked_add(range.physical_length) else {
                log::warn!(
                    "memory map refused: the range at {:#x} of {:#x} bytes overflows",
                    range.physical_address,
                    range.physical_length
                );
                return None;
            };
            ranges.push(MemoryMapRange {
                physical_address: range.physical_address,
                physical_end: end,
                physical_length: range.physical_length,
                virtual_address: range.host_address,
            });
        }

        // By start, then length: zero-length ranges come before the range with a length that
        // shares their start, whatever order the host listed them in, so the validation below
        // accepts them.
        ranges.sort_by_key(|r| (r.physical_address, r.physical_length));

        let mut end: u64 = 0;
        let mut valid = true;
        for range in &ranges {
            // Only the previous range's end is compared.
            if range.physical_address < end {
                log::warn!(
                    "memory map refused: the range at {:#x} starts before the previous one's end \
                     {end:#x}",
                    range.physical_address
                );
                valid = false;
                break;
            }
            match range.physical_address.checked_add(range.physical_length) {
                Some(e) => end = e,
                None => {
                    log::warn!(
                        "memory map refused: the range at {:#x} of {:#x} bytes overflows",
                        range.physical_address,
                        range.physical_length
                    );
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            return None;
        }

        Some(Arc::new(MemoryMap {
            platform,
            ranges,
            aliases: crate::sync::Mutex::new(AliasCache::default()),
        }))
    }

    /// A plain init (no validation): the same ranges.
    pub fn copy(&self) -> Arc<Self> {
        Arc::new(MemoryMap {
            platform: self.platform.clone(),
            ranges: self.ranges.clone(),
            aliases: crate::sync::Mutex::new(AliasCache::default()),
        })
    }

    /// The ranges, in stored (sorted) order.
    pub fn ranges(&self) -> &[MemoryMapRange] {
        &self.ranges
    }

    /// Remaps each request range into the caller's region at `address`, in request order, then
    /// protects the region; pieces remapped before a failure stay.
    pub fn map_memory_at_virtual_address(
        &self,
        address: HostPtr,
        ranges: &[PhysRange],
        read_only: bool,
    ) -> bool {
        let p = &*self.platform;
        let mut mapped: u64 = 0;
        let mut target = addr_u64(address);
        for request in ranges {
            let mut range = *request;
            // Guest physical 0 is refused, even with length 0.
            if range.physical_address == 0 {
                return false;
            }
            let Some(m) = mapped.checked_add(range.physical_length) else {
                log::warn!(
                    "mapping refused: the ranges' total length overflows at the range at {:#x} of \
                     {:#x} bytes",
                    range.physical_address,
                    range.physical_length
                );
                return false;
            };
            mapped = m;
            if range.physical_length == 0 {
                continue;
            }
            let Some(end) = range.physical_address.checked_add(range.physical_length) else {
                log::warn!(
                    "mapping refused: the range at {:#x} of {:#x} bytes overflows",
                    range.physical_address,
                    range.physical_length
                );
                return false;
            };
            for map_range in &self.ranges {
                if range.physical_address < map_range.physical_address
                    || range.physical_address >= map_range.physical_end
                {
                    continue;
                }
                if map_range.physical_end >= end {
                    if !self.map_physical_range(
                        map_range,
                        range.physical_address,
                        range.physical_length,
                        target,
                    ) {
                        log::error!(
                            "mapping failed: {:#x} bytes of guest memory at {:#x} not remapped",
                            range.physical_length,
                            range.physical_address
                        );
                        return false;
                    }
                    target = target.wrapping_add(range.physical_length);
                    range.physical_length = 0;
                    break;
                }
                let length = map_range.physical_end.wrapping_sub(range.physical_address);
                if !self.map_physical_range(map_range, range.physical_address, length, target) {
                    log::error!(
                        "mapping failed: {length:#x} bytes of guest memory at {:#x} not remapped",
                        range.physical_address
                    );
                    return false;
                }
                target = target.wrapping_add(length);
                range.physical_length = range.physical_length.wrapping_sub(length);
                range.physical_address = map_range.physical_end;
            }
            if range.physical_length != 0 {
                log::warn!(
                    "mapping refused: {:#x} bytes from {:#x} are not in guest memory",
                    range.physical_length,
                    range.physical_address
                );
                return false;
            }
        }

        let prot = if read_only {
            VmProt::READ
        } else {
            VmProt::READ | VmProt::WRITE
        };
        if let Err(kr) = p.vm().protect_max(address, mapped, prot) {
            log::error!(
                "mapping failed: protecting {mapped:#x} bytes at {:#x} returned {:#x}",
                addr_u64(address),
                kr.0
            );
            return false;
        }
        true
    }

    /// A mapping of `length` bytes from `physical_address`, as [`MemoryMap::new_range_mapping`]
    /// of the one range. `read_only` is the host's `uint32_t`.
    pub fn new_contiguous_mapping(
        &self,
        physical_address: u64,
        length: u64,
        read_only: u32,
    ) -> Option<Arc<Mapping<P>>> {
        let range = PhysRange {
            physical_address,
            physical_length: length,
        };
        self.new_range_mapping(&[range], length, read_only)
    }

    /// A fresh VM allocation with each request range remapped into it, in request order; every
    /// refusal after the allocation deallocates it, except a failed protect. A length that is not
    /// whole pages dies.
    pub fn new_range_mapping(
        &self,
        ranges: &[PhysRange],
        total_length: u64,
        read_only: u32,
    ) -> Option<Arc<Mapping<P>>> {
        if !self.platform.vm().aliases_guest_in_place() {
            return self.alias_ranges(ranges, total_length, read_only);
        }
        let key: AliasKey = (
            ranges
                .iter()
                .map(|r| (r.physical_address, r.physical_length))
                .collect(),
            total_length,
            read_only,
        );
        if let Some(mapping) = self.aliases.lock().get(&key) {
            return Some(mapping);
        }
        let mapping = self.alias_ranges(ranges, total_length, read_only)?;
        // A mapping of the map's own (one run, in place) costs nothing to make again.
        if matches!(mapping.teardown(), Teardown::None) {
            return Some(mapping);
        }
        let evicted = self.aliases.lock().insert(key, Arc::clone(&mapping));
        // Unmapped here, outside the lock: an unmap costs as a map does.
        drop(evicted);
        Some(mapping)
    }

    /// [`new_range_mapping`](Self::new_range_mapping)'s mapping, made afresh.
    fn alias_ranges(
        &self,
        ranges: &[PhysRange],
        total_length: u64,
        read_only: u32,
    ) -> Option<Arc<Mapping<P>>> {
        let vm = self.platform.vm();
        // A total length of 0 passes the assert.
        crate::fatal_unless!(
            total_length % const { NonZeroU64::new(APV_PAGE_SIZE).unwrap() } == 0,
            DeathKind::MappingLengthUnaligned
        );
        // One contiguous run of guest pages, inside one range, on a platform whose aliases cost
        // more than they give (`HostVm::aliases_guest_in_place`): the map's own host mapping of
        // them, which the host keeps, so the mapping tears nothing down. Its protection stays
        // the host's: a read-only mapping is only read through.
        if vm.aliases_guest_in_place() {
            let mut runs = ranges.iter().filter(|r| r.physical_length != 0);
            if let (Some(run), None) = (runs.next(), runs.next())
                && run.physical_address != 0
                && run.physical_length == total_length
            {
                let address =
                    self.virtual_address_for_physical(run.physical_address, total_length);
                if !address.is_null() {
                    return Some(Mapping::new(
                        address,
                        total_length,
                        read_only == 0,
                        Teardown::None,
                    ));
                }
            }
        }
        let Ok(address) = vm.allocate_anywhere(total_length) else {
            return None;
        };
        // The whole allocation freed, the result ignored.
        let free_all = || {
            let _ = vm.deallocate(address, total_length);
        };

        let mut mapped: u64 = 0;
        let mut target = addr_u64(address);
        for request in ranges {
            let mut range = *request;
            // Guest physical 0 is refused, even with length 0.
            if range.physical_address == 0 {
                free_all();
                return None;
            }
            let Some(m) = mapped.checked_add(range.physical_length) else {
                free_all();
                return None;
            };
            mapped = m;
            if mapped > total_length {
                free_all();
                return None;
            }
            if range.physical_length == 0 {
                continue;
            }
            // `end` stays the request's original end
            let Some(end) = range.physical_address.checked_add(range.physical_length) else {
                free_all();
                return None;
            };
            // Pieces split across contiguous map ranges, in stored order.
            for map_range in &self.ranges {
                if range.physical_address < map_range.physical_address
                    || range.physical_address >= map_range.physical_end
                {
                    continue;
                }
                let length = map_range.physical_end.wrapping_sub(range.physical_address);
                if map_range.physical_end < end {
                    if !self.map_physical_range(map_range, range.physical_address, length, target) {
                        free_all();
                        return None;
                    }
                    target = target.wrapping_add(length);
                    range.physical_length = range.physical_length.wrapping_sub(length);
                    range.physical_address = map_range.physical_end;
                    continue;
                }
                if !self.map_physical_range(
                    map_range,
                    range.physical_address,
                    range.physical_length,
                    target,
                ) {
                    free_all();
                    return None;
                }
                target = target.wrapping_add(range.physical_length);
                range.physical_length = 0;
                break;
            }
            if range.physical_length != 0 {
                free_all();
                return None;
            }
        }

        if mapped != total_length {
            free_all();
            return None;
        }
        let prot = if read_only != 0 {
            VmProt::READ
        } else {
            VmProt::READ | VmProt::WRITE
        };
        if vm.protect_max(address, total_length, prot).is_err() {
            // Deallocated, then the mapping is made anyway.
            free_all();
        }
        // Writable unless read-only; making the mapping never fails.
        Some(Mapping::new(
            address,
            total_length,
            read_only == 0,
            Teardown::Vm {
                platform: self.platform.clone(),
            },
        ))
    }

    /// Copies `dst.len()` bytes from guest physical `physical`; false when no range holds them.
    pub fn read(&self, physical: u64, dst: &mut [u8]) -> bool {
        let source = self.virtual_address_for_physical(physical, dst.len().to_u64());
        if !source.is_null() && !dst.is_empty() {
            // A copy of 0 bytes touches nothing.
            crate::guest::GuestPtr::new(source).copy_to(0, dst, DeathKind::MapReadFault);
        }
        !source.is_null()
    }

    /// Where `length` bytes from guest physical `physical` are mapped here; NULL when no range
    /// holds them.
    pub fn virtual_address_for_physical(&self, physical: u64, length: u64) -> HostPtr {
        let Some(end) = physical.checked_add(length) else {
            log::warn!("{length:#x} bytes from guest address {physical:#x} overflow");
            return HostPtr::NULL;
        };
        // The first containing range decides; no straddling.
        for range in &self.ranges {
            if physical >= range.physical_address && physical < range.physical_end {
                if end > range.physical_end {
                    log::warn!(
                        "{length:#x} bytes from guest address {physical:#x} run past the end of \
                         their memory map range"
                    );
                    return HostPtr::NULL;
                }
                // u64 arithmetic, wrapping.
                return ptr_u64(
                    physical
                        .wrapping_sub(range.physical_address)
                        .wrapping_add(addr_u64(range.virtual_address)),
                );
            }
        }
        HostPtr::NULL
    }

    /// Copies `src` to guest physical `physical`. `src` is never NULL here, so this always
    /// answers true.
    pub fn write(&self, physical: u64, src: &[u8]) -> bool {
        let destination = self.virtual_address_for_physical(physical, src.len().to_u64());
        // `src` is tested, not `destination`: a NULL translation is copied to, which is the
        // process's death when `src` is not empty.
        if !src.is_empty() {
            crate::guest::GuestPtr::new(destination).copy_from(0, src, DeathKind::MapWriteNull);
        }
        true
    }
}

/// Guest pages read and written through the memory map's own host mapping of guest RAM, one
/// run of physical pages at a time, with no mapping made for them: for platforms where an alias
/// costs more than copying its bytes (`HostVm::aliases_guest_in_place`). Offsets count from the
/// first page of the first run.
pub struct GuestPages<P: Platform> {
    map: Arc<MemoryMap<P>>,
    ranges: Vec<PhysRange>,
    writable: bool,
}

impl<P: Platform> GuestPages<P> {
    /// The runs `ranges` (in order) of `map`'s guest RAM; `writable` where no page is read-only.
    pub fn new(map: Arc<MemoryMap<P>>, ranges: Vec<PhysRange>, writable: bool) -> Self {
        GuestPages {
            map,
            ranges,
            writable,
        }
    }

    /// Whether every page is writable.
    pub fn writable(&self) -> bool {
        self.writable
    }

    /// Visits the parts of `offset..offset + length` run by run, as (guest physical address,
    /// offset into the request, length); false where the runs end first or `visit` refuses.
    fn parts(&self, offset: u64, length: u64, mut visit: impl FnMut(u64, usize, usize) -> bool) -> bool {
        let Some(end) = offset.checked_add(length) else {
            return false;
        };
        let mut first = 0u64;
        for range in &self.ranges {
            let last = first.saturating_add(range.physical_length);
            let (from, to) = (offset.max(first), end.min(last));
            if from < to
                && !visit(
                    range.physical_address.wrapping_add(from - first),
                    (from - offset).to_usize(),
                    (to - from).to_usize(),
                )
            {
                return false;
            }
            first = last;
            if first >= end {
                return true;
            }
        }
        end <= first || length == 0
    }

    /// Copies the pages' bytes from `offset` into `dst`; false where a part has no host address.
    pub fn read(&self, offset: u64, dst: &mut [u8]) -> bool {
        self.parts(offset, dst.len().to_u64(), |physical, at, length| {
            self.map.read(physical, &mut dst[at..at + length])
        })
    }

    /// Copies `src` into the pages from `offset`; false (that part and the rest unwritten) where
    /// a part has no host address.
    pub fn write(&self, offset: u64, src: &[u8]) -> bool {
        self.parts(offset, src.len().to_u64(), |physical, at, length| {
            !self
                .map
                .virtual_address_for_physical(physical, length.to_u64())
                .is_null()
                && self.map.write(physical, &src[at..at + length])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::descriptor::GuestPhysicalRange;
    use crate::platform::os::vm::KernReturn;
    use crate::testkit::calllog::{Arg, Call};
    use crate::testkit::capture::catch_death;
    use crate::testkit::guest_ram::GuestRam;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Map = MemoryMap<FakePlatform>;

    const PAGE: u64 = 0x4000;
    const RW: i32 = 3;
    const RO: i32 = 1;

    fn plat() -> Arc<FakePlatform> {
        Arc::new(FakePlatform::new())
    }

    /// The memory area's lines `logs` took since its last take, in order.
    fn memory_lines(logs: &crate::testkit::logs::Capture) -> Vec<String> {
        logs.take()
            .into_iter()
            .filter(|(_, target, _)| {
                target.starts_with(concat!(env!("CARGO_CRATE_NAME"), "::memory"))
            })
            .map(|(_, _, message)| message)
            .collect()
    }

    fn gpr(pa: u64, len: u64, host: HostPtr) -> GuestPhysicalRange {
        GuestPhysicalRange {
            physical_address: pa,
            physical_length: len,
            host_address: host,
        }
    }

    fn desc(ranges: &[GuestPhysicalRange]) -> MemoryMapDescriptor {
        let mut d = MemoryMapDescriptor::new();
        for r in ranges {
            d.add_range(*r);
        }
        d
    }

    fn fake_host(n: usize) -> HostPtr {
        HostPtr::from_addr(0x1_0000_0000 + n * 0x10_0000)
    }

    /// A map over real fake RAM: one range per `(pa, len)`, adopted by the fake VM.
    fn ram_map(p: &Arc<FakePlatform>, ranges: &[(u64, u64)]) -> (GuestRam, Arc<Map>) {
        let ram = GuestRam::new();
        for (pa, len) in ranges {
            ram.add_range(*pa, *len).unwrap();
        }
        p.fake_vm().adopt(&ram);
        let d = desc(
            &ram.ranges()
                .iter()
                .map(|r| gpr(r.physical_address, r.region.len(), r.host()))
                .collect::<Vec<_>>(),
        );
        let m = Map::new(p.clone(), &d).unwrap();
        (ram, m)
    }

    fn pr(pa: u64, len: u64) -> PhysRange {
        PhysRange {
            physical_address: pa,
            physical_length: len,
        }
    }

    fn vm_calls(p: &FakePlatform) -> Vec<Call> {
        p.call_log().calls_of("mach_vm")
    }

    fn sels(p: &FakePlatform) -> Vec<&'static str> {
        vm_calls(p).iter().map(|c| c.sel).collect()
    }

    fn pattern(n: usize, seed: u8) -> Vec<u8> {
        (0..n)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
            .collect()
    }

    mod new {
        use super::*;

        #[test]
        fn empty_descriptor_is_an_empty_map() {
            let p = plat();
            let m = Map::new(p.clone(), &MemoryMapDescriptor::new()).unwrap();
            assert!(m.ranges().is_empty());
            assert!(p.call_log().is_empty());
        }

        #[test]
        fn sorts_by_start_and_fills_end() {
            let p = plat();
            let d = desc(&[
                gpr(0x10_0000, 0x8000, fake_host(2)),
                gpr(0x4000, 0x4000, fake_host(0)),
                gpr(0x8000, 0x6000, fake_host(1)),
            ]);
            let m = Map::new(p.clone(), &d).unwrap();
            let got: Vec<_> = m
                .ranges()
                .iter()
                .map(|r| {
                    (
                        r.physical_address,
                        r.physical_end,
                        r.physical_length,
                        r.virtual_address,
                    )
                })
                .collect();
            assert_eq!(
                got,
                vec![
                    (0x4000, 0x8000, 0x4000, fake_host(0)),
                    (0x8000, 0xe000, 0x6000, fake_host(1)),
                    (0x10_0000, 0x10_8000, 0x8000, fake_host(2)),
                ]
            );
            // No platform call.
            assert!(p.call_log().is_empty());
        }

        /// The overflow is refused by the first loop, before the sort, so the validation's own
        /// overflow check is dead code.
        #[test]
        fn overflow_in_descriptor_order_before_the_sort() {
            let p = plat();
            let d = desc(&[
                gpr(0x8000, 0x8000, fake_host(0)),
                gpr(0x4000, 0x8000, fake_host(1)), // would be an intersection after the sort
                gpr(u64::MAX, 1, fake_host(2)),
            ]);
            let logs = crate::testkit::logs::capture();
            assert!(Map::new(p.clone(), &d).is_none());
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "memory map refused: the range at 0xffffffffffffffff of 0x1 bytes overflows"
                        .to_owned()
                ]
            );
        }

        #[test]
        fn overflow_to_exactly_zero_is_refused_but_the_top_byte_is_not() {
            let p = plat();
            let d = desc(&[gpr(u64::MAX - 0x3fff, 0x4000, fake_host(0))]);
            assert!(Map::new(p.clone(), &d).is_none());
            let d = desc(&[gpr(u64::MAX - 0x3fff, 0x3fff, fake_host(0))]);
            assert!(Map::new(p.clone(), &d).is_some());
        }

        #[test]
        fn an_intersection_is_refused_with_a_warning() {
            let p = plat();
            let d = desc(&[
                gpr(0x8000, 0x4000, fake_host(1)),
                gpr(0x4000, 0x8000, fake_host(0)),
            ]);
            let logs = crate::testkit::logs::capture();
            assert!(Map::new(p.clone(), &d).is_none());
            assert_eq!(memory_lines(&logs), vec![INTERSECTION.to_owned()]);
        }

        /// The warning for a range at 0x8000 that starts inside the range before it.
        const INTERSECTION: &str =
            "memory map refused: the range at 0x8000 starts before the previous one's end 0xc000";

        #[test]
        fn adjacent_and_duplicate_empty_ranges_pass() {
            let p = plat();
            let d = desc(&[
                gpr(0x4000, 0x4000, fake_host(0)),
                gpr(0x8000, 0x4000, fake_host(1)),
                gpr(0xc000, 0, HostPtr::NULL),
                gpr(0xc000, 0, HostPtr::NULL),
            ]);
            assert_eq!(Map::new(p.clone(), &d).unwrap().ranges().len(), 4);
        }

        /// Zero-length ranges sort before the range with a length that shares their start, in
        /// either descriptor order, so both pass.
        #[test]
        fn a_zero_length_twin_passes_before_or_after() {
            for d in [
                desc(&[
                    gpr(0x8000, 0, fake_host(0)),
                    gpr(0x8000, 0x4000, fake_host(1)),
                ]),
                desc(&[
                    gpr(0x8000, 0x4000, fake_host(1)),
                    gpr(0x8000, 0, fake_host(0)),
                ]),
            ] {
                let p = plat();
                let m = Map::new(p.clone(), &d).unwrap();
                assert_eq!(m.ranges()[0].physical_length, 0);
                assert_eq!(m.ranges()[1].physical_length, 0x4000);
            }
        }

        /// Two ranges with a length that share a start intersect, the shorter sorted first.
        #[test]
        fn two_ranges_with_a_length_sharing_a_start_intersect() {
            let p = plat();
            let d = desc(&[
                gpr(0x8000, 0x8000, fake_host(1)),
                gpr(0x8000, 0x4000, fake_host(0)),
            ]);
            let logs = crate::testkit::logs::capture();
            assert!(Map::new(p.clone(), &d).is_none());
            assert_eq!(memory_lines(&logs), vec![INTERSECTION.to_owned()]);
        }

        /// Any number of ranges: by start, then length, and ties (equal start and length) in
        /// descriptor order.
        #[test]
        fn many_ranges_sort_by_start_then_length() {
            let p = plat();
            let mut input = Vec::new();
            for i in 0..40u64 {
                // Every start twice, the zero-length twin at odd positions only.
                let start = 0x10_0000 + (39 - i) / 2 * 0x10_0000;
                let len = if i % 2 == 0 { 0 } else { 0x4000 };
                input.push(gpr(start, len, fake_host(i as usize)));
            }
            // Two zero-length ranges more at one start, after its range with a length.
            input.push(gpr(0x10_0000, 0, fake_host(40)));
            input.push(gpr(0x10_0000, 0, fake_host(41)));
            let m = Map::new(p.clone(), &desc(&input)).unwrap();
            let got: Vec<_> = m
                .ranges()
                .iter()
                .map(|r| (r.physical_address, r.physical_length, r.virtual_address))
                .collect();
            let mut want: Vec<_> = input
                .iter()
                .map(|r| (r.physical_address, r.physical_length, r.host_address))
                .collect();
            want.sort_by_key(|&(start, len, _)| (start, len));
            assert_eq!(got, want);
            assert_eq!(
                &got[..3],
                &[
                    (0x10_0000, 0, fake_host(38)),
                    (0x10_0000, 0, fake_host(40)),
                    (0x10_0000, 0, fake_host(41)),
                ]
            );
        }
    }

    #[test]
    fn copy_shares_the_ranges_without_validation() {
        let p = plat();
        let d = desc(&[
            gpr(0x8000, 0x8000, fake_host(0)),
            gpr(0x4000, 0x4000, fake_host(1)),
        ]);
        let m = Map::new(p.clone(), &d).unwrap();
        let c = m.copy();
        assert!(!Arc::ptr_eq(&m, &c));
        assert_eq!(m.ranges(), c.ranges());
        assert!(p.call_log().is_empty());
        assert!(format!("{c:?}").contains("MemoryMap"));
    }

    mod virtual_address_for_physical {
        use super::*;

        fn map(p: &Arc<FakePlatform>) -> Arc<Map> {
            let d = desc(&[
                gpr(0x4000, 0x4000, fake_host(0)),
                gpr(0x8000, 0x4000, fake_host(1)),
                gpr(0x10000, 0, fake_host(2)),
                gpr(0x20000, 0x4000, HostPtr::NULL),
            ]);
            Map::new(p.clone(), &d).unwrap()
        }

        #[test]
        fn translates_inside_a_range() {
            let p = plat();
            let m = map(&p);
            assert_eq!(m.virtual_address_for_physical(0x4000, 0x4000), fake_host(0));
            assert_eq!(
                m.virtual_address_for_physical(0x9234, 0x10),
                fake_host(1).wrapping_add(0x1234)
            );
            // Length 0 is fine, also at the last byte.
            assert_eq!(
                m.virtual_address_for_physical(0xbfff, 0),
                fake_host(1).wrapping_add(0x3fff)
            );
        }

        /// Contiguous ranges, still refused.
        #[test]
        fn a_straddle_is_refused_with_a_log() {
            let p = plat();
            let m = map(&p);
            let logs = crate::testkit::logs::capture();
            assert!(m.virtual_address_for_physical(0x7ff0, 0x20).is_null());
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "0x20 bytes from guest address 0x7ff0 run past the end of their memory map \
                     range"
                        .to_owned()
                ]
            );
        }

        #[test]
        fn overflow_is_refused_with_a_log() {
            let p = plat();
            let m = map(&p);
            let logs = crate::testkit::logs::capture();
            assert!(m.virtual_address_for_physical(0x4000, u64::MAX).is_null());
            assert_eq!(
                memory_lines(&logs),
                vec!["0xffffffffffffffff bytes from guest address 0x4000 overflow".to_owned()]
            );
        }

        #[test]
        fn outside_every_range_and_empty_ranges_are_null_silently() {
            let p = plat();
            let m = map(&p);
            let logs = crate::testkit::logs::capture();
            assert!(m.virtual_address_for_physical(0x0, 1).is_null());
            assert!(m.virtual_address_for_physical(0xc000, 1).is_null());
            assert!(m.virtual_address_for_physical(0x10000, 0).is_null());
            assert!(memory_lines(&logs).is_empty());
        }

        #[test]
        fn a_null_host_range_gives_near_null_pointers() {
            let p = plat();
            let m = map(&p);
            assert!(m.virtual_address_for_physical(0x20000, 4).is_null());
            assert_eq!(
                m.virtual_address_for_physical(0x20010, 4),
                HostPtr::from_addr(0x10)
            );
        }
    }

    mod read {
        use super::*;

        #[test]
        fn copies_guest_bytes() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            let mark = p.call_log().mark();
            let bytes = pattern(0x40, 3);
            assert!(ram.write(0x5000, &bytes));
            let mut dst = vec![0u8; 0x40];
            assert!(m.read(0x5000, &mut dst));
            assert_eq!(dst, bytes);
            assert!(p.call_log().since(mark).is_empty());
        }

        #[test]
        fn refusal_leaves_dst_alone() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let mut dst = vec![0xaau8; 8];
            let logs = crate::testkit::logs::capture();
            assert!(!m.read(0x100_0000, &mut dst));
            assert!(!m.read(0x7ffc, &mut dst));
            assert_eq!(dst, vec![0xaa; 8]);
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "0x8 bytes from guest address 0x7ffc run past the end of their memory map \
                     range"
                        .to_owned()
                ]
            );
        }

        #[test]
        fn an_empty_read_inside_a_range_is_yes() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            assert!(m.read(0x4000, &mut []));
            assert!(!m.read(0x1_0000, &mut []));
        }

        #[test]
        fn a_near_null_source_faults_at_the_memcpy() {
            let p = plat();
            let m = Map::new(p, &desc(&[gpr(0x4000, 0x4000, HostPtr::NULL)])).unwrap();
            let e = catch_death(|| {
                let mut dst = [0u8; 4];
                m.read(0x4010, &mut dst)
            });
            assert_eq!(e.unwrap_err(), DeathKind::MapReadFault.site());
        }
    }

    mod write {
        use super::*;
        use crate::raise::death::DeathSite;

        const SITE: DeathSite = DeathKind::MapWriteNull.site();

        #[test]
        fn copies_into_guest_ram() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            let bytes = pattern(0x21, 9);
            assert!(m.write(0x9ff0, &bytes));
            assert_eq!(ram.read_vec(0x9ff0, 0x21).unwrap(), bytes);
        }

        /// Outside every range, the copy goes to NULL.
        #[test]
        fn outside_every_range_copies_to_null() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let e = catch_death(|| m.write(0x100_0000, &[1, 2, 3]));
            assert_eq!(e.unwrap_err(), SITE);
        }

        /// A straddle logs, then copies to NULL.
        #[test]
        fn a_straddle_logs_then_copies_to_null() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0x8000, 0x4000)]);
            let logs = crate::testkit::logs::capture();
            let e = catch_death(|| m.write(0x7ffe, &[1, 2, 3, 4]));
            assert_eq!(e.unwrap_err(), SITE);
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "0x4 bytes from guest address 0x7ffe run past the end of their memory map \
                     range"
                        .to_owned()
                ]
            );
        }

        /// With nothing to copy, true even outside every range.
        #[test]
        fn an_empty_write_is_yes_anywhere() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            assert!(m.write(0x100_0000, &[]));
            assert!(m.write(0x4000, &[]));
        }
    }

    mod new_range_mapping {
        use super::*;

        #[test]
        fn unaligned_total_asserts_before_any_vm_call() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            // Half a guest page.
            let half = APV_PAGE_SIZE / 2;
            let e = catch_death(|| m.new_range_mapping(&[pr(0x4000, half)], half, 0));
            assert_eq!(e.unwrap_err(), DeathKind::MappingLengthUnaligned.site());
            assert!(vm_calls(&p).is_empty());
        }

        /// Length 0 passes the assert and makes an empty mapping.
        #[test]
        fn zero_total_makes_an_empty_mapping() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            let mp = m.new_range_mapping(&[], 0, 0).unwrap();
            assert_eq!(mp.length(), 0);
            assert!(mp.writable());
            assert_eq!(sels(&p), vec!["mach_vm_allocate", "mach_vm_protect"]);
            let calls = vm_calls(&p);
            assert_eq!(
                calls[0].args,
                vec![Arg::U64(0), Arg::Str("ANYWHERE".into())]
            );
            assert_eq!(
                calls[1].args,
                vec![
                    Arg::Ptr(mp.virtual_address().addr()),
                    Arg::U64(0),
                    Arg::Bool(true),
                    Arg::I32(RW)
                ]
            );
        }

        #[test]
        fn allocation_failure_is_nil_and_nothing_else() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            p.script()
                .push("mach_vm_allocate", Step::value(KernReturn(3)));
            let logs = crate::testkit::logs::capture();
            assert!(m.new_range_mapping(&[pr(0x4000, PAGE)], PAGE, 0).is_none());
            assert_eq!(sels(&p), vec!["mach_vm_allocate"]);
            assert!(memory_lines(&logs).is_empty());
        }

        #[test]
        fn maps_shared_guest_ram_and_deallocates_at_last_release() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            let bytes = pattern(0x4000, 1);
            assert!(ram.write(0x8000, &bytes));
            let mp = m.new_range_mapping(&[pr(0x8000, PAGE)], PAGE, 0).unwrap();
            let va = mp.virtual_address();
            assert_eq!(mp.length(), PAGE);
            assert!(mp.writable());
            assert_eq!(p.fake_vm().read(va, PAGE).unwrap(), bytes);
            let calls = vm_calls(&p);
            assert_eq!(
                calls.iter().map(|c| c.sel).collect::<Vec<_>>(),
                vec!["mach_vm_allocate", "mach_vm_remap", "mach_vm_protect"]
            );
            let host = ram.host_addr(0x8000).unwrap();
            assert_eq!(
                calls[1].args,
                vec![Arg::Ptr(va.addr()), Arg::Ptr(host.addr()), Arg::U64(PAGE)]
            );
            assert_eq!(
                calls[2].args,
                vec![
                    Arg::Ptr(va.addr()),
                    Arg::U64(PAGE),
                    Arg::Bool(true),
                    Arg::I32(RW)
                ]
            );
            // Shared: a write through the mapping is guest RAM.
            assert!(p.fake_vm().write(va, &[0xee]));
            p.fake_vm().sync();
            assert_eq!(ram.read_vec(0x8000, 1).unwrap(), vec![0xee]);
            let mark = p.call_log().mark();
            drop(mp);
            let after = p.call_log().since(mark);
            assert_eq!(after.len(), 1);
            assert_eq!(after[0].sel, "mach_vm_deallocate");
            assert_eq!(after[0].args, vec![Arg::Ptr(va.addr()), Arg::U64(PAGE)]);
        }

        #[test]
        fn any_nonzero_read_only_is_read_only() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            let mp = m.new_range_mapping(&[pr(0x4000, PAGE)], PAGE, 2).unwrap();
            assert!(!mp.writable());
            let c = &vm_calls(&p)[2];
            assert_eq!(c.sel, "mach_vm_protect");
            assert_eq!(c.args[3], Arg::I32(RO));
            assert!(!p.fake_vm().is_writable(mp.virtual_address(), PAGE));
        }

        #[test]
        fn request_ranges_concatenate_in_order_with_zero_lengths_skipped() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x10000)]);
            let a = pattern(0x4000, 1);
            let b = pattern(0x4000, 2);
            assert!(ram.write(0xc000, &a));
            assert!(ram.write(0x4000, &b));
            let mp = m
                .new_range_mapping(
                    &[pr(0xc000, PAGE), pr(0x8000, 0), pr(0x4000, PAGE)],
                    2 * PAGE,
                    0,
                )
                .unwrap();
            let va = mp.virtual_address();
            assert_eq!(p.fake_vm().read(va, PAGE).unwrap(), a);
            assert_eq!(p.fake_vm().read(va.wrapping_add(PAGE), PAGE).unwrap(), b);
            let remaps = p.call_log().calls_to("mach_vm_remap");
            assert_eq!(remaps.len(), 2);
            assert_eq!(remaps[1].args[0], Arg::Ptr(va.addr() + PAGE as usize));
        }

        /// A request over contiguous map ranges is split, not refused.
        #[test]
        fn a_request_over_contiguous_ranges_is_split() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0x8000, 0x4000)]);
            let mp = m.new_range_mapping(&[pr(0x6000, 0x4000), pr(0xa000, 0x4000)], 0x8000, 0);
            // 0xa000 + 0x4000 runs past the last range: refused.
            assert!(mp.is_none());
            let mark = p.call_log().mark();
            let bytes = pattern(0x8000, 5);
            assert!(ram.write(0x4000, &bytes[..0x4000]));
            assert!(ram.write(0x8000, &bytes[0x4000..]));
            let mp = m
                .new_range_mapping(&[pr(0x4000, 0x8000)], 0x8000, 0)
                .unwrap();
            let va = mp.virtual_address();
            let remaps: Vec<_> = p
                .call_log()
                .since(mark)
                .into_iter()
                .filter(|c| c.sel == "mach_vm_remap")
                .collect();
            assert_eq!(remaps.len(), 2);
            assert_eq!(
                remaps[0].args,
                vec![
                    Arg::Ptr(va.addr()),
                    Arg::Ptr(ram.host_addr(0x4000).unwrap().addr()),
                    Arg::U64(0x4000)
                ]
            );
            assert_eq!(
                remaps[1].args,
                vec![
                    Arg::Ptr(va.addr() + 0x4000),
                    // `host_addr(0x8000)` would name the first range's end.
                    Arg::Ptr(ram.ranges()[1].host().addr()),
                    Arg::U64(0x4000)
                ]
            );
            assert_eq!(p.fake_vm().read(va, 0x8000).unwrap(), bytes);
        }

        #[test]
        fn a_remainder_with_no_next_range_is_refused_after_the_first_piece() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0xc000, 0x4000)]);
            assert!(
                m.new_range_mapping(&[pr(0x4000, 0x8000)], 0x8000, 0)
                    .is_none()
            );
            assert_eq!(
                sels(&p),
                vec!["mach_vm_allocate", "mach_vm_remap", "mach_vm_deallocate"]
            );
            let c = vm_calls(&p);
            assert_eq!(c[2].args, vec![c[1].args[0].clone(), Arg::U64(0x8000)]);
        }

        /// Guest physical 0, even with length 0.
        #[test]
        fn physical_zero_is_refused_even_empty() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x0, 0x8000)]);
            assert!(m.new_range_mapping(&[pr(0, 0)], 0, 0).is_none());
            assert_eq!(sels(&p), vec!["mach_vm_allocate", "mach_vm_deallocate"]);
            // The same zero-length request elsewhere is skipped.
            assert!(m.new_range_mapping(&[pr(0x4000, 0)], 0, 0).is_some());
        }

        #[test]
        fn physical_zero_is_checked_before_the_total() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            // Both refusals deallocate; the first request range decides which check fires,
            // and neither remaps.
            assert!(m.new_range_mapping(&[pr(0, 0x10_0000)], PAGE, 0).is_none());
            assert_eq!(sels(&p), vec!["mach_vm_allocate", "mach_vm_deallocate"]);
        }

        #[test]
        fn more_than_the_total_is_refused_before_mapping() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            assert!(
                m.new_range_mapping(&[pr(0x4000, PAGE), pr(0x8000, PAGE)], PAGE, 0)
                    .is_none()
            );
            assert_eq!(
                sels(&p),
                vec!["mach_vm_allocate", "mach_vm_remap", "mach_vm_deallocate"]
            );
        }

        #[test]
        fn less_than_the_total_is_refused_after_mapping() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            assert!(
                m.new_range_mapping(&[pr(0x4000, PAGE)], 2 * PAGE, 0)
                    .is_none()
            );
            assert_eq!(
                sels(&p),
                vec!["mach_vm_allocate", "mach_vm_remap", "mach_vm_deallocate"]
            );
            assert_eq!(vm_calls(&p)[2].args[1], Arg::U64(2 * PAGE));
        }

        #[test]
        fn an_end_overflow_is_refused() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            assert!(
                m.new_range_mapping(&[pr(u64::MAX - 0xfff, PAGE)], PAGE, 0)
                    .is_none()
            );
            assert_eq!(sels(&p), vec!["mach_vm_allocate", "mach_vm_deallocate"]);
        }

        #[test]
        fn outside_every_range_is_refused() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            assert!(
                m.new_range_mapping(&[pr(0x10_0000, PAGE)], PAGE, 0)
                    .is_none()
            );
            assert_eq!(sels(&p), vec!["mach_vm_allocate", "mach_vm_deallocate"]);
        }

        #[test]
        fn a_failed_remap_is_refused() {
            for (req, n_ok) in [
                (pr(0x4000, PAGE), 0usize),
                (pr(0x6000, PAGE), 0),
                (pr(0x6000, PAGE), 1),
            ] {
                let p = plat();
                let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0x8000, 0x4000)]);
                if n_ok > 0 {
                    p.script()
                        .push_n("mach_vm_remap", n_ok as u64, Step::Default);
                }
                p.script().push("mach_vm_remap", Step::value(KernReturn(1)));
                assert!(m.new_range_mapping(&[req], PAGE, 0).is_none());
                let s = sels(&p);
                assert_eq!(s.first(), Some(&"mach_vm_allocate"));
                assert_eq!(s.last(), Some(&"mach_vm_deallocate"));
                assert_eq!(
                    s.iter().filter(|x| **x == "mach_vm_remap").count(),
                    n_ok + 1
                );
            }
        }

        #[test]
        fn a_null_host_range_fails_the_remap() {
            let p = plat();
            let m = Map::new(p.clone(), &desc(&[gpr(0x4000, 0x4000, HostPtr::NULL)])).unwrap();
            assert!(m.new_range_mapping(&[pr(0x4000, PAGE)], PAGE, 0).is_none());
            let c = vm_calls(&p);
            assert_eq!(c[1].sel, "mach_vm_remap");
            assert_eq!(c[1].args[1], Arg::Ptr(0));
            assert_eq!(c[2].sel, "mach_vm_deallocate");
        }

        /// A failed protect deallocates, then returns the mapping anyway.
        #[test]
        fn a_failed_protect_still_returns_the_mapping() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
            p.script()
                .push("mach_vm_protect", Step::value(KernReturn(4)));
            let mp = m.new_range_mapping(&[pr(0x4000, PAGE)], PAGE, 1).unwrap();
            assert!(!mp.writable());
            assert_eq!(
                sels(&p),
                vec![
                    "mach_vm_allocate",
                    "mach_vm_remap",
                    "mach_vm_protect",
                    "mach_vm_deallocate"
                ]
            );
            drop(mp);
            assert_eq!(p.call_log().count("mach_vm_deallocate"), 2);
        }

        #[test]
        fn map_physical_range_computes_the_source_with_wrapping() {
            let p = plat();
            let m = Map::new(p.clone(), &MemoryMapDescriptor::new()).unwrap();
            let r = MemoryMapRange {
                physical_address: 0x8000,
                physical_end: 0xc000,
                physical_length: 0x4000,
                virtual_address: HostPtr::from_addr(0x1000),
            };
            assert!(!m.map_physical_range(&r, 0x4000, 0x10, 0x7000));
            let c = vm_calls(&p);
            assert_eq!(
                c[0].args,
                vec![
                    Arg::Ptr(0x7000),
                    Arg::Ptr(0x1000usize.wrapping_sub(0x4000)),
                    Arg::U64(0x10)
                ]
            );
        }
    }

    #[test]
    fn new_contiguous_mapping_is_one_range() {
        let p = plat();
        let (ram, m) = ram_map(&p, &[(0x4000, 0x8000)]);
        let mp = m.new_contiguous_mapping(0x8000, PAGE, 0).unwrap();
        assert!(mp.writable());
        let c = vm_calls(&p);
        assert_eq!(c[0].args[0], Arg::U64(PAGE));
        assert_eq!(
            c[1].args[1],
            Arg::Ptr(ram.host_addr(0x8000).unwrap().addr())
        );
        assert_eq!(c[2].args[3], Arg::I32(RW));
        let mp2 = m.new_contiguous_mapping(0x4000, PAGE, 1).unwrap();
        assert!(!mp2.writable());
        let e = catch_death(|| m.new_contiguous_mapping(0x4000, 1, 0));
        assert_eq!(e.unwrap_err(), DeathKind::MappingLengthUnaligned.site());
    }

    mod map_memory_at_virtual_address {
        use super::*;
        use crate::platform::os::vm::HostVm;

        fn region(p: &FakePlatform, len: u64) -> HostPtr {
            p.fake_vm().unlogged().allocate_anywhere(len).unwrap()
        }

        #[test]
        fn maps_in_order_then_protects_what_was_mapped() {
            let p = plat();
            let (ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0x8000, 0x4000)]);
            let a = pattern(0x8000, 4);
            assert!(ram.write(0x4000, &a[..0x4000]));
            assert!(ram.write(0x8000, &a[0x4000..]));
            let dst = region(&p, 0xc000);
            assert!(m.map_memory_at_virtual_address(
                dst,
                &[pr(0x4000, 0x8000), pr(0x4000, 0)],
                false
            ));
            assert_eq!(p.fake_vm().read(dst, 0x8000).unwrap(), a);
            let c = vm_calls(&p);
            assert_eq!(
                c.iter().map(|c| c.sel).collect::<Vec<_>>(),
                vec!["mach_vm_remap", "mach_vm_remap", "mach_vm_protect"]
            );
            assert_eq!(c[1].args[0], Arg::Ptr(dst.addr() + 0x4000));
            assert_eq!(
                c[2].args,
                vec![
                    Arg::Ptr(dst.addr()),
                    Arg::U64(0x8000),
                    Arg::Bool(true),
                    Arg::I32(RW)
                ]
            );
        }

        #[test]
        fn read_only_protects_read() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            assert!(m.map_memory_at_virtual_address(dst, &[pr(0x4000, PAGE)], true));
            assert_eq!(vm_calls(&p)[1].args[3], Arg::I32(RO));
        }

        /// `newRangeMapping` and `mapMemoryAtVirtualAddress` refuse any request range whose guest
        /// physical address is 0, even a zero-length one
        #[test]
        fn physical_zero_is_refused_silently() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0, 0)], false));
            assert!(vm_calls(&p).is_empty());
            assert!(memory_lines(&logs).is_empty());
        }

        #[test]
        fn an_accumulation_overflow_logs_after_earlier_pieces() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(
                dst,
                &[pr(0x4000, PAGE), pr(0x4000, u64::MAX)],
                false
            ));
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "mapping refused: the ranges' total length overflows at the range at 0x4000 \
                     of 0xffffffffffffffff bytes"
                        .to_owned()
                ]
            );
            // No cleanup of the piece already remapped.
            assert_eq!(sels(&p), vec!["mach_vm_remap"]);
        }

        #[test]
        fn an_end_overflow_logs() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(u64::MAX, 1)], false));
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "mapping refused: the range at 0xffffffffffffffff of 0x1 bytes overflows"
                        .to_owned()
                ]
            );
        }

        #[test]
        fn a_failed_whole_remap_logs() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            p.script().push("mach_vm_remap", Step::value(KernReturn(1)));
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0x4000, PAGE)], false));
            assert_eq!(memory_lines(&logs), vec![NOT_REMAPPED.to_owned()]);
            assert_eq!(sels(&p), vec!["mach_vm_remap"]);
        }

        #[test]
        fn a_failed_piece_remap_logs() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000), (0x8000, 0x4000)]);
            let dst = region(&p, 2 * PAGE);
            p.script().push("mach_vm_remap", Step::value(KernReturn(1)));
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0x4000, 2 * PAGE)], false));
            assert_eq!(memory_lines(&logs), vec![NOT_REMAPPED.to_owned()]);
        }

        /// The error for the first page of guest memory at 0x4000, not remapped.
        const NOT_REMAPPED: &str =
            "mapping failed: 0x4000 bytes of guest memory at 0x4000 not remapped";

        #[test]
        fn an_incomplete_range_logs_and_keeps_its_pieces() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, 2 * PAGE);
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0x4000, 2 * PAGE)], false));
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "mapping refused: 0x4000 bytes from 0x8000 are not in guest memory".to_owned()
                ]
            );
            assert_eq!(sels(&p), vec!["mach_vm_remap"]);
            // Outside every range: nothing mapped at all.
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0x10_0000, 0x10)], false));
            assert_eq!(
                memory_lines(&logs),
                vec![
                    "mapping refused: 0x10 bytes from 0x100000 are not in guest memory".to_owned()
                ]
            );
        }

        #[test]
        fn a_failed_protect_logs_the_kern_return() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            p.script()
                .push("mach_vm_protect", Step::value(KernReturn(-1)));
            let logs = crate::testkit::logs::capture();
            assert!(!m.map_memory_at_virtual_address(dst, &[pr(0x4000, PAGE)], false));
            // `kr` is an `int`, printed in hex as its bits.
            assert_eq!(
                memory_lines(&logs),
                vec![format!(
                    "mapping failed: protecting 0x4000 bytes at {:#x} returned 0xffffffff",
                    dst.addr()
                )]
            );
        }

        #[test]
        fn no_ranges_protects_zero_bytes() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, 0x4000)]);
            let dst = region(&p, PAGE);
            assert!(m.map_memory_at_virtual_address(dst, &[], false));
            assert_eq!(vm_calls(&p)[0].args[1], Arg::U64(0));
        }
    }

    mod guest_pages {
        use super::*;

        fn run(pa: u64, len: u64) -> PhysRange {
            PhysRange {
                physical_address: pa,
                physical_length: len,
            }
        }

        /// Two runs out of order: offsets count through the runs as they come.
        fn scattered(p: &Arc<FakePlatform>) -> (GuestRam, GuestPages<FakePlatform>) {
            let (ram, m) = ram_map(p, &[(0x4000, 4 * PAGE)]);
            let pages = GuestPages::new(m, vec![run(0xc000, PAGE), run(0x4000, PAGE)], true);
            (ram, pages)
        }

        #[test]
        fn a_read_across_runs_follows_their_order() {
            let p = plat();
            let (ram, pages) = scattered(&p);
            assert!(ram.write(0xc000 + PAGE - 3, &[1, 2, 3]));
            assert!(ram.write(0x4000, &[4, 5]));
            let mut got = [0u8; 5];
            assert!(pages.read(PAGE - 3, &mut got));
            assert_eq!(got, [1, 2, 3, 4, 5]);
        }

        #[test]
        fn a_write_across_runs_lands_in_each() {
            let p = plat();
            let (ram, pages) = scattered(&p);
            let bytes = pattern(8, 0x31);
            assert!(pages.write(PAGE - 4, &bytes));
            assert_eq!(ram.read_vec(0xc000 + PAGE - 4, 4).unwrap(), bytes[..4]);
            assert_eq!(ram.read_vec(0x4000, 4).unwrap(), bytes[4..]);
        }

        #[test]
        fn past_the_runs_is_refused() {
            let p = plat();
            let (_ram, pages) = scattered(&p);
            let mut got = [0u8; 2];
            assert!(!pages.read(2 * PAGE - 1, &mut got));
            assert!(!pages.write(2 * PAGE, &[1]));
        }

        #[test]
        fn a_run_outside_the_map_is_refused_without_a_write() {
            let p = plat();
            let (_ram, m) = ram_map(&p, &[(0x4000, PAGE)]);
            let pages = GuestPages::new(m, vec![run(0x100_0000, PAGE)], true);
            assert!(!pages.write(0, &[1, 2, 3]));
        }
    }
}

/// What [`AliasCache`] keys a mapping by: its ranges (physical address and length each, in
/// order), its total length and whether it is read-only.
type AliasKey = (Vec<(u64, u64)>, u64, u32);

/// The most ranges [`AliasCache`] keeps aliased, a view each at most: about 256 MiB of 4 KiB
/// pages, and as many views, which the process holds beside the guest's.
const ALIAS_CACHE_RANGES: usize = 65_536;

/// Mappings of guest pages kept for reuse, where an alias costs more than its pages are worth
/// (`HostVm::aliases_guest_in_place`): core maps a texture's pages afresh for each command
/// buffer that uses them, and on Windows a view made in QEMU's process cost about 10 ms, its
/// unmap as much. A mapping aliases physical pages, so one made for the same ranges shows what
/// they hold now, whenever it was made: reusing it is making it again. The least recently used
/// go first past [`ALIAS_CACHE_RANGES`]; a mapping goes when its last holder drops it.
pub(crate) struct AliasCache<P: Platform> {
    entries: std::collections::HashMap<AliasKey, (Arc<Mapping<P>>, u64)>,
    clock: u64,
    ranges: usize,
    hits: u64,
    misses: u64,
}

impl<P: Platform> Default for AliasCache<P> {
    fn default() -> Self {
        AliasCache {
            entries: std::collections::HashMap::new(),
            clock: 0,
            ranges: 0,
            hits: 0,
            misses: 0,
        }
    }
}

impl<P: Platform> AliasCache<P> {
    /// The mapping kept for `key`, marked used now.
    fn get(&mut self, key: &AliasKey) -> Option<Arc<Mapping<P>>> {
        self.clock = self.clock.wrapping_add(1);
        let clock = self.clock;
        let found = self.entries.get_mut(key).map(|(mapping, used)| {
            *used = clock;
            Arc::clone(mapping)
        });
        match found {
            Some(_) => self.hits = self.hits.wrapping_add(1),
            None => self.misses = self.misses.wrapping_add(1),
        }
        let lookups = self.hits.wrapping_add(self.misses);
        if lookups.is_multiple_of(4096) {
            log::info!(
                "guest page aliases: {} of the last {lookups} lookups reused; {} mappings over {}                  ranges kept",
                self.hits,
                self.entries.len(),
                self.ranges
            );
            self.hits = 0;
            self.misses = 0;
        }
        found
    }

    /// Keeps `mapping` for `key`; answers the mappings it lets go, for the caller to drop
    /// outside the lock.
    fn insert(&mut self, key: AliasKey, mapping: Arc<Mapping<P>>) -> Vec<Arc<Mapping<P>>> {
        let ranges = key.0.len();
        let mut evicted = Vec::new();
        if ranges > ALIAS_CACHE_RANGES {
            return evicted;
        }
        while self.ranges.saturating_add(ranges) > ALIAS_CACHE_RANGES {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some((old, _)) = self.entries.remove(&oldest) {
                self.ranges = self.ranges.saturating_sub(oldest.0.len());
                evicted.push(old);
            }
        }
        self.clock = self.clock.wrapping_add(1);
        if let Some((old, _)) = self.entries.insert(key, (mapping, self.clock)) {
            evicted.push(old);
        } else {
            self.ranges = self.ranges.saturating_add(ranges);
        }
        evicted
    }
}
