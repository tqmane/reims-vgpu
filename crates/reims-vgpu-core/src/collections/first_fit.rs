//! [`FirstFitVa`]: first fit over a sorted free list, the root task's virtual address space.
//!
//! Its offsets reach the host as the `voff` of the root task's `mapMemory` and `unmapMemory`.
//! Every size rounds up to the granule. An allocation that would round or end past 2^64 fits
//! nowhere and fails, as one larger than every free range does; a free rounds with wrapping `u64`
//! arithmetic. A free coalesces with both neighbours, and nothing checks that a freed range was
//! allocated. The device never resets it:
//! a range that leaks at a root teardown stays allocated, and every later offset moves past it.

/// One free range, `start..=last`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeRange {
    /// The first free byte.
    pub start: u64,
    /// The last free byte.
    pub last: u64,
}

/// First fit over the free ranges of `0..=last`, in granules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstFitVa {
    /// The free ranges, sorted and disjoint.
    free: Vec<FreeRange>,
    /// `granule - 1`, or 0 for a granule of 0: every size and start rounds with it.
    mask: u64,
}

impl FirstFitVa {
    /// One free range, `0..=last`, allocated in `granule`s (0 or 1 for bytes).
    pub fn new(last: u64, granule: u64) -> Self {
        FirstFitVa {
            free: vec![FreeRange { start: 0, last }],
            mask: granule.saturating_sub(1),
        }
    }

    /// The start of the first free range that holds `size` rounded up to the granule, at its
    /// first granule boundary. A size of 0 fails, and so does one that rounds past 2^64; a range
    /// whose boundary or block end would lie past 2^64 does not hold it. A failure changes
    /// nothing.
    pub fn allocate(&mut self, size: u64) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let mask = self.mask;
        let asz = size.checked_add(mask)? & !mask;
        let (i, range, al, end) = self.free.iter().enumerate().find_map(|(i, r)| {
            let al = r.start.checked_add(mask)? & !mask;
            // `asz` is at least `size`, so at least 1.
            let end = al.checked_add(asz.wrapping_sub(1))?;
            (end <= r.last).then_some((i, *r, al, end))
        })?;
        // Used only when the block ends before the range does, so below 2^64.
        let after = al.wrapping_add(asz);
        match (al == range.start, end == range.last) {
            // The block fills the range.
            (true, true) => {
                self.free.remove(i);
            }
            (true, false) => {
                if let Some(r) = self.free.get_mut(i) {
                    r.start = after;
                }
            }
            (false, true) => {
                if let Some(r) = self.free.get_mut(i) {
                    r.last = al.wrapping_sub(1);
                }
            }
            // The block splits the range: the tail follows it.
            (false, false) => {
                if let Some(r) = self.free.get_mut(i) {
                    r.last = al.wrapping_sub(1);
                }
                self.free.insert(
                    i.wrapping_add(1),
                    FreeRange {
                        start: after,
                        last: range.last,
                    },
                );
            }
        }
        Some(al)
    }

    /// Frees `size` bytes, rounded up to the granule, at `start`, coalescing with the ranges on
    /// either side. Adjacency is inclusive: a free that touches or overlaps a neighbour merges
    /// with it.
    ///
    /// Nothing checks that the range was allocated. A free inside a free range changes nothing,
    /// and one that reaches past the next range's end moves only that range's start.
    pub fn deallocate(&mut self, start: u64, size: u64) {
        let end = start.wrapping_add(self.round(size));
        let last = end.wrapping_sub(1);
        let freed = FreeRange { start, last };
        let Some(first) = self.free.first().copied() else {
            self.free.push(freed);
            return;
        };
        let i = if first.start >= start {
            0
        } else {
            // The first range at or after `start`; range 0 starts below it.
            match self.free.iter().skip(1).position(|r| r.start >= start) {
                Some(p) => p.wrapping_add(1),
                None => {
                    // Every range starts below `start`: extend or append after the last one.
                    let Some(tail) = self.free.last_mut() else {
                        return;
                    };
                    if start > tail.last.wrapping_add(1) {
                        self.free.push(freed);
                    } else if last > tail.last {
                        tail.last = last;
                    }
                    return;
                }
            }
        };
        let prev = i.checked_sub(1);
        let touches_prev = prev
            .and_then(|p| self.free.get(p))
            .is_some_and(|p| start <= p.last.wrapping_add(1));
        let Some(next) = self.free.get(i).copied() else {
            return;
        };
        let touches_next = end >= next.start;
        match (touches_prev, prev) {
            (true, Some(p)) => {
                if touches_next {
                    // Bridges both: the previous range takes the next one's end.
                    if let Some(r) = self.free.get_mut(p) {
                        r.last = next.last;
                    }
                    self.free.remove(i);
                } else if let Some(r) = self.free.get_mut(p)
                    && last > r.last
                {
                    r.last = last;
                }
            }
            _ => {
                if !touches_next {
                    self.free.insert(i, freed);
                } else if let Some(r) = self.free.get_mut(i)
                    && start < r.start
                {
                    r.start = start;
                }
            }
        }
    }

    /// The free ranges, sorted.
    #[cfg(test)]
    pub fn free_ranges(&self) -> &[FreeRange] {
        &self.free
    }

    /// `size` rounded up to the granule, wrapping.
    fn round(&self, size: u64) -> u64 {
        size.wrapping_add(self.mask) & !self.mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64, seeded so that seed 0 still runs.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
        }

        fn below(&mut self, n: u64) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x % n
        }
    }

    fn r(start: u64, last: u64) -> FreeRange {
        FreeRange { start, last }
    }

    /// The root task's space: 16 MiB in 16 KiB granules.
    fn root() -> FirstFitVa {
        FirstFitVa::new(0xffffff, 0x4000)
    }

    #[test]
    fn the_root_space_is_16_mib() {
        let a = root();
        assert_eq!(a.free_ranges(), &[r(0, 0xffffff)]);
        assert_eq!(a.mask, 0x3fff);
    }

    #[test]
    fn a_zero_granule_allocates_bytes() {
        let mut a = FirstFitVa::new(0xff, 0);
        assert_eq!(a.mask, 0);
        assert_eq!(a.allocate(3), Some(0));
        assert_eq!(a.free_ranges(), &[r(3, 0xff)]);
        let mut b = FirstFitVa::new(0xff, 1);
        assert_eq!(b.allocate(1), Some(0));
        assert_eq!(b.allocate(1), Some(1));
    }

    #[test]
    fn allocating_zero_fails_and_changes_nothing() {
        let mut a = root();
        assert_eq!(a.allocate(0), None);
        assert_eq!(a, root());
    }

    #[test]
    fn sizes_round_to_the_granule_and_bump_from_0() {
        let mut a = root();
        assert_eq!(a.allocate(1), Some(0));
        assert_eq!(a.allocate(0x4000), Some(0x4000));
        assert_eq!(a.allocate(0x4001), Some(0x8000));
        assert_eq!(a.free_ranges(), &[r(0x10000, 0xffffff)]);
    }

    #[test]
    fn a_block_that_fills_a_range_removes_it() {
        let mut a = root();
        assert_eq!(a.allocate(0x1000000), Some(0));
        assert!(a.free_ranges().is_empty());
        assert_eq!(a.allocate(0x4000), None);
    }

    #[test]
    fn a_block_too_big_fails() {
        let mut a = root();
        assert_eq!(a.allocate(0x1004000), None);
        assert_eq!(a, root());
    }

    /// A size that rounds past 2^64 fails wherever the free ranges start, and changes nothing.
    #[test]
    fn a_size_rounding_past_2_64_fails() {
        let mut a = root();
        assert_eq!(a.allocate(u64::MAX - 0x3ffe), None);
        assert_eq!(a.allocate(u64::MAX), None);
        assert_eq!(a, root());
        assert_eq!(a.allocate(0x4000), Some(0));
        let before = a.clone();
        assert_eq!(a.allocate(u64::MAX - 0x3ffe), None);
        assert_eq!(a.allocate(u64::MAX), None);
        assert_eq!(a, before);
    }

    /// A block whose end would lie past 2^64 does not fit, wherever the range starts: the
    /// allocation fails and changes nothing, as one larger than the space does.
    #[test]
    fn a_block_ending_past_2_64_fails() {
        let mut a = root();
        assert_eq!(a.allocate(0x8000), Some(0));
        let before = a.clone();
        // From 0x8000, 2^64 - 0x4000 bytes would end at 0x3fff after wrapping.
        assert_eq!(a.allocate(0xffff_ffff_ffff_c000), None);
        assert_eq!(a.allocate(u64::MAX - 0x3fff), None);
        assert_eq!(a, before);
        assert_eq!(a.allocate(0x4000), Some(0x8000));
    }

    /// A free range that starts past the last granule boundary below 2^64 holds nothing: its
    /// first boundary would lie past 2^64.
    #[test]
    fn a_range_whose_boundary_lies_past_2_64_holds_nothing() {
        let mut a = FirstFitVa::new(u64::MAX, 0x4000);
        a.free = vec![r(u64::MAX - 0xffe, u64::MAX)];
        let before = a.clone();
        assert_eq!(a.allocate(1), None);
        assert_eq!(a, before);
    }

    #[test]
    fn an_unaligned_range_trims_its_end_or_splits() {
        // A bogus free leaves a range starting off the granule: the block starts at the next
        // boundary, so the range keeps its head.
        let mut a = FirstFitVa::new(0xffff, 0x4000);
        assert_eq!(a.allocate(0x10000), Some(0));
        a.deallocate(0x1000, 0x7000);
        assert_eq!(a.free_ranges(), &[r(0x1000, 0x8fff)]);
        // Ends at the range's end: the range's end moves down to the block.
        assert_eq!(a.allocate(0x4000), Some(0x4000));
        assert_eq!(a.free_ranges(), &[r(0x1000, 0x3fff), r(0x8000, 0x8fff)]);
        let mut b = FirstFitVa::new(0xffff, 0x4000);
        assert_eq!(b.allocate(0x10000), Some(0));
        b.deallocate(0x1000, 0xb000);
        assert_eq!(b.free_ranges(), &[r(0x1000, 0xcfff)]);
        // Inside the range: it splits around the block.
        assert_eq!(b.allocate(0x4000), Some(0x4000));
        assert_eq!(b.free_ranges(), &[r(0x1000, 0x3fff), r(0x8000, 0xcfff)]);
    }

    #[test]
    fn a_free_into_an_empty_list_inserts() {
        let mut a = root();
        a.allocate(0x1000000).unwrap();
        a.deallocate(0x4000, 1);
        assert_eq!(a.free_ranges(), &[r(0x4000, 0x7fff)]);
    }

    #[test]
    fn a_free_before_the_first_range_inserts_or_moves_its_start() {
        let mut a = root();
        a.allocate(0xc000).unwrap();
        a.deallocate(0, 0x4000);
        assert_eq!(a.free_ranges(), &[r(0, 0x3fff), r(0xc000, 0xffffff)]);
        a.deallocate(0x8000, 0x4000);
        assert_eq!(a.free_ranges(), &[r(0, 0x3fff), r(0x8000, 0xffffff)]);
        // A free at exactly the next start does not move it.
        a.deallocate(0x8000, 0x4000);
        assert_eq!(a.free_ranges(), &[r(0, 0x3fff), r(0x8000, 0xffffff)]);
    }

    #[test]
    fn a_free_bridging_two_ranges_merges_them() {
        let mut a = root();
        let x = a.allocate(0x4000).unwrap();
        let y = a.allocate(0x4000).unwrap();
        let z = a.allocate(0x4000).unwrap();
        a.deallocate(x, 0x4000);
        assert_eq!(a.free_ranges(), &[r(0, 0x3fff), r(0xc000, 0xffffff)]);
        a.deallocate(y, 0x4000);
        assert_eq!(a.free_ranges(), &[r(0, 0x7fff), r(0xc000, 0xffffff)]);
        a.deallocate(z, 0x4000);
        assert_eq!(a.free_ranges(), &[r(0, 0xffffff)]);
    }

    #[test]
    fn a_free_touching_only_the_previous_range_extends_it() {
        let mut a = root();
        let x = a.allocate(0x4000).unwrap();
        let y = a.allocate(0x4000).unwrap();
        let _z = a.allocate(0x4000).unwrap();
        a.deallocate(x, 0x4000);
        a.deallocate(y, 0x2000);
        assert_eq!(a.free_ranges(), &[r(0, 0x7fff), r(0xc000, 0xffffff)]);
        // One inside it changes nothing.
        a.deallocate(0x1000, 0x10);
        assert_eq!(a.free_ranges(), &[r(0, 0x7fff), r(0xc000, 0xffffff)]);
    }

    #[test]
    fn a_free_between_two_ranges_touching_neither_inserts() {
        let mut a = root();
        for _ in 0..4 {
            a.allocate(0x4000).unwrap();
        }
        a.deallocate(0, 0x4000);
        a.deallocate(0x8000, 0x4000);
        assert_eq!(
            a.free_ranges(),
            &[r(0, 0x3fff), r(0x8000, 0xbfff), r(0x10000, 0xffffff)]
        );
    }

    #[test]
    fn a_free_past_every_range_appends_or_extends() {
        let mut a = FirstFitVa::new(0xffff, 0x4000);
        a.allocate(0x10000).unwrap();
        a.deallocate(0, 0x4000);
        a.deallocate(0xc000, 0x4000);
        assert_eq!(a.free_ranges(), &[r(0, 0x3fff), r(0xc000, 0xffff)]);
        a.deallocate(0x10000, 0x4000); // adjacent to the last: extends
        assert_eq!(a.free_ranges(), &[r(0, 0x3fff), r(0xc000, 0x13fff)]);
        a.deallocate(0xd000, 0x10); // inside the last: nothing
        assert_eq!(a.free_ranges(), &[r(0, 0x3fff), r(0xc000, 0x13fff)]);
        a.deallocate(0x20000, 0x4000); // beyond: appends
        assert_eq!(
            a.free_ranges(),
            &[r(0, 0x3fff), r(0xc000, 0x13fff), r(0x20000, 0x23fff)]
        );
    }

    #[test]
    fn a_free_past_a_single_range_appends() {
        let mut a = root();
        a.deallocate(0x2000000, 0x4000);
        assert_eq!(a.free_ranges(), &[r(0, 0xffffff), r(0x2000000, 0x2003fff)]);
    }

    #[test]
    fn a_free_reaching_past_the_next_range_keeps_its_end() {
        let mut a = FirstFitVa::new(0xffff, 0x4000);
        a.allocate(0x8000).unwrap();
        a.deallocate(0x4000, 0x10000);
        assert_eq!(a.free_ranges(), &[r(0x4000, 0xffff)]);
    }

    /// Never reset: a freed hole is reused first, and a leaked range shifts later offsets.
    #[test]
    fn first_fit_reuses_hole() {
        let mut a = root();
        let x = a.allocate(0x8000).unwrap();
        let _y = a.allocate(0x4000).unwrap();
        a.deallocate(x, 0x8000);
        assert_eq!(a.allocate(0x4000), Some(0));
        assert_eq!(a.allocate(0x8000), Some(0xc000));
        assert_eq!(a.allocate(0x4000), Some(0x4000));
    }

    /// One bool per granule: first fit at the lowest granule; the free list is the maximal runs
    /// of free granules.
    struct Naive {
        free: Vec<bool>,
    }

    impl Naive {
        const G: u64 = 0x4000;
        fn allocate(&mut self, size: u64) -> Option<u64> {
            let n = size.div_ceil(Self::G) as usize;
            let total = self.free.len();
            let mut i = 0;
            while i + n <= total {
                if self.free[i..i + n].iter().all(|f| *f) {
                    self.free[i..i + n].iter_mut().for_each(|f| *f = false);
                    return Some(i as u64 * Self::G);
                }
                i += 1;
            }
            None
        }
        fn deallocate(&mut self, at: u64, size: u64) {
            let i = (at / Self::G) as usize;
            let n = size.div_ceil(Self::G) as usize;
            self.free[i..i + n].iter_mut().for_each(|f| *f = true);
        }
        fn ranges(&self) -> Vec<FreeRange> {
            let mut out: Vec<FreeRange> = Vec::new();
            for (i, f) in self.free.iter().enumerate() {
                if !*f {
                    continue;
                }
                let s = i as u64 * Self::G;
                match out.last_mut() {
                    Some(l) if l.last + 1 == s => l.last = s + Self::G - 1,
                    _ => out.push(r(s, s + Self::G - 1)),
                }
            }
            out
        }
    }

    /// Well-formed histories (live frees only) match a granule bitmap on every result and free
    /// list.
    #[test]
    fn random_histories_match_a_granule_bitmap() {
        for seed in 1..=200u64 {
            let mut rng = Rng::new(seed);
            let granules = 64u64;
            let mut a = FirstFitVa::new(granules * 0x4000 - 1, 0x4000);
            let mut m = Naive {
                free: vec![true; granules as usize],
            };
            let mut live: Vec<(u64, u64)> = Vec::new();
            for _ in 0..300 {
                if live.is_empty() || rng.below(3) != 0 {
                    let size = 1 + rng.below(6 * 0x4000);
                    let got = a.allocate(size);
                    assert_eq!(got, m.allocate(size), "seed {seed}");
                    if let Some(at) = got {
                        live.push((at, size));
                    }
                } else {
                    let (at, size) = live.swap_remove(rng.below(live.len() as u64) as usize);
                    a.deallocate(at, size);
                    m.deallocate(at, size);
                }
                assert_eq!(a.free_ranges(), &m.ranges()[..], "seed {seed}");
            }
        }
    }
}
