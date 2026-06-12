//! Rust port of the block data structure D of `_reference.py` (ALGORITHM.md
//! S3, Lemma 3.3 / SPEC.md S4).
//!
//! The port is semantically 1:1 with the Python reference, including every
//! *observable order*: Python `dict`s are insertion-ordered, and the reference
//! leaks that order through `Pull` (the order of the returned keys feeds the
//! recursion's `S`, which ultimately determines the settlement order). Blocks
//! are therefore stored as tombstoned slot vectors that reproduce Python dict
//! semantics for the operations the reference performs (fresh insert appends;
//! delete preserves the order of the rest; value update keeps the position).
//!
//! The reference's only other source of nondeterminism is `random.randint` in
//! the quickselect; here the RNG is an explicit [`SplitMix64`] passed in by the
//! caller so a differential test can pin both sides to the same stream.

use std::collections::{HashMap, VecDeque};

/// `2**62`, the reference's `INF_INT` (used for "infinite" hop counts and the
/// third component of the `INF` key).
pub const INF_INT: i64 = 1 << 62;

/// A label key `(length, hops, vertex_or_pred_id)` compared lexicographically
/// (ALGORITHM.md S1.3). Mirrors the reference's `Key` tuple.
///
/// `len` is never NaN (edge weights are validated finite and `dhat` values are
/// sums of finite non-negatives starting from `+0.0`, so `-0.0` cannot occur
/// either); `f64::total_cmp` therefore agrees with Python float comparison on
/// every value that can appear here.
#[derive(Clone, Copy, Debug)]
pub struct Key {
    pub len: f64,
    pub hops: i64,
    pub id: i64,
}

/// The reference's `INF` key `(inf, 2**62, 2**62)`.
pub const KEY_INF: Key = Key {
    len: f64::INFINITY,
    hops: INF_INT,
    id: INF_INT,
};

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.len
            .total_cmp(&other.len)
            .then_with(|| self.hops.cmp(&other.hops))
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// SplitMix64 PRNG. Chosen because it is trivial to implement identically in
/// Python (the differential test patches the reference's `random` module with
/// the same generator so both sides draw the same pivot sequence).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Inclusive-range integer, the same contract as Python's
    /// `random.randint(lo, hi)` realized as `lo + next() % (hi - lo + 1)`.
    pub fn randint(&mut self, lo: usize, hi: usize) -> usize {
        debug_assert!(lo <= hi);
        lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
    }
}

/// A list of `(vertex, key)` pairs (a Python `list[tuple[int, Key]]`).
pub type Pairs = Vec<(u32, Key)>;

/// Partition `pairs` into the `m` smallest-by-value and the rest, with a
/// random-pivot quickselect. Exact port of `_select_smallest`, including the
/// `randint` call sequence and the swap pattern, so the (otherwise arbitrary)
/// output order matches the Python reference run with the same RNG stream.
pub fn select_smallest(
    pairs: &[(u32, Key)],
    m: usize,
    rng: &mut SplitMix64,
) -> (Pairs, Pairs) {
    let n = pairs.len();
    if m == 0 {
        return (Vec::new(), pairs.to_vec());
    }
    if m >= n {
        return (pairs.to_vec(), Vec::new());
    }

    let mut items = pairs.to_vec();
    let mut lo = 0usize;
    let mut hi = n - 1;
    let target = m - 1; // index (0-based) of the m-th smallest after partitioning
    while lo < hi {
        let pivot_idx = rng.randint(lo, hi);
        let pivot_val = items[pivot_idx].1;
        items.swap(pivot_idx, hi);
        let mut store = lo;
        for i in lo..hi {
            if items[i].1 < pivot_val {
                items.swap(store, i);
                store += 1;
            }
        }
        items.swap(store, hi);
        match store.cmp(&target) {
            std::cmp::Ordering::Equal => break,
            std::cmp::Ordering::Less => lo = store + 1,
            std::cmp::Ordering::Greater => hi = store - 1,
        }
    }
    let rest = items.split_off(m);
    (items, rest)
}

/// Split `pairs` into chunks of size <= `cap` by repeated median finding;
/// chunk `i`'s values all precede chunk `i + 1`'s. Port of `_chunk_by_median`.
fn chunk_by_median(
    pairs: Vec<(u32, Key)>,
    cap: usize,
    rng: &mut SplitMix64,
) -> Vec<Vec<(u32, Key)>> {
    if pairs.is_empty() {
        return Vec::new();
    }
    if pairs.len() <= cap {
        return vec![pairs];
    }
    let half = pairs.len() / 2;
    let (lower, upper) = select_smallest(&pairs, half, rng);
    let mut out = chunk_by_median(lower, cap, rng);
    out.extend(chunk_by_median(upper, cap, rng));
    out
}

/// One block: a Python-dict stand-in for `dict[int, Key]` restricted to the
/// operations the reference performs. Iteration order is insertion order with
/// deletions leaving the remaining order intact (tombstones).
struct Block {
    slots: Vec<Option<(u32, Key)>>,
    live: usize,
}

impl Block {
    fn from_pairs(pairs: &[(u32, Key)]) -> Self {
        Block {
            slots: pairs.iter().map(|&p| Some(p)).collect(),
            live: pairs.len(),
        }
    }

    /// Append (Python: fresh `block[key] = value`; callers guarantee the key
    /// is not currently in the block). Returns the slot index.
    fn push(&mut self, key: u32, value: Key) -> usize {
        self.slots.push(Some((key, value)));
        self.live += 1;
        self.slots.len() - 1
    }

    fn remove(&mut self, slot: usize) {
        debug_assert!(self.slots[slot].is_some());
        self.slots[slot] = None;
        self.live -= 1;
    }

    fn iter(&self) -> impl Iterator<Item = (u32, Key)> + '_ {
        self.slots.iter().filter_map(|s| *s)
    }

    fn min_value(&self) -> Option<Key> {
        self.iter().map(|(_k, v)| v).min()
    }
}

/// The partial-sorting batched priority structure of Lemma 3.3 (`BlockDS` in
/// the reference). See the module docs for the order-fidelity contract.
pub struct BlockDs {
    /// Block-size parameter `M` (>= 1).
    m: usize,
    /// Upper bound `B`; the last D1 bound is always `B`.
    b: Key,
    /// Arena of all blocks ever created (stale ones simply become garbage,
    /// mirroring Python dropping dict references).
    arena: Vec<Block>,
    /// D0 block ids (receives only `batch_prepend`).
    d0: VecDeque<usize>,
    /// D1 block ids (receives only `insert`), with parallel upper bounds.
    d1: Vec<usize>,
    d1_bounds: Vec<Key>,
    /// `self.where` of the reference: key -> (value, block id, slot index).
    registry: HashMap<u32, (Key, usize, usize)>,
}

impl BlockDs {
    pub fn new(m: usize, b: Key) -> Self {
        let mut ds = BlockDs {
            m: m.max(1),
            b,
            arena: Vec::new(),
            d0: VecDeque::new(),
            d1: Vec::new(),
            d1_bounds: vec![b],
            registry: HashMap::new(),
        };
        let empty = ds.alloc_block(&[]);
        ds.d1.push(empty);
        ds
    }

    pub fn len(&self) -> usize {
        self.registry.len()
    }

    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }

    fn alloc_block(&mut self, pairs: &[(u32, Key)]) -> usize {
        self.arena.push(Block::from_pairs(pairs));
        self.arena.len() - 1
    }

    fn register_block(&mut self, bid: usize, pairs: &[(u32, Key)]) {
        for (slot, &(k, v)) in pairs.iter().enumerate() {
            self.registry.insert(k, (v, bid, slot));
        }
    }

    /// Minimum value currently in D, or `B` if D is empty (`_min_value`).
    fn min_value(&self) -> Key {
        let mut candidates: Vec<Key> = Vec::with_capacity(2);
        for &bid in &self.d0 {
            if self.arena[bid].live > 0 {
                candidates.push(self.arena[bid].min_value().unwrap());
                break;
            }
        }
        for &bid in &self.d1 {
            if self.arena[bid].live > 0 {
                candidates.push(self.arena[bid].min_value().unwrap());
                break;
            }
        }
        candidates.into_iter().min().unwrap_or(self.b)
    }

    /// Split the over-full D1 block at `idx` at its median (`_split_d1`).
    fn split_d1(&mut self, idx: usize, rng: &mut SplitMix64) {
        let bid = self.d1[idx];
        let bnd = self.d1_bounds[idx];
        let items: Vec<(u32, Key)> = self.arena[bid].iter().collect();
        let half = items.len() / 2;
        let (lower_items, upper_items) = select_smallest(&items, half, rng);
        let new_bound = lower_items.iter().map(|&(_k, v)| v).max().unwrap();

        let lower_id = self.alloc_block(&lower_items);
        let upper_id = self.alloc_block(&upper_items);
        self.d1[idx] = lower_id;
        self.d1_bounds[idx] = new_bound;
        self.d1.insert(idx + 1, upper_id);
        self.d1_bounds.insert(idx + 1, bnd);

        self.register_block(lower_id, &lower_items);
        self.register_block(upper_id, &upper_items);
    }

    /// Insert `(key, value)` into D1; keep only the smaller-value pair on a
    /// duplicate key (`insert`).
    pub fn insert(&mut self, key: u32, value: Key, rng: &mut SplitMix64) {
        debug_assert!(value < self.b, "Insert: value must be < B");
        if let Some(&(old_value, old_bid, old_slot)) = self.registry.get(&key) {
            if value >= old_value {
                return; // keep existing smaller value
            }
            self.arena[old_bid].remove(old_slot);
            self.registry.remove(&key);
        }

        let idx = self.d1_bounds.partition_point(|bnd| *bnd < value); // bisect_left
        let bid = self.d1[idx];
        let slot = self.arena[bid].push(key, value);
        self.registry.insert(key, (value, bid, slot));
        if self.arena[bid].live > self.m {
            self.split_d1(idx, rng);
        }
    }

    /// Prepend `items` as new D0 block(s) (`batch_prepend`). Precondition:
    /// every value is smaller than every value currently in D.
    pub fn batch_prepend(&mut self, items: &[(u32, Key)], rng: &mut SplitMix64) {
        if items.is_empty() {
            return;
        }

        // Ordered dedup with keep-min, matching Python dict semantics: the
        // first occurrence fixes the position, a smaller value updates in
        // place.
        let mut dedup: Vec<(u32, Key)> = Vec::with_capacity(items.len());
        let mut pos: HashMap<u32, usize> = HashMap::with_capacity(items.len());
        for &(k, v) in items {
            match pos.get(&k) {
                None => {
                    pos.insert(k, dedup.len());
                    dedup.push((k, v));
                }
                Some(&i) => {
                    if v < dedup[i].1 {
                        dedup[i].1 = v;
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        {
            let cur_min = self.min_value();
            for &(_k, v) in &dedup {
                debug_assert!(
                    v < cur_min,
                    "BatchPrepend precondition violated: a value is not \
                     smaller than D's current minimum"
                );
            }
        }

        for &(k, v) in &dedup {
            if let Some(&(old_value, old_bid, old_slot)) = self.registry.get(&k) {
                assert!(
                    v < old_value,
                    "BatchPrepend: duplicate key with non-smaller value \
                     violates the precondition"
                );
                self.arena[old_bid].remove(old_slot);
                self.registry.remove(&k);
            }
        }

        let cap = std::cmp::max(1, self.m.div_ceil(2)); // max(1, (M + 1) // 2)
        let chunks: Vec<Vec<(u32, Key)>> = if dedup.len() <= self.m {
            vec![dedup]
        } else {
            chunk_by_median(dedup, cap, rng)
        };

        // `self._d0_blocks = chunks + self._d0_blocks`
        for chunk in chunks.into_iter().rev() {
            let bid = self.alloc_block(&chunk);
            self.register_block(bid, &chunk);
            self.d0.push_front(bid);
        }
    }

    /// Remove and return the keys of the M smallest values plus a separating
    /// bound (`pull`).
    pub fn pull(&mut self, rng: &mut SplitMix64) -> (Vec<u32>, Key) {
        let m = self.m;

        let mut s0_items: Vec<(u32, Key)> = Vec::new();
        let mut s0_seen = 0usize;
        for &bid in &self.d0 {
            s0_seen += 1;
            if self.arena[bid].live == 0 {
                continue;
            }
            s0_items.extend(self.arena[bid].iter());
            if s0_items.len() >= m {
                break;
            }
        }
        let d0_exhausted = s0_seen == self.d0.len();

        let mut s1_items: Vec<(u32, Key)> = Vec::new();
        let mut s1_seen = 0usize;
        for &bid in &self.d1 {
            s1_seen += 1;
            if self.arena[bid].live == 0 {
                continue;
            }
            s1_items.extend(self.arena[bid].iter());
            if s1_items.len() >= m {
                break;
            }
        }
        let d1_exhausted = s1_seen == self.d1.len();

        let mut union = s0_items;
        union.extend(s1_items);

        if union.len() <= m && d0_exhausted && d1_exhausted {
            for &(k, _v) in &union {
                self.registry.remove(&k);
            }
            self.d0.clear();
            let empty = self.alloc_block(&[]);
            self.d1.clear();
            self.d1.push(empty);
            self.d1_bounds.clear();
            self.d1_bounds.push(self.b);
            return (union.iter().map(|&(k, _v)| k).collect(), self.b);
        }

        let (smallest, _rest) = select_smallest(&union, m, rng);
        for &(k, _v) in &smallest {
            let (_value, bid, slot) = self.registry.remove(&k).unwrap();
            self.arena[bid].remove(slot);
        }

        while let Some(&front) = self.d0.front() {
            if self.arena[front].live == 0 {
                self.d0.pop_front();
            } else {
                break;
            }
        }
        // Drop emptied leading D1 blocks (and bounds); the last block (bound
        // B) always stays.
        while self.d1.len() > 1 && self.arena[self.d1[0]].live == 0 {
            self.d1.remove(0);
            self.d1_bounds.remove(0);
        }

        let x = self.min_value();
        (smallest.iter().map(|&(k, _v)| k).collect(), x)
    }

    /// White-box invariant checker (port of `_check_invariants`); test-only.
    #[cfg(test)]
    fn check_invariants(&self) {
        assert_eq!(self.d1.len(), self.d1_bounds.len());
        assert!(!self.d1.is_empty());
        assert_eq!(*self.d1_bounds.last().unwrap(), self.b);
        for w in self.d1_bounds.windows(2) {
            assert!(w[0] <= w[1]);
        }
        for &bid in &self.d1 {
            assert!(self.arena[bid].live <= self.m);
        }
        for (ids, bounds) in [
            (&self.d1, Some(&self.d1_bounds)),
            (&self.d0.iter().copied().collect::<Vec<_>>(), None),
        ] {
            let mut prev_max: Option<Key> = None;
            for (i, &bid) in ids.iter().enumerate() {
                let block = &self.arena[bid];
                if block.live == 0 {
                    continue;
                }
                let bmin = block.iter().map(|(_k, v)| v).min().unwrap();
                let bmax = block.iter().map(|(_k, v)| v).max().unwrap();
                if let Some(pm) = prev_max {
                    assert!(pm <= bmin);
                }
                if let Some(bounds) = bounds {
                    assert!(bmax <= bounds[i]);
                }
                prev_max = Some(bmax);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(len: f64, id: i64) -> Key {
        Key { len, hops: 1, id }
    }

    fn ds_with(m: usize, bound_len: f64) -> (BlockDs, SplitMix64) {
        (
            BlockDs::new(m, key(bound_len, INF_INT)),
            SplitMix64::new(0xDEAD_BEEF),
        )
    }

    #[test]
    fn insert_and_pull_returns_smallest() {
        let (mut ds, mut rng) = ds_with(3, 1e9);
        for (i, len) in [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0].iter().enumerate() {
            ds.insert(i as u32, key(*len, i as i64), &mut rng);
            ds.check_invariants();
        }
        assert_eq!(ds.len(), 7);
        let (keys, x) = ds.pull(&mut rng);
        ds.check_invariants();
        assert_eq!(keys.len(), 3);
        // Smallest three by value: lens 1.0 (id 1), 2.0 (id 5), 3.0 (id 3).
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 3, 5]);
        // Separating bound is the new minimum (len 5.0).
        assert_eq!(x.len, 5.0);
        assert_eq!(ds.len(), 4);
    }

    #[test]
    fn insert_duplicate_keeps_min() {
        let (mut ds, mut rng) = ds_with(4, 1e9);
        ds.insert(7, key(5.0, 7), &mut rng);
        ds.insert(7, key(9.0, 7), &mut rng); // larger: ignored
        ds.insert(7, key(2.0, 7), &mut rng); // smaller: replaces
        assert_eq!(ds.len(), 1);
        let (keys, x) = ds.pull(&mut rng);
        assert_eq!(keys, vec![7]);
        assert_eq!(x, ds.b); // emptied: bound is B
    }

    #[test]
    fn batch_prepend_comes_out_first() {
        let (mut ds, mut rng) = ds_with(2, 1e9);
        for i in 0..6u32 {
            ds.insert(i, key(100.0 + i as f64, i as i64), &mut rng);
        }
        // Prepend strictly smaller values, more than M so chunking kicks in.
        let items: Vec<(u32, Key)> = (10..15u32)
            .map(|i| (i, key(i as f64, i as i64)))
            .collect();
        ds.batch_prepend(&items, &mut rng);
        ds.check_invariants();
        assert_eq!(ds.len(), 11);
        let (k1, _) = ds.pull(&mut rng);
        ds.check_invariants();
        assert_eq!(k1.len(), 2);
        assert!(k1.iter().all(|&k| (10..15).contains(&k)));
    }

    #[test]
    fn pull_drains_everything() {
        let (mut ds, mut rng) = ds_with(3, 1e9);
        for i in 0..10u32 {
            ds.insert(i, key(i as f64, i as i64), &mut rng);
        }
        let mut seen = Vec::new();
        let mut last_max = f64::NEG_INFINITY;
        while !ds.is_empty() {
            let (keys, x) = ds.pull(&mut rng);
            ds.check_invariants();
            assert!(!keys.is_empty());
            let batch_max = keys.iter().map(|&k| k as f64).fold(f64::NEG_INFINITY, f64::max);
            assert!(keys.iter().map(|&k| k as f64).fold(f64::INFINITY, f64::min) > last_max);
            last_max = batch_max;
            // Every remaining value is >= the separating bound's length.
            assert!(x.len >= batch_max);
            seen.extend(keys);
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..10u32).collect::<Vec<_>>());
    }

    #[test]
    fn select_smallest_partitions() {
        let mut rng = SplitMix64::new(42);
        let pairs: Vec<(u32, Key)> = (0..50u32)
            .map(|i| (i, key(((i * 37) % 50) as f64, i as i64)))
            .collect();
        let (small, rest) = select_smallest(&pairs, 20, &mut rng);
        assert_eq!(small.len(), 20);
        assert_eq!(rest.len(), 30);
        let max_small = small.iter().map(|&(_k, v)| v).max().unwrap();
        let min_rest = rest.iter().map(|&(_k, v)| v).min().unwrap();
        assert!(max_small < min_rest);
    }
}
