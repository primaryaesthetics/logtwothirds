//! Shared recursion engine for the BMSSP variants (`src/variants/*`).
//!
//! This is a fork of the mainline `src/bmssp.rs` recursion, parameterized by
//! [`Config`] and generic over the Lemma 3.3 queue ([`DQueue`]). The mainline
//! stays untouched; variants only configure this engine. Unlike the mainline,
//! the engine makes **no settlement-order promises** — the correctness bar
//! for every variant is bit-exact distances vs Dijkstra (see
//! `tests/variants_correctness.rs`), not the Step E settlement-order gate.
//!
//! Invariants preserved from the paper regardless of configuration:
//! - the total order on labels (Assumption 2.1 via `Key`), the `<=`
//!   relaxation rule (Remark 3.4), and the settled-vertex filter on pulled
//!   batches (AUDIT.md F3);
//! - FindPivots' covering property (Lemma 3.2): if only `j <= k`
//!   Bellman-Ford rounds are run, the pivot tree-size threshold is lowered
//!   to `j` so that every x in U-tilde is either complete in W or covered by
//!   a pivot;
//! - the BMSSP contract (Lemma 3.1/3.7): every child call returns a complete
//!   `U = T_<B'(S)` with `B' <= B`, whatever oracle produced it.

use crate::block_queue::{BlockDs, Key, SplitMix64, INF_INT, KEY_INF};
use crate::bmssp::{compute_params, transform_to_constant_degree, BmsspError, Csr};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// Engine invariant checks: `debug_assert!` in normal builds, hard `assert!`
/// when built with `--features verify`. Release hot loops carry no
/// instrumentation unless that feature is on.
macro_rules! verify_assert {
    ($($arg:tt)*) => {{
        #[cfg(feature = "verify")]
        {
            assert!($($arg)*);
        }
        #[cfg(not(feature = "verify"))]
        {
            debug_assert!($($arg)*);
        }
    }};
}

/// Wall-clock seconds per engine phase. Populated only under
/// `--features phase-timer` (all-zero otherwise); phases are non-overlapping
/// leaves, so `total - sum(phases)` is the recursion bookkeeping.
#[derive(Default, Debug, Clone)]
pub struct EnginePhases {
    /// Constant-degree transform (zero when `Config::transform` is false).
    pub transform: f64,
    /// FindPivots (Algorithm 1), including its relaxations.
    pub find_pivots: f64,
    /// Paper BaseCase (Algorithm 2, singleton + k+1 truncation).
    pub base_case: f64,
    /// Bounded multi-source Dijkstra oracle (`hybrid_*` config).
    pub dijkstra_oracle: f64,
    /// `DQueue::pull`.
    pub q_pull: f64,
    /// The Algorithm 3 edge-relaxation loop, including `DQueue::insert`s.
    pub relax_loop: f64,
    /// `DQueue::batch_prepend`.
    pub q_batch_prepend: f64,
    /// Distance extraction + predecessor recovery.
    pub finalize: f64,
}

/// Engine operation counters; incremented only under
/// `--features phase-timer` (all-zero otherwise).
#[derive(Default, Debug, Clone)]
pub struct EngineCounters {
    pub edge_scans: u64,
    pub relaxations: u64,
    pub q_inserts: u64,
    pub q_pulls: u64,
    pub q_pulled_keys: u64,
    pub q_prepend_items: u64,
    pub oracle_calls: u64,
    pub oracle_settled: u64,
    pub findpivots_calls: u64,
    pub basecase_calls: u64,
    pub bmssp_calls: u64,
}

/// Time `$e` into `$st.phase.$field` under `phase-timer`; just `$e` otherwise.
macro_rules! phase {
    ($st:expr, $field:ident, $e:expr) => {{
        #[cfg(feature = "phase-timer")]
        let __phase_t0 = std::time::Instant::now();
        let __phase_r = $e;
        #[cfg(feature = "phase-timer")]
        {
            $st.phase.$field += __phase_t0.elapsed().as_secs_f64();
        }
        __phase_r
    }};
}

/// Add `$d` to counter `$field` under `phase-timer`; no-op otherwise.
macro_rules! count {
    ($st:expr, $field:ident, $d:expr) => {{
        #[cfg(feature = "phase-timer")]
        {
            $st.cnt.$field += $d as u64;
        }
    }};
}

/// Queue contract of Lemma 3.3 as the recursion uses it. `BlockDs` satisfies
/// it with the paper's amortized bounds; simpler structures may satisfy the
/// *semantic* contract with worse bounds (documented per variant).
pub trait DQueue {
    fn new(m: usize, b: Key) -> Self;
    fn insert(&mut self, key: u32, value: Key, rng: &mut SplitMix64);
    /// Precondition: every value in `items` is smaller than every value in D.
    fn batch_prepend(&mut self, items: &[(u32, Key)], rng: &mut SplitMix64);
    /// Returns (keys of the <= M smallest values, separating bound).
    fn pull(&mut self, rng: &mut SplitMix64) -> (Vec<u32>, Key);
    fn is_empty(&self) -> bool;
}

impl DQueue for BlockDs {
    fn new(m: usize, b: Key) -> Self {
        BlockDs::new(m, b)
    }
    fn insert(&mut self, key: u32, value: Key, rng: &mut SplitMix64) {
        BlockDs::insert(self, key, value, rng)
    }
    fn batch_prepend(&mut self, items: &[(u32, Key)], rng: &mut SplitMix64) {
        BlockDs::batch_prepend(self, items, rng)
    }
    fn pull(&mut self, rng: &mut SplitMix64) -> (Vec<u32>, Key) {
        BlockDs::pull(self, rng)
    }
    fn is_empty(&self) -> bool {
        BlockDs::is_empty(self)
    }
}

/// Engine configuration. Defaults reproduce the mainline algorithm (up to
/// settlement order, which the engine does not track).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Run on the constant-degree transform (paper Section 2) or directly on
    /// the input graph. Correctness never needs the transform; only the
    /// Lemma 3.2 / Remark 3.5 size accounting does.
    pub transform: bool,
    /// Forced (k, t); `None` = the paper's k = floor(log^(1/3) n),
    /// t = floor(log^(2/3) n). Any k >= 1, t >= 1 is correct.
    pub kt_override: Option<(usize, usize)>,
    /// Replace the recursion below-or-at this level with a bounded
    /// multi-source Dijkstra (no k+1 truncation). `-1` = never (paper
    /// BaseCase at l = 0 only). `0` = Dijkstra base case, `1` = also swallow
    /// level-1 calls, etc.
    pub hybrid_max_level: i32,
    /// Also switch to the Dijkstra oracle whenever the pulled frontier has
    /// `|S| <= hybrid_frontier` (0 = disabled). Applies at any level.
    pub hybrid_frontier: usize,
    /// Stop FindPivots' Bellman-Ford early when the frontier stops shrinking
    /// (round i produced a frontier no smaller than round i-1's, i >= 2).
    /// The pivot tree-size threshold is lowered to the number of rounds
    /// actually run, preserving Lemma 3.2's covering property.
    pub lazy_pivots: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            transform: true,
            kt_override: None,
            hybrid_max_level: -1,
            hybrid_frontier: 0,
            lazy_pivots: false,
        }
    }
}

/// Output of a variant run (distances/predecessors on the original graph).
pub struct VariantRun {
    pub dist: Vec<f64>,
    pub pred: Vec<i32>,
    pub k: usize,
    pub t: usize,
    pub levels: usize,
    pub n_inner: usize,
    /// Per-phase wall clock (all zeros unless built with `phase-timer`).
    pub phase: EnginePhases,
    /// Operation counters (all zeros unless built with `phase-timer`).
    pub cnt: EngineCounters,
    /// Total wall clock of the run (zero unless built with `phase-timer`).
    pub total_seconds: f64,
}

/// Reused buffers for the two heap oracles. The former `best` map is gone:
/// every value it held was by construction `key(v)` for the *current*
/// `dhat`/`hops` (both writers store exactly that), so the pop-time staleness
/// check recomputes the key instead. Membership in the popped set (`in_u0`)
/// lives in the engine's epoch-stamped `pop_stamp` array.
#[derive(Default)]
struct HeapScratch {
    heap: KeyHeap,
}

/// Structure-of-arrays 4-ary min-heap over the oracle heap entries
/// `(len, hops, vertex)`. Entries always satisfy `Key.id == vertex` (both
/// writers push `key(v)` / `vkey` with `id = v`), so the id is not stored;
/// the order is exactly `Key`'s: `total_cmp` on len, then hops, then vertex.
/// Same layout idea as `dijkstra::Heap` — sifting reads the dense `lens`
/// array first and touches `hops`/`vals` only on float ties.
#[derive(Default)]
struct KeyHeap {
    lens: Vec<f64>,
    hops: Vec<i64>,
    vals: Vec<u32>,
}

impl KeyHeap {
    /// Exactly `Key`'s order with `id == val` (u32 order == i64 order here).
    #[inline(always)]
    fn less(a: (f64, i64, u32), b: (f64, i64, u32)) -> bool {
        match a.0.total_cmp(&b.0) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => (a.1, a.2) < (b.1, b.2),
        }
    }

    fn clear(&mut self) {
        self.lens.clear();
        self.hops.clear();
        self.vals.clear();
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.lens.is_empty()
    }

    #[inline(always)]
    fn entry(&self, i: usize) -> (f64, i64, u32) {
        (self.lens[i], self.hops[i], self.vals[i])
    }

    #[inline(always)]
    fn set(&mut self, i: usize, e: (f64, i64, u32)) {
        self.lens[i] = e.0;
        self.hops[i] = e.1;
        self.vals[i] = e.2;
    }

    #[inline(always)]
    fn push(&mut self, e: (f64, i64, u32)) {
        let mut i = self.lens.len();
        self.lens.push(e.0);
        self.hops.push(e.1);
        self.vals.push(e.2);
        while i > 0 {
            let parent = (i - 1) >> 2;
            let pe = self.entry(parent);
            if Self::less(e, pe) {
                self.set(i, pe);
                i = parent;
            } else {
                break;
            }
        }
        self.set(i, e);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<(f64, i64, u32)> {
        let len = self.lens.len();
        if len == 0 {
            return None;
        }
        let min = self.entry(0);
        let last = self.entry(len - 1);
        self.lens.truncate(len - 1);
        self.hops.truncate(len - 1);
        self.vals.truncate(len - 1);
        let n = len - 1;
        if n > 0 {
            let mut i = 0usize;
            loop {
                let first = 4 * i + 1;
                if first >= n {
                    break;
                }
                let last_child = std::cmp::min(first + 4, n);
                let mut sm = first;
                let mut sme = self.entry(first);
                for c in first + 1..last_child {
                    let ce = self.entry(c);
                    if Self::less(ce, sme) {
                        sm = c;
                        sme = ce;
                    }
                }
                if Self::less(sme, last) {
                    self.set(i, sme);
                    i = sm;
                } else {
                    break;
                }
            }
            self.set(i, last);
        }
        Some(min)
    }
}

struct Engine<'g, Q: DQueue> {
    g: &'g Csr,
    cfg: Config,
    dhat: Vec<f64>,
    hops: Vec<i64>,
    pred: Vec<i64>,
    k: usize,
    t: usize,
    settled: Vec<bool>,
    rng: SplitMix64,
    scratch: HeapScratch,
    /// Epoch-stamped "popped this oracle call" membership (replaces a per-
    /// call hash set; never cleared, the epoch bump invalidates all stamps).
    pop_stamp: Vec<u32>,
    pop_epoch: u32,
    phase: EnginePhases,
    cnt: EngineCounters,
    _q: std::marker::PhantomData<Q>,
}

impl<'g, Q: DQueue> Engine<'g, Q> {
    fn new(g: &'g Csr, source: u32, k: usize, t: usize, seed: u64, cfg: Config) -> Self {
        let n = g.n;
        let mut dhat = vec![f64::INFINITY; n];
        let mut hops = vec![INF_INT; n];
        dhat[source as usize] = 0.0;
        hops[source as usize] = 0;
        Engine {
            g,
            cfg,
            dhat,
            hops,
            pred: vec![-1; n],
            k,
            t,
            settled: vec![false; n],
            rng: SplitMix64::new(seed),
            scratch: HeapScratch::default(),
            pop_stamp: vec![0; n],
            pop_epoch: 0,
            phase: EnginePhases::default(),
            cnt: EngineCounters::default(),
            _q: std::marker::PhantomData,
        }
    }

    #[inline]
    fn key(&self, v: u32) -> Key {
        Key {
            len: self.dhat[v as usize],
            hops: self.hops[v as usize],
            id: v as i64,
        }
    }

    /// The `<=` relaxation of Remark 3.4 (never gated on a bound).
    #[inline]
    fn try_relax(&mut self, u: u32, v: u32, w: f64) -> bool {
        let cand_len = self.dhat[u as usize] + w;
        let cand = Key {
            len: cand_len,
            hops: self.hops[u as usize] + 1,
            id: u as i64,
        };
        let cur = Key {
            len: self.dhat[v as usize],
            hops: self.hops[v as usize],
            id: self.pred[v as usize],
        };
        if cand <= cur {
            self.dhat[v as usize] = cand_len;
            self.hops[v as usize] = cand.hops;
            self.pred[v as usize] = u as i64;
            true
        } else {
            false
        }
    }

    #[inline]
    fn settle(&mut self, v: u32) {
        self.settled[v as usize] = true;
    }

    /// FindPivots (Algorithm 1), with the optional lazy early stop.
    /// Returns (P, W). Covering property (Lemma 3.2) holds with the pivot
    /// threshold equal to the number of Bellman-Ford rounds actually run.
    fn find_pivots(&mut self, b: Key, s: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let k = self.k;

        let mut w_set: HashSet<u32> = s.iter().copied().collect();
        let mut w_order: Vec<u32> = Vec::with_capacity(s.len());
        {
            let mut seen: HashSet<u32> =
                HashSet::with_capacity_and_hasher(s.len(), Default::default());
            for &x in s {
                if seen.insert(x) {
                    w_order.push(x);
                }
            }
        }
        let mut frontier: Vec<u32> = w_order.clone();
        let mut rounds_done = 0usize;

        for i in 1..=k {
            let mut nf_set: HashSet<u32> = HashSet::default();
            let mut next_frontier: Vec<u32> = Vec::new();
            for &u in &frontier {
                let (start, end) = (self.g.indptr[u as usize], self.g.indptr[u as usize + 1]);
                for e in start..end {
                    let v = self.g.indices[e];
                    let w = self.g.weights[e];
                    let passed = self.try_relax(u, v, w);
                    if passed && self.key(v) < b && nf_set.insert(v) {
                        next_frontier.push(v);
                    }
                }
            }
            rounds_done = i;
            for &v in &next_frontier {
                if w_set.insert(v) {
                    w_order.push(v);
                }
            }
            if w_set.len() > k * s.len() {
                // Early exit: P = S covers everything (Lemma 3.2 case 1).
                return (s.to_vec(), w_order);
            }
            if next_frontier.is_empty() {
                // Bellman-Ford converged below B: nothing left to relax.
                break;
            }
            // Lazy stop: the frontier is not shrinking, so trees are still
            // growing; cut the rounds and accept a lower pivot threshold.
            if self.cfg.lazy_pivots && i >= 2 && i < k && next_frontier.len() >= frontier.len() {
                break;
            }
            frontier = next_frontier;
        }

        // Tight forest F over W; pivot threshold = rounds actually run.
        let threshold = rounds_done.max(1);
        let mut children: HashMap<u32, Vec<u32>> = HashMap::default();
        let mut has_tight_parent: HashSet<u32> = HashSet::default();
        for &v in &w_order {
            let up = self.pred[v as usize];
            if up >= 0 && w_set.contains(&(up as u32)) {
                let u = up as u32;
                let (start, end) = (self.g.indptr[u as usize], self.g.indptr[u as usize + 1]);
                for e in start..end {
                    let vv = self.g.indices[e];
                    let w = self.g.weights[e];
                    #[allow(clippy::float_cmp)]
                    let tight = vv == v
                        && self.dhat[v as usize] == self.dhat[u as usize] + w
                        && self.hops[v as usize] == self.hops[u as usize] + 1;
                    if tight {
                        children.entry(u).or_default().push(v);
                        has_tight_parent.insert(v);
                        break;
                    }
                }
            }
        }

        let mut p: Vec<u32> = Vec::new();
        for &u in s {
            if has_tight_parent.contains(&u) {
                continue;
            }
            let mut size = 0usize;
            let mut stack = vec![u];
            while let Some(x) = stack.pop() {
                size += 1;
                if let Some(ch) = children.get(&x) {
                    stack.extend_from_slice(ch);
                }
            }
            if size >= threshold {
                p.push(u);
            }
        }

        (p, w_order)
    }

    /// Paper BaseCase (Algorithm 2): mini-Dijkstra from a singleton,
    /// truncated at k+1 settled vertices.
    fn base_case(&mut self, b: Key, s: &[u32]) -> (Key, Vec<u32>) {
        debug_assert!(s.len() == 1);
        let x = s[0];
        let k = self.k;

        let mut u0: Vec<u32> = vec![x];
        let mut heap = std::mem::take(&mut self.scratch.heap);
        heap.clear();
        self.pop_epoch += 1;
        let epoch = self.pop_epoch;
        self.pop_stamp[x as usize] = epoch;

        heap.push((self.dhat[x as usize], self.hops[x as usize], x));

        while !heap.is_empty() && u0.len() < k + 1 {
            let (klen, khops, u) = heap.pop().unwrap();
            if klen.total_cmp(&self.dhat[u as usize]) != std::cmp::Ordering::Equal
                || khops != self.hops[u as usize]
            {
                continue; // stale: u's label improved after this push
            }
            if self.pop_stamp[u as usize] != epoch {
                self.pop_stamp[u as usize] = epoch;
                u0.push(u);
            }
            self.relax_bounded(u, b, &mut heap);
        }

        let (bp, u_out) = if u0.len() <= k {
            (b, u0)
        } else {
            let bp = u0.iter().map(|&v| self.key(v)).max().unwrap();
            let filtered = u0
                .iter()
                .copied()
                .filter(|&v| self.key(v) < bp)
                .collect::<Vec<u32>>();
            (bp, filtered)
        };

        for &v in &u_out {
            self.settle(v);
        }
        self.scratch.heap = heap;
        (bp, u_out)
    }

    /// Bounded multi-source Dijkstra: a valid BMSSP oracle for any
    /// subproblem (returns B' = B and the complete U = T_<B(S)).
    ///
    /// Correctness: the precondition gives that every incomplete v with
    /// d(v) < B has its shortest path through a complete y in S; y sits in
    /// the heap at its true key, so the standard Dijkstra induction applies
    /// to the offset multi-source run, with the bound B only suppressing
    /// labels that U = T_<B(B'=B)(S) never contains.
    fn dijkstra_base(&mut self, b: Key, s: &[u32]) -> (Key, Vec<u32>) {
        let mut u0: Vec<u32> = Vec::new();
        let mut heap = std::mem::take(&mut self.scratch.heap);
        heap.clear();
        self.pop_epoch += 1;
        let epoch = self.pop_epoch;

        for &x in s {
            heap.push((self.dhat[x as usize], self.hops[x as usize], x));
        }

        while let Some((klen, khops, u)) = heap.pop() {
            if klen.total_cmp(&self.dhat[u as usize]) != std::cmp::Ordering::Equal
                || khops != self.hops[u as usize]
            {
                continue; // stale: u's label improved after this push
            }
            if self.pop_stamp[u as usize] != epoch {
                self.pop_stamp[u as usize] = epoch;
                u0.push(u);
            }
            self.relax_bounded(u, b, &mut heap);
        }

        for &v in &u0 {
            self.settle(v);
        }
        count!(self, oracle_settled, u0.len());
        self.scratch.heap = heap;
        (b, u0)
    }

    /// Shared relaxation step of the two heap oracles: BaseCase semantics,
    /// i.e. the relaxation itself is gated by `vkey < B` (Algorithm 2 L8).
    #[inline]
    fn relax_bounded(&mut self, u: u32, b: Key, heap: &mut KeyHeap) {
        let (start, end) = (self.g.indptr[u as usize], self.g.indptr[u as usize + 1]);
        count!(self, edge_scans, end - start);
        for e in start..end {
            let v = self.g.indices[e];
            let w = self.g.weights[e];
            let cand_len = self.dhat[u as usize] + w;
            let cand_hops = self.hops[u as usize] + 1;
            let cand = Key {
                len: cand_len,
                hops: cand_hops,
                id: u as i64,
            };
            let cur = Key {
                len: self.dhat[v as usize],
                hops: self.hops[v as usize],
                id: self.pred[v as usize],
            };
            let vkey = Key {
                len: cand_len,
                hops: cand_hops,
                id: v as i64,
            };
            if cand <= cur && vkey < b {
                self.dhat[v as usize] = cand_len;
                self.hops[v as usize] = cand_hops;
                self.pred[v as usize] = u as i64;
                heap.push((cand_len, cand_hops, v));
            }
        }
    }

    /// BMSSP (Algorithm 3), with the configured oracle switch.
    fn bmssp(&mut self, l: usize, b: Key, s: &[u32]) -> (Key, Vec<u32>) {
        count!(self, bmssp_calls, 1);
        let use_dijkstra = (l as i32) <= self.cfg.hybrid_max_level
            || (self.cfg.hybrid_frontier > 0 && s.len() <= self.cfg.hybrid_frontier);
        if use_dijkstra {
            count!(self, oracle_calls, 1);
            return phase!(self, dijkstra_oracle, self.dijkstra_base(b, s));
        }
        if l == 0 {
            count!(self, basecase_calls, 1);
            return phase!(self, base_case, self.base_case(b, s));
        }

        let (k, t) = (self.k, self.t);
        count!(self, findpivots_calls, 1);
        let (p, w_order) = phase!(self, find_pivots, self.find_pivots(b, s));

        let shift = (l - 1) * t;
        let m_cap = if shift >= 63 {
            self.g.n
        } else {
            std::cmp::min(1usize << shift, self.g.n)
        }
        .max(1);
        let mut d = Q::new(m_cap, b);

        for &x in &p {
            let kx = self.key(x);
            d.insert(x, kx, &mut self.rng);
        }

        let bp0 = p.iter().map(|&x| self.key(x)).min().unwrap_or(b);

        let mut u_set: HashSet<u32> = HashSet::default();
        let mut u_order: Vec<u32> = Vec::new();
        let mut bp_last = bp0;
        let lt = l * t;
        let bound_cap: u128 = if lt >= 100 {
            u128::MAX
        } else {
            (k as u128) << lt
        };

        while (u_order.len() as u128) < bound_cap && !d.is_empty() {
            count!(self, q_pulls, 1);
            let (si, bi) = phase!(self, q_pull, d.pull(&mut self.rng));
            count!(self, q_pulled_keys, si.len());
            verify_assert!(!si.is_empty());

            // Settled-vertex filter (AUDIT.md F3): keys whose label was
            // finalized by a sibling call are dropped, not recursed on.
            let si_fresh: Vec<u32> = si
                .iter()
                .copied()
                .filter(|&x| !self.settled[x as usize])
                .collect();
            let (bp_i, ui) = if si_fresh.is_empty() {
                (bi, Vec::new())
            } else {
                self.bmssp(l - 1, bi, &si_fresh)
            };
            for &x in &ui {
                if u_set.insert(x) {
                    u_order.push(x);
                }
            }
            bp_last = bp_i;

            #[cfg(feature = "phase-timer")]
            let __relax_t0 = std::time::Instant::now();
            let mut kk: Vec<(u32, Key)> = Vec::new();
            for &u in &ui {
                let (start, end) = (self.g.indptr[u as usize], self.g.indptr[u as usize + 1]);
                count!(self, edge_scans, end - start);
                for e in start..end {
                    let v = self.g.indices[e];
                    let w = self.g.weights[e];
                    let passed = self.try_relax(u, v, w);
                    if passed {
                        count!(self, relaxations, 1);
                        let vkey = self.key(v);
                        if bi <= vkey && vkey < b {
                            count!(self, q_inserts, 1);
                            d.insert(v, vkey, &mut self.rng);
                        } else if bp_i <= vkey && vkey < bi {
                            kk.push((v, vkey));
                        }
                    }
                }
            }

            let mut prepend = kk;
            for &x in &si_fresh {
                let kx = self.key(x);
                if bp_i <= kx && kx < bi {
                    prepend.push((x, kx));
                }
            }
            #[cfg(feature = "phase-timer")]
            {
                self.phase.relax_loop += __relax_t0.elapsed().as_secs_f64();
            }
            if !prepend.is_empty() {
                count!(self, q_prepend_items, prepend.len());
                phase!(self, q_batch_prepend, d.batch_prepend(&prepend, &mut self.rng));
            }
        }

        let bp = std::cmp::min(bp_last, b);

        let mut result_set = u_set;
        let mut result_order = u_order;
        for &x in &w_order {
            if self.key(x) < bp && result_set.insert(x) {
                result_order.push(x);
                self.settle(x);
            }
        }

        (bp, result_order)
    }
}

/// Map transformed-graph predecessors back to the original graph (same walk
/// as the mainline's `recover_pred`).
fn recover_pred_transformed(
    pred_t: &[i64],
    dhat_t: &[f64],
    rep: &[u32],
    owner: &[u32],
    n: usize,
    source: usize,
) -> Vec<i32> {
    let mut pred = vec![-1i32; n];
    for (v, pv) in pred.iter_mut().enumerate() {
        if v == source {
            continue;
        }
        let r = rep[v] as usize;
        if !dhat_t[r].is_finite() {
            continue;
        }
        let mut cur = r;
        loop {
            let p = pred_t[cur];
            if p < 0 {
                break;
            }
            let p = p as usize;
            if owner[p] as usize != v {
                *pv = owner[p] as i32;
                break;
            }
            cur = p;
        }
    }
    pred
}

/// Run the configured engine on `g` from `source`.
pub fn run<Q: DQueue>(
    g: &Csr,
    source: usize,
    seed: u64,
    cfg: Config,
) -> Result<VariantRun, BmsspError> {
    if source >= g.n {
        return Err(BmsspError::SourceOutOfRange);
    }
    for &w in &g.weights {
        if w < 0.0 || !w.is_finite() {
            return Err(BmsspError::BadWeight);
        }
    }

    #[cfg(feature = "phase-timer")]
    let __total_t0 = std::time::Instant::now();
    #[allow(unused_mut)]
    let mut run = if cfg.transform {
        let mut eng_phase = EnginePhases::default();
        let tr = phase!(
            EngineHolder { phase: &mut eng_phase },
            transform,
            transform_to_constant_degree(g, source)
        );
        let (k, t, levels) = compute_params(tr.g2.n, cfg.kt_override);
        let mut eng = Engine::<Q>::new(&tr.g2, tr.source2, k, t, seed, cfg);
        eng.phase = eng_phase;
        eng.bmssp(levels, KEY_INF, &[tr.source2]);
        let (dist, pred) = phase!(eng, finalize, {
            let dist: Vec<f64> = (0..g.n).map(|v| eng.dhat[tr.rep[v] as usize]).collect();
            let pred =
                recover_pred_transformed(&eng.pred, &eng.dhat, &tr.rep, &tr.owner, g.n, source);
            (dist, pred)
        });
        VariantRun {
            dist,
            pred,
            k,
            t,
            levels,
            n_inner: tr.g2.n,
            phase: eng.phase,
            cnt: eng.cnt,
            total_seconds: 0.0,
        }
    } else {
        let (k, t, levels) = compute_params(g.n, cfg.kt_override);
        let mut eng = Engine::<Q>::new(g, source as u32, k, t, seed, cfg);
        eng.bmssp(levels, KEY_INF, &[source as u32]);
        let pred: Vec<i32> = phase!(
            eng,
            finalize,
            eng.pred
                .iter()
                .map(|&p| if p < 0 { -1 } else { p as i32 })
                .collect()
        );
        VariantRun {
            dist: eng.dhat,
            pred,
            k,
            t,
            levels,
            n_inner: g.n,
            phase: eng.phase,
            cnt: eng.cnt,
            total_seconds: 0.0,
        }
    };
    #[cfg(feature = "phase-timer")]
    {
        run.total_seconds = __total_t0.elapsed().as_secs_f64();
    }
    Ok(run)
}

/// Adapter so the `phase!` macro (which writes `$st.phase.$field`) can time
/// code that runs before an [`Engine`] exists.
#[cfg(feature = "phase-timer")]
struct EngineHolder<'a> {
    phase: &'a mut EnginePhases,
}
