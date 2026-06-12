//! Rust port of the BMSSP algorithm of `python/logtwothirds/_reference.py`
//! (Duan–Mao–Mao–Shu–Yin, arXiv:2504.17033v2, as distilled in ALGORITHM.md).
//!
//! The port is semantically 1:1 with the reference. Every place where the
//! reference's behavior depends on an *order* (Python dict insertion order,
//! list order, heap extraction order) is reproduced exactly; the two stand-ins
//! for genuinely unspecified behavior are pinned to explicit choices that the
//! differential test also forces on the Python side:
//!
//! * `random.randint` (quickselect pivots) -> [`SplitMix64`] seeded by the
//!   caller;
//! * builtin-`set` iteration order (the reference's `U` / `result_U` in
//!   `bmssp`, leaked through `list(result_U)`) -> insertion order.
//!
//! Allowed optimizations that do not change any result: u32 vertex ids, CSR
//! vectors instead of per-vertex Python lists, and flat `Vec` state arrays.

use crate::block_queue::{BlockDs, Key, SplitMix64, INF_INT, KEY_INF};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// CSR directed graph, out-adjacency (the reference's `Graph`).
pub struct Csr {
    pub n: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub weights: Vec<f64>,
}

/// Build a CSR graph from an edge list, grouping edges by source while
/// preserving their relative order — exactly what the reference's
/// `build_graph` produces. Panics on out-of-range endpoints (callers
/// validate weights separately, as `_run_sssp` does).
pub fn build_csr(n: usize, edges: &[(u32, u32, f64)]) -> Csr {
    let mut indptr = vec![0usize; n + 1];
    for &(u, v, _w) in edges {
        assert!((u as usize) < n && (v as usize) < n, "edge endpoint out of range");
        indptr[u as usize + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let mut cursor = indptr.clone();
    let mut indices = vec![0u32; edges.len()];
    let mut weights = vec![0.0f64; edges.len()];
    for &(u, v, w) in edges {
        let slot = cursor[u as usize];
        cursor[u as usize] += 1;
        indices[slot] = v;
        weights[slot] = w;
    }
    Csr { n, indptr, indices, weights }
}

/// Result of [`transform_to_constant_degree`].
pub struct Transformed {
    pub g2: Csr,
    pub source2: u32,
    /// `rep[v]`: the designated cycle vertex of original vertex `v`.
    pub rep: Vec<u32>,
    /// `owner[x]`: the original vertex whose cycle contains transformed
    /// vertex `x` (not present in the reference; used only to map
    /// predecessors back to the original graph, which the reference never
    /// does).
    pub owner: Vec<u32>,
}

/// Constant-degree transform (ALGORITHM.md S1.1). Port of
/// `transform_to_constant_degree`: for each original vertex, one cycle vertex
/// per incident edge-endpoint in *edge-enumeration order* (out-slot before
/// in-slot for self-loops), zero-weight directed cycles for degree >= 2, then
/// the cross edges in edge order. Vertex ids are assigned exactly as the
/// reference does (vertices in order, then their slots in order).
pub fn transform_to_constant_degree(g: &Csr, source: usize) -> Transformed {
    let n = g.n;
    // Edge list in CSR enumeration order (the reference's `edges`).
    let m = g.indices.len();
    let mut edge_src = vec![0u32; m];
    for u in 0..n {
        edge_src[g.indptr[u]..g.indptr[u + 1]].fill(u as u32);
    }

    // Incident-endpoint counts (the reference's `len(slots[v])`).
    let mut degree = vec![0usize; n];
    for ei in 0..m {
        degree[edge_src[ei] as usize] += 1;
        degree[g.indices[ei] as usize] += 1;
    }

    // Assign transformed ids: for each vertex in order, one id per slot in
    // slot order (or a single id if isolated). `base[v]` is the first id.
    let mut base = vec![0usize; n];
    let mut rep = vec![0u32; n];
    let mut new_n = 0usize;
    for v in 0..n {
        base[v] = new_n;
        rep[v] = new_n as u32;
        new_n += degree[v].max(1);
    }

    let mut owner = vec![0u32; new_n];
    for v in 0..n {
        owner[base[v]..base[v] + degree[v].max(1)].fill(v as u32);
    }

    // Slot positions per edge occurrence, in the reference's append order:
    // for each edge ei, the out-slot of u is appended, then the in-slot of v.
    let mut cursor = vec![0usize; n];
    let mut out_id = vec![0u32; m];
    let mut in_id = vec![0u32; m];
    for ei in 0..m {
        let u = edge_src[ei] as usize;
        out_id[ei] = (base[u] + cursor[u]) as u32;
        cursor[u] += 1;
        let v = g.indices[ei] as usize;
        in_id[ei] = (base[v] + cursor[v]) as u32;
        cursor[v] += 1;
    }

    // New edge list: zero-weight cycles first (vertices in order), then the
    // cross edges (edges in order) — the reference's `new_edges` order.
    let mut new_edges: Vec<(u32, u32, f64)> = Vec::new();
    for v in 0..n {
        let d = degree[v];
        if d >= 2 {
            for i in 0..d {
                let a = (base[v] + i) as u32;
                let b = (base[v] + (i + 1) % d) as u32;
                new_edges.push((a, b, 0.0));
            }
        }
    }
    for ei in 0..m {
        new_edges.push((out_id[ei], in_id[ei], g.weights[ei]));
    }

    let g2 = build_csr(new_n, &new_edges);
    Transformed {
        g2,
        source2: rep[source],
        rep,
        owner,
    }
}

/// `k, t, L` from the vertex count of the transformed graph (`compute_params`
/// / SPEC.md S5), with an optional `(k, t)` override that mirrors the test
/// suite's `_small_params` monkeypatch (`L` always follows the same
/// `ceil(log_n / t)` formula).
pub fn compute_params(n: usize, kt_override: Option<(usize, usize)>) -> (usize, usize, usize) {
    let log_n = f64::max(1.0, (n.max(2) as f64).log2());
    let (k, t) = match kt_override {
        Some((k, t)) => (k, t),
        None => {
            let k = (log_n.powf(1.0 / 3.0).floor() as i64).max(1) as usize;
            let t = (log_n.powf(2.0 / 3.0).floor() as i64).max(1) as usize;
            (k, t)
        }
    };
    let l = ((log_n / t as f64).ceil() as i64).max(1) as usize;
    (k, t, l)
}

/// Operation counters (`OpCounter`, SPEC.md S7.a).
#[derive(Default, Debug, Clone)]
pub struct OpCounter {
    pub edge_scans: u64,
    pub relaxations: u64,
    pub ds_inserts: u64,
    pub ds_prepend_items: u64,
    pub ds_pulls: u64,
    pub ds_pulled_items: u64,
    pub heap_ops: u64,
    pub findpivots_calls: u64,
    pub bmssp_calls: u64,
    pub basecase_calls: u64,
}

/// True iff the dhat values in the settlement log are non-decreasing
/// (`is_globally_sorted`).
pub fn is_globally_sorted(events: &[(u32, f64)]) -> bool {
    events.windows(2).all(|w| w[0].1 <= w[1].1)
}

/// Per-run mutable state (`State`).
struct State<'g> {
    g: &'g Csr,
    dhat: Vec<f64>,
    hops: Vec<i64>,
    pred: Vec<i64>,
    k: usize,
    t: usize,
    counter: OpCounter,
    settle_log: Vec<(u32, f64)>,
    settled: Vec<bool>,
    rng: SplitMix64,
}

impl<'g> State<'g> {
    fn new(g: &'g Csr, source: u32, k: usize, t: usize, seed: u64) -> Self {
        let n = g.n;
        let mut dhat = vec![f64::INFINITY; n];
        let mut hops = vec![INF_INT; n];
        dhat[source as usize] = 0.0;
        hops[source as usize] = 0;
        State {
            g,
            dhat,
            hops,
            pred: vec![-1; n],
            k,
            t,
            counter: OpCounter::default(),
            settle_log: Vec::new(),
            settled: vec![false; n],
            rng: SplitMix64::new(seed),
        }
    }

    /// Vertex v's label key `(dhat[v], hops[v], v)` (`key`).
    #[inline]
    fn key(&self, v: u32) -> Key {
        Key {
            len: self.dhat[v as usize],
            hops: self.hops[v as usize],
            id: v as i64,
        }
    }

    /// Shared relaxation helper (`try_relax`): the "<=" test of Remark 3.4,
    /// never gated on a bound. Returns whether the relaxation passed.
    #[inline]
    fn try_relax(&mut self, u: u32, v: u32, w: f64) -> bool {
        self.counter.edge_scans += 1;
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
        if cand <= cur {
            self.dhat[v as usize] = cand_len;
            self.hops[v as usize] = cand_hops;
            self.pred[v as usize] = u as i64;
            self.counter.relaxations += 1;
            true
        } else {
            false
        }
    }

    /// Append a settlement event for `v` if not already settled (`_settle`).
    fn settle(&mut self, v: u32) {
        if !self.settled[v as usize] {
            self.settled[v as usize] = true;
            self.settle_log.push((v, self.dhat[v as usize]));
        }
    }

    /// FindPivots(B, S) — ALGORITHM.md S4.1, Algorithm 1 (`find_pivots`).
    fn find_pivots(&mut self, b: Key, s: &[u32]) -> (Vec<u32>, Vec<u32>) {
        self.counter.findpivots_calls += 1;
        let k = self.k;

        let mut w_set: HashSet<u32> = s.iter().copied().collect();
        // L2-3: W <- S; W_0 <- S (order-preserving dedup, dict.fromkeys).
        let mut w_order: Vec<u32> = Vec::with_capacity(s.len());
        {
            let mut seen: HashSet<u32> = HashSet::with_capacity(s.len());
            for &x in s {
                if seen.insert(x) {
                    w_order.push(x);
                }
            }
        }
        let mut frontier: Vec<u32> = w_order.clone();

        for _i in 0..k {
            // L4: for i <- 1 to k
            let mut nf_set: HashSet<u32> = HashSet::new();
            let mut next_frontier: Vec<u32> = Vec::new();
            for &u in &frontier {
                // L6: edges (u, v) with u in W_{i-1}
                let (start, end) = (self.g.indptr[u as usize], self.g.indptr[u as usize + 1]);
                for e in start..end {
                    let v = self.g.indices[e];
                    let w = self.g.weights[e];
                    // L7-8: relax (updates dhat even if the candidate is >= B)
                    let passed = self.try_relax(u, v, w);
                    // L9: bound check uses v's own key (see reference NOTE).
                    if passed && self.key(v) < b && nf_set.insert(v) {
                        next_frontier.push(v);
                    }
                }
            }
            for &v in &next_frontier {
                // L11: W <- W u W_i
                if w_set.insert(v) {
                    w_order.push(v);
                }
            }
            if w_set.len() > k * s.len() {
                // L12-14: early exit, P <- S
                return (s.to_vec(), w_order);
            }
            frontier = next_frontier;
        }

        // L15: recover the tight-edge forest F as a child map (under
        // Assumption 2.1, v's unique tight in-edge is (pred[v], v)).
        let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut has_tight_parent: HashSet<u32> = HashSet::new();
        for &v in &w_order {
            let up = self.pred[v as usize];
            if up >= 0 && w_set.contains(&(up as u32)) {
                let u = up as u32;
                let (start, end) = (self.g.indptr[u as usize], self.g.indptr[u as usize + 1]);
                for e in start..end {
                    let vv = self.g.indices[e];
                    let w = self.g.weights[e];
                    #[allow(clippy::float_cmp)] // exact tightness test, as in the reference
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

        // L16: P = roots of S-rooted trees in F with >= k vertices.
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
            if size >= k {
                p.push(u);
            }
        }

        (p, w_order) // L17
    }

    /// BaseCase(B, S) — ALGORITHM.md S4.2, Algorithm 2 (`base_case`).
    fn base_case(&mut self, b: Key, s: &[u32]) -> (Key, Vec<u32>) {
        self.counter.basecase_calls += 1;
        assert!(s.len() == 1, "BaseCase requires |S| == 1");
        let x = s[0];
        let k = self.k;

        let mut u0: Vec<u32> = vec![x]; // L2
        let mut in_u0: HashSet<u32> = HashSet::new();
        in_u0.insert(x);

        // Binary heap with lazy deletion; `best` tracks each vertex's current
        // key so stale pops are skipped. Keys are unique (they embed the
        // vertex id), so any min-heap yields the same extraction sequence as
        // Python's heapq.
        let mut heap: BinaryHeap<std::cmp::Reverse<(Key, u32)>> = BinaryHeap::new();
        let mut best: HashMap<u32, Key> = HashMap::new();

        let kx0 = self.key(x);
        best.insert(x, kx0);
        heap.push(std::cmp::Reverse((kx0, x)));
        self.counter.heap_ops += 1; // L3: H <- {<x, dhat[x]>}

        while !heap.is_empty() && u0.len() < k + 1 {
            // L4
            let std::cmp::Reverse((kx, u)) = heap.pop().unwrap();
            self.counter.heap_ops += 1;
            if best.get(&u) != Some(&kx) {
                continue; // stale entry
            }
            if in_u0.insert(u) {
                u0.push(u); // L6
            }
            let (start, end) = (self.g.indptr[u as usize], self.g.indptr[u as usize + 1]);
            for e in start..end {
                // L7
                let v = self.g.indices[e];
                let w = self.g.weights[e];
                self.counter.edge_scans += 1;
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
                // L8: here the relaxation itself is gated by "< B", and the
                // gate compares v's own key (see reference NOTE).
                if cand <= cur && vkey < b {
                    self.dhat[v as usize] = cand_len;
                    self.hops[v as usize] = cand_hops;
                    self.pred[v as usize] = u as i64;
                    self.counter.relaxations += 1;
                    best.insert(v, vkey);
                    heap.push(std::cmp::Reverse((vkey, v)));
                    self.counter.heap_ops += 1; // L10-13 (lazy DecreaseKey)
                }
            }
        }

        let (bp, u_out) = if u0.len() <= k {
            (b, u0) // L14-15
        } else {
            let bp = u0.iter().map(|&v| self.key(v)).max().unwrap(); // L16
            let filtered = u0
                .iter()
                .copied()
                .filter(|&v| self.key(v) < bp)
                .collect::<Vec<u32>>(); // L17
            (bp, filtered)
        };

        for &v in &u_out {
            self.settle(v); // SPEC.md S7.b item 1
        }

        (bp, u_out)
    }

    /// BMSSP(l, B, S) — ALGORITHM.md S4.3, Algorithm 3 (`bmssp`).
    fn bmssp(&mut self, l: usize, b: Key, s: &[u32]) -> (Key, Vec<u32>) {
        self.counter.bmssp_calls += 1;
        if l == 0 {
            return self.base_case(b, s); // L2-3
        }

        let (k, t) = (self.k, self.t);
        let (p, w_order) = self.find_pivots(b, s); // L4

        // M = 2^((l-1)*t), capped at n (SPEC.md S5).
        let shift = (l - 1) * t;
        let m_cap = if shift >= 63 {
            self.g.n
        } else {
            std::cmp::min(1usize << shift, self.g.n)
        }
        .max(1);
        let mut d = BlockDs::new(m_cap, b); // L5

        for &x in &p {
            // L6
            let kx = self.key(x);
            d.insert(x, kx, &mut self.rng);
            self.counter.ds_inserts += 1;
        }

        // L7 (+ footnote: if P = empty, B'_0 <- B).
        let bp0 = p.iter().map(|&x| self.key(x)).min().unwrap_or(b);

        // The reference's `U` is a Python set whose iteration order leaks via
        // `list(result_U)`; here it is pinned to insertion order (the
        // differential test patches the reference's `set` to match).
        let mut u_set: HashSet<u32> = HashSet::new();
        let mut u_order: Vec<u32> = Vec::new();
        let mut bp_last = bp0;
        let lt = l * t;
        let bound_cap: u128 = if lt >= 100 {
            u128::MAX
        } else {
            (k as u128) << lt
        };

        while (u_order.len() as u128) < bound_cap && !d.is_empty() {
            // L8
            let (si, bi) = d.pull(&mut self.rng); // L10 (Pull returns (S', x))
            self.counter.ds_pulls += 1;
            self.counter.ds_pulled_items += si.len() as u64;
            assert!(!si.is_empty(), "Pull returned an empty set while D was non-empty");

            // TODO(spec) carried over from the reference: filter out vertices
            // already settled by a sibling call (see `_reference.py`).
            let si_fresh: Vec<u32> = si
                .iter()
                .copied()
                .filter(|&x| !self.settled[x as usize])
                .collect();
            let (bp_i, ui) = if si_fresh.is_empty() {
                (bi, Vec::new())
            } else {
                self.bmssp(l - 1, bi, &si_fresh) // L11
            };
            debug_assert!(
                ui.iter().all(|x| !u_set.contains(x)),
                "the U_i must be pairwise disjoint"
            );
            for &x in &ui {
                // L12: U <- U u U_i
                if u_set.insert(x) {
                    u_order.push(x);
                }
            }
            bp_last = bp_i;

            let mut kk: Vec<(u32, Key)> = Vec::new();
            for &u in &ui {
                // L14: for edge e = (u, v) with u in U_i
                let (start, end) = (self.g.indptr[u as usize], self.g.indptr[u as usize + 1]);
                for e in start..end {
                    let v = self.g.indices[e];
                    let w = self.g.weights[e];
                    let passed = self.try_relax(u, v, w); // L15-16
                    if passed {
                        // Bucket decision uses v's OWN key (see reference NOTE).
                        let vkey = self.key(v);
                        if bi <= vkey && vkey < b {
                            // L17-18
                            d.insert(v, vkey, &mut self.rng);
                            self.counter.ds_inserts += 1;
                        } else if bp_i <= vkey && vkey < bi {
                            // L19-20
                            kk.push((v, vkey));
                        }
                    }
                }
            }

            // L21: K plus the unfinished part of the pulled batch.
            let mut prepend = kk;
            for &x in &si_fresh {
                let kx = self.key(x);
                if bp_i <= kx && kx < bi {
                    prepend.push((x, kx));
                }
            }
            if !prepend.is_empty() {
                d.batch_prepend(&prepend, &mut self.rng);
                self.counter.ds_prepend_items += prepend.len() as u64;
            }
        }

        let bp = std::cmp::min(bp_last, b); // L22
        assert!(bp <= b);

        // L22: U <- U u {x in W : key(x) < B'}, settling the additions
        // (SPEC.md S7.b item 2). `result_U` starts as a copy of `U`.
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

/// Errors of [`sssp_bmssp`] (mirroring `_run_sssp`'s raises).
#[derive(Debug, PartialEq, Eq)]
pub enum BmsspError {
    /// `source` out of range (Python `IndexError`).
    SourceOutOfRange,
    /// A negative, NaN, or infinite edge weight (Python `ValueError`).
    BadWeight,
}

/// Full output of a BMSSP run.
#[derive(Debug)]
pub struct BmsspRun {
    /// Shortest-path lengths on the original graph (`inf` if unreachable).
    pub dist: Vec<f64>,
    /// Predecessor of each original vertex on a shortest path (`-1` for the
    /// source and unreachable vertices). Not part of the reference; derived
    /// from the transformed graph's predecessor labels.
    pub pred: Vec<i32>,
    /// Settlement-order log over *transformed* vertex ids (SPEC.md S7.b).
    pub settle_log: Vec<(u32, f64)>,
    pub counter: OpCounter,
    pub k: usize,
    pub t: usize,
    pub levels: usize,
    /// Vertex count of the transformed graph.
    pub n_transformed: usize,
}

/// Map transformed-graph predecessor labels back to the original graph: walk
/// the predecessor chain from `rep[v]` through `v`'s zero-weight cycle until
/// it leaves `v`'s cycle; the owner of that vertex is `v`'s predecessor.
fn recover_pred(st: &State<'_>, tr: &Transformed, n: usize, source: usize) -> Vec<i32> {
    let mut pred = vec![-1i32; n];
    for (v, pv) in pred.iter_mut().enumerate() {
        if v == source {
            continue;
        }
        let r = tr.rep[v] as usize;
        if !st.dhat[r].is_finite() {
            continue;
        }
        let mut cur = r;
        loop {
            let p = st.pred[cur];
            if p < 0 {
                break; // defensive: only the transformed source has pred -1
            }
            let p = p as usize;
            if tr.owner[p] as usize != v {
                *pv = tr.owner[p] as i32;
                break;
            }
            cur = p;
        }
    }
    pred
}

/// Top-level SSSP via BMSSP (`_run_sssp` / `sssp_instrumented`).
///
/// `seed` drives the quickselect pivots (any value yields a correct run;
/// the differential test pins it to match a patched reference run).
/// `kt_override` forces `(k, t)` as the test suite's `_small_params` does.
pub fn sssp_bmssp(
    g: &Csr,
    source: usize,
    seed: u64,
    kt_override: Option<(usize, usize)>,
) -> Result<BmsspRun, BmsspError> {
    if source >= g.n {
        return Err(BmsspError::SourceOutOfRange);
    }
    for &w in &g.weights {
        // Rejects negatives, NaN, and inf (`w < 0 or not isfinite(w)`).
        if w < 0.0 || !w.is_finite() {
            return Err(BmsspError::BadWeight);
        }
    }

    let tr = transform_to_constant_degree(g, source); // step 1
    let (k, t, levels) = compute_params(tr.g2.n, kt_override); // step 2

    let mut st = State::new(&tr.g2, tr.source2, k, t, seed); // step 3
    st.bmssp(levels, KEY_INF, &[tr.source2]); // step 4: BMSSP(L, inf, {s'})

    let dist: Vec<f64> = (0..g.n).map(|v| st.dhat[tr.rep[v] as usize]).collect(); // step 5
    let pred = recover_pred(&st, &tr, g.n, source);
    let n_transformed = tr.g2.n;
    Ok(BmsspRun {
        dist,
        pred,
        settle_log: st.settle_log,
        counter: st.counter,
        k,
        t,
        levels,
        n_transformed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(n: usize, edges: &[(u32, u32, f64)], source: usize) -> BmsspRun {
        let g = build_csr(n, edges);
        sssp_bmssp(&g, source, 0x5EED, None).unwrap()
    }

    /// Simple Dijkstra oracle on the original graph.
    fn oracle(n: usize, edges: &[(u32, u32, f64)], source: usize) -> Vec<f64> {
        let g = build_csr(n, edges);
        let mut dist = vec![f64::INFINITY; n];
        let mut heap = BinaryHeap::new();
        dist[source] = 0.0;
        heap.push(std::cmp::Reverse((ordered_float(0.0), source)));
        while let Some(std::cmp::Reverse((d, u))) = heap.pop() {
            let d = f64::from_bits(d);
            if d > dist[u] {
                continue;
            }
            for e in g.indptr[u]..g.indptr[u + 1] {
                let v = g.indices[e] as usize;
                let nd = d + g.weights[e];
                if nd < dist[v] {
                    dist[v] = nd;
                    heap.push(std::cmp::Reverse((ordered_float(nd), v)));
                }
            }
        }
        dist
    }

    /// Order-preserving bits for non-negative finite/inf f64.
    fn ordered_float(x: f64) -> u64 {
        x.to_bits()
    }

    #[test]
    fn single_vertex() {
        let r = run(1, &[], 0);
        assert_eq!(r.dist, vec![0.0]);
        assert_eq!(r.pred, vec![-1]);
    }

    #[test]
    fn single_edge() {
        let r = run(2, &[(0, 1, 3.5)], 0);
        assert_eq!(r.dist, vec![0.0, 3.5]);
        assert_eq!(r.pred, vec![-1, 0]);
    }

    #[test]
    fn two_vertex_zero_weight_cycle() {
        let r = run(2, &[(0, 1, 0.0), (1, 0, 0.0)], 0);
        assert_eq!(r.dist, vec![0.0, 0.0]);
    }

    #[test]
    fn chain_of_1000() {
        let edges: Vec<(u32, u32, f64)> = (0..999).map(|i| (i, i + 1, 1.0)).collect();
        let r = run(1000, &edges, 0);
        assert_eq!(r.dist[999], 999.0);
        assert_eq!(r.pred[999], 998);
    }

    #[test]
    fn unreachable_component() {
        let r = run(6, &[(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0)], 0);
        assert_eq!(&r.dist[..3], &[0.0, 1.0, 2.0]);
        assert!(r.dist[3..].iter().all(|d| d.is_infinite()));
        assert_eq!(&r.pred[3..], &[-1, -1, -1]);
    }

    #[test]
    fn self_loop_and_parallel_edges() {
        let r = run(2, &[(0, 0, 2.0), (0, 1, 5.0), (0, 1, 2.0)], 0);
        assert_eq!(r.dist, vec![0.0, 2.0]);
        assert_eq!(r.pred, vec![-1, 0]);
    }

    #[test]
    fn matches_oracle_on_random_graphs() {
        let mut rng = SplitMix64::new(7);
        for _case in 0..40 {
            let n = 1 + (rng.next_u64() % 120) as usize;
            let m = (rng.next_u64() % (3 * n as u64 + 1)) as usize;
            let mut edges = Vec::with_capacity(m);
            for _ in 0..m {
                let u = (rng.next_u64() % n as u64) as u32;
                let v = (rng.next_u64() % n as u64) as u32;
                let w = if rng.next_u64() % 10 == 0 {
                    0.0
                } else {
                    ((rng.next_u64() % 1000) + 1) as f64 / 1000.0
                };
                edges.push((u, v, w));
            }
            let source = (rng.next_u64() % n as u64) as usize;
            let got = run(n, &edges, source);
            let want = oracle(n, &edges, source);
            for (v, &w) in want.iter().enumerate() {
                assert!(
                    got.dist[v] == w || (got.dist[v].is_infinite() && w.is_infinite()),
                    "n={n} v={v}: {} vs {}",
                    got.dist[v],
                    w
                );
            }
        }
    }

    #[test]
    fn errors() {
        let g = build_csr(2, &[(0, 1, 1.0)]);
        assert_eq!(
            sssp_bmssp(&g, 5, 0, None).unwrap_err(),
            BmsspError::SourceOutOfRange
        );
        let g = build_csr(2, &[(0, 1, -1.0)]);
        assert_eq!(sssp_bmssp(&g, 0, 0, None).unwrap_err(), BmsspError::BadWeight);
    }

    #[test]
    fn settle_log_covers_reachable() {
        let edges: Vec<(u32, u32, f64)> = (0..49).map(|i| (i, i + 1, 1.0)).collect();
        let g = build_csr(50, &edges);
        let r = sssp_bmssp(&g, 0, 1, None).unwrap();
        let tr = transform_to_constant_degree(&g, 0);
        let settled: HashSet<u32> = r.settle_log.iter().map(|&(v, _)| v).collect();
        for v in 0..50 {
            assert!(settled.contains(&tr.rep[v]), "rep of {v} not settled");
        }
    }
}
