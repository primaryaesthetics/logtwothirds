//! Dijkstra's single-source shortest paths on a CSR graph.
//!
//! Uses an implicit **4-ary** min-heap with **lazy deletion**: a vertex may be
//! pushed multiple times as its tentative distance improves, and stale entries
//! are discarded when popped.
//!
//! The heap is stored **structure-of-arrays** — keys (`f64`) and vertex ids
//! (`u32`) in two parallel arrays. Sifting is comparison-bound and touches only
//! the dense keys array, halving the cache footprint of the hottest code versus
//! an array-of-16-byte-structs layout. Both arrays are pre-reserved by the
//! caller, so the relaxation loop performs **no allocations**.
//!
//! The relaxation loop and heap sifts use unchecked indexing for speed. This is
//! sound because [`dijkstra`] runs an up-front validation pass ([`validate`]
//! plus the fused edge scan) proving every CSR offset and vertex id is in
//! range; given that, all indices computed in the hot loop are in bounds, and
//! the heap never exceeds its reserved capacity of `nnz + 1`.

/// Signals that a structural or value precondition failed.
pub enum DijkstraError {
    /// A negative or NaN edge weight was found (maps to Python `ValueError`).
    NegativeWeight,
    /// `indptr`/`indices` are not a valid CSR structure (maps to `ValueError`).
    MalformedCsr,
}

/// A structure-of-arrays 4-ary min-heap keyed on `f64` distance.
///
/// `keys[i]` is the priority of the entry whose vertex id is `vals[i]`; the two
/// vectors are kept the same length. Reused across calls by the caller so its
/// backing allocations persist.
pub struct Heap {
    keys: Vec<f64>,
    vals: Vec<u32>,
}

impl Heap {
    /// A new empty heap with no backing allocation.
    pub fn new() -> Self {
        Heap {
            keys: Vec::new(),
            vals: Vec::new(),
        }
    }

    /// Ensure capacity for at least `cap` entries and empty the heap.
    #[inline]
    fn reset_with_capacity(&mut self, cap: usize) {
        self.keys.clear();
        self.vals.clear();
        if self.keys.capacity() < cap {
            self.keys.reserve(cap - self.keys.capacity());
            self.vals.reserve(cap - self.vals.capacity());
        }
    }

    /// Push `(key, val)`.
    ///
    /// # Safety
    /// The heap must have spare capacity (`len < capacity`). The caller reserves
    /// `nnz + 1` slots and performs at most that many pushes, so this holds.
    #[inline(always)]
    unsafe fn push(&mut self, key: f64, val: u32) {
        let i = self.keys.len();
        debug_assert!(i < self.keys.capacity());
        self.keys.set_len(i + 1);
        self.vals.set_len(i + 1);
        self.sift_up(i, key, val);
    }

    /// Pop the minimum `(key, val)`, or `None` if empty.
    #[inline(always)]
    fn pop(&mut self) -> Option<(f64, u32)> {
        let len = self.keys.len();
        if len == 0 {
            return None;
        }
        // SAFETY: `len >= 1`, so indices 0 and `len - 1` are valid.
        unsafe {
            let min_key = *self.keys.get_unchecked(0);
            let min_val = *self.vals.get_unchecked(0);
            let last_key = *self.keys.get_unchecked(len - 1);
            let last_val = *self.vals.get_unchecked(len - 1);
            self.keys.set_len(len - 1);
            self.vals.set_len(len - 1);
            if len > 1 {
                self.sift_down(last_key, last_val);
            }
            Some((min_key, min_val))
        }
    }

    /// Restore the heap property upward from index `i`.
    ///
    /// # Safety
    /// `i` must be `< len` and the arrays equal-length.
    #[inline(always)]
    unsafe fn sift_up(&mut self, mut i: usize, key: f64, val: u32) {
        while i > 0 {
            let parent = (i - 1) >> 2; // 4-ary: parent = (i-1)/4
            let pk = *self.keys.get_unchecked(parent);
            if key < pk {
                *self.keys.get_unchecked_mut(i) = pk;
                *self.vals.get_unchecked_mut(i) = *self.vals.get_unchecked(parent);
                i = parent;
            } else {
                break;
            }
        }
        *self.keys.get_unchecked_mut(i) = key;
        *self.vals.get_unchecked_mut(i) = val;
    }

    /// Restore the heap property downward, placing `(key, val)` from index 0.
    ///
    /// # Safety
    /// The arrays must be non-empty and equal-length.
    #[inline(always)]
    unsafe fn sift_down(&mut self, key: f64, val: u32) {
        let len = self.keys.len();
        let mut i = 0usize;
        loop {
            let first_child = 4 * i + 1;
            if first_child >= len {
                break;
            }
            // Smallest of the (up to four) children — reads only dense `keys`.
            let last_child = core::cmp::min(first_child + 4, len);
            let mut smallest = first_child;
            let mut smallest_key = *self.keys.get_unchecked(first_child);
            let mut c = first_child + 1;
            while c < last_child {
                let k = *self.keys.get_unchecked(c);
                if k < smallest_key {
                    smallest = c;
                    smallest_key = k;
                }
                c += 1;
            }
            if smallest_key < key {
                *self.keys.get_unchecked_mut(i) = smallest_key;
                *self.vals.get_unchecked_mut(i) = *self.vals.get_unchecked(smallest);
                i = smallest;
            } else {
                break;
            }
        }
        *self.keys.get_unchecked_mut(i) = key;
        *self.vals.get_unchecked_mut(i) = val;
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate the `indptr` structure of a CSR graph over `n` vertices: length
/// `n + 1`, non-decreasing, starts at 0, ends at `nnz`. This proves every
/// `indptr[u]..indptr[u+1]` slice is a valid range into `indices`/`weights`.
/// Neighbor-id bounds (`indices[e] < n`) are checked in the fused edge pass in
/// [`dijkstra`]; together they make the relaxation loop's indexing sound.
fn validate(indptr: &[i64], indices: &[i32], n: usize) -> Result<(), DijkstraError> {
    let nnz = indices.len();
    if indptr.len() != n + 1 {
        return Err(DijkstraError::MalformedCsr);
    }
    if indptr[0] != 0 || indptr[n] != nnz as i64 {
        return Err(DijkstraError::MalformedCsr);
    }
    // Non-decreasing and bounded by nnz (so every offset is a valid index).
    let mut prev: i64 = 0;
    for &p in &indptr[1..] {
        if p < prev || p > nnz as i64 {
            return Err(DijkstraError::MalformedCsr);
        }
        prev = p;
    }
    Ok(())
}

/// Issue non-faulting prefetch hints for the `dist[v]` slots of every neighbor
/// `v` in `indices[start..end]`. These are random locations in an array far
/// larger than L2, so prefetching them before the relaxation loop overlaps the
/// (otherwise serialized) cache misses. On non-x86 targets this is a no-op.
#[inline(always)]
unsafe fn prefetch_neighbors(dist: &[f64], indices: &[i32], start: usize, end: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let base = dist.as_ptr();
        for e in start..end {
            let v = *indices.get_unchecked(e) as usize;
            // SAFETY: `v < dist.len()` (validated); prefetch never dereferences,
            // so the computed address cannot fault regardless.
            _mm_prefetch(base.add(v) as *const i8, _MM_HINT_T0);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (dist, indices, start, end);
    }
}

/// Run Dijkstra from `source`, filling `dist` (lengths, `INFINITY` if
/// unreachable) and `pred` (predecessor vertex, `-1` for the source and
/// unreachable vertices).
///
/// `indptr` has length `n + 1`; for vertex `u`, its out-edges are
/// `indices[indptr[u]..indptr[u+1]]` with the matching `weights`. The caller
/// guarantees `dist.len() == pred.len() == n`, `weights.len() == indices.len()`,
/// and `source < n`.
pub fn dijkstra(
    indptr: &[i64],
    indices: &[i32],
    weights: &[f64],
    source: usize,
    dist: &mut [f64],
    pred: &mut [i32],
    heap: &mut Heap,
) -> Result<(), DijkstraError> {
    let n = dist.len();

    // Validate CSR structure (indptr monotone, ends at nnz) — proves every
    // offset is a valid index so the hot loop can skip bounds checks.
    validate(indptr, indices, n)?;

    // Single fused pass over the edge arrays: reject negative/NaN weights and
    // verify every neighbor id is in `[0, n)`. Fusing halves the memory traffic
    // of two separate prepasses. `w < 0.0` is false for NaN, tested separately.
    debug_assert_eq!(weights.len(), indices.len());
    for (&w, &v) in weights.iter().zip(indices.iter()) {
        if w < 0.0 || w.is_nan() {
            return Err(DijkstraError::NegativeWeight);
        }
        if v < 0 || (v as usize) >= n {
            return Err(DijkstraError::MalformedCsr);
        }
    }

    for d in dist.iter_mut() {
        *d = f64::INFINITY;
    }
    for p in pred.iter_mut() {
        *p = -1;
    }

    // At most one push per successful relaxation (<= nnz) plus the source.
    heap.reset_with_capacity(indices.len() + 1);

    // SAFETY (whole block): `validate` + the fused scan proved every CSR offset
    // and neighbor id is in range, so `indptr[u]`, `indptr[u+1]`, `dist[v]`,
    // `pred[v]`, `weights[e]`, `indices[e]` are all in bounds for any `u < n`
    // reached here. The heap holds <= nnz+1 entries, never exceeding its
    // reserved capacity, so every `push` has spare capacity.
    unsafe {
        *dist.get_unchecked_mut(source) = 0.0;
        heap.push(0.0, source as u32);

        while let Some((d, u)) = heap.pop() {
            let u = u as usize;
            // Lazy deletion: skip stale entries superseded by a better distance.
            if d > *dist.get_unchecked(u) {
                continue;
            }
            let start = *indptr.get_unchecked(u) as usize;
            let end = *indptr.get_unchecked(u + 1) as usize;
            // `indices`/`weights` are scanned sequentially (hardware-prefetched);
            // the random `dist[v]` reads dominate, so prefetch them up front.
            prefetch_neighbors(dist, indices, start, end);
            for e in start..end {
                let v = *indices.get_unchecked(e) as usize;
                let nd = d + *weights.get_unchecked(e);
                if nd < *dist.get_unchecked(v) {
                    *dist.get_unchecked_mut(v) = nd;
                    *pred.get_unchecked_mut(v) = u as i32;
                    heap.push(nd, v as u32);
                }
            }
        }
    }

    Ok(())
}
