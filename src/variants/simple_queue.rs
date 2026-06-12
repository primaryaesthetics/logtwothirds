//! Variant `bmssp-simpleq`: replace the Lemma 3.3 block structure with a
//! plain lazy-deletion binary heap that satisfies the same semantic
//! Insert / BatchPrepend / Pull contract.
//!
//! What is lost (documented in VARIANTS.md): the amortized bounds of
//! Lemma 3.3. Insert and every batch-prepended element cost O(log N) instead
//! of O(t) / O(log k); Pull costs O(M log N) instead of O(M). Regression #4
//! of ALGORITHM.md S7 applies: the worst case returns to O(m log n). What is
//! gained: no per-block linked lists, no median splits, no bound BST, no
//! quickselects — one contiguous heap with excellent constants.

use crate::block_queue::{Key, SplitMix64};
use crate::bmssp::{BmsspError, Csr};
use crate::variants::engine::{self, Config, DQueue, VariantRun};
use rustc_hash::FxHashMap as HashMap;
use std::collections::BinaryHeap;

/// Lazy-deletion binary heap with keep-min duplicate handling.
///
/// `best` holds the authoritative (smallest) live value per key; heap entries
/// not matching `best` are stale and skipped on pop. The Pull contract's
/// separating bound is the next live minimum (or B when drained), which by
/// heap order satisfies `max(S') < x <= min(remaining)` (keys are unique
/// under Assumption 2.1's total order).
pub struct FlatHeap {
    heap: BinaryHeap<std::cmp::Reverse<(Key, u32)>>,
    best: HashMap<u32, Key>,
    m: usize,
    b: Key,
}

impl DQueue for FlatHeap {
    fn new(m: usize, b: Key) -> Self {
        FlatHeap {
            heap: BinaryHeap::new(),
            best: HashMap::default(),
            m: m.max(1),
            b,
        }
    }

    fn insert(&mut self, key: u32, value: Key, _rng: &mut SplitMix64) {
        match self.best.get(&key) {
            Some(&old) if old <= value => {} // keep existing smaller value
            _ => {
                self.best.insert(key, value);
                self.heap.push(std::cmp::Reverse((value, key)));
            }
        }
    }

    fn batch_prepend(&mut self, items: &[(u32, Key)], rng: &mut SplitMix64) {
        // The precondition (all smaller than current contents) makes prepend
        // semantically identical to insert here; keep-min handles in-batch
        // duplicates.
        for &(k, v) in items {
            self.insert(k, v, rng);
        }
    }

    fn pull(&mut self, _rng: &mut SplitMix64) -> (Vec<u32>, Key) {
        let mut out: Vec<u32> = Vec::with_capacity(self.m);
        while out.len() < self.m {
            match self.heap.pop() {
                None => break,
                Some(std::cmp::Reverse((v, k))) => {
                    if self.best.get(&k) == Some(&v) {
                        self.best.remove(&k);
                        out.push(k);
                    }
                }
            }
        }
        // Separating bound: the next live minimum, B if drained.
        let x = loop {
            match self.heap.peek() {
                None => break self.b,
                Some(&std::cmp::Reverse((v, k))) => {
                    if self.best.get(&k) == Some(&v) {
                        break v;
                    }
                    self.heap.pop(); // stale: discard and keep looking
                }
            }
        };
        (out, x)
    }

    fn is_empty(&self) -> bool {
        self.best.is_empty()
    }
}

pub fn sssp(g: &Csr, source: usize, seed: u64) -> Result<VariantRun, BmsspError> {
    engine::run::<FlatHeap>(g, source, seed, Config::default())
}
