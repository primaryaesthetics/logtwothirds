//! Variant `bmssp-notransform`: run BMSSP directly on the input graph,
//! skipping the constant-degree transform of paper Section 2.
//!
//! Why correctness still holds: no correctness lemma uses the degree bound.
//! Lemma 3.7's induction, Lemma 3.6, Lemma 3.2's covering property, and the
//! Algorithm 2 Dijkstra argument are all degree-free. The constant-degree
//! assumption is consumed exclusively by the *time* analysis: Lemma 3.2's
//! "|W| = O(k|S|) since out-degrees are constant" after the early exit,
//! Remark 3.5's N = O(k 2^(lt)) insertion count, and the heap-size remark in
//! the base case. Without it those become volume bounds over actual degrees
//! (|W| can overshoot k|S| by one round's worth of out-neighbors, i.e. up to
//! max-degree, before the early exit fires), so the worst-case bound becomes
//! O(m log^(2/3) n) only for bounded-degree inputs and degrades toward
//! O(m k + m t) factors driven by degree skew otherwise — stated honestly in
//! VARIANTS.md. What is bought: the transform turns an (n, m) graph into one
//! with n2 = 2m + isolated vertices and 3m edges (n2 = 8e6 for n = 1e6,
//! m = 4e6), so skipping it shrinks every state array, every set, and every
//! queue by ~8x and removes the zero-weight cycle walks entirely.
//!
//! k, t, L are computed from the original n.

use crate::block_queue::BlockDs;
use crate::bmssp::{BmsspError, Csr};
use crate::variants::engine::{self, Config, VariantRun};

pub fn sssp(g: &Csr, source: usize, seed: u64) -> Result<VariantRun, BmsspError> {
    engine::run::<BlockDs>(
        g,
        source,
        seed,
        Config {
            transform: false,
            ..Config::default()
        },
    )
}
