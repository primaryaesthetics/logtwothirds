//! Variant `bmssp-fast`: the winning combination from VARIANTS.md.
//!
//! Composition (each piece's correctness argument lives with its own
//! variant; they are independent, so they compose):
//! - no constant-degree transform (`no_transform`): correctness lemmas are
//!   degree-free;
//! - bounded multi-source Dijkstra oracle for small subproblems
//!   (`hybrid_base`): a valid Lemma 3.1 oracle;
//! - lazy-deletion binary heap for D (`simple_queue`): satisfies the
//!   Lemma 3.3 semantic contract;
//! - tuned (k, t) (`param_tuning`): correctness is parameter-free.
//!
//! The exact knob values are the measured optimum; see VARIANTS.md.

use crate::bmssp::{BmsspError, Csr};
use crate::variants::engine::{self, Config, VariantRun};
use crate::variants::simple_queue::FlatHeap;

/// Tuned (k, t) for the no-transform engine, from the grid search in
/// VARIANTS.md (benchmarks/results/grid_fast_*.json; computed on the
/// *original* vertex count). The surface is flat within ~15% for
/// t in [12, 16] with k = 1 best or tied everywhere measured: with the
/// Dijkstra oracle swallowing levels <= 1, large pull batches (big t)
/// matter and FindPivots depth (k) does not.
pub fn fast_kt(n: usize) -> (usize, usize) {
    let log_n = f64::max(1.0, (n.max(2) as f64).log2());
    if log_n < 13.0 {
        let k = (log_n.powf(1.0 / 3.0).floor() as i64).max(1) as usize;
        let t = (log_n.powf(2.0 / 3.0).floor() as i64).max(1) as usize;
        (k, t)
    } else {
        (1, 12)
    }
}

pub fn sssp(
    g: &Csr,
    source: usize,
    seed: u64,
    kt: Option<(usize, usize)>,
) -> Result<VariantRun, BmsspError> {
    let kt = kt.or_else(|| Some(fast_kt(g.n)));
    engine::run::<FlatHeap>(
        g,
        source,
        seed,
        Config {
            transform: false,
            kt_override: kt,
            hybrid_max_level: 1,
            hybrid_frontier: 1024,
            lazy_pivots: false,
        },
    )
}
