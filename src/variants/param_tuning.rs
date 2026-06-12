//! Variant `bmssp-tuned`: the paper's algorithm with empirically chosen
//! (k, t) instead of k = floor(log^(1/3) n), t = floor(log^(2/3) n).
//!
//! Correctness does not depend on the values (ALGORITHM.md S2; Lemmas 3.2
//! and 3.7 hold for any k >= 1, t >= 1 — k and t only enter the *time*
//! bounds of Lemma 3.12 / Remark 3.5). The default table below is filled in
//! from the grid search reported in VARIANTS.md; explicit (k, t) can be
//! passed for experiments.

use crate::block_queue::BlockDs;
use crate::bmssp::{BmsspError, Csr};
use crate::variants::engine::{self, Config, VariantRun};

/// Empirically chosen (k, t) for a transformed-graph vertex count `n2`.
/// Source: the grid search in VARIANTS.md (benchmarks/grid_kt.py;
/// benchmarks/results/grid_tuned_*.json). The measured optimum sits at
/// (k=8, t=12) across random n=1e5..1e6 and the NY road graph — roughly
/// 4x the paper's k and 1.5x its t — and the surface is flat within ~10%
/// for k in [6, 8], t in [8, 12]. Past t=14 the level structure collapses
/// (L=2 with M=1 children) and times blow up 5-40x, so t is capped.
pub fn tuned_kt(n2: usize) -> (usize, usize) {
    let log_n = f64::max(1.0, (n2.max(2) as f64).log2());
    if log_n < 14.0 {
        // Small graphs: sub-100ms either way; keep the paper's formula.
        let k = (log_n.powf(1.0 / 3.0).floor() as i64).max(1) as usize;
        let t = (log_n.powf(2.0 / 3.0).floor() as i64).max(1) as usize;
        (k, t)
    } else {
        (8, 12)
    }
}

pub fn sssp(
    g: &Csr,
    source: usize,
    seed: u64,
    kt: Option<(usize, usize)>,
) -> Result<VariantRun, BmsspError> {
    // The table needs the transformed vertex count n2 = sum over vertices of
    // max(1, deg_in + deg_out); estimating it as max(n, 2m) is within one
    // floor() of the table's log-granularity.
    let kt = kt.or_else(|| Some(tuned_kt((2 * g.indices.len()).max(g.n))));
    engine::run::<BlockDs>(
        g,
        source,
        seed,
        Config {
            kt_override: kt,
            ..Config::default()
        },
    )
}
