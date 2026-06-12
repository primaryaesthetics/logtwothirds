//! Variant `bmssp-hybrid`: replace the recursion's base case with a bounded
//! multi-source Dijkstra whenever the subproblem is small (level <= D or
//! frontier <= B vertices).
//!
//! Correctness: a bounded multi-source Dijkstra run to exhaustion is a valid
//! BMSSP oracle — given Algorithm 3's requirement 2 (every incomplete v with
//! d(v) < B has its shortest path through a complete y in S), it returns
//! B' = B and the complete set U = T_<B(S), which is exactly Lemma 3.1's
//! "successful execution" contract. The k+1 truncation of Algorithm 2 (and
//! with it the |U| size guarantee of Lemma 3.9 at the swallowed levels) is
//! given up; the parent's |U| < k*2^(lt) loop guard still bounds its own
//! accumulation, so termination is unaffected. The worst-case bound regresses
//! toward O(m log n) (the oracle sorts its subproblem). Tunables: D
//! (`hybrid_max_level`) and B (`hybrid_frontier`); defaults from VARIANTS.md.

use crate::block_queue::BlockDs;
use crate::bmssp::{BmsspError, Csr};
use crate::variants::engine::{self, Config, VariantRun};

pub fn sssp_with(
    g: &Csr,
    source: usize,
    seed: u64,
    max_level: i32,
    frontier: usize,
) -> Result<VariantRun, BmsspError> {
    engine::run::<BlockDs>(
        g,
        source,
        seed,
        Config {
            hybrid_max_level: max_level,
            hybrid_frontier: frontier,
            ..Config::default()
        },
    )
}

pub fn sssp(g: &Csr, source: usize, seed: u64) -> Result<VariantRun, BmsspError> {
    // Defaults from the sweep in VARIANTS.md (benchmarks/sweep_hybrid.py):
    // D=1, B=1024 was best on both the random family and the road graph;
    // the sweep is monotone — the more work the Dijkstra oracle takes over,
    // the faster the run.
    sssp_with(g, source, seed, 1, 1024)
}
