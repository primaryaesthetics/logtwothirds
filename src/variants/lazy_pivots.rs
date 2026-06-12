//! Variant `bmssp-lazypiv`: FindPivots with early termination of the
//! Bellman-Ford rounds when the frontier stops shrinking.
//!
//! Correctness (Lemma 3.2): the covering property is what the parent needs —
//! every x in U-tilde is either complete in W, or its shortest path visits a
//! complete pivot y in P. The paper proves it with k rounds and pivot
//! tree-size threshold k: a vertex <= k-1 tight edges below its last
//! complete-in-S ancestor is completed by k rounds; otherwise that ancestor
//! roots a tight tree of >= k vertices. The same proof goes through verbatim
//! with any j <= k rounds and threshold j. Stopping early therefore only
//! weakens the *size* bound |P| <= |W|/k to |P| <= |W|/j (more pivots, more
//! D.Insert traffic) — it can never lose a vertex.
//!
//! The paper's part (b) — "running rounds only over edges whose tail was
//! relaxed in the previous round" — is already how Algorithm 1 is stated
//! (line 6 scans edges out of W_{i-1} only) and how the mainline implements
//! it, so this variant adds only (a), the adaptive round count. Note the
//! interaction with k: at production sizes the paper's k is 2-3, so few
//! rounds exist to cut; lazy termination pays off mainly when combined with
//! a larger tuned k (measured in VARIANTS.md).

use crate::block_queue::BlockDs;
use crate::bmssp::{BmsspError, Csr};
use crate::variants::engine::{self, Config, VariantRun};

pub fn sssp(g: &Csr, source: usize, seed: u64) -> Result<VariantRun, BmsspError> {
    engine::run::<BlockDs>(
        g,
        source,
        seed,
        Config {
            lazy_pivots: true,
            ..Config::default()
        },
    )
}
