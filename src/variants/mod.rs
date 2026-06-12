//! Algorithm-level BMSSP variants (research track; see VARIANTS.md).
//!
//! Every variant is exposed to Python as `method="bmssp-<name>"`. The
//! mainline `src/bmssp.rs` is untouched; variants share the parameterized
//! engine in [`engine`] and must pass the distance-correctness suite in
//! `tests/variants_correctness.rs` (bit-exact vs Dijkstra; settlement order
//! is *not* part of the variants' contract).

pub mod engine;
pub mod fast;
pub mod hybrid_base;
pub mod lazy_pivots;
pub mod no_transform;
pub mod param_tuning;
pub mod simple_queue;

use crate::block_queue::BlockDs;
use crate::bmssp::{BmsspError, Csr};
use engine::Config;
pub use engine::VariantRun;
use simple_queue::FlatHeap;

/// All variant names, as accepted by [`run_variant`] (without the "bmssp-"
/// prefix).
pub const VARIANT_NAMES: &[&str] = &[
    "tuned",
    "hybrid",
    "simpleq",
    "lazypiv",
    "notransform",
    "fast",
];

/// Dispatch a variant by name (the part after "bmssp-"). `kt` optionally
/// forces (k, t) for any variant (used by the grid search); when `None`,
/// each variant applies its own default. Returns `Ok(None)` for an unknown
/// name.
///
/// The hybrid oracle thresholds can be swept without new bindings via
/// `hybrid:<max_level>` or `hybrid:<max_level>:<frontier>` (e.g.
/// "hybrid:2:64"); plain "hybrid" uses the defaults from VARIANTS.md.
pub fn run_variant(
    name: &str,
    g: &Csr,
    source: usize,
    seed: u64,
    kt: Option<(usize, usize)>,
) -> Result<Option<VariantRun>, BmsspError> {
    let base = Config {
        kt_override: kt,
        ..Config::default()
    };
    if let Some(rest) = name.strip_prefix("hybrid:") {
        let mut parts = rest.split(':');
        let max_level: i32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
        let frontier: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        return Ok(Some(engine::run::<BlockDs>(
            g,
            source,
            seed,
            Config {
                hybrid_max_level: max_level,
                hybrid_frontier: frontier,
                ..base
            },
        )?));
    }
    let run = match name {
        "tuned" => param_tuning::sssp(g, source, seed, kt)?,
        "hybrid" => engine::run::<BlockDs>(
            g,
            source,
            seed,
            Config {
                hybrid_max_level: 1,
                hybrid_frontier: 1024,
                ..base
            },
        )?,
        "simpleq" => engine::run::<FlatHeap>(g, source, seed, base)?,
        "lazypiv" => engine::run::<BlockDs>(
            g,
            source,
            seed,
            Config {
                lazy_pivots: true,
                ..base
            },
        )?,
        "notransform" => engine::run::<BlockDs>(
            g,
            source,
            seed,
            Config {
                transform: false,
                ..base
            },
        )?,
        "fast" => fast::sssp(g, source, seed, kt)?,
        _ => return Ok(None),
    };
    Ok(Some(run))
}
