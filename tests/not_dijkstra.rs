//! Acceptance criterion 4: the `not_dijkstra` check from the Python suite
//! (`tests/test_verification.py::test_not_dijkstra`), ported to the Rust
//! BMSSP. With (k, t) forced to (2, 2) — the regime where the algorithm's
//! distinctive settlement mechanism is active at testable sizes — the
//! settlement log must NOT be globally sorted by distance in at least 15 of
//! 20 random runs. An implementation degenerated into Dijkstra settles in
//! sorted order under any parameters and fails here.

mod common;

use _logtwothirds::block_queue::SplitMix64;
use _logtwothirds::bmssp::{build_csr, is_globally_sorted, sssp_bmssp};

#[test]
fn not_dijkstra() {
    let mut unsorted_count = 0;
    for seed in 0..20u64 {
        let mut r = SplitMix64::new(seed.wrapping_mul(7919) ^ 0x5A5A);
        let n = 500 + (r.next_u64() % 1001) as usize; // n in [500, 1500]
        let edges = common::random_constant_degree_graph(n, 2 * n, seed);
        let g = build_csr(n, &edges);
        let run = sssp_bmssp(&g, 0, seed ^ 0xBADC_AB1E, Some((2, 2))).unwrap();
        assert!(
            !run.settle_log.is_empty(),
            "seed {seed}: settlement log empty"
        );
        if !is_globally_sorted(&run.settle_log) {
            unsorted_count += 1;
        }
    }
    assert!(
        unsorted_count >= 15,
        "only {unsorted_count}/20 runs unsorted"
    );
}

/// Sanity companion (mirrors `test_not_globally_sorted`): a single forced
/// (k, t) = (2, 2) run on a 4096-vertex graph is out of order, while the
/// distances still match Dijkstra-style relaxation results (checked by the
/// property tests).
#[test]
fn not_globally_sorted_single_run() {
    let n = 4096;
    let edges = common::random_constant_degree_graph(n, 2 * n, 1);
    let g = build_csr(n, &edges);
    let run = sssp_bmssp(&g, 0, 42, Some((2, 2))).unwrap();
    assert!(!is_globally_sorted(&run.settle_log));
}
