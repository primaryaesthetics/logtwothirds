//! Shared helpers for the Rust integration tests.
#![allow(dead_code)]

use _logtwothirds::block_queue::SplitMix64;

/// A randomly generated directed graph for the differential test.
///
/// IMPORTANT: `gen_diff_graph` is mirrored, draw for draw, by
/// `gen_diff_graph` in `tests/diff_driver.py`; the Python driver re-generates
/// the same graphs from the seed instead of parsing them from a file. Any
/// change here must be made there as well.
pub struct DiffGraph {
    pub n: usize,
    pub edges: Vec<(u32, u32, f64)>,
    pub source: usize,
    /// Seed for the algorithm's quickselect RNG (both sides use SplitMix64).
    pub algo_seed: u64,
}

pub fn gen_diff_graph(seed: u64) -> DiffGraph {
    let mut r = SplitMix64::new(seed ^ 0xD1FF_E12E_5EED_5EED);
    let n = match seed % 4 {
        0 => 1 + (r.next_u64() % 40) as usize,
        1 => 2 + (r.next_u64() % 459) as usize,
        _ => 500 + (r.next_u64() % 4501) as usize,
    };
    let m = (r.next_u64() % (3 * n as u64 + 1)) as usize;
    let mut edges = Vec::with_capacity(m);
    for _ in 0..m {
        let u = (r.next_u64() % n as u64) as u32;
        let v = (r.next_u64() % n as u64) as u32;
        let w = if r.next_u64() % 20 == 0 {
            0.0
        } else {
            ((r.next_u64() % 1_000_000) + 1) as f64 / 1e6
        };
        edges.push((u, v, w));
    }
    let source = (r.next_u64() % n as u64) as usize;
    let algo_seed = r.next_u64();
    DiffGraph {
        n,
        edges,
        source,
        algo_seed,
    }
}

/// Hamiltonian-cycle backbone plus random extra edges (the shape of the
/// Python suite's `random_constant_degree_graph`): out-degree >= 1
/// everywhere, whole graph reachable from vertex 0, weights in [0.1, 1.0]
/// rounded to 4 decimals.
pub fn random_constant_degree_graph(n: usize, m: usize, seed: u64) -> Vec<(u32, u32, f64)> {
    let mut r = SplitMix64::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xC0FF_EE00);
    let mut edges: Vec<(u32, u32, f64)> = Vec::with_capacity(m);
    for i in 0..n {
        let w = (1000 + r.next_u64() % 9001) as f64 / 10000.0;
        edges.push((i as u32, ((i + 1) % n) as u32, w));
    }
    for _ in 0..m.saturating_sub(n) {
        let u = (r.next_u64() % n as u64) as u32;
        let v = (r.next_u64() % n as u64) as u32;
        let w = (1000 + r.next_u64() % 9001) as f64 / 10000.0;
        edges.push((u, v, w));
    }
    edges
}
