//! Head-to-head timing of the production Dijkstra and the `bmssp-fast`
//! variant on one random graph (m = 4n, weights in [0.01, 1), source 0 —
//! the BENCHMARKS.md family). Median of five runs each, plus a distance
//! cross-check so a timing win can never hide a wrong answer.
//!
//! ```text
//! cargo run --release --example bench_fast -- [n]
//! ```

use _logtwothirds::block_queue::SplitMix64;
use _logtwothirds::bmssp::{build_csr, Csr};
use _logtwothirds::dijkstra;
use _logtwothirds::variants::fast;
use std::time::Instant;

fn random_graph(n: usize, seed: u64) -> Csr {
    let m = 4 * n;
    let mut rng = SplitMix64::new(seed);
    let mut edges = Vec::with_capacity(m);
    for _ in 0..m {
        let u = (rng.next_u64() % n as u64) as u32;
        let v = (rng.next_u64() % n as u64) as u32;
        let w = 0.01 + (rng.next_u64() % 990_000) as f64 / 1_000_000.0;
        edges.push((u, v, w));
    }
    build_csr(n, &edges)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1_000_000);
    let g = random_graph(n, 0xC0FFEE);
    eprintln!("graph: random m=4n, n={n}, m={}", g.indices.len());

    // Dijkstra takes the Python-facing CSR types (i64 indptr, i32 indices).
    let indptr: Vec<i64> = g.indptr.iter().map(|&p| p as i64).collect();
    let indices: Vec<i32> = g.indices.iter().map(|&v| v as i32).collect();

    let mut dist = vec![0.0f64; n];
    let mut pred = vec![0i32; n];
    let mut heap = dijkstra::Heap::new();
    let mut dj_times = Vec::new();
    for _ in 0..6 {
        let t0 = Instant::now();
        let r = dijkstra::dijkstra(&indptr, &indices, &g.weights, 0, &mut dist, &mut pred, &mut heap);
        let dt = t0.elapsed().as_secs_f64();
        assert!(r.is_ok());
        dj_times.push(dt);
    }
    dj_times.remove(0); // warmup
    let dj = median(dj_times);

    let mut fast_times = Vec::new();
    let mut fast_dist = Vec::new();
    for _ in 0..6 {
        let t0 = Instant::now();
        let run = fast::sssp(&g, 0, 0, None).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        fast_times.push(dt);
        fast_dist = run.dist;
    }
    fast_times.remove(0); // warmup
    let bf = median(fast_times);

    let mismatches = dist
        .iter()
        .zip(fast_dist.iter())
        .filter(|(a, b)| a.total_cmp(b) != std::cmp::Ordering::Equal)
        .count();
    assert_eq!(mismatches, 0, "bmssp-fast distances diverge from dijkstra");

    println!("dijkstra   median {dj:8.3} s");
    println!("bmssp-fast median {bf:8.3} s   ({:.2}x dijkstra)", bf / dj);
}
