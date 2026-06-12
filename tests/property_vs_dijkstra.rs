//! Acceptance criterion 3: property-based comparison of Rust BMSSP against
//! Rust Dijkstra on random graphs up to 10^6 edges — distances must match
//! exactly (bit-for-bit).
//!
//! Bit-exact equality is the correct expectation: both algorithms compute,
//! for every vertex, the minimum over paths of the left-to-right rounded
//! weight sum, and the constant-degree transform only interleaves `+ 0.0`
//! terms (exact for the non-negative values that occur here).

mod common;

use _logtwothirds::block_queue::SplitMix64;
use _logtwothirds::bmssp::{build_csr, sssp_bmssp, Csr};
use _logtwothirds::dijkstra;

/// Run the production Dijkstra on a `Csr` (converting to its i64/i32 view).
fn dijkstra_dist(g: &Csr) -> (Vec<f64>, Vec<i32>) {
    let indptr: Vec<i64> = g.indptr.iter().map(|&p| p as i64).collect();
    let indices: Vec<i32> = g.indices.iter().map(|&v| v as i32).collect();
    let mut dist = vec![0.0f64; g.n];
    let mut pred = vec![0i32; g.n];
    let mut heap = dijkstra::Heap::new();
    let ok = dijkstra::dijkstra(
        &indptr,
        &indices,
        &g.weights,
        0,
        &mut dist,
        &mut pred,
        &mut heap,
    );
    assert!(ok.is_ok());
    (dist, pred)
}

fn assert_dist_equal(bmssp: &[f64], dij: &[f64], ctx: &str) {
    assert_eq!(bmssp.len(), dij.len());
    for (v, (a, b)) in bmssp.iter().zip(dij.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{ctx}: distance mismatch at vertex {v}: bmssp={a:?} dijkstra={b:?}"
        );
    }
}

/// Predecessors must reconstruct the reported distances exactly: for every
/// reachable non-source vertex there is an edge (pred[v], v) with
/// dist[pred[v]] + w == dist[v].
fn assert_pred_consistent(g: &Csr, dist: &[f64], pred: &[i32], source: usize) {
    assert_eq!(pred[source], -1);
    for v in 0..g.n {
        if v == source {
            continue;
        }
        if dist[v].is_infinite() {
            assert_eq!(pred[v], -1, "unreachable vertex {v} must have pred -1");
            continue;
        }
        let u = pred[v];
        assert!(u >= 0, "reachable vertex {v} must have a predecessor");
        let u = u as usize;
        let found = (g.indptr[u]..g.indptr[u + 1]).any(|e| {
            g.indices[e] as usize == v && (dist[u] + g.weights[e]).to_bits() == dist[v].to_bits()
        });
        assert!(found, "no tight edge ({u}, {v}) backs pred[{v}]");
    }
}

fn random_graph(n: usize, m: usize, seed: u64, connected_backbone: bool) -> Csr {
    let mut r = SplitMix64::new(seed ^ 0xABCD_EF01_2345_6789);
    let mut edges: Vec<(u32, u32, f64)> = Vec::with_capacity(m);
    if connected_backbone {
        for i in 0..n {
            let w = ((r.next_u64() % 1_000_000) + 1) as f64 / 1e6;
            edges.push((i as u32, ((i + 1) % n) as u32, w));
        }
    }
    while edges.len() < m {
        let u = (r.next_u64() % n as u64) as u32;
        let v = (r.next_u64() % n as u64) as u32;
        let w = if r.next_u64() % 25 == 0 {
            0.0
        } else {
            ((r.next_u64() % 1_000_000) + 1) as f64 / 1e6
        };
        edges.push((u, v, w));
    }
    build_csr(n, &edges)
}

#[test]
fn distances_match_dijkstra_on_random_graphs() {
    let mut r = SplitMix64::new(99);
    for case in 0..30u64 {
        let n = 1 + (r.next_u64() % 2000) as usize;
        let m = (r.next_u64() % (4 * n as u64 + 1)) as usize;
        let g = random_graph(n, m, case, case % 3 == 0);
        let run = sssp_bmssp(&g, 0, case.wrapping_mul(0x1234_5678), None).unwrap();
        let (dij_dist, _) = dijkstra_dist(&g);
        assert_dist_equal(&run.dist, &dij_dist, &format!("case {case} (n={n}, m={m})"));
        assert_pred_consistent(&g, &run.dist, &run.pred, 0);
    }
}

#[test]
fn distances_match_dijkstra_on_million_edge_graph() {
    // 10^6 edges: cycle backbone over 250k vertices plus random edges.
    let n = 250_000;
    let m = 1_000_000;
    let g = random_graph(n, m, 0xFEED, true);
    assert_eq!(g.indices.len(), m);
    let run = sssp_bmssp(&g, 0, 0x600D_5EED, None).unwrap();
    let (dij_dist, _) = dijkstra_dist(&g);
    assert_dist_equal(&run.dist, &dij_dist, "million-edge graph");
    assert_pred_consistent(&g, &run.dist, &run.pred, 0);
}

#[test]
fn distances_match_dijkstra_on_sparse_disconnected_graph() {
    // Many unreachable vertices and zero-weight edges.
    let n = 100_000;
    let m = 150_000;
    let g = random_graph(n, m, 0xD15C, false);
    let run = sssp_bmssp(&g, 0, 7, None).unwrap();
    let (dij_dist, _) = dijkstra_dist(&g);
    assert_dist_equal(&run.dist, &dij_dist, "sparse disconnected graph");
    assert_pred_consistent(&g, &run.dist, &run.pred, 0);
}
