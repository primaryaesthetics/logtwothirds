//! Correctness gate for the BMSSP variants (src/variants/): distances must
//! match the production Dijkstra bit-for-bit on
//!   * at least 500 property graphs per variant, covering zero-weight edges,
//!     tied path lengths (integer and equal weights), self-loops, parallel
//!     edges, disconnected graphs, and random sources;
//!   * a 10^6-edge stress graph per variant.
//!
//! Settlement order vs the Python reference is deliberately NOT checked —
//! variants legitimately change it. Distance equality is the bar, and
//! bit-exactness is the right form of it: every algorithm here computes a
//! minimum over identical left-to-right rounded path sums.

use _logtwothirds::block_queue::SplitMix64;
use _logtwothirds::bmssp::{build_csr, Csr};
use _logtwothirds::dijkstra;
use _logtwothirds::variants::{run_variant, VARIANT_NAMES};

fn dijkstra_dist(g: &Csr, source: usize) -> Vec<f64> {
    let indptr: Vec<i64> = g.indptr.iter().map(|&p| p as i64).collect();
    let indices: Vec<i32> = g.indices.iter().map(|&v| v as i32).collect();
    let mut dist = vec![0.0f64; g.n];
    let mut pred = vec![0i32; g.n];
    let mut heap = dijkstra::Heap::new();
    dijkstra::dijkstra(&indptr, &indices, &g.weights, source, &mut dist, &mut pred, &mut heap)
        .map_err(|_| ())
        .expect("oracle dijkstra failed");
    dist
}

fn assert_dist_equal(got: &[f64], want: &[f64], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (v, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{ctx}: distance mismatch at vertex {v}: variant={a:?} dijkstra={b:?}"
        );
    }
}

/// Predecessors must reconstruct the reported distances exactly.
fn assert_pred_consistent(g: &Csr, dist: &[f64], pred: &[i32], source: usize, ctx: &str) {
    assert_eq!(pred[source], -1, "{ctx}: source pred");
    for v in 0..g.n {
        if v == source {
            continue;
        }
        if dist[v].is_infinite() {
            assert_eq!(pred[v], -1, "{ctx}: unreachable {v} must have pred -1");
            continue;
        }
        let u = pred[v];
        assert!(u >= 0, "{ctx}: reachable {v} lacks a predecessor");
        let u = u as usize;
        let found = (g.indptr[u]..g.indptr[u + 1]).any(|e| {
            g.indices[e] as usize == v && (dist[u] + g.weights[e]).to_bits() == dist[v].to_bits()
        });
        assert!(found, "{ctx}: no tight edge ({u}, {v}) backs pred[{v}]");
    }
}

/// Weight regimes that stress ties and zero weights.
#[derive(Clone, Copy, Debug)]
enum WeightKind {
    /// U(0.001, 1.0) floats with ~4% exact zeros.
    Float,
    /// Integer weights in {0..9} (as f64): massive tie pressure.
    SmallInt,
    /// All edges weight 1.0: every equal-hop path ties.
    Unit,
    /// ~50% zeros, rest small integers: zero-weight chains and cycles.
    ZeroHeavy,
}

fn random_graph(
    n: usize,
    m: usize,
    seed: u64,
    kind: WeightKind,
    backbone: bool,
) -> Csr {
    let mut r = SplitMix64::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut edges: Vec<(u32, u32, f64)> = Vec::with_capacity(m + n);
    let draw_w = |r: &mut SplitMix64| -> f64 {
        match kind {
            WeightKind::Float => {
                if r.next_u64() % 25 == 0 {
                    0.0
                } else {
                    ((r.next_u64() % 999_000) + 1_000) as f64 / 1e6
                }
            }
            WeightKind::SmallInt => (r.next_u64() % 10) as f64,
            WeightKind::Unit => 1.0,
            WeightKind::ZeroHeavy => {
                if r.next_u64() % 2 == 0 {
                    0.0
                } else {
                    ((r.next_u64() % 5) + 1) as f64
                }
            }
        }
    };
    if backbone {
        for i in 0..n {
            let w = draw_w(&mut r);
            edges.push((i as u32, ((i + 1) % n) as u32, w));
        }
    }
    while edges.len() < m {
        let u = (r.next_u64() % n as u64) as u32;
        let v = (r.next_u64() % n as u64) as u32;
        let w = draw_w(&mut r);
        edges.push((u, v, w));
    }
    build_csr(n, &edges)
}

/// The >= 500-graph property suite, run for one variant.
fn property_suite(variant: &str) {
    let kinds = [
        WeightKind::Float,
        WeightKind::SmallInt,
        WeightKind::Unit,
        WeightKind::ZeroHeavy,
    ];
    let mut r = SplitMix64::new(0xA11C_E5ED);
    let mut cases = 0usize;
    for round in 0..130u64 {
        for (ki, &kind) in kinds.iter().enumerate() {
            let seed = round * 31 + ki as u64;
            let n = 1 + (r.next_u64() % 250) as usize;
            let m = (r.next_u64() % (4 * n as u64 + 1)) as usize;
            let backbone = round % 3 == 0;
            let g = random_graph(n, m, seed, kind, backbone);
            let source = (r.next_u64() % n as u64) as usize;
            let run = run_variant(variant, &g, source, seed.wrapping_mul(0x5851_F42D), None)
                .expect("variant errored")
                .expect("unknown variant");
            let want = dijkstra_dist(&g, source);
            let ctx = format!("{variant}: case {cases} (n={n}, m={m}, {kind:?}, src={source})");
            assert_dist_equal(&run.dist, &want, &ctx);
            assert_pred_consistent(&g, &run.dist, &run.pred, source, &ctx);
            cases += 1;
        }
    }
    assert!(cases >= 500, "property suite must cover >= 500 graphs, got {cases}");
}

/// The 10^6-edge stress graph, run for one variant.
fn stress_suite(variant: &str) {
    let n = 250_000;
    let m = 1_000_000;
    let g = random_graph(n, m, 0xFEED_F00D, WeightKind::Float, true);
    assert_eq!(g.indices.len(), m);
    let run = run_variant(variant, &g, 0, 0x600D_5EED, None)
        .expect("variant errored")
        .expect("unknown variant");
    let want = dijkstra_dist(&g, 0);
    assert_dist_equal(&run.dist, &want, &format!("{variant}: million-edge stress"));
    assert_pred_consistent(&g, &run.dist, &run.pred, 0, &format!("{variant}: stress pred"));
}

/// The 10^6-edge stress graph with SmallInt weights: at this size the exact
/// `(len, hops)` ties of integer weights are pervasive — a road-network-like
/// regime the small property graphs cannot reproduce at scale. Guards
/// against the `<=`-relaxation duplicate-heap-entry cascade that once made
/// the oracles' work combinatorial on tie-rich graphs: without the
/// duplicate-pop skip in the oracle loops this test hangs and exhausts
/// memory rather than merely failing.
fn stress_ties_suite(variant: &str) {
    let n = 250_000;
    let m = 1_000_000;
    let g = random_graph(n, m, 0xFEED_F00D, WeightKind::SmallInt, true);
    assert_eq!(g.indices.len(), m);
    let run = run_variant(variant, &g, 0, 0x600D_5EED, None)
        .expect("variant errored")
        .expect("unknown variant");
    let want = dijkstra_dist(&g, 0);
    assert_dist_equal(&run.dist, &want, &format!("{variant}: tie-rich stress"));
    assert_pred_consistent(&g, &run.dist, &run.pred, 0, &format!("{variant}: tie-rich pred"));
}

macro_rules! variant_tests {
    ($($name:ident => $variant:literal),* $(,)?) => {
        $(
            mod $name {
                #[test]
                fn properties_500_graphs() {
                    super::property_suite($variant);
                }
                #[test]
                fn stress_million_edges() {
                    super::stress_suite($variant);
                }
                #[test]
                fn stress_million_edges_tie_rich() {
                    super::stress_ties_suite($variant);
                }
            }
        )*
    };
}

variant_tests! {
    tuned => "tuned",
    hybrid => "hybrid",
    simpleq => "simpleq",
    lazypiv => "lazypiv",
    notransform => "notransform",
    fast => "fast",
}

/// Sanity: the dispatcher knows exactly the documented set of variants.
#[test]
fn variant_names_dispatch() {
    let g = build_csr(3, &[(0, 1, 1.0), (1, 2, 0.5)]);
    for &name in VARIANT_NAMES {
        let run = run_variant(name, &g, 0, 0, None).unwrap();
        assert!(run.is_some(), "{name} should dispatch");
        let run = run.unwrap();
        assert_eq!(run.dist, vec![0.0, 1.0, 1.5], "{name} wrong on the toy graph");
    }
    assert!(run_variant("nope", &g, 0, 0, None).unwrap().is_none());
}
