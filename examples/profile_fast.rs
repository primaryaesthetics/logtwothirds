//! Phase-level profile of the `bmssp-fast` variant (src/variants/fast.rs).
//!
//! Build with the timer enabled:
//!
//! ```text
//! cargo run --release --features phase-timer --example profile_fast -- [n] [gr-file]
//! ```
//!
//! Same graph family as `examples/profile_phases.rs` (random m = 4n,
//! weights in [0.01, 1), source 0; or a DIMACS `.gr` file).

use _logtwothirds::block_queue::SplitMix64;
use _logtwothirds::bmssp::{build_csr, Csr};
use _logtwothirds::variants::fast;

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

fn dimacs_graph(path: &str) -> Csr {
    let text = std::fs::read_to_string(path).expect("read .gr file");
    let mut n = 0usize;
    let mut edges = Vec::new();
    for line in text.lines() {
        let mut it = line.split_ascii_whitespace();
        match it.next() {
            Some("p") => {
                assert_eq!(it.next(), Some("sp"));
                n = it.next().unwrap().parse().unwrap();
            }
            Some("a") => {
                let u: u32 = it.next().unwrap().parse::<u32>().unwrap() - 1;
                let v: u32 = it.next().unwrap().parse::<u32>().unwrap() - 1;
                let w: f64 = it.next().unwrap().parse().unwrap();
                edges.push((u, v, w));
            }
            _ => {}
        }
    }
    build_csr(n, &edges)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1_000_000);
    let g = match args.get(2) {
        Some(path) => {
            eprintln!("graph: {path}");
            dimacs_graph(path)
        }
        None => {
            eprintln!("graph: random m=4n, n={n}");
            random_graph(n, 0xC0FFEE)
        }
    };

    // One warmup + one measured run.
    let _ = fast::sssp(&g, 0, 0, None).unwrap();
    let run = fast::sssp(&g, 0, 0, None).unwrap();

    let p = &run.phase;
    let phases = [
        ("transform", p.transform),
        ("find_pivots", p.find_pivots),
        ("base_case (Algorithm 2)", p.base_case),
        ("dijkstra oracle (hybrid base)", p.dijkstra_oracle),
        ("FlatHeap::pull", p.q_pull),
        ("bmssp relax loop (incl. q inserts)", p.relax_loop),
        ("FlatHeap::batch_prepend", p.q_batch_prepend),
        ("finalize (dist + pred recovery)", p.finalize),
    ];
    let accounted: f64 = phases.iter().map(|(_, t)| *t).sum();
    let total = run.total_seconds;

    println!(
        "n={} m={} | n_inner={} k={} t={} L={}",
        g.n,
        g.indices.len(),
        run.n_inner,
        run.k,
        run.t,
        run.levels
    );
    println!("total {total:9.3} s");
    let mut sorted = phases;
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (name, secs) in sorted {
        println!("  {name:38} {secs:9.3} s  {:5.1}%", 100.0 * secs / total);
    }
    println!(
        "  {:38} {:9.3} s  {:5.1}%   (recursion bookkeeping, sets, settle)",
        "bmssp body (unattributed)",
        total - accounted,
        100.0 * (total - accounted) / total
    );
    let c = &run.cnt;
    println!(
        "counters: edge_scans={} relaxations={} q_inserts={} q_pulls={} \
         q_pulled_keys={} q_prepend_items={} oracle_calls={} oracle_settled={} \
         findpivots_calls={} basecase_calls={} bmssp_calls={}",
        c.edge_scans,
        c.relaxations,
        c.q_inserts,
        c.q_pulls,
        c.q_pulled_keys,
        c.q_prepend_items,
        c.oracle_calls,
        c.oracle_settled,
        c.findpivots_calls,
        c.basecase_calls,
        c.bmssp_calls
    );
}
