//! Acceptance criterion 2: differential test Rust-BMSSP vs the pure-Python
//! reference (`python/logtwothirds/_reference.py`) on 200 random graphs with
//! n up to 5000. Both the distances AND the settlement order must match
//! exactly (bit-for-bit on the f64s).
//!
//! Protocol: this test re-runs every graph in Rust, writes per-seed results
//! (distance bits + settlement log) to a file, then invokes
//! `tests/diff_driver.py`, which regenerates the identical graphs from the
//! seeds, runs the reference, and compares. The driver pins the reference's
//! two sources of unspecified behavior to the same choices the Rust port
//! makes (SplitMix64 quickselect pivots, insertion-ordered sets) — see the
//! driver's docstring. On divergence the driver reports the first differing
//! settlement-log index and its context, then exits nonzero.

mod common;

use _logtwothirds::bmssp::{build_csr, sssp_bmssp};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 200 by the acceptance criterion; override with LOGTWOTHIRDS_DIFF_GRAPHS
/// for quick debugging runs.
fn num_graphs() -> u64 {
    std::env::var("LOGTWOTHIRDS_DIFF_GRAPHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200)
}

fn find_python(manifest: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LOGTWOTHIRDS_PYTHON") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    for candidate in [".venv/Scripts/python.exe", ".venv/bin/python"] {
        let p = manifest.join(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[test]
fn differential_vs_python_reference() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(python) = find_python(&manifest) else {
        eprintln!(
            "SKIP: no Python interpreter found (.venv missing and \
             LOGTWOTHIRDS_PYTHON not set); differential test not run"
        );
        return;
    };

    let num_graphs = num_graphs();
    let mut out = String::new();
    for seed in 0..num_graphs {
        let g = common::gen_diff_graph(seed);
        let csr = build_csr(g.n, &g.edges);
        let run = sssp_bmssp(&csr, g.source, g.algo_seed, None)
            .unwrap_or_else(|e| panic!("seed {seed}: bmssp failed: {e:?}"));

        writeln!(out, "GRAPH {seed}").unwrap();
        writeln!(
            out,
            "PARAMS {} {} {} {}",
            run.k, run.t, run.levels, run.n_transformed
        )
        .unwrap();
        out.push_str("DIST");
        for d in &run.dist {
            write!(out, " {:x}", d.to_bits()).unwrap();
        }
        out.push('\n');
        out.push_str("SETTLE");
        for &(v, d) in &run.settle_log {
            write!(out, " {}:{:x}", v, d.to_bits()).unwrap();
        }
        out.push('\n');
        writeln!(out, "END").unwrap();
    }

    let results_path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("bmssp_differential.txt");
    std::fs::write(&results_path, out).expect("write results file");

    let driver = manifest.join("tests").join("diff_driver.py");
    let output = Command::new(&python)
        .arg(&driver)
        .arg(&results_path)
        .arg(num_graphs.to_string())
        .output()
        .expect("failed to spawn the Python differential driver");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "differential driver reported divergence or crashed \
         (status {:?})\n--- driver stdout ---\n{stdout}\n--- driver stderr ---\n{stderr}",
        output.status.code()
    );
    // Belt and braces: the driver's last line is the summary.
    assert!(
        stdout.contains(&format!("ALL OK {num_graphs}/{num_graphs}")),
        "driver did not confirm all graphs:\n{stdout}\n{stderr}"
    );
}
