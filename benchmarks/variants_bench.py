"""Benchmark the BMSSP variants (src/variants/) against lt-dijkstra and the
mainline lt-bmssp.

Graphs are generated EXACTLY like benchmarks/run.py's random family
(numpy default_rng(0xC0FFEE + i), m = 4n, weights U[0.01, 1), duplicate
edges summed by the COO->CSR conversion), so rows here are directly
comparable to BENCHMARKS.md. The DIMACS family reuses run.py's parser.

Every timed implementation is cross-checked against lt-dijkstra with
np.array_equal (bit-exact distances, inf-aware) before timing; a mismatch
aborts the run loudly.

Usage:
    python benchmarks/variants_bench.py --sizes 100000,1000000 \
        --variants tuned,hybrid,simpleq,lazypiv,notransform,fast \
        --include-mainline --dimacs --runs 3 --tag v1
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from logtwothirds import _logtwothirds as lt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import parse_dimacs_gr, DIMACS_FILE  # noqa: E402

RESULTS_DIR = HERE / "results"
SEED_RANDOM = 0xC0FFEE
BMSSP_SEED = 0

ALL_VARIANTS = ["tuned", "hybrid", "simpleq", "lazypiv", "notransform", "fast"]


def gen_random(n: int):
    """run.py's gen_random, returning the canonical CSR triple."""
    i = round(math.log10(n)) - 4
    m = 4 * n
    rng = np.random.default_rng(SEED_RANDOM + i)
    rows = rng.integers(0, n, size=m, dtype=np.int64).astype(np.int32)
    cols = rng.integers(0, n, size=m, dtype=np.int64).astype(np.int32)
    data = rng.uniform(0.01, 1.0, size=m)
    coo = sp.coo_array((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    csr = coo.tocsr()
    csr.sort_indices()
    return (
        np.ascontiguousarray(csr.indptr, dtype=np.int64),
        np.ascontiguousarray(csr.indices, dtype=np.int32),
        np.ascontiguousarray(csr.data, dtype=np.float64),
    )


def gen_dimacs():
    n, src, dst, wgt = parse_dimacs_gr(DIMACS_FILE)
    coo = sp.coo_array((wgt, (src, dst)), shape=(n, n), dtype=np.float64)
    csr = coo.tocsr()
    csr.sort_indices()
    return (
        np.ascontiguousarray(csr.indptr, dtype=np.int64),
        np.ascontiguousarray(csr.indices, dtype=np.int32),
        np.ascontiguousarray(csr.data, dtype=np.float64),
    )


def bench(fn, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    gc.collect()
    gcold = gc.isenabled()
    gc.disable()
    try:
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    finally:
        if gcold:
            gc.enable()
    return {"median": statistics.median(times), "min": min(times),
            "max": max(times), "runs": runs}


def fmt(t: float) -> str:
    return f"{t*1e3:.1f} ms" if t < 1.0 else f"{t:.2f} s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="100000,1000000",
                    help="comma list of random-family sizes (m = 4n each)")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--include-mainline", action="store_true",
                    help="also time the mainline lt-bmssp (slow at 1e7)")
    ap.add_argument("--mainline-runs", type=int, default=0,
                    help="override run count for mainline bmssp (0 = --runs)")
    ap.add_argument("--dimacs", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the per-impl distance cross-check (use only "
                    "when the correctness gate already covers the impl and "
                    "the extra untimed run would blow the time budget)")
    ap.add_argument("--tag", default="variants")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    graphs = [("random", f"n=10^{round(math.log10(n))}", gen_random(n))
              for n in sizes]
    if args.dimacs:
        graphs.append(("dimacs", "USA-road-d.NY", gen_dimacs()))

    results = []
    for family, label, (indptr, indices, weights) in graphs:
        n = len(indptr) - 1
        m = len(indices)
        print(f"\n=== {family}/{label}: n={n:,} m={m:,} ===", flush=True)

        ref_dist, _ = lt.dijkstra(indptr, indices, weights, 0)

        impls = {"lt-dijkstra":
                 lambda: lt.dijkstra(indptr, indices, weights, 0)}
        if args.include_mainline:
            impls["lt-bmssp"] = (
                lambda: lt.bmssp(indptr, indices, weights, 0, BMSSP_SEED))
        for v in variants:
            impls[f"bmssp-{v}"] = (
                lambda v=v: lt.bmssp_variant(indptr, indices, weights, 0, v,
                                             BMSSP_SEED))

        for name, fn in impls.items():
            if not args.skip_verify:
                got_dist, _ = fn()
                if not np.array_equal(got_dist, ref_dist):
                    bad = int(np.sum(got_dist != ref_dist))
                    print(f"  !! {name}: DISTANCE MISMATCH on {bad} vertices "
                          "— aborting", flush=True)
                    return 1
                del got_dist
                gc.collect()
            runs = args.runs
            if name == "lt-bmssp" and args.mainline_runs > 0:
                runs = args.mainline_runs
            r = bench(fn, runs=runs, warmup=args.warmup)
            ratio = ""
            results.append({"family": family, "label": label, "n": n, "m": m,
                            "impl": name, "timing": r})
            base = next((x["timing"]["median"] for x in results
                         if x["label"] == label and x["impl"] == "lt-dijkstra"),
                        None)
            if base and name != "lt-dijkstra":
                ratio = f"  ({r['median']/base:5.1f}x dijkstra)"
            print(f"  {name:20s} median {fmt(r['median']):>10s}   "
                  f"(min {fmt(r['min'])}, max {fmt(r['max'])}){ratio}",
                  flush=True)

    out = RESULTS_DIR / f"results_{args.tag}.json"
    out.write_text(json.dumps({"meta": {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "runs": args.runs, "warmup": args.warmup,
        "seeds": f"random={SEED_RANDOM:#x} bmssp={BMSSP_SEED}",
    }, "results": results}, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
