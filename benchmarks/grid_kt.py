"""Grid-search (k, t) for the BMSSP variants.

Correctness does not depend on (k, t) (ALGORITHM.md S2: any k >= 1, t >= 1;
the values only enter Lemma 3.12's time bound), verified by the variants
correctness suite. This script measures the (k, t) -> time surface for a
chosen variant on the benchmark graphs.

Usage:
    python benchmarks/grid_kt.py --graph random:100000 --variant tuned
    python benchmarks/grid_kt.py --graph dimacs --variant fast --runs 2
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from logtwothirds import _logtwothirds as lt

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from variants_bench import gen_random, gen_dimacs  # noqa: E402

RESULTS_DIR = HERE / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph", default="random:100000",
                    help="random:<n> or dimacs")
    ap.add_argument("--variant", default="tuned")
    ap.add_argument("--ks", default="1,2,3,4,6,8")
    ap.add_argument("--ts", default="2,4,6,8,12,16,20,24")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--budget", type=float, default=30.0,
                    help="skip a config if a single run exceeds this many s")
    args = ap.parse_args()

    if args.graph == "dimacs":
        indptr, indices, weights = gen_dimacs()
        glabel = "dimacs"
    else:
        n = int(args.graph.split(":")[1])
        indptr, indices, weights = gen_random(n)
        glabel = f"random{n}"
    n = len(indptr) - 1
    ref, _ = lt.dijkstra(indptr, indices, weights, 0)

    ks = [int(x) for x in args.ks.split(",")]
    ts = [int(x) for x in args.ts.split(",")]
    rows = []
    print(f"graph={glabel} n={n:,} m={len(indices):,} variant={args.variant}")
    print("k\\t " + "".join(f"{t:>9d}" for t in ts))
    for k in ks:
        line = f"{k:3d} "
        for t in ts:
            gc.collect()
            times = []
            skipped = False
            for _ in range(args.runs):
                t0 = time.perf_counter()
                dist, _ = lt.bmssp_variant(indptr, indices, weights, 0,
                                           args.variant, 0, k, t)
                dt = time.perf_counter() - t0
                if not np.array_equal(dist, ref):
                    print(f"\n!! MISMATCH at k={k} t={t}")
                    return 1
                times.append(dt)
                if dt > args.budget:
                    skipped = True
                    break
            best = min(times)
            rows.append({"k": k, "t": t, "seconds": best,
                         "skipped_rest": skipped})
            line += f"{best:9.3f}"
            print(line, end="\r" if t != ts[-1] else "\n", flush=True)

    best = min(rows, key=lambda r: r["seconds"])
    print(f"\nbest: k={best['k']} t={best['t']} -> {best['seconds']:.3f}s")
    out = RESULTS_DIR / f"grid_{args.variant}_{glabel}.json"
    out.write_text(json.dumps({"graph": glabel, "variant": args.variant,
                               "rows": rows}, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
