"""Lazy-pivots vs fixed-k FindPivots at matched (k, t).

Usage: python benchmarks/sweep_lazypiv.py [random_n|dimacs ...]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants_bench import gen_random, gen_dimacs  # noqa: E402
from logtwothirds import _logtwothirds as lt  # noqa: E402

CONFIGS = [("tuned", 8, 12), ("lazypiv", 8, 12),
           ("tuned", 16, 12), ("lazypiv", 16, 12),
           ("tuned", 2, 7), ("lazypiv", 2, 7)]

graphs = []
for arg in (sys.argv[1:] or ["100000"]):
    if arg == "dimacs":
        graphs.append(("dimacs", gen_dimacs()))
    else:
        graphs.append((f"random{arg}", gen_random(int(arg))))

for label, (indptr, indices, weights) in graphs:
    ref, _ = lt.dijkstra(indptr, indices, weights, 0)
    print(f"\n{label}: n={len(indptr)-1:,} m={len(indices):,}")
    for name, k, t in CONFIGS:
        best = None
        for _ in range(2):
            t0 = time.perf_counter()
            dist, _ = lt.bmssp_variant(indptr, indices, weights, 0, name, 0,
                                       k, t)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
            assert np.array_equal(dist, ref), f"MISMATCH {name} k={k} t={t}"
        print(f"  {name:8s} k={k:<3d} t={t:<3d} {best:8.3f}s")
