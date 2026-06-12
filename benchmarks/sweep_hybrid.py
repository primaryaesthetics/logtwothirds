"""Sweep the hybrid-base thresholds D (max level swallowed by the Dijkstra
oracle) and B (frontier-size switch) — variant strings hybrid:<D>:<B>.

Usage: python benchmarks/sweep_hybrid.py [random_n ...]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants_bench import gen_random, gen_dimacs  # noqa: E402
from logtwothirds import _logtwothirds as lt  # noqa: E402

CONFIGS = ["hybrid:0:0", "hybrid:1:0", "hybrid:2:0",
           "hybrid:0:32", "hybrid:1:64", "hybrid:1:1024", "hybrid:2:1024"]

graphs = []
for arg in (sys.argv[1:] or ["100000"]):
    if arg == "dimacs":
        graphs.append(("dimacs", gen_dimacs()))
    else:
        graphs.append((f"random{arg}", gen_random(int(arg))))

for label, (indptr, indices, weights) in graphs:
    ref, _ = lt.dijkstra(indptr, indices, weights, 0)
    print(f"\n{label}: n={len(indptr)-1:,} m={len(indices):,}")
    for name in CONFIGS:
        best = None
        for _ in range(2):
            t0 = time.perf_counter()
            dist, _ = lt.bmssp_variant(indptr, indices, weights, 0, name, 0)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
            assert np.array_equal(dist, ref), f"MISMATCH {name}"
        print(f"  {name:16s} {best:8.3f}s")
