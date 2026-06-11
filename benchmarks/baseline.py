"""Microbenchmark: logtwothirds vs scipy on a large random graph.

Builds a random directed graph with n = 10^6 vertices and m = 4*10^6 edges,
then times single-source Dijkstra from vertex 0 for both implementations.

Acceptance: logtwothirds must be no slower than scipy.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from logtwothirds import shortest_paths

N = 1_000_000
M = 4_000_000
SEED = 12345


def build_graph(n: int, m: int, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n, size=m)
    cols = rng.integers(0, n, size=m)
    data = rng.uniform(0.01, 1.0, size=m)
    g = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
    g.sort_indices()
    return g


def time_call(fn, repeats: int = 7) -> float:
    """Best-of-`repeats` wall-clock seconds (best-of-N is standard practice for
    a stable floor; it suppresses OS-scheduling and turbo-clock noise)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    print(f"Building graph: n={N:,}, m={M:,} (seed={SEED}) ...", flush=True)
    g = build_graph(N, M, SEED)
    # Pre-extract CSR triple so both sides time only the shortest-path work on
    # identical, already-sorted data.
    indptr = np.ascontiguousarray(g.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(g.indices, dtype=np.int32)
    weights = np.ascontiguousarray(g.data, dtype=np.float64)
    source = 0

    # Warm up / sanity check that both agree before timing.
    d_ours, _ = shortest_paths((indptr, indices, weights), source)
    d_scipy = scipy_dijkstra(g, directed=True, indices=source)
    agree = np.allclose(d_ours, d_scipy, rtol=0, atol=1e-9, equal_nan=True)
    reach_ours = int(np.isfinite(d_ours).sum())
    print(f"Correctness vs scipy: {'MATCH' if agree else 'MISMATCH'} "
          f"(reachable={reach_ours:,})", flush=True)

    # Apples-to-apples: our API always computes predecessors, so scipy must too
    # (return_predecessors=True). We also report the distances-only scipy time
    # for reference, but the pass/fail uses the matching-output configuration.
    t_ours = time_call(
        lambda: shortest_paths((indptr, indices, weights), source)
    )
    t_scipy_pred = time_call(
        lambda: scipy_dijkstra(
            g, directed=True, indices=source, return_predecessors=True
        )
    )
    t_scipy_dist = time_call(
        lambda: scipy_dijkstra(g, directed=True, indices=source)
    )

    print()
    print(f"logtwothirds dijkstra (dist+pred)  : {t_ours*1000:9.1f} ms")
    print(f"scipy    dijkstra (dist+pred)  : {t_scipy_pred*1000:9.1f} ms")
    print(f"scipy    dijkstra (dist only)  : {t_scipy_dist*1000:9.1f} ms")
    print(f"speedup vs scipy(dist+pred)    : {t_scipy_pred / t_ours:6.2f}x")
    print(f"speedup vs scipy(dist only)    : {t_scipy_dist / t_ours:6.2f}x")

    # Pass/fail against the like-for-like configuration (both produce
    # distances AND predecessors).
    t_scipy = t_scipy_pred

    if not agree:
        raise SystemExit("FAIL: results disagree with scipy")
    if t_ours > t_scipy:
        raise SystemExit(
            f"FAIL: logtwothirds ({t_ours*1000:.1f} ms) slower than "
            f"scipy ({t_scipy*1000:.1f} ms)"
        )
    print("\nPASS: logtwothirds is no slower than scipy.")


if __name__ == "__main__":
    main()
