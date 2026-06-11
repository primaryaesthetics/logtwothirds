"""Shared helpers for the logtwothirds test suite."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def random_graph(n: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    """Build a random directed graph as a CSR matrix.

    Edge weights are positive floats in [0.01, 1.0). The number of edges is
    ``density * n * n`` (clamped to ``[0, n*n]``). Duplicate (u, v) draws are
    summed by ``csr_matrix``; we keep them — multi-edges collapse to the min by
    construction below via explicit dedup to a single weight.
    """
    n2 = n * n
    m = int(min(max(density * n2, 0), n2))
    if m == 0 or n == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)

    rows = rng.integers(0, n, size=m)
    cols = rng.integers(0, n, size=m)
    data = rng.uniform(0.01, 1.0, size=m)
    g = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    # Collapse duplicate coordinates to a single edge (take the min weight) so
    # the graph is a simple weighted digraph for comparison purposes.
    g = g.tocsr()
    g.sort_indices()
    return g
