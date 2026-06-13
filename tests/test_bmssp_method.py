"""Tests for ``shortest_paths(..., method="bmssp")`` (the Rust BMSSP port)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from logtwothirds import shortest_paths

from .conftest import random_graph


def test_bmssp_matches_dijkstra_on_random_graphs():
    rng = np.random.default_rng(2024)
    for n, density in [(1, 0.0), (2, 0.5), (50, 0.05), (300, 0.01), (1000, 0.002)]:
        g = random_graph(n, density, rng)
        d_dij, _p_dij = shortest_paths(g, 0, method="dijkstra")
        d_bms, p_bms = shortest_paths(g, 0, method="bmssp")
        # Bit-exact: both compute the min over paths of the rounded sums.
        assert np.array_equal(d_dij, d_bms), (n, density)
        # Predecessors reconstruct the distances exactly.
        csr = g.tocsr()
        for v in range(n):
            u = p_bms[v]
            if u == -1:
                assert v == 0 or np.isinf(d_bms[v])
                continue
            assert d_bms[u] + csr[u, v] == pytest.approx(d_bms[v], rel=0, abs=1e-9)


def test_bmssp_edge_cases():
    # Single vertex.
    g = sp.csr_matrix((1, 1), dtype=np.float64)
    dist, pred = shortest_paths(g, 0, method="bmssp")
    assert dist.tolist() == [0.0]
    assert pred.tolist() == [-1]

    # Multi-edges take the minimum (CSR triple keeps parallel edges).
    indptr = np.array([0, 2, 2], dtype=np.int64)
    indices = np.array([1, 1], dtype=np.int32)
    weights = np.array([5.0, 2.0], dtype=np.float64)
    dist, pred = shortest_paths((indptr, indices, weights), 0, method="bmssp")
    assert dist.tolist() == [0.0, 2.0]
    assert pred.tolist() == [-1, 0]

    # Self-loop does not corrupt results.
    g = sp.csr_matrix(([2.0, 3.0], ([0, 0], [0, 1])), shape=(2, 2), dtype=np.float64)
    dist, pred = shortest_paths(g, 0, method="bmssp")
    assert dist.tolist() == [0.0, 3.0]
    assert pred.tolist() == [-1, 0]


def test_bmssp_errors():
    g = sp.csr_matrix((3, 3), dtype=np.float64)
    with pytest.raises(IndexError):
        shortest_paths(g, 3, method="bmssp")
    g = sp.csr_matrix(([-1.0], ([0], [1])), shape=(3, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        shortest_paths(g, 0, method="bmssp")


def test_unknown_method_still_raises():
    g = sp.csr_matrix((2, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        shortest_paths(g, 0, method="bellman-ford")


@pytest.mark.parametrize(
    "method",
    ["bmssp-fast", "bmssp-tuned", "bmssp-hybrid", "bmssp-simpleq",
     "bmssp-lazypiv", "bmssp-notransform"],
)
def test_variant_methods_match_dijkstra(method):
    """Every public bmssp-<name> method is distance-bit-exact vs dijkstra
    (the deep gate is tests/variants_correctness.rs; this pins the Python
    dispatch)."""
    rng = np.random.default_rng(777)
    for n, density in [(2, 0.5), (120, 0.05), (500, 0.01)]:
        g = random_graph(n, density, rng)
        d_dij, _ = shortest_paths(g, 0, method="dijkstra")
        d_var, _ = shortest_paths(g, 0, method=method)
        assert np.array_equal(d_dij, d_var), (method, n, density)


def test_auto_selects_dijkstra():
    """method="auto" (the default) always selects dijkstra — the
    BENCHMARKS.md verdict: no BMSSP crossover at practical sizes."""
    rng = np.random.default_rng(99)
    g = random_graph(300, 0.02, rng)
    d_auto, p_auto = shortest_paths(g, 0, method="auto")
    d_default, p_default = shortest_paths(g, 0)
    d_dij, p_dij = shortest_paths(g, 0, method="dijkstra")
    np.testing.assert_array_equal(d_auto, d_dij)
    np.testing.assert_array_equal(p_auto, p_dij)
    np.testing.assert_array_equal(d_default, d_dij)
    np.testing.assert_array_equal(p_default, p_dij)
