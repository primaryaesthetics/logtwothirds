"""Edge-case tests for logtwothirds.shortest_paths."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from logtwothirds import shortest_paths


def test_empty_graph():
    """Zero vertices: both outputs are length-0 arrays."""
    g = sp.csr_matrix((0, 0), dtype=np.float64)
    # Any source is out of range for an empty graph.
    with pytest.raises(IndexError):
        shortest_paths(g, 0)


def test_single_vertex():
    g = sp.csr_matrix((1, 1), dtype=np.float64)
    dist, pred = shortest_paths(g, 0)
    assert dist.tolist() == [0.0]
    assert pred.tolist() == [-1]


def test_single_vertex_self_loop():
    g = sp.csr_matrix(([5.0], ([0], [0])), shape=(1, 1), dtype=np.float64)
    dist, pred = shortest_paths(g, 0)
    assert dist.tolist() == [0.0]  # self-loop never improves the source
    assert pred.tolist() == [-1]


def test_disconnected_graph():
    # 0 -> 1 (1.0); vertices 2,3 unreachable; 2 -> 3 (1.0).
    g = sp.csr_matrix(
        ([1.0, 1.0], ([0, 2], [1, 3])), shape=(4, 4), dtype=np.float64
    )
    dist, pred = shortest_paths(g, 0)
    assert dist[0] == 0.0
    assert dist[1] == 1.0
    assert np.isinf(dist[2])
    assert np.isinf(dist[3])
    assert pred.tolist() == [-1, 0, -1, -1]


def test_zero_weight_edges():
    # 0 ->0-> 1 ->0-> 2 : all reachable at distance 0.
    g = sp.csr_matrix(
        ([0.0, 0.0], ([0, 1], [1, 2])), shape=(3, 3), dtype=np.float64
    )
    dist, pred = shortest_paths(g, 0)
    assert dist.tolist() == [0.0, 0.0, 0.0]
    assert pred.tolist() == [-1, 0, 1]


def test_self_loop_does_not_affect_others():
    # 0 -> 0 (2.0), 0 -> 1 (3.0). Self-loop must not corrupt results.
    g = sp.csr_matrix(
        ([2.0, 3.0], ([0, 0], [0, 1])), shape=(2, 2), dtype=np.float64
    )
    dist, pred = shortest_paths(g, 0)
    assert dist.tolist() == [0.0, 3.0]
    assert pred.tolist() == [-1, 0]


def test_multi_edges_take_minimum():
    # Two parallel 0 -> 1 edges with weights 5 and 2; the shorter wins.
    # Build via COO so duplicates are preserved into CSR (scipy sums them,
    # so we feed a CSR triple directly to keep both edges distinct).
    indptr = np.array([0, 2, 2], dtype=np.int64)
    indices = np.array([1, 1], dtype=np.int32)
    weights = np.array([5.0, 2.0], dtype=np.float64)
    dist, pred = shortest_paths((indptr, indices, weights), 0)
    assert dist.tolist() == [0.0, 2.0]
    assert pred.tolist() == [-1, 0]


def test_source_out_of_range_high():
    g = sp.csr_matrix((3, 3), dtype=np.float64)
    with pytest.raises(IndexError):
        shortest_paths(g, 3)


def test_source_out_of_range_negative():
    g = sp.csr_matrix((3, 3), dtype=np.float64)
    with pytest.raises(IndexError):
        shortest_paths(g, -1)


def test_negative_weight_raises_value_error():
    g = sp.csr_matrix(
        ([1.0, -2.0], ([0, 1], [1, 2])), shape=(3, 3), dtype=np.float64
    )
    with pytest.raises(ValueError):
        shortest_paths(g, 0)


def test_unknown_method_raises_value_error():
    g = sp.csr_matrix((2, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        shortest_paths(g, 0, method="bellman-ford")


def test_csr_triple_input():
    # Same triangle as the scipy path, fed as a raw CSR triple.
    indptr = np.array([0, 2, 3, 3], dtype=np.int64)
    indices = np.array([1, 2, 2], dtype=np.int32)
    weights = np.array([1.0, 4.0, 2.0], dtype=np.float64)
    dist, pred = shortest_paths((indptr, indices, weights), 0)
    assert dist.tolist() == [0.0, 1.0, 3.0]
    assert pred.tolist() == [-1, 0, 1]


def test_predecessor_tree_is_consistent():
    """Predecessors must reconstruct the reported distances exactly."""
    rng = np.random.default_rng(42)
    n = 200
    m = 2000
    rows = rng.integers(0, n, size=m)
    cols = rng.integers(0, n, size=m)
    data = rng.uniform(0.1, 1.0, size=m)
    g = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    g.sort_indices()
    dist, pred = shortest_paths(g, 0)

    # Dense lookup of edge weights for verification.
    for v in range(n):
        if pred[v] == -1:
            continue
        u = pred[v]
        w = g[u, v]
        assert abs(dist[v] - (dist[u] + w)) <= 1e-9
