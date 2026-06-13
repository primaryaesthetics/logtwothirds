"""shortest_paths_multi: parallel rows == sequential single-source."""

import numpy as np
import pytest
import scipy.sparse as sp

from logtwothirds import shortest_paths_multi, shortest_paths


def _random_csr(n, m, seed):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n, size=m)
    cols = rng.integers(0, n, size=m)
    data = rng.uniform(0.0, 1.0, size=m)
    g = sp.coo_array((data, (rows, cols)), shape=(n, n)).tocsr()
    g.sort_indices()
    return g


@pytest.mark.parametrize("method", ["dijkstra", "bmssp"])
def test_rows_match_single_source(method):
    g = _random_csr(500, 2000, seed=7)
    sources = [0, 17, 499, 17]  # duplicates allowed
    dist, pred = shortest_paths_multi(g, sources, method=method)
    assert dist.shape == (4, 500)
    assert pred.shape == (4, 500)
    assert dist.dtype == np.float64
    assert pred.dtype == np.int32
    for i, s in enumerate(sources):
        d1, p1 = shortest_paths(g, s, method=method)
        np.testing.assert_array_equal(dist[i], d1)
        np.testing.assert_array_equal(pred[i], p1)


def test_empty_sources():
    g = _random_csr(50, 100, seed=1)
    dist, pred = shortest_paths_multi(g, [])
    assert dist.shape == (0, 50)
    assert pred.shape == (0, 50)


@pytest.mark.parametrize("method", ["dijkstra", "bmssp"])
def test_source_out_of_range(method):
    g = _random_csr(10, 30, seed=2)
    with pytest.raises(IndexError):
        shortest_paths_multi(g, [0, 10], method=method)
    with pytest.raises(IndexError):
        shortest_paths_multi(g, [-1], method=method)


def test_negative_weight_rejected():
    indptr = np.array([0, 1, 1], dtype=np.int64)
    indices = np.array([1], dtype=np.int32)
    weights = np.array([-1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        shortest_paths_multi((indptr, indices, weights), [0])


def test_unknown_method():
    g = _random_csr(10, 30, seed=3)
    with pytest.raises(ValueError, match="unknown method"):
        shortest_paths_multi(g, [0], method="bfs")
    # The single-source-only research variants are rejected here too.
    with pytest.raises(ValueError, match="unknown method"):
        shortest_paths_multi(g, [0], method="bmssp-fast")


def test_auto_method_is_dijkstra():
    """method="auto" (the default) selects dijkstra always (BENCHMARKS.md)."""
    g = _random_csr(200, 800, seed=11)
    d_auto, p_auto = shortest_paths_multi(g, [0, 5], method="auto")
    d_dij, p_dij = shortest_paths_multi(g, [0, 5], method="dijkstra")
    np.testing.assert_array_equal(d_auto, d_dij)
    np.testing.assert_array_equal(p_auto, p_dij)
    d1, p1 = shortest_paths(g, 0, method="auto")
    np.testing.assert_array_equal(d1, d_dij[0])
    np.testing.assert_array_equal(p1, p_dij[0])
