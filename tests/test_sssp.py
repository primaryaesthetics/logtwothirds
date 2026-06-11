"""End-to-end tests for sssp (ALGORITHM.md S4.4 / SPEC.md S8.6).

Oracle: reference Dijkstra on the original (untransformed) graph.
"""

from __future__ import annotations

import math

import pytest

from logtwothirds._reference import build_graph, sssp

from .reference import dijkstra


def assert_matches_dijkstra(g, source):
    got = sssp(g, source)
    want = dijkstra(g, source)
    assert len(got) == len(want)
    for v, (a, b) in enumerate(zip(got, want)):
        if math.isinf(a) and math.isinf(b):
            continue
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-9), (v, a, b)


def test_single_vertex():
    g = build_graph(1, [])
    assert sssp(g, 0) == [0.0]


def test_single_edge():
    g = build_graph(2, [(0, 1, 3.5)])
    assert_matches_dijkstra(g, 0)


def test_two_vertex_zero_weight_cycle():
    g = build_graph(2, [(0, 1, 0.0), (1, 0, 0.0)])
    assert_matches_dijkstra(g, 0)
    assert sssp(g, 0) == [0.0, 0.0]


def test_chain_of_1000():
    n = 1000
    g = build_graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])
    assert_matches_dijkstra(g, 0)
    got = sssp(g, 0)
    assert got[-1] == 999.0


def test_star():
    n_leaves = 20
    g = build_graph(
        n_leaves + 1, [(0, i + 1, float(i + 1)) for i in range(n_leaves)]
    )
    assert_matches_dijkstra(g, 0)


def test_complete_digraph_30():
    n = 30
    edges = [(u, v, float((u + 2 * v) % 7) + 0.5) for u in range(n) for v in range(n) if u != v]
    g = build_graph(n, edges)
    assert_matches_dijkstra(g, 0)


def test_all_zero_weights():
    n = 12
    edges = [(i, (i + 1) % n, 0.0) for i in range(n)]
    edges += [(0, 5, 0.0), (5, 0, 0.0)]
    g = build_graph(n, edges)
    assert_matches_dijkstra(g, 0)
    got = sssp(g, 0)
    assert all(d == 0.0 for d in got)


def test_all_equal_weights_forces_ties():
    n = 16
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n, 1.0))
        edges.append((i, (i + 3) % n, 1.0))
        edges.append((i, (i + 5) % n, 1.0))
    g = build_graph(n, edges)
    assert_matches_dijkstra(g, 0)


def test_unreachable_component():
    g = build_graph(6, [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0)])
    got = sssp(g, 0)
    assert got[0] == 0.0
    assert got[1] == 1.0
    assert got[2] == 2.0
    assert math.isinf(got[3])
    assert math.isinf(got[4])
    assert math.isinf(got[5])
    assert_matches_dijkstra(g, 0)


def test_source_with_no_out_edges():
    g = build_graph(3, [(1, 0, 1.0), (1, 2, 1.0)])
    got = sssp(g, 0)
    assert got[0] == 0.0
    assert math.isinf(got[1])
    assert math.isinf(got[2])
    assert_matches_dijkstra(g, 0)


def test_large_magnitude_spread():
    g = build_graph(
        4, [(0, 1, 1e-9), (1, 2, 1e9), (0, 2, 5e8), (2, 3, 1e-9)]
    )
    assert_matches_dijkstra(g, 0)


def test_negative_weight_raises():
    g = build_graph(2, [(0, 1, 1.0)])
    # Manually corrupt a weight to bypass build_graph's validation.
    object.__setattr__(g, "weights", [-1.0])
    with pytest.raises(ValueError):
        sssp(g, 0)


def test_source_out_of_range_raises():
    g = build_graph(2, [(0, 1, 1.0)])
    with pytest.raises(IndexError):
        sssp(g, 5)
    with pytest.raises(IndexError):
        sssp(g, -1)
