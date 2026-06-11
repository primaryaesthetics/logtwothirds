"""Tests for transform_to_constant_degree (ALGORITHM.md S1.1 / SPEC.md S8.5)."""

from __future__ import annotations

import random

from logtwothirds._reference import build_graph, transform_to_constant_degree

from .reference import dijkstra


def in_out_degrees(g) -> tuple[list[int], list[int]]:
    out_deg = [g.indptr[v + 1] - g.indptr[v] for v in range(g.n)]
    in_deg = [0] * g.n
    for v in g.indices:
        in_deg[v] += 1
    return in_deg, out_deg


def test_degrees_at_most_two():
    rng = random.Random(0)
    n = 30
    edges = [
        (rng.randrange(n), rng.randrange(n), round(rng.uniform(0, 5), 3))
        for _ in range(120)
    ]
    g = build_graph(n, edges)
    g2, source2, rep = transform_to_constant_degree(g, 0)

    in_deg, out_deg = in_out_degrees(g2)
    assert all(d <= 2 for d in in_deg)
    assert all(d <= 2 for d in out_deg)


def test_distances_preserved_random_graphs():
    rng = random.Random(1)
    for trial in range(10):
        n = rng.randint(2, 25)
        m = rng.randint(0, 3 * n)
        edges = [
            (rng.randrange(n), rng.randrange(n), round(rng.uniform(0, 5), 3))
            for _ in range(m)
        ]
        g = build_graph(n, edges)
        source = rng.randrange(n)

        g2, source2, rep = transform_to_constant_degree(g, source)

        d_g = dijkstra(g, source)
        d_g2 = dijkstra(g2, source2)

        for v in range(n):
            assert abs(d_g[v] - d_g2[rep[v]]) < 1e-9 or (
                d_g[v] == float("inf") and d_g2[rep[v]] == float("inf")
            )


def test_no_neighbors():
    g = build_graph(2, [])
    g2, source2, rep = transform_to_constant_degree(g, 0)
    assert g2.n == 2
    assert dijkstra(g2, source2)[rep[1]] == float("inf")


def test_one_neighbor():
    g = build_graph(2, [(0, 1, 1.5)])
    g2, source2, rep = transform_to_constant_degree(g, 0)
    d = dijkstra(g2, source2)
    assert abs(d[rep[1]] - 1.5) < 1e-9


def test_self_loop():
    g = build_graph(2, [(0, 0, 1.0), (0, 1, 2.0)])
    g2, source2, rep = transform_to_constant_degree(g, 0)
    d = dijkstra(g2, source2)
    assert abs(d[rep[0]] - 0.0) < 1e-9
    assert abs(d[rep[1]] - 2.0) < 1e-9
    in_deg, out_deg = in_out_degrees(g2)
    assert all(x <= 2 for x in in_deg)
    assert all(x <= 2 for x in out_deg)


def test_parallel_edges():
    g = build_graph(2, [(0, 1, 3.0), (0, 1, 1.0), (0, 1, 2.0)])
    g2, source2, rep = transform_to_constant_degree(g, 0)
    d = dijkstra(g2, source2)
    assert abs(d[rep[1]] - 1.0) < 1e-9
