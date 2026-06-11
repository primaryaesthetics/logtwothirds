"""Tests for base_case (Algorithm 2, ALGORITHM.md S4.2 / SPEC.md S8.3)."""

from __future__ import annotations

import math

from logtwothirds._reference import (
    INF,
    INF_INT,
    Graph,
    Key,
    State,
    base_case,
    build_graph,
)

from .reference import dijkstra_with_order, t_lt


def fabricate_singleton_state(
    g: Graph, source: int, x: int, k: int, t: int
) -> tuple[State, list[float], list[int], list[int]]:
    """SPEC.md S8.3: ``S = {x}``, ``x`` complete; everything else unset.

    ``source`` anchors the oracle distances ``d(.)``; ``x`` need not equal
    ``source``. ``x``'s out-edges are relaxed by ``base_case`` itself, not
    here, so the bound-gating test can observe what it does/doesn't write.
    """
    dist, hops, pred = dijkstra_with_order(g, source)
    st = State.new(g, source, k, t)
    st.dhat[x] = dist[x]
    st.hops[x] = hops[x]
    st.pred[x] = pred[x]
    return st, dist, hops, pred


def chain_graph(n: int, w: float = 1.0) -> Graph:
    return build_graph(n, [(i, i + 1, w) for i in range(n - 1)])


def test_heap_exhausted_returns_B_and_full_subtree():
    # Chain of 4 vertices, k large enough that the whole reachable set fits
    # within k+1 settled vertices.
    g = chain_graph(4)
    st, dist, hops, pred = fabricate_singleton_state(g, 0, 0, k=5, t=1)
    B: Key = INF

    Bp, U = base_case(st, B, [0])

    assert Bp == B
    expected = t_lt(dist, hops, pred, [0], B)
    assert set(U) == expected
    for v in U:
        assert st.dhat[v] == dist[v]
        assert st.settled[v]


def test_truncation_returns_max_settled_key_and_filtered_set():
    # Chain of 10 vertices, k=2 -> heap stops once |U0| == k+1 == 3.
    g = chain_graph(10)
    st, dist, hops, pred = fabricate_singleton_state(g, 0, 0, k=2, t=1)
    B: Key = INF

    Bp, U = base_case(st, B, [0])

    assert len(U) == 2  # |U0| == k+1 == 3, truncated to {v : key(v) < B'}
    assert Bp == (2.0, 2, 2)
    assert set(U) == {0, 1}

    expected = t_lt(dist, hops, pred, [0], Bp)
    assert set(U) == expected
    for v in U:
        assert st.dhat[v] == dist[v]


def test_bound_respected_no_write_beyond_B():
    # 0 -(5.0)-> 1: candidate for vertex 1 is (5.0, 1, 0), well past B.
    g = build_graph(2, [(0, 1, 5.0)])
    st, dist, hops, pred = fabricate_singleton_state(g, 0, 0, k=5, t=1)
    B: Key = (3.0, INF_INT, INF_INT)

    Bp, U = base_case(st, B, [0])

    assert Bp == B
    assert U == [0]
    # vertex 1's candidate (5.0, 1, 0) is not < B, so it must be untouched.
    assert math.isinf(st.dhat[1])
    assert st.hops[1] == INF_INT
    assert st.pred[1] == -1


def test_zero_weight_cycle_terminates_with_hops_tiebreak():
    # 0 -(1)-> 1 <-> 2 (zero-weight cycle) -(1)-> 3.
    g = build_graph(
        4, [(0, 1, 1.0), (1, 2, 0.0), (2, 1, 0.0), (2, 3, 1.0)]
    )
    st, dist, hops, pred = fabricate_singleton_state(g, 0, 0, k=5, t=1)
    B: Key = INF

    Bp, U = base_case(st, B, [0])

    assert Bp == B
    expected = t_lt(dist, hops, pred, [0], B)
    assert set(U) == expected
    for v in U:
        assert st.dhat[v] == dist[v]
        assert st.hops[v] == hops[v]
