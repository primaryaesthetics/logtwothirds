"""Tests for find_pivots (Algorithm 1, ALGORITHM.md S4.1 / SPEC.md S8.2)."""

from __future__ import annotations

import math

import pytest

from logtwothirds._reference import (
    INF,
    INF_INT,
    Graph,
    Key,
    State,
    build_graph,
    find_pivots,
    out_edges,
    try_relax,
)

from .reference import dijkstra_with_order, subtree, t_lt


def fabricate_state(
    g: Graph, source: int, b: Key, B: Key, k: int, t: int
) -> tuple[State, list[int], list[float], list[int], list[int]]:
    """SPEC.md S8.2 "Easiest valid fabrication".

    Mark ``complete = {v : key(v) < b}`` with dhat/hops/pred from the
    reference tree and relax their out-edges once; let
    ``S = {v : b <= key(v) < B, dhat[v] < inf}``.
    """
    dist, hops, pred = dijkstra_with_order(g, source)
    st = State.new(g, source, k, t)

    complete = [v for v in range(g.n) if (dist[v], hops[v], v) < b]
    for v in complete:
        st.dhat[v] = dist[v]
        st.hops[v] = hops[v]
        st.pred[v] = pred[v]
    for u in complete:
        for v, w in out_edges(g, u):
            try_relax(st, u, v, w)

    S = [
        v
        for v in range(g.n)
        if b <= (dist[v], hops[v], v) < B and st.dhat[v] < math.inf
    ]
    return st, S, dist, hops, pred


def check_find_pivots(
    g: Graph, source: int, b: Key, B: Key, k: int, t: int = 1
) -> tuple[list[int], list[int], list[int], State]:
    """Run find_pivots on a fabricated state and check Lemma 3.2's contract.

    Returns (P, W, S, st) for case-specific extra assertions.
    """
    st, S, dist, hops, pred = fabricate_state(g, source, b, B, k, t)
    assert S, "fabricated S must be non-empty for a meaningful test"

    u_tilde = t_lt(dist, hops, pred, S, B)

    P, W = find_pivots(st, B, S)

    assert set(P) <= set(S)
    assert set(W) <= u_tilde

    if len(W) > k * len(S):
        # Early-exit branch (ALGORITHM.md S4.1 L12-14). The |W| = O(k|S|)
        # bound assumes constant out-degree (post-transform graphs); these
        # fixtures use arbitrary degree, so only check the qualitative
        # branch conditions here.
        assert set(P) == set(S)
    else:
        assert len(P) <= len(W) // k if k > 0 else True

    # Lemma 3.2 disjunction.
    complete_P = {y for y in P if st.dhat[y] == dist[y]}
    for x in u_tilde:
        if x in W and st.dhat[x] == dist[x]:
            continue
        assert any(x in subtree(pred, y) for y in complete_P), (
            f"vertex {x} satisfies neither branch of Lemma 3.2's disjunction"
        )

    # dhat soundness: never below the true distance, never increased.
    for v in range(g.n):
        if dist[v] < math.inf:
            assert st.dhat[v] >= dist[v] - 1e-12

    return P, W, S, st


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------


def chain_graph(n: int, w: float = 1.0) -> Graph:
    return build_graph(n, [(i, i + 1, w) for i in range(n - 1)])


def star_graph(n_leaves: int, w: float = 1.0) -> Graph:
    return build_graph(n_leaves + 1, [(0, i + 1, w) for i in range(n_leaves)])


def parallel_branches_graph() -> Graph:
    # 0 -> 1 (weight 1)         : dist 1, hops 1
    # 0 -> 2 -> 3 (weight 0.5 each): dist 1, hops 2 -- tie broken by hops
    return build_graph(
        4, [(0, 1, 1.0), (0, 2, 0.5), (2, 3, 0.5)]
    )


def zero_weight_cycle_graph() -> Graph:
    # 0 -> 1 (w=1); 1 <-> 2 zero-weight cycle; 2 -> 3 (w=1)
    return build_graph(
        4, [(0, 1, 1.0), (1, 2, 0.0), (2, 1, 0.0), (2, 3, 1.0)]
    )


def explosion_graph(fanout: int) -> Graph:
    # source 0 fans out to `fanout` leaves with weight 1 each.
    return build_graph(fanout + 1, [(0, i + 1, 1.0) for i in range(fanout)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chain_k_step_propagation():
    g = chain_graph(6)
    b: Key = (0.0, 0, 0)  # nothing is "complete" except via S itself
    B: Key = INF
    check_find_pivots(g, 0, b, B, k=2)


def test_star_graph():
    g = star_graph(5)
    b: Key = (0.0, 0, 0)
    B: Key = INF
    check_find_pivots(g, 0, b, B, k=2)


def test_parallel_branches_hops_tiebreak():
    g = parallel_branches_graph()
    b: Key = (0.0, 0, 0)
    B: Key = INF
    P, W, S, st = check_find_pivots(g, 0, b, B, k=3)
    # Both branch endpoints (1 and 3, distance 1 with hops 1 and 2) should be
    # reachable in W under a sufficiently large k.
    assert {1, 3} <= set(W)


def test_zero_weight_cycle():
    g = zero_weight_cycle_graph()
    b: Key = (0.0, 0, 0)
    B: Key = INF
    check_find_pivots(g, 0, b, B, k=3)


def test_w_explosion_triggers_early_exit():
    g = explosion_graph(fanout=5)
    b: Key = (0.0, 0, 0)
    B: Key = INF
    P, W, S, st = check_find_pivots(g, 0, b, B, k=1)
    assert set(S) == {0}
    assert len(W) > 1 * len(S)
    assert set(P) == set(S)


def test_empty_pivot_set_when_everything_within_k_hops():
    # 0 -> 1, weight 1; with k=3 the whole reachable set is within k hops of
    # the single source, but the resulting tight tree has only 2 < k=3
    # vertices, so no pivot qualifies (ALGORITHM.md S6.5).
    g = build_graph(2, [(0, 1, 1.0)])
    b: Key = (0.0, 0, 0)
    B: Key = INF
    P, W, S, st = check_find_pivots(g, 0, b, B, k=3)
    assert S == [0]
    assert set(W) == {0, 1}
    assert P == []


@pytest.mark.parametrize("b_idx,B_idx", [(0, 5), (1, 4), (2, 3)])
def test_chain_various_bound_windows(b_idx: int, B_idx: int):
    g = chain_graph(8)
    dist, hops, pred = dijkstra_with_order(g, 0)
    keys = [(dist[v], hops[v], v) for v in range(g.n)]
    keys_sorted = sorted(set(keys))
    # Pick a bound strictly between two consecutive keys.
    b = (keys_sorted[b_idx][0], keys_sorted[b_idx][1], -1)
    B = (keys_sorted[B_idx][0] + 0.5, INF_INT, INF_INT)
    check_find_pivots(g, 0, b, B, k=2)
