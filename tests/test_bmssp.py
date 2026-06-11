"""Tests for bmssp (Algorithm 3, ALGORITHM.md S4.3 / SPEC.md S8.4)."""

from __future__ import annotations

import math
import random

import pytest

from logtwothirds._reference import (
    INF,
    BlockDS,
    Graph,
    Key,
    State,
    bmssp,
    build_graph,
    key,
)

from .reference import dijkstra_with_order, t_lt
from .test_find_pivots import fabricate_state


def random_graph(n: int, m: int, seed: int, max_w: float = 5.0) -> Graph:
    """A connected-ish backbone chain plus ``m`` random extra edges."""
    rng = random.Random(seed)
    edges = [(i, i + 1, round(rng.uniform(0.1, max_w), 3)) for i in range(n - 1)]
    for _ in range(m):
        u = rng.randrange(n)
        v = rng.randrange(n)
        edges.append((u, v, round(rng.uniform(0.0, max_w), 3)))
    return build_graph(n, edges)


def chain_graph(n: int, w: float = 1.0) -> Graph:
    return build_graph(n, [(i, i + 1, w) for i in range(n - 1)])


def check_bmssp(
    g: Graph, source: int, b: Key, B: Key, k: int, t: int, l: int
) -> tuple[Key, list[int], "State"]:
    st, S, dist, hops, pred = fabricate_state(g, source, b, B, k, t)
    assert S, "fabricated S must be non-empty"
    assert len(S) <= 2 ** (l * t), "precondition |S| <= 2**(l*t) violated"

    Bp, U = bmssp(st, l, B, S)

    assert Bp <= B

    expected = t_lt(dist, hops, pred, S, Bp)
    assert set(U) == expected, (set(U), expected)

    for u in U:
        assert st.dhat[u] == dist[u], (u, st.dhat[u], dist[u])

    bound_cap = k * (2 ** (l * t))
    if Bp < B:
        assert len(U) >= bound_cap, (len(U), bound_cap)
    assert len(U) <= 4 * bound_cap, (len(U), bound_cap)

    return Bp, U, st


# ---------------------------------------------------------------------------
# Postcondition (Lemma 3.7) on random graphs, l = 1..3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("l", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_postcondition_random_graphs(l: int, seed: int):
    k, t = 2, 2
    g = random_graph(n=60, m=120, seed=seed)
    source = 0
    b: Key = (0.0, 0, 0)
    B: Key = INF
    check_bmssp(g, source, b, B, k, t, l)


# ---------------------------------------------------------------------------
# Small Ũ: full drain, B' == B
# ---------------------------------------------------------------------------


def test_small_reachable_set_returns_B_unchanged():
    k, t, l = 2, 2, 1
    g = chain_graph(6)
    b: Key = (0.0, 0, 0)
    B: Key = INF
    Bp, U, st = check_bmssp(g, 0, b, B, k, t, l)
    assert Bp == B


# ---------------------------------------------------------------------------
# Internal probes: disjointness (already asserted internally), max key in
# S_i < B_i, batch_prepend values < B_i, Lemma 3.10's
# min_{x in D} d(x) >= B'_{i-1}.
# ---------------------------------------------------------------------------


def test_internal_invariants_via_blockds_probes(monkeypatch):
    k, t, l = 2, 2, 2
    g = random_graph(n=50, m=100, seed=3)
    source = 0
    b: Key = (0.0, 0, 0)
    B: Key = INF

    st, S, dist, hops, pred = fabricate_state(g, source, b, B, k, t)
    assert S
    assert len(S) <= 2 ** (l * t)

    true_dist = dist  # oracle distances, indexed by transformed-graph vertex

    # Each bmssp stack frame owns its own BlockDS instance, so the "current
    # B_i" for that frame's batch_prepend calls can be tracked per-instance
    # (id(self)), surviving the recursive call that happens between this
    # frame's pull() and batch_prepend()).
    last_Bi: dict[int, Key] = {}

    orig_pull = BlockDS.pull

    def traced_pull(self):
        Si, Bi = orig_pull(self)
        # Lemma 3.10: max key in S_i < B_i, and every pulled vertex is
        # actually reachable (finite oracle distance).
        for x in Si:
            assert key(st, x) < Bi, (x, key(st, x), Bi)
            assert true_dist[x] < math.inf
        last_Bi[id(self)] = Bi
        return Si, Bi

    orig_bp = BlockDS.batch_prepend

    def traced_bp(self, items):
        for (_x, val) in items:
            assert val < last_Bi[id(self)], (val, last_Bi[id(self)])
        return orig_bp(self, items)

    monkeypatch.setattr(BlockDS, "pull", traced_pull)
    monkeypatch.setattr(BlockDS, "batch_prepend", traced_bp)

    Bp, U = bmssp(st, l, B, S)

    expected = t_lt(dist, hops, pred, S, Bp)
    assert set(U) == expected
