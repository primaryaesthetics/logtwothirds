"""Oracles for the BMSSP test suite. SPEC.md S8 ("tests/reference.py").

Plain Dijkstra and Bellman-Ford (distances only), plus a "with order"
Dijkstra variant that additionally produces the (dist, hops, pred) shortest
path tree under the Assumption 2.1 tie-breaking total order
(ALGORITHM.md S1.3) -- used to fabricate mid-algorithm states and to compute
T(S), T_<B(S), etc. for postcondition checks.
"""

from __future__ import annotations

import heapq
import math

from logtwothirds._reference import INF_INT, Graph, Key, out_edges


def dijkstra(g: Graph, source: int) -> list[float]:
    """Textbook Dijkstra; distances only."""
    n = g.n
    dist = [math.inf] * n
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in out_edges(g, u):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def bellman_ford(g: Graph, source: int) -> list[float]:
    """Textbook Bellman-Ford; distances only."""
    n = g.n
    dist = [math.inf] * n
    dist[source] = 0.0
    for _ in range(max(0, n - 1)):
        changed = False
        for u in range(n):
            if dist[u] == math.inf:
                continue
            for v, w in out_edges(g, u):
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    changed = True
        if not changed:
            break
    return dist


def dijkstra_with_order(
    g: Graph, source: int
) -> tuple[list[float], list[int], list[int]]:
    """Dijkstra producing the canonical (dist, hops, pred) shortest path tree
    under the ALGORITHM.md S1.3 total order: a candidate
    ``(cand_len, cand_hops, u)`` replaces vertex v's current
    ``(dist[v], hops[v], pred[v])`` whenever it is lexicographically <=.

    This makes Pred a tree (Assumption 2.1) and gives a unique answer for
    "the" shortest path even with zero-weight edges and ties.
    """
    n = g.n
    dist = [math.inf] * n
    hops = [INF_INT] * n
    pred = [-1] * n
    dist[source] = 0.0
    hops[source] = 0

    heap: list[tuple[float, int, int]] = [(0.0, 0, source)]
    while heap:
        d, h, u = heapq.heappop(heap)
        if (d, h, u) != (dist[u], hops[u], u):
            continue  # stale entry
        for v, w in out_edges(g, u):
            cand = (d + w, h + 1, u)
            cur = (dist[v], hops[v], pred[v])
            if cand <= cur:
                dist[v], hops[v], pred[v] = cand
                heapq.heappush(heap, (dist[v], hops[v], v))
    return dist, hops, pred


def subtree(pred: list[int], root: int) -> set[int]:
    """All descendants of ``root`` (inclusive) in the tree given by ``pred``."""
    children: dict[int, list[int]] = {}
    for v, p in enumerate(pred):
        if p != -1:
            children.setdefault(p, []).append(v)
    out: set[int] = set()
    stack = [root]
    while stack:
        x = stack.pop()
        out.add(x)
        stack.extend(children.get(x, ()))
    return out


def t_of(pred: list[int], roots: list[int]) -> set[int]:
    """``T(S) = union_{v in S} T(v)``. ALGORITHM.md S1.4."""
    out: set[int] = set()
    for r in roots:
        out |= subtree(pred, r)
    return out


def t_lt(
    dist: list[float],
    hops: list[int],
    pred: list[int],
    roots: list[int],
    bound: Key,
) -> set[int]:
    """``T_<B(S) = {v in T(S) : key(v) < B}``. ALGORITHM.md S1.4."""
    cand = t_of(pred, roots)
    return {v for v in cand if (dist[v], hops[v], v) < bound}


def t_in_range(
    dist: list[float],
    hops: list[int],
    pred: list[int],
    roots: list[int],
    lo: Key,
    hi: Key,
) -> set[int]:
    """``T_[lo,hi)(S) = {v in T(S) : lo <= key(v) < hi}``. ALGORITHM.md S1.4."""
    cand = t_of(pred, roots)
    return {v for v in cand if lo <= (dist[v], hops[v], v) < hi}
