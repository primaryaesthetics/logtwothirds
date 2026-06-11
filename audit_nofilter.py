"""Audit experiment: paper-literal BMSSP (QUESTIONS.md item-3 filter REMOVED).

Runs Algorithm 3 exactly as in the paper (no settled-vertex filter on pulled
batches, no U_i-disjointness assert) on the same random-graph corpus as
audit_stress.py, and reports:
  - output mismatches vs the Dijkstra oracle (read from dhat, as specified);
  - how often a pulled batch contained an already-settled vertex;
  - how often U_i-disjointness was violated;
  - how often the returned B'_i regressed below B'_{i-1} (monotonicity loss);
  - crashes (e.g. BatchPrepend precondition violations).
"""
from __future__ import annotations

import heapq
import math
import random

import logtwothirds._reference as ref
from logtwothirds._reference import (
    BlockDS, INF, Key, State, base_case, find_pivots, key, out_edges,
    try_relax, _settle,
)

STATS = {
    "runs": 0,
    "output_mismatch_runs": 0,
    "stale_pulls": 0,
    "disjoint_violations": 0,
    "bprime_regressions": 0,
    "crashes": 0,
}


def dijkstra(g: ref.Graph, s: int) -> list[float]:
    dist = [math.inf] * g.n
    dist[s] = 0.0
    done = [False] * g.n
    h = [(0.0, s)]
    while h:
        d, u = heapq.heappop(h)
        if done[u]:
            continue
        done[u] = True
        for v, w in out_edges(g, u):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist


def bmssp_literal(st: State, l: int, B: Key, S: list[int]):
    """Algorithm 3 verbatim: no settled filter, no disjointness assert."""
    if l == 0:
        return base_case(st, B, S)
    g = st.g
    k, t = st.k, st.t
    P, W = find_pivots(st, B, S)
    M = max(1, min(2 ** ((l - 1) * t), g.n))
    D = BlockDS(M, B)
    for x in P:
        D.insert(x, key(st, x))
    Bp_0 = min((key(st, x) for x in P), default=B)
    U: set[int] = set()
    Bp_last = Bp_0
    bound_cap = k * (2 ** (l * t))
    while len(U) < bound_cap and len(D) > 0:
        Si, Bi = D.pull()
        if any(st.settled[x] for x in Si):
            STATS["stale_pulls"] += 1
        Bp_i, Ui = bmssp_literal(st, l - 1, Bi, Si)
        if not U.isdisjoint(Ui):
            STATS["disjoint_violations"] += 1
        if Bp_i < Bp_last:
            STATS["bprime_regressions"] += 1
        U |= set(Ui)
        Bp_last = Bp_i
        K: list[tuple[int, Key]] = []
        for u in Ui:
            for v, w in out_edges(g, u):
                outcome = try_relax(st, u, v, w)
                if outcome.passed:
                    vkey = key(st, v)
                    if Bi <= vkey < B:
                        D.insert(v, vkey)
                    elif Bp_i <= vkey < Bi:
                        K.append((v, vkey))
        prepend = K + [(x, key(st, x)) for x in Si if Bp_i <= key(st, x) < Bi]
        if prepend:
            D.batch_prepend(prepend)
    Bp = min(Bp_last, B)
    result_U = set(U)
    for x in W:
        if (st.dhat[x], st.hops[x], x) < Bp and x not in result_U:
            result_U.add(x)
            _settle(st, x)
    return Bp, list(result_U)


def random_graph(rng: random.Random, n: int, m: int) -> ref.Graph:
    weights = [0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 3.0]
    edges = [(rng.randrange(n), rng.randrange(n), rng.choice(weights))
             for _ in range(m)]
    return ref.build_graph(n, edges)


def run_one(g: ref.Graph, source: int, k: int, t: int) -> None:
    g2, source2, rep = ref.transform_to_constant_degree(g, source)
    log_n = max(1.0, math.log2(max(2, g2.n)))
    L = max(1, math.ceil(log_n / t))
    st = State.new(g2, source2, k, t)
    oracle = dijkstra(g2, source2)
    try:
        bmssp_literal(st, L, INF, [source2])
    except (AssertionError, RecursionError) as exc:
        STATS["crashes"] += 1
        print("  crash:", type(exc).__name__, str(exc)[:100])
        return
    STATS["runs"] += 1
    if any(st.dhat[v] != oracle[v] for v in range(g2.n)):
        STATS["output_mismatch_runs"] += 1


def main() -> None:
    param_sets = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2)]
    rng = random.Random(12345)
    for trial in range(300):
        n = rng.randint(1, 120)
        m = rng.randint(0, 3 * n)
        g = random_graph(rng, n, m)
        k, t = param_sets[trial % len(param_sets)]
        run_one(g, 0, k, t)
    for trial in range(20):
        n = rng.randint(400, 900)
        g = random_graph(rng, n, 2 * n)
        run_one(g, 0, 2, 2)
    print(STATS)


if __name__ == "__main__":
    main()
