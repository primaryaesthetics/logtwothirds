# Historical audit script (not part of the test suite). One-off stress-check
# from the AUDIT.md investigation; run manually from the repo root with
# `python tests/audit/audit_stress.py`. Kept for reproducibility.
"""Audit stress-check: recursion invariants of bmssp on random graphs.

Wraps ref.bmssp with pre/postcondition asserts (Lemma 3.1/3.7/3.9 of the
paper) checked against a Dijkstra oracle on the transformed graph, and
instruments BlockDS.pull to measure how often a pulled batch contains an
already-settled vertex (the QUESTIONS.md item-3 stale-entry scenario).

Run:  python audit_stress.py
"""
from __future__ import annotations

import heapq
import math
import random
import sys

import logtwothirds._reference as ref

# ---------------------------------------------------------------------------
# Oracle: textbook Dijkstra (numeric; weights are small integers/0.5 so all
# additions are exact in binary floating point).
# ---------------------------------------------------------------------------

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
        for v, w in ref.out_edges(g, u):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist


# ---------------------------------------------------------------------------
# Instrumented bmssp wrapper
# ---------------------------------------------------------------------------

ORACLE: list[float] = []        # true distances on the transformed graph
STATS = {
    "calls": 0,
    "partial_small_U": 0,       # B' < B with |U| < k*2^(lt) (the ALGORITHM.md
                                # S4.3 "partial child emptied D" corner)
    "stale_pulls": 0,           # pulled batches containing a settled vertex
    "pulled_batches": 0,
    "runs": 0,
}
CUR_ST: ref.State | None = None

orig_bmssp = ref.bmssp
orig_pull = ref.BlockDS.pull


def checked_bmssp(st: ref.State, l: int, B, S):
    STATS["calls"] += 1
    # --- preconditions (Lemma 3.1 / requirement 1, 2's checkable parts) ---
    assert l >= 0
    assert len(S) <= 2 ** (l * st.t), "requirement 1: |S| <= 2^(l*t)"
    assert len(set(S)) == len(S), "S must not contain duplicates"
    for x in S:
        assert ref.key(st, x) < B, "B > max key over S"
        # every x in S must already carry a sound finite label
        assert st.dhat[x] >= ORACLE[x] or math.isclose(st.dhat[x], ORACLE[x])

    dhat_before = list(st.dhat)

    Bp, U = orig_bmssp(st, l, B, S)

    # --- postconditions ---
    assert Bp <= B, "B' <= B"
    cap = st.k * 2 ** (l * st.t)
    assert len(U) <= 4 * cap, "Lemma 3.9: |U| <= 4k*2^(lt)"
    if Bp < B and len(U) < cap:
        # Lemma 3.9's lower bound has a known corner (ALGORITHM.md S4.3 /
        # S9 uncertainty: a partial child emptied D). Count, don't fail.
        STATS["partial_small_U"] += 1
    assert len(U) == len(set(U)), "U must not contain duplicates"
    for u in U:
        # U is complete at return (Lemma 3.7)
        assert st.dhat[u] == ORACLE[u], (
            f"U member {u} incomplete: dhat={st.dhat[u]} d={ORACLE[u]}"
        )
        # U subset of {v : key(v) < B'}
        assert ref.key(st, u) < Bp, "U member with key >= B'"
    # dhat only improves and stays sound
    for v in range(st.g.n):
        assert st.dhat[v] <= dhat_before[v], "dhat increased"
        assert st.dhat[v] >= ORACLE[v], f"dhat unsound at {v}"
    return Bp, U


def spying_pull(self):
    Si, Bi = orig_pull(self)
    STATS["pulled_batches"] += 1
    if CUR_ST is not None and any(CUR_ST.settled[x] for x in Si):
        STATS["stale_pulls"] += 1
    return Si, Bi


ref.bmssp = checked_bmssp
ref.BlockDS.pull = spying_pull
# make the module-internal recursive call site use the wrapper too
ref_globals = orig_bmssp.__globals__
ref_globals["bmssp"] = checked_bmssp


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def random_graph(rng: random.Random, n: int, m: int) -> ref.Graph:
    weights = [0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 3.0]  # exact in fp, many ties
    edges = []
    for _ in range(m):
        edges.append(
            (rng.randrange(n), rng.randrange(n), rng.choice(weights))
        )
    return ref.build_graph(n, edges)


def run_one(g: ref.Graph, source: int, k: int, t: int) -> None:
    global ORACLE, CUR_ST
    g2, source2, rep = ref.transform_to_constant_degree(g, source)
    log_n = max(1.0, math.log2(max(2, g2.n)))
    L = max(1, math.ceil(log_n / t))
    st = ref.State.new(g2, source2, k, t)
    ORACLE = dijkstra(g2, source2)
    CUR_ST = st

    checked_bmssp(st, L, ref.INF, [source2])
    STATS["runs"] += 1

    # top level: every reachable transformed vertex must be complete
    for v in range(g2.n):
        assert st.dhat[v] == ORACLE[v], f"final dhat wrong at {v}"
    # settle log entries carry true distances, each vertex at most once
    seen = set()
    for (v, d) in st.settle_log.events:
        assert v not in seen
        seen.add(v)
        assert d == ORACLE[v], f"settle log value wrong at {v}"
    # answer mapped back through rep equals oracle on the original graph
    orig_oracle = dijkstra(g, source)
    for v in range(g.n):
        assert st.dhat[rep[v]] == orig_oracle[v], f"original dist wrong at {v}"


def main() -> None:
    param_sets = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2)]
    rng = random.Random(12345)

    # small graphs, all parameter regimes
    for trial in range(300):
        n = rng.randint(1, 120)
        m = rng.randint(0, 3 * n)
        g = random_graph(rng, n, m)
        k, t = param_sets[trial % len(param_sets)]
        run_one(g, 0, k, t)

    # medium graphs, (2,2): deep recursion + frequent partial executions
    for trial in range(20):
        n = rng.randint(400, 900)
        g = random_graph(rng, n, 2 * n)
        run_one(g, 0, 2, 2)

    # medium graphs, default parameters
    for trial in range(10):
        n = rng.randint(400, 900)
        g = random_graph(rng, n, 2 * n)
        g2, _, _ = ref.transform_to_constant_degree(g, 0)
        k, t, _L = ref.compute_params(g2.n)
        run_one(g, 0, k, t)

    print("OK")
    print(STATS)


if __name__ == "__main__":
    main()
