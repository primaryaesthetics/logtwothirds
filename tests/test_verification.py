"""Tests for the VERIFICATION instrumentation (SPEC.md S7).

Covers the operations counter / empirical-complexity smoke test (S7.a) and
the settlement-order log / non-sorting proof (S7.b).
"""

from __future__ import annotations

import heapq
import math
import random

import logtwothirds._reference as ref
from logtwothirds._reference import (
    SettleLog,
    State,
    bmssp,
    build_graph,
    compute_params,
    is_globally_sorted,
    out_edges,
    sssp_instrumented,
    transform_to_constant_degree,
    INF,
)


def random_constant_degree_graph(n: int, m: int, seed: int):
    """A Hamiltonian-cycle backbone (out-degree 1) plus random extra edges,
    so the average out-degree is ``m / n`` and the whole graph is reachable
    from vertex 0.
    """
    rng = random.Random(seed)
    edges = [
        (i, (i + 1) % n, round(rng.uniform(0.1, 1.0), 4)) for i in range(n)
    ]
    for _ in range(max(0, m - n)):
        u = rng.randrange(n)
        v = rng.randrange(n)
        edges.append((u, v, round(rng.uniform(0.1, 1.0), 4)))
    return build_graph(n, edges)


# ---------------------------------------------------------------------------
# 7.a Operations counter -- empirical complexity (smoke test)
# ---------------------------------------------------------------------------


def test_empirical_complexity():
    sizes = [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16]
    ratios = []
    for n in sizes:
        g = random_constant_degree_graph(n, 2 * n, seed=42)
        _dist, counter, _log = sssp_instrumented(g, 0)
        g2, _src2, _rep = transform_to_constant_degree(g, 0)
        n2 = g2.n
        m2 = len(g2.indices)
        r = counter.total() / (m2 * (math.log2(n2) ** (2.0 / 3.0)))
        ratios.append(r)

    print("n, r(n):", list(zip(sizes, ratios)))

    assert max(ratios) / min(ratios) < 4

    for prev, cur in zip(ratios, ratios[1:]):
        assert cur <= prev * 1.30, (prev, cur)


# ---------------------------------------------------------------------------
# 7.b Settlement-order log
# ---------------------------------------------------------------------------


def dijkstra_with_settle_log(g, source) -> tuple[list[float], SettleLog]:
    """Reference Dijkstra, instrumented with a settlement log in
    extraction order (always globally sorted -- validates the checker).
    """
    n = g.n
    dist = [math.inf] * n
    dist[source] = 0.0
    settled = [False] * n
    log = SettleLog()
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if settled[u]:
            continue
        settled[u] = True
        log.events.append((u, d))
        for v, w in out_edges(g, u):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist, log


def test_sorted_oracle_sanity():
    g = random_constant_degree_graph(4096, 2 * 4096, seed=7)
    _dist, log = dijkstra_with_settle_log(g, 0)
    assert is_globally_sorted(log) is True


def _small_params(n: int) -> tuple[int, int, int]:
    """(k, t) = (2, 2): the regime where the algorithm's distinctive
    settlement mechanism is active at testable sizes. See QUESTIONS.md item 4.

    TODO(spec): SPEC.md S7.b's mandatory non-sorting check doesn't pin down
    k/t, and under the *default* compute_params the settlement order is
    provably (Lemma 3.10 / Remark 3.8) sorted for all practically testable
    n: k = floor(log2(n2)**(1/3)) stays 2 until n2 >= 2**27, and at k = 2
    FindPivots' early exit (|W| > k|S|) fires on essentially every fresh
    frontier of the constant-out-degree-(<=2) transformed graph, forcing
    P = S. With P = S every vertex is settled through the BlockDS pipeline,
    whose batches are monotone by Lemma 3.10; the only out-of-order source
    -- the L22 W-sweep of vertices in *non-pivot* tight subtrees -- never
    has candidates. Forcing (k, t) = (2, 2) (per SPEC.md S8.4's "monkeypatch
    compute_params, e.g. k=2, t=2"; correctness is parameter-independent,
    ALGORITHM.md S2) activates the pivot branch and the W-sweep. This does
    not weaken the check: an implementation degenerated into Dijkstra
    settles in sorted order under *any* parameters and still fails here.
    """
    log_n = max(1.0, math.log2(max(2, n)))
    return 2, 2, max(1, math.ceil(log_n / 2))


def test_not_globally_sorted(monkeypatch):
    monkeypatch.setattr(ref, "compute_params", _small_params)
    g = random_constant_degree_graph(4096, 2 * 4096, seed=1)
    _dist, _counter, log = sssp_instrumented(g, 0)
    assert is_globally_sorted(log) is False


def test_not_dijkstra(monkeypatch):
    """Acceptance criterion: on 20 random graphs with n >= 500, the
    settlement log must NOT be sorted by distance in >= 15 cases.
    Parameter regime per _small_params (see its docstring / QUESTIONS.md
    item 4)."""
    monkeypatch.setattr(ref, "compute_params", _small_params)

    unsorted_count = 0
    for seed in range(20):
        rng = random.Random(seed)
        n = rng.randint(500, 1500)
        g = random_constant_degree_graph(n, 2 * n, seed=seed)
        _dist, _counter, log = sssp_instrumented(g, 0)
        if not is_globally_sorted(log):
            unsorted_count += 1

    assert unsorted_count >= 15, f"only {unsorted_count}/20 runs unsorted"


def test_settlement_complete():
    g = random_constant_degree_graph(256, 2 * 256, seed=11)
    source = 0

    g2, source2, rep = transform_to_constant_degree(g, source)
    k, t, L = compute_params(g2.n)
    st = State.new(g2, source2, k, t)
    bmssp(st, L, INF, [source2])

    oracle, _log = dijkstra_with_settle_log(g2, source2)

    # Every original vertex reachable from source has its designated
    # cycle-vertex (one of its cycle vertices) settled.
    for v in range(g.n):
        if math.isfinite(oracle[rep[v]]):
            assert st.settled[rep[v]], v

    # Every logged dhat equals the oracle distance of that (transformed)
    # vertex.
    for (v, d) in st.settle_log.events:
        assert math.isclose(d, oracle[v], rel_tol=0, abs_tol=1e-9), (v, d, oracle[v])
