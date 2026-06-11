"""Pure-Python reference implementation of the BMSSP algorithm.

Implements, as literally as practical, the algorithm distilled in
``ALGORITHM.md`` (itself a distillation of Duan, Mao, Mao, Shu, Yin,
"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths",
arXiv:2504.17033v2). ``SPEC.md`` fixes the engineering decisions used here
(single-module layout, data types, instrumentation).

Every public function/class carries a comment pointing at the relevant
section of ``ALGORITHM.md`` (e.g. "ALGORITHM.md S4.3" = section 4.3,
Algorithm 3). Where the spec documents flag an ambiguity, the implementation
follows the closest literal reading and marks the spot with
``# TODO(spec)``; see ``QUESTIONS.md`` for the corresponding question.
"""

from __future__ import annotations

import bisect
import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Graph representation and constant-degree transform (ALGORITHM.md S1.1,
#    SPEC.md S2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Graph:
    """CSR (compressed sparse row) directed graph, out-adjacency.

    Out-edges of vertex ``u`` are ``indices[indptr[u]:indptr[u+1]]`` with
    matching ``weights``. SPEC.md S2.
    """

    n: int
    indptr: list[int]
    indices: list[int]
    weights: list[float]


def build_graph(n: int, edges: list[tuple[int, int, float]]) -> Graph:
    """Build a CSR ``Graph`` from an edge list. SPEC.md S2.

    Validates ``0 <= u, v < n`` and ``w >= 0`` finite. Parallel edges and
    self-loops are allowed (harmless).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    out_edges: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (u, v, w) in edges:
        if not (0 <= u < n) or not (0 <= v < n):
            raise ValueError(f"edge endpoint out of range for n={n}: ({u}, {v})")
        if not (w >= 0) or not math.isfinite(w):
            raise ValueError(f"edge weight must be finite and >= 0, got {w}")
        out_edges[u].append((v, float(w)))

    indptr = [0] * (n + 1)
    indices: list[int] = []
    weights: list[float] = []
    for u in range(n):
        for (v, w) in out_edges[u]:
            indices.append(v)
            weights.append(w)
        indptr[u + 1] = len(indices)
    return Graph(n=n, indptr=indptr, indices=indices, weights=weights)


def transform_to_constant_degree(
    g: Graph, source: int
) -> tuple[Graph, int, list[int]]:
    """Constant-degree transform of ALGORITHM.md S1.1 / SPEC.md S2.

    For each original vertex ``v`` with ``d`` incident edge-endpoints
    (counting both in- and out-edges, with multiplicity for parallel edges
    and self-loops), creates ``d`` "cycle vertices", one per incident edge
    occurrence (``d == 0`` -> a single isolated vertex; ``d == 1`` -> a
    single vertex with no cycle edge; ``d >= 2`` -> a directed zero-weight
    cycle through all ``d`` cycle vertices). For each original edge
    ``(u, v, w)`` adds a weight-``w`` edge from ``u``'s "out to v" cycle
    vertex to ``v``'s "in from u" cycle vertex.

    Returns ``(g2, source2, rep)`` where ``rep[v]`` is a designated cycle
    vertex of ``v`` and ``source2 = rep[source]``.
    """
    n = g.n
    edges: list[tuple[int, int, float]] = []
    for u in range(n):
        for e in range(g.indptr[u], g.indptr[u + 1]):
            edges.append((u, g.indices[e], g.weights[e]))

    # slots[v]: list of (kind, edge_index) descriptors for the cycle
    # vertices belonging to original vertex v, in deterministic order.
    slots: list[list[tuple[str, int]]] = [[] for _ in range(n)]
    for ei, (u, v, _w) in enumerate(edges):
        slots[u].append(("out", ei))
        slots[v].append(("in", ei))

    slot_id: list[dict[tuple[str, int], int]] = [dict() for _ in range(n)]
    rep = [0] * n
    new_n = 0
    for v in range(n):
        d = len(slots[v])
        if d == 0:
            rep[v] = new_n
            new_n += 1
        else:
            for s in slots[v]:
                slot_id[v][s] = new_n
                new_n += 1
            rep[v] = slot_id[v][slots[v][0]]

    new_edges: list[tuple[int, int, float]] = []
    # Zero-weight directed cycles, one per original vertex with d >= 2.
    for v in range(n):
        d = len(slots[v])
        if d >= 2:
            for i in range(d):
                a = slot_id[v][slots[v][i]]
                b = slot_id[v][slots[v][(i + 1) % d]]
                new_edges.append((a, b, 0.0))
    # Cross edges realizing the original edges.
    for ei, (u, v, w) in enumerate(edges):
        a = slot_id[u][("out", ei)]
        b = slot_id[v][("in", ei)]
        new_edges.append((a, b, w))

    g2 = build_graph(new_n, new_edges)
    return g2, rep[source], rep


# ---------------------------------------------------------------------------
# 2. Path-key total order (ALGORITHM.md S1.3, SPEC.md S3)
# ---------------------------------------------------------------------------

# A label key is (length, hops, vertex_or_pred_id), compared lexicographically
# by Python's native tuple order (Assumption 2.1's O(1) realization).
Key = tuple[float, int, int]

INF_INT = 2 ** 62
INF: Key = (math.inf, INF_INT, INF_INT)


def key(st: "State", v: int) -> Key:
    """Vertex v's label key (dhat[v], hops[v], v). ALGORITHM.md S1.3."""
    return (st.dhat[v], st.hops[v], v)


def relax_leq(
    cand_len: float,
    cand_hops: int,
    u: int,
    cur_len: float,
    cur_hops: int,
    cur_pred: int,
) -> bool:
    """The "<=" relaxation test of Remark 3.4 (ALGORITHM.md S6.1).

    Equality holds exactly when the candidate is the *same path*
    (``u == cur_pred`` with matching length/hops) and must take the
    "relax" branch.
    """
    return (cand_len, cand_hops, u) <= (cur_len, cur_hops, cur_pred)


# ---------------------------------------------------------------------------
# 3. Verification instrumentation (SPEC.md S7 / ALGORITHM.md "VERIFICATION")
# ---------------------------------------------------------------------------


@dataclass
class OpCounter:
    """Operation counters for the empirical-complexity check. SPEC.md S7.a."""

    edge_scans: int = 0
    relaxations: int = 0
    ds_inserts: int = 0
    ds_prepend_items: int = 0
    ds_pulls: int = 0
    ds_pulled_items: int = 0
    heap_ops: int = 0
    findpivots_calls: int = 0
    bmssp_calls: int = 0
    basecase_calls: int = 0

    def total(self) -> int:
        return (
            self.edge_scans
            + self.relaxations
            + self.ds_inserts
            + self.ds_prepend_items
            + self.ds_pulls
            + self.ds_pulled_items
            + self.heap_ops
            + self.findpivots_calls
            + self.bmssp_calls
            + self.basecase_calls
        )


@dataclass
class SettleLog:
    """Append-only settlement-order log. SPEC.md S7.b."""

    events: list[tuple[int, float]] = field(default_factory=list)


def is_globally_sorted(log: SettleLog) -> bool:
    """True iff the dhat values in ``log.events`` are non-decreasing."""
    vals = [d for (_v, d) in log.events]
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# 4. Per-run state, parameters, and the shared relaxation helper
#    (ALGORITHM.md S2, SPEC.md S3 / S5 / S6)
# ---------------------------------------------------------------------------


def compute_params(n: int) -> tuple[int, int, int]:
    """k, t, L from the vertex count of the *transformed* graph. SPEC.md S5.

    Correctness does not depend on these values (ALGORITHM.md S2); only the
    running-time bound does.
    """
    log_n = max(1.0, math.log2(max(2, n)))
    k = max(1, math.floor(log_n ** (1.0 / 3.0)))
    t = max(1, math.floor(log_n ** (2.0 / 3.0)))
    L = max(1, math.ceil(log_n / t))
    return k, t, L


@dataclass
class State:
    """Per-run mutable state shared by all procedures. SPEC.md S3."""

    g: Graph
    dhat: list[float]
    hops: list[int]
    pred: list[int]
    k: int
    t: int
    counter: OpCounter
    settle_log: SettleLog
    settled: list[bool]

    @staticmethod
    def new(g: Graph, source: int, k: int, t: int) -> "State":
        n = g.n
        dhat = [math.inf] * n
        hops = [INF_INT] * n
        pred = [-1] * n
        dhat[source] = 0.0
        hops[source] = 0
        return State(
            g=g,
            dhat=dhat,
            hops=hops,
            pred=pred,
            k=k,
            t=t,
            counter=OpCounter(),
            settle_log=SettleLog(),
            settled=[False] * n,
        )


@dataclass
class RelaxOutcome:
    """Result of :func:`try_relax`."""

    passed: bool
    cand: Key


def try_relax(st: State, u: int, v: int, w: float) -> RelaxOutcome:
    """Shared relaxation helper for edge ``(u, v)`` with weight ``w``.

    Evaluates the "<=" test of Remark 3.4 (ALGORITHM.md S6.1; this is
    Algorithm 1 line 7, Algorithm 2 line 8, and Algorithm 3 line 15) and, if
    it passes, writes ``dhat[v]``/``hops[v]``/``pred[v]``. Callers implement
    the differing bound-gating logic of each call site themselves
    (ALGORITHM.md S4.1/S4.2/S4.3 notes); this helper never gates on a bound.
    """
    st.counter.edge_scans += 1
    cand_len = st.dhat[u] + w
    cand_hops = st.hops[u] + 1
    cand: Key = (cand_len, cand_hops, u)
    if relax_leq(cand_len, cand_hops, u, st.dhat[v], st.hops[v], st.pred[v]):
        st.dhat[v] = cand_len
        st.hops[v] = cand_hops
        st.pred[v] = u
        st.counter.relaxations += 1
        return RelaxOutcome(passed=True, cand=cand)
    return RelaxOutcome(passed=False, cand=cand)


def out_edges(g: Graph, u: int):
    """Iterate ``(v, w)`` over the out-edges of ``u``."""
    for e in range(g.indptr[u], g.indptr[u + 1]):
        yield g.indices[e], g.weights[e]


# ---------------------------------------------------------------------------
# 5. The block data structure D (ALGORITHM.md S3, Lemma 3.3 / SPEC.md S4)
# ---------------------------------------------------------------------------


def _select_smallest(
    pairs: list[tuple[int, Key]], m: int
) -> tuple[list[tuple[int, Key]], list[tuple[int, Key]]]:
    """Partition ``pairs`` into the ``m`` smallest-by-value and the rest.

    Random-pivot quickselect (expected O(len(pairs))), per ALGORITHM.md S3.2
    ("Median selection ... random-pivot quickselect"). ``pairs`` is not
    mutated; the returned lists are not sorted relative to each other beyond
    "all of the first list's values < all of the second list's values" not
    being guaranteed for ties, which cannot occur (Assumption 2.1: distinct
    values).
    """
    n = len(pairs)
    if m <= 0:
        return [], list(pairs)
    if m >= n:
        return list(pairs), []

    items = list(pairs)
    lo, hi = 0, n - 1
    target = m - 1  # index (0-based) of the m-th smallest after partitioning
    while lo < hi:
        pivot_idx = random.randint(lo, hi)
        pivot_val = items[pivot_idx][1]
        items[pivot_idx], items[hi] = items[hi], items[pivot_idx]
        store = lo
        for i in range(lo, hi):
            if items[i][1] < pivot_val:
                items[store], items[i] = items[i], items[store]
                store += 1
        items[store], items[hi] = items[hi], items[store]
        if store == target:
            break
        elif store < target:
            lo = store + 1
        else:
            hi = store - 1
    return items[:m], items[m:]


def _chunk_by_median(
    pairs: list[tuple[int, Key]], cap: int
) -> list[dict[int, Key]]:
    """Split ``pairs`` into blocks of size <= ``cap`` by repeated median
    finding, ordered so that every value in chunk ``i`` < every value in
    chunk ``i + 1``. ALGORITHM.md S3.2 ("BatchPrepend ... split L by repeated
    median-finding into O(|L|/M) blocks").
    """
    if not pairs:
        return []
    if len(pairs) <= cap:
        return [dict(pairs)]
    lower, upper = _select_smallest(pairs, len(pairs) // 2)
    return _chunk_by_median(lower, cap) + _chunk_by_median(upper, cap)


class BlockDS:
    """The partial-sorting batched priority structure of Lemma 3.3.

    ALGORITHM.md S3 / SPEC.md S4. Two sequences of blocks:

    - ``D1`` (``self._d1_blocks`` / ``self._d1_bounds``): receives only
      :meth:`insert`. Each block is a ``dict[key, value]`` of size <= ``M``,
      paired with an upper bound; bounds are non-decreasing and the last
      bound is always ``B``. ``insert`` routes to the block with the
      smallest bound >= value (binary search over ``_d1_bounds``); a block
      exceeding ``M`` pairs is split at its median.
    - ``D0`` (``self._d0_blocks``): receives only :meth:`batch_prepend`,
      which splits its argument into <= ceil(M/2)-sized blocks and prepends
      them, preserving the inter-block ordering invariant.

    A registry ``self.where: dict[key -> (value, block_dict)]`` gives O(1)
    duplicate-handling and deletion (``del block_dict[key]``).
    """

    def __init__(self, M: int, B: Key) -> None:
        self.M = max(1, M)
        self.B = B
        self._d0_blocks: list[dict[int, Key]] = []
        self._d1_blocks: list[dict[int, Key]] = [dict()]
        self._d1_bounds: list[Key] = [B]
        self.where: dict[int, tuple[Key, dict[int, Key]]] = {}

    def __len__(self) -> int:
        return len(self.where)

    # -- internal helpers ---------------------------------------------------

    def _min_value(self) -> Key:
        """Minimum value currently in D, or ``B`` if D is empty.

        By the inter-block ordering invariant the minimum (if any) lies in
        the first non-empty block of D0 or of D1. ALGORITHM.md S3.2.
        """
        candidates: list[Key] = []
        for block in self._d0_blocks:
            if block:
                candidates.append(min(block.values()))
                break
        for block in self._d1_blocks:
            if block:
                candidates.append(min(block.values()))
                break
        if not candidates:
            return self.B
        return min(candidates)

    def _split_d1(self, idx: int) -> None:
        """Split an over-full D1 block at ``idx`` at its median.

        ALGORITHM.md S3.2 "Split". The lower half keeps the original block's
        position (with a new, smaller bound = its max value); the upper half
        becomes a new block at ``idx + 1`` retaining the original bound.
        """
        block = self._d1_blocks[idx]
        bnd = self._d1_bounds[idx]
        items = list(block.items())
        lower_items, upper_items = _select_smallest(items, len(items) // 2)
        new_bound = max(v for _k, v in lower_items)

        lower_block = dict(lower_items)
        upper_block = dict(upper_items)
        self._d1_blocks[idx] = lower_block
        self._d1_bounds[idx] = new_bound
        self._d1_blocks.insert(idx + 1, upper_block)
        self._d1_bounds.insert(idx + 1, bnd)

        for k, v in lower_items:
            self.where[k] = (v, lower_block)
        for k, v in upper_items:
            self.where[k] = (v, upper_block)

    # -- public operations (Lemma 3.3) --------------------------------------

    def insert(self, key: int, value: Key) -> None:
        """Insert ``(key, value)`` into D1. ALGORITHM.md S3.1 "Insert".

        If ``key`` is already present, keep only the smaller-value pair.
        """
        if __debug__:
            assert value < self.B, "Insert: value must be < B"
        if key in self.where:
            old_value, old_block = self.where[key]
            if not (value < old_value):
                return  # keep existing smaller value
            del old_block[key]
            del self.where[key]

        idx = bisect.bisect_left(self._d1_bounds, value)
        block = self._d1_blocks[idx]
        block[key] = value
        self.where[key] = (value, block)
        if len(block) > self.M:
            self._split_d1(idx)

    def batch_prepend(self, items: list[tuple[int, Key]]) -> None:
        """Prepend ``items`` as new D0 block(s). ALGORITHM.md S3.1
        "BatchPrepend".

        PRECONDITION: every value in ``items`` is smaller than every value
        currently in D (checked under ``__debug__``).
        """
        if not items:
            return

        dedup: dict[int, Key] = {}
        for k, v in items:
            if k not in dedup or v < dedup[k]:
                dedup[k] = v

        if __debug__:
            cur_min = self._min_value()
            for v in dedup.values():
                assert v < cur_min, (
                    "BatchPrepend precondition violated: a value is not "
                    "smaller than D's current minimum"
                )

        for k, v in dedup.items():
            if k in self.where:
                old_value, old_block = self.where[k]
                assert v < old_value, (
                    "BatchPrepend: duplicate key with non-smaller value "
                    "violates the precondition"
                )
                del old_block[k]
                del self.where[k]

        pairs = list(dedup.items())
        cap = max(1, (self.M + 1) // 2)
        if len(pairs) <= self.M:
            chunks = [dict(pairs)]
        else:
            chunks = _chunk_by_median(pairs, cap)

        self._d0_blocks = chunks + self._d0_blocks
        for block in chunks:
            for k, v in block.items():
                self.where[k] = (v, block)

    def pull(self) -> tuple[list[int], Key]:
        """Remove and return the keys of the M smallest values in D, plus a
        separating bound. ALGORITHM.md S3.1 "Pull".
        """
        M = self.M

        s0_items: list[tuple[int, Key]] = []
        s0_seen = 0
        for block in self._d0_blocks:
            s0_seen += 1
            if not block:
                continue
            s0_items.extend(block.items())
            if len(s0_items) >= M:
                break
        d0_exhausted = s0_seen == len(self._d0_blocks)

        s1_items: list[tuple[int, Key]] = []
        s1_seen = 0
        for block in self._d1_blocks:
            s1_seen += 1
            if not block:
                continue
            s1_items.extend(block.items())
            if len(s1_items) >= M:
                break
        d1_exhausted = s1_seen == len(self._d1_blocks)

        union = s0_items + s1_items

        if len(union) <= M and d0_exhausted and d1_exhausted:
            for k, _v in union:
                del self.where[k]
            self._d0_blocks = []
            self._d1_blocks = [dict()]
            self._d1_bounds = [self.B]
            return [k for k, _v in union], self.B

        smallest, _rest = _select_smallest(union, M)
        for k, _v in smallest:
            _value, block = self.where[k]
            del block[k]
            del self.where[k]

        while self._d0_blocks and not self._d0_blocks[0]:
            self._d0_blocks.pop(0)

        x = self._min_value()
        return [k for k, _v in smallest], x

    # -- debug / test helpers -------------------------------------------------

    def _check_invariants(self) -> None:
        """White-box invariant checker used only by tests/debug code."""
        assert len(self._d1_blocks) == len(self._d1_bounds) >= 1
        assert self._d1_bounds[-1] == self.B
        for i in range(len(self._d1_bounds) - 1):
            assert self._d1_bounds[i] <= self._d1_bounds[i + 1]
        for block in self._d1_blocks:
            assert len(block) <= self.M

        for blocks, bounds in (
            (self._d1_blocks, self._d1_bounds),
            (self._d0_blocks, None),
        ):
            prev_max: Optional[Key] = None
            for i, block in enumerate(blocks):
                if not block:
                    continue
                bmin, bmax = min(block.values()), max(block.values())
                if prev_max is not None:
                    assert prev_max <= bmin
                if bounds is not None:
                    assert bmax <= bounds[i]
                prev_max = bmax


# ---------------------------------------------------------------------------
# 6. FindPivots (ALGORITHM.md S4.1, Algorithm 1 / SPEC.md S6)
# ---------------------------------------------------------------------------


def find_pivots(st: State, B: Key, S: list[int]) -> tuple[list[int], list[int]]:
    """FindPivots(B, S). ALGORITHM.md S4.1 (Algorithm 1), Lemma 3.2.

    Precondition: every incomplete vertex ``v`` with ``d(v) < B`` has its
    shortest path through some complete vertex in ``S``.
    Returns ``(P, W)`` with ``P subseteq S``, ``W subseteq T_<B(S)``.
    """
    st.counter.findpivots_calls += 1
    g = st.g
    k = st.k

    W_set: set[int] = set(S)
    W_order: list[int] = list(dict.fromkeys(S))  # L2-3: W <- S; W_0 <- S
    frontier: list[int] = list(W_order)

    for _i in range(k):  # L4: for i <- 1 to k
        next_frontier_set: set[int] = set()
        next_frontier: list[int] = []
        for u in frontier:  # L6: for all edges (u, v) with u in W_{i-1}
            for v, w in out_edges(g, u):
                outcome = try_relax(st, u, v, w)  # L7-8: relax (updates dhat
                # even if the candidate is >= B, per the S4.1 note)
                # NOTE(spec): bound check against B must use v's own key
                # (dhat[v], hops[v], v), not outcome.cand whose third
                # component is the predecessor u -- see the analogous fix
                # in bmssp's L17-20 (ALGORITHM.md S3.2 / S1.3).
                if outcome.passed and key(st, v) < B:  # L9
                    if v not in next_frontier_set:
                        next_frontier_set.add(v)
                        next_frontier.append(v)
        for v in next_frontier:  # L11: W <- W u W_i
            if v not in W_set:
                W_set.add(v)
                W_order.append(v)
        if len(W_set) > k * len(S):  # L12: early exit
            return list(S), W_order  # L13-14: P <- S

        frontier = next_frontier

    # L15: F = tight edges (u, v) with u, v in W and dhat[v] == dhat[u] + w_uv.
    # Under Assumption 2.1, v's unique tight in-edge (if any) is (pred[v], v);
    # F is therefore a forest, recovered as a child map below.
    children: dict[int, list[int]] = {}
    has_tight_parent_in_W: set[int] = set()
    for v in W_order:
        u = st.pred[v]
        if u in W_set:
            for vv, w in out_edges(g, u):
                if (
                    vv == v
                    and st.dhat[v] == st.dhat[u] + w
                    and st.hops[v] == st.hops[u] + 1
                ):
                    children.setdefault(u, []).append(v)
                    has_tight_parent_in_W.add(v)
                    break

    # L16: P = roots of S-rooted trees in F with >= k vertices. Tree sizes
    # via iterative DFS (no Python recursion, per SPEC.md S6).
    P: list[int] = []
    for u in S:
        if u in has_tight_parent_in_W:
            continue
        size = 0
        stack = [u]
        while stack:
            x = stack.pop()
            size += 1
            stack.extend(children.get(x, ()))
        if size >= k:
            P.append(u)

    return P, W_order  # L17


# ---------------------------------------------------------------------------
# 7. BaseCase (ALGORITHM.md S4.2, Algorithm 2 / SPEC.md S6)
# ---------------------------------------------------------------------------


def _settle(st: State, v: int) -> None:
    """Append a settlement event for ``v`` if not already settled.
    SPEC.md S7.b.
    """
    if not st.settled[v]:
        st.settled[v] = True
        st.settle_log.events.append((v, st.dhat[v]))


def base_case(st: State, B: Key, S: list[int]) -> tuple[Key, list[int]]:
    """BaseCase(B, S). ALGORITHM.md S4.2 (Algorithm 2).

    Pre: ``S = {x}``, ``x`` complete, ``B > dhat[x]``, every incomplete
    ``v`` with ``d(v) < B`` is in ``T(x)``.
    Post: returns ``B' <= B`` and ``U = T_<B'({x})``, ``|U| <= k``.
    """
    st.counter.basecase_calls += 1
    assert len(S) == 1, "BaseCase requires |S| == 1"
    x = S[0]
    g = st.g
    k = st.k

    U0: list[int] = [x]  # L2
    in_U0: set[int] = {x}

    # Binary heap H keyed by the S1.3 total order, with lazy deletion: a
    # vertex may be re-pushed on "DecreaseKey"; ``best`` tracks each
    # vertex's current key so stale pops are skipped.
    heap: list[tuple[Key, int]] = []
    best: dict[int, Key] = {}

    def push(v: int) -> None:
        kv = key(st, v)
        best[v] = kv
        heapq.heappush(heap, (kv, v))
        st.counter.heap_ops += 1

    push(x)  # L3: H <- {<x, dhat[x]>}

    while heap and len(U0) < k + 1:  # L4
        kx, u = heapq.heappop(heap)
        st.counter.heap_ops += 1
        if best.get(u) != kx:
            continue  # stale entry (superseded by a later DecreaseKey)
        if u not in in_U0:  # L6: U_0 <- U_0 u {u}
            in_U0.add(u)
            U0.append(u)
        for v, w in out_edges(g, u):  # L7: for edge e = (u, v)
            st.counter.edge_scans += 1
            cand_len = st.dhat[u] + w
            cand_hops = st.hops[u] + 1
            cand: Key = (cand_len, cand_hops, u)
            cur: Key = (st.dhat[v], st.hops[v], st.pred[v])
            vkey: Key = (cand_len, cand_hops, v)
            # L8: unlike Algorithm 1/3, the relaxation itself is gated by
            # "< B" -- candidates >= B are not written at all.
            # NOTE(spec): the "< B" half of this gate compares against B,
            # a vertex-key bound (dhat,hops,vertex); it must use vkey
            # (third component v), not cand (third component u, the
            # predecessor) -- see the analogous fix in bmssp's L17-20 and
            # find_pivots's L9 (ALGORITHM.md S3.2 / S1.3).
            if cand <= cur and vkey < B:
                st.dhat[v] = cand_len
                st.hops[v] = cand_hops
                st.pred[v] = u
                st.counter.relaxations += 1
                push(v)  # L10-13: H.Insert / H.DecreaseKey (lazy variant)

    if len(U0) <= k:  # L14
        Bp, U = B, U0  # L15
    else:  # L16
        Bp = max(key(st, v) for v in U0)
        U = [v for v in U0 if key(st, v) < Bp]  # L17

    for v in U:  # SPEC.md S7.b item 1: settlement order = heap-extraction order
        _settle(st, v)

    return Bp, U


# ---------------------------------------------------------------------------
# 8. BMSSP and the top-level SSSP procedure
#    (ALGORITHM.md S4.3 / S4.4, Algorithm 3 / SPEC.md S6)
# ---------------------------------------------------------------------------


def bmssp(st: State, l: int, B: Key, S: list[int]) -> tuple[Key, list[int]]:
    """BMSSP(l, B, S). ALGORITHM.md S4.3 (Algorithm 3).

    Pre: ``|S| <= 2^(l*t)``; ``B > max_{x in S} dhat[x]``; every incomplete
    ``v`` with ``d(v) < B`` is in ``T(S*)``.
    Post: returns ``B' <= B`` and ``U = T_<B'(S)``, complete at return.
    """
    st.counter.bmssp_calls += 1
    if l == 0:  # L2-3
        return base_case(st, B, S)

    g = st.g
    k, t = st.k, st.t

    P, W = find_pivots(st, B, S)  # L4

    # M = 2^((l-1)*t), capped at n (SPEC.md S5: a Pull can never return more
    # than n keys, so any M >= n behaves identically).
    M = min(2 ** ((l - 1) * t), g.n)
    M = max(1, M)
    D = BlockDS(M, B)  # L5

    for x in P:  # L6
        D.insert(x, key(st, x))
        st.counter.ds_inserts += 1

    if P:  # L7
        Bp_0 = min(key(st, x) for x in P)
    else:
        Bp_0 = B  # L7 footnote: if P = empty, B'_0 <- B

    U: set[int] = set()
    Bp_last = Bp_0
    bound_cap = k * (2 ** (l * t))

    while len(U) < bound_cap and len(D) > 0:  # L8
        # NOTE: Algorithm 3 L10 writes "B_i, S_i <- D.Pull()", but
        # ALGORITHM.md S9 / SPEC.md S4 fix Pull()'s return order as
        # (S', x) regardless of that pseudocode naming -- so the set
        # comes first here.
        Si, Bi = D.pull()  # L10
        st.counter.ds_pulls += 1
        st.counter.ds_pulled_items += len(Si)
        assert Si, "Pull returned an empty set while D was non-empty"

        # TODO(spec): D's "value" for a key v is supposed to always equal
        # the current key(v) (ALGORITHM.md S3.2's "derived invariant"), kept
        # in sync by Insert's keep-min rule. But a vertex v can also be
        # relaxed as a *side effect* of an unrelated recursive call deep in
        # the tree (e.g. during a sibling's BaseCase mini-Dijkstra, S6.6),
        # which updates st.dhat[v]/settles v globally without touching this
        # ancestor's D entry for v. If that happens, D may later Pull a
        # vertex that has already been settled (and is already part of an
        # earlier U_i), which would violate the U_i-disjointness and
        # T_<B'(S) postconditions (Lemma 3.7) if passed to a fresh recursive
        # call unchanged. Not addressed anywhere in ALGORITHM.md/SPEC.md, so
        # we filter out already-settled vertices here as the closest literal
        # fix consistent with "U_i are pairwise disjoint and complete"
        # (ALGORITHM.md S4.3 invariants).
        Si_fresh = [x for x in Si if not st.settled[x]]
        if not Si_fresh:
            Bp_i, Ui = Bi, []
        else:
            Bp_i, Ui = bmssp(st, l - 1, Bi, Si_fresh)  # L11
        if __debug__:
            assert U.isdisjoint(Ui), "the U_i must be pairwise disjoint"
        U |= set(Ui)  # L12
        Bp_last = Bp_i

        K: list[tuple[int, Key]] = []
        for u in Ui:  # L14: for edge e = (u, v) where u in U_i
            for v, w in out_edges(g, u):
                outcome = try_relax(st, u, v, w)  # L15-16
                if outcome.passed:
                    # NOTE(spec): the bucket decision and the value stored in
                    # D/K must use key(v) = (dhat[v], hops[v], v) -- the
                    # vertex's OWN key after relaxation -- not
                    # outcome.cand = (dhat[u]+w, hops[u]+1, u), whose third
                    # component is the predecessor u, not v. These two are
                    # numerically the same (length, hops) post-relaxation but
                    # differ in the tie-breaking third component, which can
                    # place a "tight" vertex on the wrong side of B'_i and
                    # silently drop it from both buckets (ALGORITHM.md S3.2's
                    # "the value stored in D for key v always equals the
                    # current dhat[v]", realized in the totally ordered key
                    # space per S1.3).
                    vkey = key(st, v)
                    if Bi <= vkey < B:  # L17-18
                        D.insert(v, vkey)
                        st.counter.ds_inserts += 1
                    elif Bp_i <= vkey < Bi:  # L19-20
                        K.append((v, vkey))

        # L21: K plus the unfinished part of the pulled batch.
        prepend = K + [
            (x, key(st, x)) for x in Si_fresh if Bp_i <= key(st, x) < Bi
        ]
        if prepend:
            D.batch_prepend(prepend)
            st.counter.ds_prepend_items += len(prepend)

    Bp = min(Bp_last, B)  # L22
    assert Bp <= B

    result_U = set(U)
    for x in W:  # L22: U <- U u {x in W : dhat[x] < B'}
        if (st.dhat[x], st.hops[x], x) < Bp and x not in result_U:
            result_U.add(x)
            _settle(st, x)  # SPEC.md S7.b item 2

    return Bp, list(result_U)


# ---------------------------------------------------------------------------
# Top-level SSSP procedure (ALGORITHM.md S4.4)
# ---------------------------------------------------------------------------


def _run_sssp(g: Graph, source: int) -> tuple[list[float], State]:
    if not (0 <= source < g.n):
        raise IndexError(
            f"source {source} out of range for graph with {g.n} vertices"
        )
    for w in g.weights:
        if w < 0 or not math.isfinite(w):
            raise ValueError(f"edge weight must be finite and >= 0, got {w}")

    g2, source2, rep = transform_to_constant_degree(g, source)  # step 1
    n2 = g2.n
    k, t, L = compute_params(n2)  # step 2

    st = State.new(g2, source2, k, t)  # step 3
    bmssp(st, L, INF, [source2])  # step 4: BMSSP(L, inf, {s'})

    dist = [st.dhat[rep[v]] for v in range(g.n)]  # step 5
    return dist, st  # step 6


def sssp(g: Graph, source: int) -> list[float]:
    """``dist[v]`` = shortest-path length from ``source``, ``inf`` if
    unreachable. ALGORITHM.md S4.4.
    """
    dist, _st = _run_sssp(g, source)
    return dist


def sssp_instrumented(
    g: Graph, source: int
) -> tuple[list[float], OpCounter, SettleLog]:
    """Like :func:`sssp`, but also returns the operation counters and the
    settlement-order log. SPEC.md S7.
    """
    dist, st = _run_sssp(g, source)
    return dist, st.counter, st.settle_log
