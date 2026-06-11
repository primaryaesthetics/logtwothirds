# SPEC.md — Python implementation specification

Target: a pure-Python implementation of the Duan–Mao–Mao–Shu–Yin O(m·log^(2/3) n) directed
SSSP algorithm, exactly as specified in [ALGORITHM.md](ALGORITHM.md). Read ALGORITHM.md
first; this file only fixes engineering decisions, signatures, and the test plan. **Where this
file and ALGORITHM.md seem to disagree, ALGORITHM.md (i.e. the paper) wins.**

Ground rules for the implementer:

- Python ≥ 3.10, standard library only at runtime (`heapq`, `math`, `random`, `dataclasses`).
  Tests may use `pytest` and `hypothesis`.
- No recursion-elimination cleverness needed: recursion depth is `⌈log2(n)/t⌉ + 1` (single
  digits for any feasible n); the default recursion limit is fine.
- Do not "optimize" any control flow away. Several conditions that look redundant (the `≤` in
  relaxations, re-inserting pulled keys, updating `d̂` for candidates ≥ B) are load-bearing —
  see ALGORITHM.md §6.1, §7.

---

## 1. Module structure

```
dmssp/
  __init__.py          # re-export sssp
  graph.py             # CSR graph type + constant-degree transformation
  order.py             # path-key total order (Assumption 2.1)
  block_ds.py          # the Lemma 3.3 structure (class BlockDS)
  params.py            # k, t, top level l from n
  state.py             # per-run mutable state: dhat/hops/pred arrays, instrumentation
  bmssp.py             # find_pivots, base_case, bmssp, sssp
  instrumentation.py   # OpCounter, SettleLog, is_globally_sorted
tests/
  test_block_ds.py
  test_find_pivots.py
  test_base_case.py
  test_bmssp.py
  test_sssp.py
  test_transform.py
  test_properties.py   # hypothesis-based
  test_verification.py # VERIFICATION section checks
  reference.py         # textbook Dijkstra + Bellman-Ford oracles (tests only)
```

---

## 2. Graph representation (`graph.py`)

CSR (compressed sparse row), out-adjacency:

```python
@dataclass(frozen=True)
class Graph:
    n: int                  # number of vertices, ids 0..n-1
    indptr: list[int]       # len n+1; out-edges of u are positions indptr[u]:indptr[u+1]
    indices: list[int]      # len m; head (target) vertex of each edge
    weights: list[float]    # len m; non-negative

def build_graph(n: int, edges: list[tuple[int, int, float]]) -> Graph
    # validates: 0 <= u,v < n, w >= 0 and finite; sorts/buckets edges by tail into CSR.
    # parallel edges and self-loops are allowed (harmless).

def transform_to_constant_degree(g: Graph, source: int) -> tuple[Graph, int, list[int]]
    # Returns (g2, source2, rep) implementing ALGORITHM.md §1.1:
    #   - for each vertex v with neighbor multiset {w1..wd} (in- and out-neighbors,
    #     d = deg(v)), create d cycle vertices x[v][w1..wd]; if d == 0 create 1 plain vertex;
    #     if d >= 2 add zero-weight cycle edges x_i -> x_{i+1 mod d};
    #     (d == 1: a single vertex, no cycle edge)
    #   - for each original edge (u, v, w): add edge x[u][v] -> x[v][u] with weight w
    #     (for parallel edges, one cycle slot per edge occurrence is acceptable)
    #   - rep[v] = id of one designated cycle vertex of v (used to read the answer);
    #     source2 = rep[source]
    # POST: every vertex of g2 has in-degree <= 2 and out-degree <= 2;
    #       d_{g2}(rep[v]) == d_g(v) for all v.
```

Output of the whole algorithm:

```python
def sssp(g: Graph, source: int) -> list[float]
    # dist[v] = shortest-path length, math.inf if unreachable. Defined in bmssp.py.
```

---

## 3. Path-key total order (`order.py`)

Realization of Assumption 2.1 (ALGORITHM.md §1.3). A **label key** is the tuple

```python
Key = tuple[float, int, int]   # (length, hops, vertex_or_pred_id)
```

compared by Python's native lexicographic tuple order.

- Cross-vertex comparisons (heap of Algorithm 2, all values stored in `BlockDS`, all bounds):
  vertex `v`'s key is `(dhat[v], hops[v], v)`.
- Relaxation test for edge `(u, v)` (the "≤" of Remark 3.4):

```python
def relax_leq(cand_len: float, cand_hops: int, u: int,
              cur_len: float, cur_hops: int, cur_pred: int) -> bool:
    return (cand_len, cand_hops, u) <= (cur_len, cur_hops, cur_pred)
```

  Equality (`==`) happens exactly when the candidate is the same path (u == cur_pred, same
  length and hops) — and must return True.
- Bounds (`B`, `B_i`, `B'_i`, …) are values of type `Key`. Infinity bound:
  `INF: Key = (math.inf, INF_INT, INF_INT)` with `INF_INT = 2**62`. An unsettled vertex's
  label is `INF` as well; note `INF < INF` is False, so unreachable vertices never pass
  `key(v) < B` for any bound — which is the desired behavior.
- Provide `def key(state, v) -> Key` returning `(dhat[v], hops[v], v)`.

State arrays (in `state.py`):

```python
@dataclass
class State:
    g: Graph                  # the constant-degree graph
    dhat: list[float]         # init inf; dhat[source] = 0.0
    hops: list[int]           # init INF_INT; hops[source] = 0
    pred: list[int]           # init -1
    k: int; t: int
    counter: OpCounter        # see §7
    settle_log: SettleLog     # see §7
    settled: list[bool]       # dedupe flags for the settle log
```

---

## 4. `BlockDS` (`block_ds.py`) — Lemma 3.3

```python
class BlockDS:
    def __init__(self, M: int, B: Key): ...
    def insert(self, key: int, value: Key) -> None: ...
    def batch_prepend(self, items: list[tuple[int, Key]]) -> None: ...
    def pull(self) -> tuple[list[int], Key]: ...        # (S', x) — set first, bound second
    def __len__(self) -> int: ...                        # number of live keys
    # is-empty test used by BMSSP line 8: len(D) > 0
```

Semantics, invariants, and costs: ALGORITHM.md §3 verbatim. Implementation decisions:

- Blocks: Python lists used as unsorted bags, with lazy deletion **not** allowed — deletion
  must be O(1)-ish real removal so that Pull's prefix scan stays O(M). Recommended concrete
  layout: each block is a `dict[int, Key]` (key → value); the registry maps key →
  (sequence_id, block_ref). Removing a key from a dict block is O(1) and the block's live
  size is `len(block)`.
- `D1` block upper bounds: keep a sorted list of `(upper_bound: Key, block_ref)` and use
  `bisect` for "smallest upper bound ≥ value". (A sorted list with bisect-insert is O(#blocks)
  per split in the worst case; that is acceptable for this implementation — note it as a
  deliberate deviation from the red-black tree of the paper, affecting constants/worst-case
  only, not correctness. A real BST is optional.)
- Median selection for `split` and for Pull's candidate selection: random-pivot quickselect
  (expected O(M)); `sorted()` fallback is acceptable but must be flagged with a comment as a
  complexity deviation (O(M log M)).
- Duplicate keys: a single registry `self.where: dict[int, ...]`; `insert` and
  `batch_prepend` consult it and keep the smaller value (delete old pair first). Within a
  `batch_prepend` batch, dedupe first (keep min), then check the registry.
- `pull()` on an empty structure returns `([], self.B)` (defensive; BMSSP never calls it).
- `batch_prepend` MUST assert (in debug mode) its precondition: every incoming value `<`
  current global minimum value. Keep a cheap `self._min_value` or scan first blocks under
  `__debug__` only.
- After `pull()` with remaining elements: returned bound `x` = minimum remaining value
  (scan first non-empty block of D0 and of D1). The contract `max(values(S')) < x` must hold.

---

## 5. Parameters (`params.py`)

```python
def compute_params(n: int) -> tuple[int, int, int]:
    # n: vertex count of the TRANSFORMED graph, n >= 1
    log_n = max(1.0, math.log2(max(2, n)))
    k = max(1, math.floor(log_n ** (1/3)))
    t = max(1, math.floor(log_n ** (2/3)))
    L = max(1, math.ceil(log_n / t))
    return k, t, L
```

`M` at level `l` is `2 ** ((l - 1) * t)` computed inside `bmssp` (Python ints, exact).
Cap `M` at `n` (a pull can never return more than n keys; capping avoids huge ints; allowed
because any M ≥ n behaves identically).

---

## 6. Procedures (`bmssp.py`)

Signatures (all operate on a shared `State`; sets of vertices are `list[int]` or `set[int]` —
fix `list[int]` for returned U/W and document non-duplication):

```python
def find_pivots(st: State, B: Key, S: list[int]) -> tuple[list[int], list[int]]:
    # returns (P, W); Algorithm 1 lines 1-17, including the in-loop early exit
    # (|W| > k*|S| -> P = S). F/tree-size computation: build pred-restricted adjacency
    # over W via the tight-edge test  key-equality  (dhat[v], hops[v]) == (dhat[u]+w, hops[u]+1)
    # and pred[v] == u  — under Assumption 2.1 simply: pred[v] == u and
    # dhat[v] == dhat[u] + w and hops[v] == hops[u] + 1.  Tree sizes by iterative DFS from
    # roots in S (no Python recursion).

def base_case(st: State, B: Key, S: list[int]) -> tuple[Key, list[int]]:
    # Algorithm 2; S must be a singleton. Heap entries (Key, vertex) via heapq with
    # stale-entry skipping OR a decrease-key wrapper; stale skipping is fine since the
    # heap holds O(k) live entries — but skipped stale pops must not count toward |U0|.
    # Returns (B', U).

def bmssp(st: State, l: int, B: Key, S: list[int]) -> tuple[Key, list[int]]:
    # Algorithm 3 lines 1-22 verbatim, including:
    #   line 7 footnote (P empty -> B'_0 = B), line 21 re-prepend of unfinished S_i keys,
    #   line 22 B' = min(B'_last, B) and the W-completion sweep.

def sssp(g: Graph, source: int) -> list[float]:
    # ALGORITHM.md §4.4: transform, params, init state, top-level bmssp call,
    # map dhat back through rep[].
```

Implementation notes:

- Relaxation is one shared helper `try_relax(st, u, v, w) -> RelaxOutcome` returning the
  candidate key and whether the `≤` test passed; callers then do their own bound/bucket logic
  (the three call sites gate differently — see ALGORITHM.md §4.1/4.2/4.3 notes).
- `U` accumulation in `bmssp`: use a Python `set[int]`; `|U|` checks are O(1). Union with
  each `U_i` must not double-count (the paper guarantees disjointness — assert it in debug
  mode, it is a strong correctness probe).
- Membership test "v is in H" (Algorithm 2 line 10): maintain a `dict` vertex → best key
  pushed; with stale-skip heaps this collapses to "push always, skip stale on pop".
- Termination sanity: assert `B' <= B` on every return, and `S_i` non-empty for every pull
  while the loop guard holds.

---

## 7. VERIFICATION (mandatory instrumentation)

Both checks live in `instrumentation.py` and are always-on (cheap O(1) per event); the
*assertions* about them live in `tests/test_verification.py`.

### 7.a Operations counter — empirical complexity

```python
@dataclass
class OpCounter:
    edge_scans: int = 0        # every evaluation of the relax test (Alg.1 L7, Alg.2 L8, Alg.3 L15)
    relaxations: int = 0       # every successful dhat write
    ds_inserts: int = 0        # BlockDS.insert calls
    ds_prepend_items: int = 0  # total items across batch_prepend calls
    ds_pulls: int = 0          # BlockDS.pull calls
    ds_pulled_items: int = 0
    heap_ops: int = 0          # base-case heap push/pop
    findpivots_calls: int = 0
    bmssp_calls: int = 0
    basecase_calls: int = 0

    def total(self) -> int     # sum of all fields
```

`sssp` must accept `counter: OpCounter | None` and return it via
`sssp_instrumented(g, s) -> tuple[list[float], OpCounter, SettleLog]`.

Required test (`test_verification.py::test_empirical_complexity`):
for random constant-out-degree-ish digraphs with `n ∈ {2**10, 2**12, 2**14, 2**16}`
(m ≈ 2n, fixed seed), compute `r(n) = counter.total() / (m_transformed * log2(n_transformed)**(2/3))`
and assert `max r(n) / min r(n) < 4` (the normalized cost is flat-ish) **and**
`r(n)` is not monotonically growing by more than 30% per size step. This is a smoke-level
empirical check of the O(m log^(2/3) n) claim, not a proof; keep thresholds loose and the
seed fixed.

### 7.b Settlement-order log — proof of non-sorting

```python
@dataclass
class SettleLog:
    events: list[tuple[int, float]]   # (vertex, dhat-at-settlement), append-only

def is_globally_sorted(log: SettleLog) -> bool:
    # True iff the dhat values in log.events are non-decreasing.
```

Settlement event definition (append exactly once per vertex, guarded by `st.settled[]`):

1. in `base_case`, for each vertex placed into the **returned** `U` (i.e. after the
   truncation of line 17 — not for the excluded max vertex), in heap-extraction order;
2. in `bmssp` line 22, for each `x ∈ W` with `d̂[x] < B'` not yet settled, in iteration order.

These are the two origin points of every member of every returned `U`; vertices reaching `U`
via `U ← U ∪ U_i` were already logged inside the recursive call.

Required tests:

- `test_settlement_complete`: on random graphs, the settled vertex set equals the reachable
  set of the transformed graph restricted to top-level-U membership; simpler robust form:
  every original vertex `v` with finite reference distance has at least one of its cycle
  vertices settled, and every logged dhat equals the true distance of that (transformed)
  vertex per the Dijkstra oracle run on the transformed graph.
- `test_not_globally_sorted` (**the mandatory check**): for a fixed-seed random digraph with
  `n = 4096`, `m ≈ 2n`, uniform weights in [0,1]:
  `assert is_globally_sorted(log) is False`.
  Rationale: Dijkstra's settlement order is always sorted; this algorithm settles the
  `W`-vertices of FindPivots in batches *after* deeper recursive calls have settled vertices
  with larger distances, so a sorted log on a non-trivial input would indicate the
  implementation has degenerated into Dijkstra (see ALGORITHM.md §7 item 1/2). If this
  assertion ever fails on such graphs, treat it as a bug in the implementation, not as a
  flaky test. (On *trivial* inputs — paths, tiny n where k = t = 1 and the recursion
  degenerates — a sorted order is possible and the test must not use such inputs.)
- `test_sorted_oracle_sanity`: the reference Dijkstra in `tests/reference.py`, instrumented
  the same way, yields `is_globally_sorted(...) is True` on the same graph (validates the
  checker itself).

---

## 8. Unit-test plan (per procedure)

`tests/reference.py`: textbook Dijkstra (heapq) and Bellman–Ford over `Graph`; both return
`list[float]`. Used as oracles everywhere; also a brute-force shortest-path-tree builder
(needed to compute `T(S)`, `T_<B(S)`, completeness) for graphs with ≤ ~200 vertices, using
the §3 tie-breaking order so that "the" shortest path is unique.

### 8.1 `test_block_ds.py`

Deterministic unit tests (small M, e.g. M ∈ {1, 2, 3, 8}):

- insert n items, pull repeatedly: concatenated pulls are a partition of the items; each
  pull's max value < returned bound ≤ next pull's min value; final pull returns bound `B`.
- `|S'| ≤ M` always; non-final pulls return exactly M.
- duplicate key, smaller value → value replaced; larger value → ignored (both for insert
  and across insert/batch_prepend).
- batch_prepend then pull: prepended items come out first.
- batch_prepend with |L| > M: still correct partition/order.
- empty pull → `([], B)`.
- white-box: after many inserts, every D1 block holds ≤ M pairs; bounds list is
  non-decreasing; inter-block ordering invariant holds (validator method
  `BlockDS._check_invariants()` used only by tests/debug).

Model-based randomized test: run a random script of valid operations against a naive model
(a dict key→value; pull = sort all, take min(M, len) smallest, bound = next smallest value or
B). Valid script generation must respect batch_prepend's precondition (generate values
strictly below the model's current minimum). Compare pulled key-sets and bounds exactly.

### 8.2 `test_find_pivots.py`

Setup helper: build a small graph, run reference Dijkstra to get true `d(·)`, then *fabricate*
a mid-algorithm state: choose a bound `B` and a frontier `S`, set `dhat[v] = d(v)` (complete)
for chosen vertices, `inf` for others, such that the Lemma 3.2 precondition holds (every
incomplete `v` with `d(v) < B` has its shortest path through a complete vertex of `S`).
Easiest valid fabrication: pick a threshold `b ≤ B`, mark exactly `{v : d(v) < b}` complete
with edges relaxed (set their dhat/hops/pred from the reference tree, and relax their
out-edges once into dhat), and let `S = {v : b ≤ key(v) < B, dhat[v] < inf}`.

Assertions (computing `Ũ = T_<B(S)` by oracle):

- `P ⊆ S`; `W ⊆ Ũ`. Size bounds, branch-exact: in the non-early-exit branch assert
  `len(W) <= k*len(S)` and `len(P) <= len(W)//k`; in the early-exit branch assert
  `P == S` (as sets), `len(W) > k*len(S)`, and `len(W) <= k*len(S) + 2*len(W_prev_round)`
  (one extra constant-degree round of growth, ≤ 2 out-edges per vertex — exposed to the
  test via a debug hook or recomputed bound `len(W) <= 3*k*len(S) + len(S)`).
- Lemma 3.2 disjunction: for every `x ∈ Ũ`, either (`x ∈ W` and `dhat[x] == d(x)`) or the
  oracle shortest path to `x` passes through some `y ∈ P` with `dhat[y] == d(y)`.
- dhat soundness preserved: `dhat[v] ≥ d(v)` for all v, and dhat never increased.
- Graph cases: chain (forces k-step propagation), star, two parallel branches with a tie
  broken by hops, a graph with zero-weight edges, a case engineered so `|W|` explodes
  (binary-tree fanout from S) to hit the early exit, and `P = ∅` case (everything within k
  hops).

### 8.3 `test_base_case.py`

Fabricated states as in 8.2 with `S = {x}` complete:

- heap exhausted with ≤ k settled → `B' == B` and `U == oracle T_<B({x})`.
- truncated → `len(U) == k`, `B' == max settled key`, `U == {v : key(v) < B'} ∩ U0`, all
  of `U` complete vs oracle.
- bound respected: no vertex with `d(v) ≥ B` (key-order) is written at all (dhat unchanged
  for those — distinguishes Alg. 2's gated relaxation from Alg. 1/3).
- zero-weight cycle inside the ball (transform-like) → terminates, hops tie-break used.

### 8.4 `test_bmssp.py`

Fabricated states (as 8.2) at small parameters (force `k`, `t` small by monkeypatching
`compute_params`, e.g. k=2, t=2, so l up to 3 on graphs of 50–200 vertices):

- Postcondition (Lemma 3.7): `U == oracle T_<B'(S)` (as vertex sets, key-order comparison
  against B'), and every `u ∈ U` has `dhat[u] == d(u)`.
- `B' <= B`; if `B' < B` then `len(U) >= k * 2**(l*t)` (Lemma 3.9 lower bound), and always
  `len(U) <= 4*k*2**(l*t)`.
- successful executions on small `Ũ` return `B' == B`.
- Internal probes (enabled via a debug flag): disjointness of `U_i`; `max key in S_i < B_i`;
  every batch_prepend value < B_i; Lemma 3.10 (`min d(x) over D ≥ B'_{i-1}` — checkable
  against the oracle since debug mode knows true distances).

### 8.5 `test_transform.py`

- degrees of transformed graph ≤ 2 in and out.
- `d_{g2}(rep[v]) == d_g(v)` for random graphs (compare reference Dijkstra on both).
- vertex with no neighbors, with one neighbor, with self-loop, parallel edges.

### 8.6 `test_sssp.py` (end-to-end, oracle = reference Dijkstra on the original graph)

Fixed cases: single vertex; single edge; two-vertex zero-weight cycle; chain of 1000;
star; complete digraph on 30 vertices; all-zero weights; all-equal weights (forces massive
ties); graph with an unreachable component (asserts `inf`); source with no out-edges;
weights with large magnitude spread (1e-9 … 1e9).

### 8.7 `test_properties.py` (hypothesis)

Strategy `digraphs()`: `n ∈ [1, 60]`, edge list drawn as up-to `3n` random `(u, v)` pairs,
weights from `one_of(just(0.0), floats(0, 1, allow_nan=False), sampled_from([0.5, 1.0]))`
(mixing forced ties), `source = 0`.

Properties:

1. `sssp(g, 0) == reference_dijkstra(g, 0)` element-wise (exact equality is required when
   weights are dyadic/0/1-like; for general floats compare with `math.isclose(rel=1e-9)` and
   additionally assert exact equality of the reachable set).
2. Idempotence/purity: running twice gives identical results (no leaked global state).
3. Monotonicity probe: settle log distances per vertex equal final distances.
4. With `k`, `t` monkeypatched to (1,1), (1,2), (2,1), (3,2): same answers (correctness is
   parameter-independent — ALGORITHM.md §2).
5. BlockDS model-based property test (operation scripts as in 8.1, hypothesis-generated).

Shrinking note: keep `n` small and parameters monkeypatched small so hypothesis
counterexamples stay readable.

---

## 9. Acceptance checklist

- [ ] All tests in §7–§8 pass (`pytest -q`), including `test_not_globally_sorted`.
- [ ] No procedure deviates from ALGORITHM.md pseudocode line structure; each function body
      carries the paper's line numbers as comments (`# L15`, …) for review.
- [ ] The five regressions of ALGORITHM.md §7 are each covered by at least one test that
      would fail if the regression were introduced (1→`test_not_globally_sorted` +
      pull-batch unit tests; 2→FindPivots pivot-bound assertions; 3→FindPivots loop-count
      via OpCounter on a chain graph; 4→ds_prepend_items > 0 on large random graphs;
      5→Lemma 3.9 size assertions in `test_bmssp.py`).
- [ ] `sssp` raises `ValueError` on negative weights and on `source` out of range.
