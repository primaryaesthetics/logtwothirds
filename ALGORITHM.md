# ALGORITHM.md — Breaking the Sorting Barrier for Directed SSSP

Self-contained distillation of: Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, Longhui Yin,
*"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"* (arXiv:2504.17033v2, July 31 2025).
Deterministic **O(m·log^(2/3) n)** SSSP on directed graphs with real non-negative edge
weights, in the comparison-addition model. This document is written so the algorithm can be
implemented without access to the PDF. Algorithm/line numbers match the paper.

---

## 1. Model, assumptions, notation

- Directed graph `G = (V, E)`, weight `w : E → R≥0` (`w_uv` for edge `(u,v)`), `n = |V|`,
  `m = |E|`, source `s`. Goal: `d(v)` = length of shortest s→v path, for all `v`.
- **Comparison-addition model**: only comparisons and additions on edge weights, unit cost each.
- The paper assumes WLOG every vertex is reachable from `s` (so `m ≥ n−1`). An implementation
  simply leaves unreachable vertices at `d̂ = ∞`; they never enter any frontier set.

### 1.1 Constant-degree transformation (Section 2 of the paper)

The algorithm is stated for graphs with **constant in-degree and out-degree (≤ 2)**.
Any graph is first transformed (classical construction, similar to [Fre83]):

- Replace each vertex `v` by a **cycle** of new vertices connected by **zero-weight** directed
  edges; the cycle has one vertex `x_vw` for every in- or out-neighbor `w` of `v`.
- For every original edge `(u,v)` with weight `w_uv`, add directed edge `x_uv → x_vu` with
  weight `w_uv`.

The transformed graph `G'` has `O(m)` vertices, `O(m)` edges, in/out-degree ≤ 2, and preserves
shortest-path lengths (`d_G(v)` = distance to any vertex on `v`'s cycle, since the cycle is
zero-weight). Degenerate cases: a vertex with 0 neighbors maps to a single node with no cycle
edges; a vertex with 1 neighbor maps to a single node (no cycle edge needed).
**All parameters `k, t, l` below are computed from `n' = |V(G')| = O(m)` of the transformed
graph**, on which `m' = O(n')`.

### 1.2 Labels and completeness

- Global array `d̂[·]`: sound estimate, `d̂[u] ≥ d(u)` always. Initially `d̂[s] = 0`, `d̂[v] = ∞`
  otherwise. Updated only by **relaxation** of an edge `(u,v)`:
  `d̂[v] ← d̂[u] + w_uv` when `d̂[u] + w_uv` is no greater than the old `d̂[v]`; also set
  `Pred[v] ← u`. Every finite `d̂[v]` value corresponds to an actual s→v path.
- Vertex `x` is **complete** when `d̂[x] = d(x)`; otherwise **incomplete**. A set is complete
  if all its members are. Completeness is monotone in time (complete stays complete).

### 1.3 Total order on paths (Assumption 2.1) — REQUIRED, not optional

The paper assumes **all paths obtained have distinct lengths**, realized by a tie-breaking
total order: a path of length `l` through `α` vertices `v1 = s, v2, …, vα` is the tuple
`⟨l, α, vα, vα−1, …, v1⟩` (vertex sequence **reversed**, endpoint first), compared
lexicographically. O(1) comparison suffices in practice:

- **Comparing `d̂[u]` vs `d̂[v]` for `u ≠ v`**: compare lengths; tie → compare hop counts `α`;
  tie → compare endpoints `u` vs `v` (decisive since `u ≠ v`).
- **Relaxing `(u,v)` against the current label of `v`**: compare lengths; tie → compare hop
  counts; tie → compare `u` vs `Pred[v]`; if `u = Pred[v]`, the candidate is the *same* path
  (i.e. `u`'s label was re-confirmed) and the update must proceed (this is the "≤" of
  Remark 3.4, see §6.1).

Concrete realization: store per-vertex `(d̂[v], hops[v], Pred[v])` and compare keys
`(d̂[v], hops[v], v)` across vertices, `(d̂[u]+w_uv, hops[u]+1, u)` vs `(d̂[v], hops[v], Pred[v])`
inside relaxation. All bounds `B, B', B_i, B'_i` are then values in this same totally ordered
key space (with `∞` = a key greater than every real key). This matters because the
constant-degree transform introduces **zero-weight edges**, which guarantee ties; without the
total order, the forest `F` in FindPivots need not be a forest and termination/correctness
arguments break. The `α` (hops) component makes a path with an extra zero-weight edge strictly
"longer".

This assumption gives: (1) `Pred[·]` always forms a tree; (2) a strict order among vertices
with equal numeric `d̂`; (3) uniqueness of shortest paths, so the shortest path tree `T` rooted
at `s` is unambiguous.

### 1.4 Set notation used in the contracts

- `T(u)`: subtree of the (unique) shortest path tree rooted at `u`; equivalently
  `v ∈ T(u)` ⟺ the shortest s→v path passes through `u`.
- `T(S) = ⋃_{v∈S} T(v)`; `S* = {v ∈ S : v complete}` (time-dependent); `T(S*)` likewise.
- `T_<B(S) = {v ∈ T(S) : d(v) < B}`; `T_[b,B)(S) = {v ∈ T(S) : d(v) ∈ [b,B)}`.
- `Ũ := T_<B(S)` — "the vertices of interest" of a call with bound `B` and frontier `S`.

---

## 2. Parameters

Computed once, from `n` = number of vertices of the **transformed** (constant-degree) graph.
Logarithms base 2 (any fixed base only changes constants).

| Parameter | Value | Role |
|---|---|---|
| `k` | `⌊(log n)^(1/3)⌋` | Bellman-Ford depth in FindPivots; pivot tree-size threshold; base-case size `k+1` |
| `t` | `⌊(log n)^(2/3)⌋` | level width: a level-`l` call handles a frontier of size ≤ `2^(l·t)` |
| `l` (top) | `⌈(log n)/t⌉` | number of recursion levels; depth of recursion ≈ `(log n)/t = O((log n)^(1/3))` |
| `M` (per call at level `l`) | `2^((l−1)·t)` | Pull batch size of the data structure `D` at that level |
| `N` (per call at level `l`) | `O(k·2^(l·t))` | max #insertions into that call's `D` (analysis only; the structure need not know it) |

Top-level invocation: `BMSSP(l = ⌈(log n)/t⌉, B = ∞, S = {s})`. Since
`k·2^(l·t) ≥ k·n > |V|`, the top call can never hit the partial-execution size cap, so it
terminates with `D` empty and **every reachable vertex complete**; the answer is read from
`d̂[·]` (not from the returned set `U`).

Correctness does **not** depend on the parameter values (any `k ≥ 1`, `t ≥ 1` is correct);
only the running-time bound does. Implementations must clamp `k ← max(1, ·)`,
`t ← max(1, ·)` for tiny `n` (e.g. `log n < 8` gives `k = 1`).

---

## 3. The data structure `D` (Lemma 3.3)

A partial-sorting, batched priority structure parameterized by block size `M` and a global
value upper bound `B` (all values ever stored are `< B`; with key-space values per §1.3).
Given at most `N` total insertions:

### 3.1 Operation signatures and contracts

```
Initialize(M, B)
    D0 ← empty block sequence; D1 ← one empty block with upper bound B.

Insert(key, value)                       — amortized O(max{1, log(N/M)})
    If key already present: keep only the pair with the smaller value
    (i.e. replace iff value < old value; otherwise no-op).
    Inserted pairs always go to sequence D1.

BatchPrepend(L)                          — amortized O(|L| · max{1, log(|L|/M)}) total
    PRECONDITION: every value in L is smaller than every value currently in D.
    Insert all pairs of L (into sequence D0). If L contains several pairs with the
    same key, keep the smallest value; if a key already exists in D (necessarily with
    a larger value, by the precondition), the old pair is replaced.

Pull() -> (S', x)                        — amortized O(|S'|)
    Remove and return a set S' of keys, |S'| ≤ M, holding the |S'| SMALLEST values
    in D, together with a separating bound x:
      - if D becomes empty: S' = all former contents, x = B;
      - else: |S'| = M and  max(value over S') < x ≤ min(value remaining in D).
    (Values are pairwise distinct by Assumption 2.1, so "the M smallest" is unambiguous.)
```

In Algorithm 3 the call site is written `B_i, S_i ← D.Pull()` — i.e. the bound is used first;
the lemma states the pair as `(S', x)`. Pick one order in the implementation and keep it.

### 3.2 Internal design and invariants

- Two sequences of **blocks**: `D0` (receives only BatchPrepend batches) and `D1` (receives
  only Insert). A block is an unsorted doubly-linked list of ≤ `M` key/value pairs.
- **Inter-block ordering invariant** (within each sequence separately): if block `B_i`
  precedes block `B_j`, every value in `B_i` ≤ every value in `B_j`. (No ordering is
  maintained *inside* a block, and no global order between `D0` and `D1` is required —
  Pull inspects prefixes of both.)
- Each `D1` block carries an **upper bound**; a block's bound is ≤ every value in the next
  block; the last `D1` block's bound is `B`. The bounds live in a self-balancing BST
  (e.g. red-black tree), so Insert finds its target block — the one with the smallest upper
  bound ≥ value — in `O(max{1, log(N/M)})` time.
- **Split**: when a `D1` block exceeds `M` pairs, find the median value in `O(M)` (linear-time
  selection), split into two blocks of ≤ `⌈M/2⌉` (smaller-than-median / rest), update the BST
  in `O(max{1, log(N/M)})`. Hence every split-created `D1` block holds `Θ(M)` elements
  (counting later-deleted ones), so `D1` has `O(max{1, N/M})` blocks. `D0`'s block count is
  not bounded this way (and needn't be).
- **BatchPrepend(L)**: if `|L| ≤ M`, one new block at the front of `D0`. Else split `L` by
  repeated median-finding into `O(|L|/M)` blocks of ≤ `⌈M/2⌉` elements each,
  `O(|L|·log(|L|/M))` total, prepended in order.
- **Key registry** (for duplicate handling): a dictionary key → (value, block/node pointer),
  giving O(1) `Delete(key)` from a block's linked list. Deleting may empty a `D1` block, whose
  bound must then be removed from the BST — `O(log(N/M))`, charged (amortized) to the Insert
  that created the element. It is *not* necessary to update block upper bounds on deletion.
- **Pull()**: collect a prefix of blocks from `D0` and from `D1` separately, stopping in each
  sequence once that sequence is exhausted or ≥ `M` elements are collected (collected sets
  `S0'`, `S1'`). If `|S0' ∪ S1'| ≤ M`, that is all of `D`: return it with `x = B`. Otherwise
  select the `M` smallest of `S0' ∪ S1'` in `O(M)` (linear selection), delete exactly those,
  return them with `x` = the minimum value remaining in `D` (found in `O(M)`: by the
  inter-block invariant the remaining minimum lies in the first non-empty block of `D0` or of
  `D1`).
- Derived *near*-invariant at the BMSSP call sites (useful for testing): the value stored in
  `D` for key `v` equals the `d̂[v]` current at the moment of the Insert/BatchPrepend, and the
  keep-min rule discards stale larger values whenever an update passes through this `D`.
  **Correction (audit):** it is *not* true that the stored value always equals the current
  `d̂[v]`: `d̂[v]` can be improved by a relaxation deep inside a descendant recursive call
  without that update ever reaching this `D`'s Insert (the re-relaxation of Remark 3.4 then
  fails both bucket tests at line 17/19 because the new value lies below `B'_i`). Such a key
  is already settled when later pulled; see QUESTIONS.md item 3 and AUDIT.md finding F3 —
  the paper's proofs of Lemmas 3.6/3.7/3.10 implicitly assume stored = current, which is the
  gap the implementation's settled-vertex filter closes.

### 3.3 Costs as used inside BMSSP (Remark 3.5)

At level `l`: `M = 2^((l−1)t)`, total insertions `N = O(k·2^(l·t))` (by `|U| = O(k·2^(lt))`,
constant degree, and disjointness of the `U_i`), so `log(N/M) = O(log k + t) = O(t)` per
Insert. Each BatchPrepend batch has size `O(|U_i|) = O(k·2^((l−1)t))`, so
`log(|L|/M) = O(log k) = O(log log n)` per prepended element. Each pulled element: amortized
O(1) (chargeable to its insertion).

---

## 4. Pseudocode

### 4.1 Algorithm 1 — FindPivots(B, S)   (paper page 6)

```
 1: function FindPivots(B, S)
    • requirement: for every incomplete vertex v with d(v) < B, the shortest
      path to v visits some complete vertex in S
    • returns: sets P, W satisfying the conditions in Lemma 3.2
 2:   W ← S
 3:   W_0 ← S
 4:   for i ← 1 to k do                                ▷ relax k steps
 5:     W_i ← ∅
 6:     for all edges (u,v) with u ∈ W_{i−1} do
 7:       if d̂[u] + w_uv ≤ d̂[v] then
 8:         d̂[v] ← d̂[u] + w_uv                          (and Pred[v] ← u)
 9:         if d̂[u] + w_uv < B then
10:           W_i ← W_i ∪ {v}
11:     W ← W ∪ W_i
12:     if |W| > k·|S| then
13:       P ← S
14:       return P, W
15:   F ← {(u,v) ∈ E : u,v ∈ W, d̂[v] = d̂[u] + w_uv}    ▷ F is a directed forest
                                                          under Assumption 2.1
16:   P ← {u ∈ S : u is a root of a tree with ≥ k vertices in F}
17:   return P, W
```

Notes:
- Lines 11–14 are **inside** the `for i` loop (early exit as soon as `W` outgrows `k|S|`;
  the proof of Lemma 3.2 — "if the algorithm returns due to |W| > k|S| … |W| = O(k|S|) since
  out-degrees are constant" — and the `O(k|W|)` time bound require checking after every
  round). Confirmed against the paper's original indentation.
- Line 8 updates `d̂[v]` even when `d̂[u]+w_uv ≥ B`; the bound `B` only gates membership in
  `W_i` (propagation), not the relaxation itself.
- Line 15: `F` is the set of **tight** edges inside `W`; equality is in the total order of
  §1.3, i.e. `u = Pred[v]` with matching length and hop count. Under Assumption 2.1 each
  vertex has at most one incoming tight edge, so `F` is a forest. Line 16 needs only the
  restriction of `F` to edges with both endpoints in `W` and tree sizes computable in `O(|W|)`.

**Lemma 3.2 (contract).**
*Precondition:* every incomplete `v` with `d(v) < B` has its shortest path through some
complete `u ∈ S`. (`B` need not exceed all `d̂[x]`, `x ∈ S`, for this procedure, but at its
call site it does.)
*Postcondition:* returns `P ⊆ S` and `W ⊆ Ũ` (`Ũ = T_<B(S)`) with `|W| = O(k|S|)` and
`|P| ≤ |W|/k` (early-exit case: `P = S` and `|W| > k|S|` ⟹ `|S| < |W|/k`; normal case: each
pivot owns a disjoint tight subtree of ≥ k vertices, so `|P| ≤ |W|/k ≤ |Ũ|/k`), such that for
every `x ∈ Ũ` at least one of:
1. `x ∈ W` and `x` is complete at return, or
2. the shortest path to `x` visits some complete `y ∈ P`.

Intuition: after `k` rounds of bounded Bellman–Ford from `S`, every `x ∈ Ũ` whose shortest
path has < k edges after its last complete-in-`S` ancestor is complete; otherwise that
ancestor roots a tight tree of ≥ k vertices and is kept as a pivot.
*Cost:* `O(k|W|) = O(min{k²|S|, k|Ũ|})`.

### 4.2 Algorithm 2 — BaseCase(B, S)   (paper page 9)

```
 1: function BaseCase(B, S)
    • requirement 1: S = {x} is a singleton, and x is complete
    • requirement 2: for every incomplete vertex v with d(v) < B,
      the shortest path to v visits x
    • returns 1: a boundary B' ≤ B
    • returns 2: a set U
 2:   U_0 ← S
 3:   initialize a binary heap H with the single element ⟨x, d̂[x]⟩
 4:   while H is non-empty and |U_0| < k + 1 do
 5:     ⟨u, d̂[u]⟩ ← H.ExtractMin()
 6:     U_0 ← U_0 ∪ {u}
 7:     for edge e = (u,v) do                            ▷ all out-edges of u
 8:       if d̂[u] + w_uv ≤ d̂[v] and d̂[u] + w_uv < B then
 9:         d̂[v] ← d̂[u] + w_uv                           (and Pred[v] ← u)
10:         if v is not in H then
11:           H.Insert(⟨v, d̂[v]⟩)
12:         else
13:           H.DecreaseKey(⟨v, d̂[v]⟩)
14:   if |U_0| ≤ k then
15:     return B' ← B,  U ← U_0
16:   else
17:     return B' ← max_{v∈U_0} d̂[v],  U ← {v ∈ U_0 : d̂[v] < B'}
```

Notes:
- A plain "mini Dijkstra" from `x`, truncated after `k+1` settled vertices and bounded by `B`.
  Unlike Algorithm 1/3, here the relaxation itself is gated by `< B` (line 8): values ≥ B are
  not even written. The heap is keyed by the total order of §1.3.
- `U_0` starts as `{x}` and `x` is also the first extraction (idempotent union).
- Successful case (heap exhausted with `|U_0| ≤ k`): everything reachable from `x` below `B`
  was settled; `B' = B`, `U = U_0`.
- Truncated case (`|U_0| = k + 1`): `B'` is set to the **largest** settled label; the vertex
  attaining it (unique by Assumption 2.1) is excluded, so `|U| = k`. Vertices extracted from
  the heap are complete by the standard Dijkstra argument (precondition 2 ensures no shorter
  path can enter from outside).
- Cost: `O(k log k)` — heap of size `O(k)` (constant out-degree), ≤ `k+1` extractions.

**Contract.**
*Pre:* `S = {x}`, `x` complete, `B > d̂[x]`, and every incomplete `v` with `d(v) < B` is in
`T({x}*) = T(x)`.
*Post:* returns `B' ≤ B` and `U = T_<B'({x})`; `U` is complete; `|U| ≤ k` (the successful
branch returns `U_0` with `|U_0| ≤ k`; the truncated branch starts from `|U_0| = k+1` and
excludes exactly the unique maximum, returning exactly `k`); if `B' < B` then `|U| ≥ k`.

### 4.3 Algorithm 3 — BMSSP(l, B, S)   (paper page 10)

```
 1: function BMSSP(l, B, S)
    • requirement 1: |S| ≤ 2^(l·t)
    • requirement 2: for every incomplete vertex x with d(x) < B, the shortest
      path to x visits some complete vertex y ∈ S
    • returns 1: a boundary B' ≤ B
    • returns 2: a set U
 2:   if l = 0 then
 3:     return B', U ← BaseCase(B, S)
 4:   P, W ← FindPivots(B, S)
 5:   D.Initialize(M, B) with M = 2^((l−1)·t)        ▷ D : instance of Lemma 3.3
 6:   D.Insert(⟨x, d̂[x]⟩) for each x ∈ P
 7:   i ← 0;  B'_0 ← min_{x∈P} d̂[x];  U ← ∅          ▷ if P = ∅, set B'_0 ← B
 8:   while |U| < k·2^(l·t) and D is non-empty do
 9:     i ← i + 1
10:     B_i, S_i ← D.Pull()
11:     B'_i, U_i ← BMSSP(l − 1, B_i, S_i)
12:     U ← U ∪ U_i
13:     K ← ∅
14:     for edge e = (u,v) where u ∈ U_i do           ▷ all out-edges of U_i
15:       if d̂[u] + w_uv ≤ d̂[v] then
16:         d̂[v] ← d̂[u] + w_uv                        (and Pred[v] ← u)
17:         if d̂[u] + w_uv ∈ [B_i, B) then
18:           D.Insert(⟨v, d̂[u] + w_uv⟩)
19:         else if d̂[u] + w_uv ∈ [B'_i, B_i) then
20:           K ← K ∪ {⟨v, d̂[u] + w_uv⟩}
21:     D.BatchPrepend(K ∪ {⟨x, d̂[x]⟩ : x ∈ S_i and d̂[x] ∈ [B'_i, B_i)})
22:   return B' ← min{B'_i, B},  U ← U ∪ {x ∈ W : d̂[x] < B'}
```

Notes:
- Line 7's `B'_0` matters when the loop body never runs (then line 22 uses `B'_0`).
- Line 10: `B_i` is simultaneously the upper bound for the recursive call and a lower bound
  on everything left in `D` (Pull's separation guarantee). `|S_i| ≤ M = 2^((l−1)t)` and
  `max_{x∈S_i} d̂[x] < B_i`, so the recursive call's requirements hold (with Lemma 3.6
  supplying requirement 2).
- Lines 15–20: the relaxation (line 16) is performed for any valid candidate, even one
  `≥ B` (which is then neither queued nor recorded); only candidates in `[B_i, B)` go to `D`
  (they belong to *later* batches), and candidates in `[B'_i, B_i)` — possible only when the
  recursive call was partial, `B'_i < B_i` — are collected in `K` for batch-prepending
  (they precede everything now in `D`). The branch is taken **even when** `d̂[v]` already
  equals `d̂[u] + w_uv` (Remark 3.4, §6.1).
- Line 21 also re-inserts the *unfinished* part of the pulled batch: pulled keys `x ∈ S_i`
  whose `d̂[x]` is still in `[B'_i, B_i)` (not completed by the partial child call). All
  prepended values are `< B_i ≤` every value currently in `D`, so BatchPrepend's
  precondition holds.
- Line 22, exit cases:
  - loop exited because `D` is empty → "successful execution"; the last pull emptied `D`
    so `B_i = B`; if the last child call was itself successful, `B'_i = B_i = B` and
    `B' = B`. (A partial child can empty `D` only when nothing remained in `[B'_i, B)`;
    then `B' = B'_i < B` but `T_[B',B)(S)` is empty, so `U = T_<B'(S) = T_<B(S)` anyway —
    the dichotomy of Lemma 3.1 is stated up to this harmless corner.)
  - loop exited because `|U| ≥ k·2^(l·t)` → "partial execution due to large workload";
    `B' = B'_i < B` (the prose of the paper describes this as step 6: "If |U| > k·2^(lt),
    set B' ← B'_i"; the pseudocode's loop guard uses `<`, i.e. exit at `≥` — follow the
    pseudocode).
  - Finally the vertices completed by FindPivots' Bellman–Ford rounds that fall below the
    final boundary — `{x ∈ W : d̂[x] < B'}` — are added to `U`.

**Lemma 3.1 / 3.7 / 3.9 (contract).**
*Pre:* `l ∈ [0, ⌈(log n)/t⌉]`; `|S| ≤ 2^(l·t)`; `B > max_{x∈S} d̂[x]`; every incomplete `v`
with `d(v) < B` is in `T(S*)` (its shortest path visits a complete vertex of `S`).
*Post:* returns `B' ≤ B` and `U = T_<B'(S)` (every vertex `v` with `d(v) < B'` whose shortest
path visits a vertex of `S`, and only those); `U` is complete at return; `d̂`/`Pred` have only
improved. Size: `|U| ≤ 4k·2^(l·t)` always, and `B' < B ⟹ |U| ≥ k·2^(l·t)` (Lemma 3.9).
Dichotomy (Lemma 3.1): **successful** `B' = B`, or **partial** `B' < B` with
`|U| = Θ(k·2^(l·t))`.
*Cost (Lemma 3.12):* `C(k + 2t/k)(l+1)|U| + C(t + l·log k)·|N⁺_[min_{x∈S} d(x), B)(U)|`,
where `N⁺_[c,d)(U) = {(u,v) ∈ E : u ∈ U, d(u)+w_uv ∈ [c,d)}`; with constant degree this is
`O((k·l + t·l/k + t)·|U|)` (statement of Lemma 3.1).

Internal invariants of the loop (used in the correctness proof, and directly checkable by an
instrumented implementation; `D_i` = keys in `D` just before iteration `i`):
- (a) every incomplete `v` with `d(v) < B` lies in `T_[B'_{i−1}, B)(P)`;
- (b) `T_[B'_{i−1}, B)(P) = T_<B(D_i) = T_<B(D_i*)`;
- `B'_0 ≤ B'_1 ≤ … ` and `min_{x∈D} d(x) ≥ B'_{i−1}` before iteration `i` (Lemma 3.10);
- `U_i = T_[B'_{i−1}, B'_i)(P)`, the `U_i` are pairwise disjoint and complete, and
  distances in `U_i` strictly precede those in `U_j` for `i < j` (Remark 3.8);
- `min{|Ũ|, k|S|} ≤ |U|` and `|S| ≤ |U|` at return (Lemma 3.11).

### 4.4 Top-level procedure   (Section 3.1 prose; numbering ours, not the paper's)

```
function SSSP(G, s):
  1. G' ← ConstantDegreeTransform(G); s' ← any cycle-vertex of s
  2. n ← |V(G')|
     k ← max(1, ⌊(log2 n)^(1/3)⌋); t ← max(1, ⌊(log2 n)^(2/3)⌋); L ← ⌈(log2 n)/t⌉
  3. d̂[v] ← ∞ for all v; d̂[s'] ← 0; hops[s'] ← 0; Pred[s'] ← ⊥
  4. (B', U) ← BMSSP(L, ∞, {s'})
  5. for each original vertex v: dist(v) ← d̂[any cycle-vertex of v]   (they are equal;
     using min over the cycle is a safe implementation choice)
  6. return dist
```

*Pre:* none beyond a valid non-negative-weight digraph.
*Post:* `dist(v) = d(v)` for all reachable `v`, `∞` otherwise. The top call satisfies
requirement 2 vacuously (`s` is the only complete vertex initially, and every shortest path
visits `s`); `|{s}| = 1 ≤ 2^(L·t)`; `∞ > d̂[s]`. Because `|U| ≤ |V| < k·2^(L·t)` (for `k ≥ 2`;
for clamped tiny parameters a partial exit can only happen with `U` already covering
everything below `B'`, and `d̂` is correct for all completed vertices), the top call runs until
`D` is empty, at which point **no incomplete vertex exists** — read results from `d̂`.

---

## 5. Recursion shape (what the divide-and-conquer achieves)

Each level-`l` call partitions its workload into ≤ `2^t`-ish child calls by *distance ranges*
`[B'_0, B'_1), [B'_1, B'_2), …` discovered on the fly by `Pull`. The recursion tree has depth
`≤ ⌈(log n)/t⌉ = O((log n)^(1/3))`; the `U_x` of all nodes at one depth are disjoint, so the
total `Σ|U_x|` over the tree is `O(n·(log n)/t)`. FindPivots costs `O(k|U_x|)` per node →
`O(nk·(log n)/t) = O(n (log n)^(2/3))` overall. Each edge causes at most one *direct* `Insert`
into a `D` over the whole algorithm at the level where it "exits" its bound (`O(t)` each), and
at most one `K`-prepend per level (`O(log k)` each, × `(log n)/t` levels). Pulled-batch
re-prepends cost `O(log k)` per `|U_i|`. Summing: `O(m·(log n)^(2/3))`.

---

## 6. Edge cases and gotchas

### 6.1 The "≤" in relaxations (Remark 3.4)

On Algorithm 1 line 7, Algorithm 2 line 8, and Algorithm 3 line 15, the test is
`d̂[u] + w_uv ≤ d̂[v]` — **with equality** (equality in the total order of §1.3, i.e. same
path). Equality must take the branch so that an edge relaxed at a lower recursion level is
*re-used* at upper levels: the upper level needs `v` to be (re-)inserted into its own `D`/`K`
even though `d̂[v]` doesn't change. Replacing `≤` by `<` loses vertices.

### 6.2 Equal numeric distances / zero-weight edges

Guaranteed to occur (the constant-degree transform creates zero-weight cycles). Handled
entirely by the total order of §1.3 (lengths, then hop counts, then endpoint/predecessor).
With it, `F` (Alg. 1 line 15) is a forest, heap/`D` orderings are strict, `max`/`min` over
labels are unique, and `B' = max_{v∈U_0} d̂[v]` excludes exactly one vertex.

### 6.3 Unreachable vertices

Stay at `d̂ = ∞` forever; never enter `W`, any heap, `D`, or any `U`. The paper assumes them
away; implementations just report `∞`.

### 6.4 Base case of the recursion

`l = 0` ⇒ `|S| ≤ 2^0 = 1` and the pulled singleton's vertex is complete (it carries the
minimum remaining value in the parent's `D`; by invariant (b) an incomplete vertex would have
to sit in some complete subtree below another `D`-key of smaller value — impossible for the
minimum, by the total order). BaseCase is the only place vertices are settled one at a time.

### 6.5 Empty pivot set

If `P = ∅` (everything within `k` hops settled, no large tight trees), `D` starts empty, the
loop body never runs, `B'_0 = B` (line 7's footnote), and line 22 returns `B' = B`,
`U = {x ∈ W : d̂[x] < B}` — exactly the vertices completed by FindPivots.

### 6.6 Returned `U` vs. global state

`U = T_<B'(S)` may *exclude* vertices that happen to be complete with `d(x) ≥ B'`
(e.g. the truncated vertex of BaseCase, or `W`-vertices above `B'`). That is fine: the parent
re-discovers them later via its own `D`/`W`. Only at the top level is "every reachable vertex
complete" guaranteed — which is why the final answer is read from `d̂[·]`, never from `U`.

### 6.7 Numbers vs. the comparison-addition model

All values are sums of input weights; no division/subtraction/rounding is used anywhere
(medians, BST keys, heap keys are all weight-sums under the §1.3 order). Floating-point
implementations inherit the usual caveats but no algorithmic dependence on precision.

---

## 7. What this is NOT — five regressions a reviewer must catch

1. **If `Pull` always returns the single global minimum** (i.e. `M = 1` everywhere, or the
   implementation "simplifies" `D` to an ordinary heap popped one element at a time) — this
   is **Dijkstra**: a total order over the whole frontier is being maintained, Θ(log n) per
   vertex, and the log^(2/3) bound is gone. `Pull` must return an *unsorted batch* of up to
   `M = 2^((l−1)t)` keys plus only a *separating bound*.

2. **If FindPivots is skipped (or `P ← S` is always returned)** — every frontier vertex is
   fed into the recursion, the recursion then effectively *sorts all of `S`* across its
   `(log n)/t` levels at `Θ(t)` per vertex, i.e. `Θ(log n)` per vertex — **Dijkstra-grade
   total work**. The whole point is `|P| ≤ |Ũ|/k`: only roots of tight subtrees of size ≥ k
   pay the `Θ(t)` insertion cost.

3. **If the `k`-round loop in FindPivots is run to exhaustion** (`k ← n−1`, or "repeat until
   no `d̂` changes") — this is **Bellman–Ford**, `O(mn)`. The rounds must stop at
   `k = ⌊(log n)^(1/3)⌋`, and additionally bail out as soon as `|W| > k|S|`; both cutoffs are
   load-bearing for the `O(k|W|)` bound.

4. **If `BatchPrepend` is replaced by repeated `Insert`** — each of the up-to-`O(|U_i|)`
   re-queued elements (the `K` set and the unfinished part of `S_i`) then costs `O(t)`
   instead of `O(log k)`; since an edge can re-enter via `K` on **every** level of an
   ancestor chain, the total becomes `O(m·t·(log n)/t) = O(m log n)` — the sorting barrier
   reappears inside the data structure. The two-sequence (`D0`/`D1`) design exists precisely
   so that cheaper-than-everything elements can be queued in `O(log(L/M))` each.

5. **If the partial-execution cutoff is removed** (the `|U| < k·2^(l·t)` guard on line 8 of
   Algorithm 3, and correspondingly the `k+1` cap in BaseCase) — child calls may settle
   arbitrarily many vertices, the `|U_x| = O(k·2^(l_x·t))` balance across the recursion tree
   fails, and per-level work is no longer `O(nk)`; the running time degrades toward the naive
   `Θ(t)`-per-vertex-per-level, i.e. `Θ(n log n)` again (correctness survives; the bound does
   not). The early return with `B' < B` and `|U| = Θ(k·2^(lt))` is what keeps every recursive
   subproblem ~`2^t` times smaller than its parent.

Also watch for: gating the *relaxation itself* by `B` outside BaseCase (Algorithms 1 and 3
must update `d̂` even for candidates ≥ B, see §4.1/§4.3 notes); and using strict `<` in the
relaxation tests (see §6.1) — both subtly break correctness rather than only the bound.

---

## 8. Complexity summary

| Piece | Cost |
|---|---|
| FindPivots, per call | `O(min{k²·|S|, k·|Ũ|})` |
| BaseCase, per call | `O(k log k)` |
| `D.Insert` at level `l` | amortized `O(t)` |
| `D.BatchPrepend` at level `l` | amortized `O(log k)` per element |
| `D.Pull` | amortized `O(1)` per returned element |
| Whole algorithm (constant-degree, `m = O(n)`) | `O(n·(log n)^(2/3))` |
| Whole algorithm (general graph, after transform) | **`O(m·(log n)^(2/3))`** |

Recursion depth `O((log n)^(1/3))`; `Σ|U_x|` per depth ≤ `n`; total over depths
`O(n·(log n)^(1/3))`.

---

## 9. Cross-check status

Cross-referenced once more against the paper (Duan–Mao–Mao–Shu–Yin, arXiv:2504.17033 — https://arxiv.org/abs/2504.17033):
Algorithm 1 lines 1–17, Algorithm 2 lines 1–17, Algorithm 3 lines 1–22, Lemma 3.1, 3.2, 3.3
(operations, costs, internal blocks/BST/split/pull mechanics), Remarks 3.4, 3.5, 3.8,
Lemmas 3.6, 3.7, 3.9, 3.10, 3.11, 3.12, the constant-degree transform, Assumption 2.1 and its
O(1) comparison realization, and the parameter choices `k = ⌊log^(1/3) n⌋`,
`t = ⌊log^(2/3) n⌋`, top `l = ⌈(log n)/t⌉`, `M = 2^((l−1)t)`.

Remaining explicit uncertainties (everything else is verbatim-faithful):

- [UNCERTAIN: Lemma 3.1's clean dichotomy ("successful ⟹ B' = B") vs. pseudocode line 22
  returning `min{B'_i, B}`: in the corner case where a *partial* child empties `D`, line 22
  yields `B' = B'_i < B` even though nothing in `[B'_i, B)` remains; `U = T_<B'(S)` still
  holds (Lemma 3.7), so correctness is unaffected. The paper's prose (step 5: "If D is empty,
  then it is a successful execution") glosses over this; implement line 22 as written.]
- [UNCERTAIN: Lemma 3.3 presents Pull as returning `(S', x)` while Algorithm 3 line 10 binds
  `B_i, S_i ← D.Pull()` (bound first). Order is cosmetic; this document and SPEC.md fix the
  order as `Pull() -> (S', x)`.]
