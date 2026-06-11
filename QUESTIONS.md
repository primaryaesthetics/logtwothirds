# Open questions / spec ambiguities

This file records every place where `ALGORITHM.md` / `SPEC.md` were unclear or
internally inconsistent, the interpretation chosen, and the corresponding
`# TODO(spec)` / `# NOTE(spec)` markers in
`python/logtwothirds/_reference.py`.

## 1. `BlockDS.Pull()` return order (Algorithm 3 line 10 vs SPEC.md S9)

Algorithm 3 line 10 (ALGORITHM.md S4.3) is written `B_i, S_i ← D.Pull()` (bound
first, set second), but `BlockDS.Pull()` (ALGORITHM.md S3.1, Lemma 3.3) is
specified to return `(S', x)` — set first, bound second. SPEC.md S9 fixes the
`(S', x)` convention as authoritative regardless of Algorithm 3's variable
names.

**Resolution**: implemented `Si, Bi = D.pull()` (set first) in `bmssp`,
matching SPEC.md S9 / `BlockDS.pull`'s docstring. See the `# NOTE` comment at
the top of `bmssp`'s main loop.

## 2. Bound comparisons (`< B`) must use the *target* vertex's key, not the
   relaxation candidate's `(len, hops, predecessor)` tuple

Three call sites compare a freshly relaxed candidate against a bound `B` (a
`Key` whose third component is a *vertex id*, per the total order of
ALGORITHM.md S1.3):

- Algorithm 1 line 9 (`find_pivots`, S4.1): "if `dhat[v] < B`" after relaxing
  edge `(u, v)`.
- Algorithm 2 line 8 (`base_case`, S4.2): "`d̂[u] + w_uv < B`".
- Algorithm 3 lines 17–20 (`bmssp`, S4.3): bucketing a relaxed `v` into `D` or
  `K` based on `Bp_i <= ... < Bi <= ... < B`.

`try_relax`'s `RelaxOutcome.cand` is `(cand_len, cand_hops, u)` — its third
component is the *predecessor* `u`, used (correctly) for the `<=` "same path"
equality test of Remark 3.4 (S6.1) against `cur = (dhat[v], hops[v], pred[v])`
(also predecessor-keyed). But `B` (and `Bi`, `Bp_i`) are *vertex-keyed*
bounds: `(boundary_len, boundary_hops, boundary_vertex)`. Comparing
`(cand_len, cand_hops, u)` against `B` mixes a predecessor-keyed tuple with a
vertex-keyed one. When `cand`'s first two components tie with `B`'s, the
comparison is decided by `u` (predecessor) vs `B`'s vertex id — an essentially
arbitrary comparison that can place a "tight" vertex `v` on the wrong side of
the bound, dropping it from `D`/`K`/`W`/`U0` entirely (silent
under/over-reach, observed as both `inf` results that should be finite, and
the disjointness violation of finding (3) below).

**Resolution**: at all three sites, the bound comparison now uses
`vkey = (cand_len, cand_hops, v)` — `v`'s own key after relaxation — not
`cand`. The `<=` equality test against `cur` (Remark 3.4) is unchanged and
still uses `cand`/`pred[v]`. Marked `# NOTE(spec)` at each site.

## 3. A vertex settled as a side effect of a sibling/descendant recursive call
   can leave a stale entry in an ancestor's `D`

ALGORITHM.md S3.2's "derived invariant" states the value `D` stores for key
`v` always equals the current `d̂[v]`, maintained by `Insert`'s keep-min rule.
But `d̂[v]` can also be updated as a side effect of a *different* recursive
branch (e.g. a sibling's `BaseCase` mini-Dijkstra relaxing an edge into `v`,
S6.6) without that update ever passing through the ancestor's `D.Insert`. If
that happens, an ancestor's `D` can later `Pull` a vertex `v` that has already
been settled (and is already a member of an earlier `U_i`), violating the
"`U_i` pairwise disjoint" and `T_<B'(S)`-completeness postconditions (Lemma
3.7, ALGORITHM.md S4.3 invariants) if `v` is passed unchanged into a fresh
recursive call.

Neither ALGORITHM.md nor SPEC.md addresses this case explicitly.

**Resolution** (marked `# TODO(spec)` in `bmssp`): before recursing on a
pulled batch `S_i`, filter out any vertex already marked `st.settled[...]`
(`Si_fresh = [x for x in Si if not st.settled[x]]`); if nothing remains,
treat the batch as already-resolved (`Bp_i, Ui = Bi, []`) and continue. The
`L21` "unfinished remainder of `S_i`" check is likewise computed over
`Si_fresh`. This is the literal-est fix consistent with the stated `U_i`
disjointness/completeness invariants; it does not change any case where the
"derived invariant" actually holds.

## 4. The non-sorting checks (SPEC.md S7.b / acceptance criterion 3) vs the
   *default* `compute_params` at practically testable sizes

SPEC.md S7.b's mandatory check ("`is_globally_sorted(log) is False` on a
random `n = 4096`, `m ≈ 2n` digraph; if this fails, treat it as an
implementation bug, not a flaky test") and the acceptance criterion "20
random graphs with `n ≥ 500`, log NOT sorted in ≥ 15 cases" both **fail
under the default `compute_params`** — measured 1/20 unsorted, and the
single n=4096 fixture is sorted on every seed tried.

This was investigated to root cause and it is *not* an implementation
deviation. Instrumented runs show every settlement comes from `base_case`
and **zero** from the Algorithm 3 line-22 `W`-sweep — the only out-of-order
settlement source the algorithm has. The chain is structural:

1. `k = max(1, floor(log2(n2)^(1/3)))` (SPEC.md S5) equals **2 for every
   `n2 < 2^27`** — i.e. for every graph a test can realistically run.
2. With `k = 2` on the constant-degree transformed graph (out-degree ≤ 2),
   FindPivots' early exit `|W| > k|S|` fires on essentially every fresh
   frontier (measured 84/85 calls): one Bellman–Ford round from `S` relaxes
   ~2 new vertices per frontier vertex, immediately exceeding `2|S|`.
3. Early exit returns `P = S`, so *every* completed vertex lies in `T(P)`
   and is (re)discovered inside the recursive `D`-pipeline calls — whose
   settlement order is **provably monotone** (Lemma 3.10:
   `min_{x∈D} d(x) ≥ B'_{i−1}`; Remark 3.8: distances in `U_i` strictly
   precede those in `U_j`, `i < j`).
4. The line-22 `W`-sweep — settling vertices in **non-pivot** tight subtrees
   *after* deeper calls have settled larger distances, exactly the
   reordering S7.b's rationale describes — only ever has candidates when
   FindPivots takes the pivot branch (`P ⊊ S`), which (2) makes vanishingly
   rare at default parameters. Direct measurement confirms: every inversion
   observed in any run was a `W`-sweep settlement.

The pivot branch activates when frontiers are *re-processed* (relaxations
into already-complete regions fail, keeping `|W| ≤ k|S|`), which requires
small `bound_cap = k·2^(l·t)`, i.e. small `t`. Sweep over the parameter
space (20 graphs, `n ∈ [500, 2000]`, `m = 2n`): `t=2` → 20/20 unsorted,
`t=3` → 16/20, `t=4` → 0/20, default `t=5` → 1/20.

**Resolution**: `compute_params` itself stays exactly as SPEC.md S5
specifies (deviating there is forbidden). The two non-sorting tests
(`test_not_globally_sorted`, `test_not_dijkstra` in
`tests/test_verification.py`) monkeypatch `(k, t) = (2, 2)` — the value
SPEC.md S8.4 itself suggests for forcing the interesting regime ("force k, t
small by monkeypatching compute_params, e.g. k=2, t=2"); correctness is
parameter-independent (ALGORITHM.md S2). Under this regime the acceptance
criterion passes 20/20 (threshold 15). This is **not a weakening of the
test**: an implementation that degenerated into Dijkstra (the regression
S7.b exists to catch) settles in sorted order under *any* parameters and
still fails both tests. Marked `TODO(spec)` at `_small_params` in
`tests/test_verification.py`.

Open question for the spec authors: should `compute_params` cap `t` (e.g.
`t ≤ log2(n)/3`) so that the bound-cap truncation — and with it the
algorithm's distinctive batch-settlement behavior — is exercised at
practical sizes? As written, for all `n2 < 2^27` the default parameters put
the algorithm in a regime where it is observationally order-equivalent to
Dijkstra, even though its cost profile (criterion 4's ops/m table) is not.
