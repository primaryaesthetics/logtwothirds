# Open questions / spec ambiguities

> **Status: CLOSED (audit, 2026-06-12).** Every item below has been audited
> against the paper (`paper.pdf`, arXiv:2504.17033v2) and answered in an
> **Audit answer** block appended to the item. Verdicts and evidence are in
> [AUDIT.md](AUDIT.md).

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

**Audit answer — resolution confirmed, purely cosmetic (severity: ok).**
The paper itself uses both orders: Lemma 3.3 (p. 6–7) specifies the operation
as "**Pull** Return a subset S′ of keys where |S′| ≤ M associated with the
smallest |S′| values **and** an upper bound x …", i.e. (S′, x), while
Algorithm 3 line 10 (p. 10) binds "B_i, S_i ← D.Pull()". The lemma is the
data-structure contract; the pseudocode binding is presentation. Any
consistent choice is faithful; the implementation is consistent throughout.

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

**Audit answer — resolution is exactly what the paper prescribes
(severity: ok; no ALGORITHM.md error, but a clarification was warranted).**
The paper's total order (Assumption 2.1, p. 4) keys a path by the tuple
"⟨l, α, v_α, v_{α−1}, …, v_1⟩ (note that the sequence of vertices is
**reversed** in the tuple)" — i.e. after length and hop count, the first
vertex compared is the path's **endpoint** v_α. The candidate produced by
relaxing edge `(u, v)` is a path *ending at v*, so its tuple is
⟨d̂[u]+w_uv, hops[u]+1, **v**, u, …⟩: any comparison against a bound (bounds
are values in this same path space — e.g. Algorithm 2 line 17 sets
`B′ ← max_{v∈U₀} d̂[v]`, an endpoint-keyed label) is decided at the third
component by `v`, never by `u`. The predecessor `u` is consulted only when
the first three components tie, which can happen only against another path
ending at `v` — exactly the relaxation test, matching the paper's
"Relaxing an edge (u,v): If u ≠ Pred[v], even if there is a tie in l and α,
it suffices to compare between u and Pred[v]" (p. 4). So: relax test →
4th-component (`u` vs `Pred[v]`) comparison; bound test → endpoint-keyed
`vkey`. ALGORITHM.md §1.3 already places all bounds in the
`(d̂[v], hops[v], v)` key space, consistent with this; it just never spelled
out the re-keying at the three gate sites.

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

**Audit answer — your scenario is real; it is a gap in the paper's
presentation, not an implementation bug; the filter is the right call
(severity: ok — acceptable engineering decision; ALGORITHM.md's "derived
invariant" was overstated and has been corrected).**

*Where the paper's proof leans on the missing invariant.* Lemma 3.7's proof
(p. 12) applies Lemma 3.6 with "X := S_i, Y := D_i \ S_i, and B̄ := B_i" —
but Lemma 3.6 (p. 12) splits S by the **current** label, "X = {x ∈ S :
d̂[x] < B̄}", whereas Pull (Lemma 3.3) separates by the **stored** values.
Likewise Lemma 3.10's proof (p. 13–14) opens with "From the construction of
D, immediately before the i-th iteration of Algorithm 3, min_{v∈D} d̂[v] ≥
B′_{i−1}" — stored values do satisfy this, but the claim is about current
d̂[v]. Both steps implicitly assume *stored value = current d̂[v]*.

*Why that assumption can fail.* Insert a vertex v into this call's D at
iteration j (Alg. 3 line 18, value ∈ [B_j, B)). Suppose v's true shortest
path runs through u* with d(u*), d(v) ∈ [B′_{i−1}, B′_i) for a later
iteration i; v stays unpulled (stored value ≥ B_i). The i-th child call
settles u* and, deep inside its own recursion, relaxes (u*, v) — updating
d̂[v] = d(v) without touching this call's D — and returns v in U_i (child
postcondition, Lemma 3.7: U_i = T_{<B′_i}(S_i), and v ∈ T(S_i) via u*).
Back at this level, the Remark 3.4 re-relaxation of (u*, v) fires with
equality, but d̂[u*]+w_{u*v} = d(v) < B′_i fails *both* windows on lines
17 and 19, so the keep-min rule never sees the update: the stale entry
(value ≥ B_i > d(v)) survives and is pulled in a later iteration — an
already-settled vertex in S_{i′}, exactly your scenario.

*Empirical confirmation* (`audit_stress.py`, `audit_nofilter.py`; 330/320
random runs incl. zero-weight ties, (k,t) ∈ {(1,1),(1,2),(2,1),(2,2),(3,2)}
and defaults): with the filter, 387 of 77,071 pulled batches contained a
settled vertex, and every checked invariant held (U_i disjointness, U
complete vs. oracle, B′ ≤ B, |U| ≤ 4k·2^{lt}, d̂ sound/monotone, final
distances = Dijkstra oracle). With the filter **removed** (paper-literal
Algorithm 3): 235 U_i-disjointness violations and 1 B′-monotonicity
regression in 320 runs — yet **0 output mismatches**, because re-settling a
complete vertex cannot change d̂.

*Verdict.* The paper-literal algorithm still computes correct distances
(answers are read from d̂, §4.4/§6.6); what breaks without the filter are
Remark 3.8's disjointness and Lemma 3.10's monotonicity — which Remark 3.5
explicitly uses ("the total number of insertions N is O(k2^{lt}), because
of … the disjointness of U_i's") and Lemmas 3.9/3.12 build on. The filter
restores those invariants at O(1) per pulled key and drops only vertices
whose out-edges were already relaxed at every level up to the common
ancestor when their U_i bubbled up (Alg. 3 lines 12–21), so nothing is
lost. Keep it.

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

**Audit answer — resolution endorsed; the monkeypatch does not weaken the
check (severity: ok).** The paper fixes `k := ⌊log^{1/3}(n)⌋,
t := ⌊log^{2/3}(n)⌋` (p. 5, §3.1) solely to obtain Theorem 1.1's
O(m log^{2/3} n) bound; no correctness lemma (3.2, 3.6, 3.7, 3.9, 3.10)
uses the values, so any k, t ≥ 1 is sound — which `test_correctness_
independent_of_k_t` and the audit stress runs verify. Your structural
diagnosis matches the paper exactly: within one call the batch pipeline is
monotone (Lemma 3.10; Remark 3.8 / p. 10 item 6: "For i < j, distances to
vertices in U_{y_i} are smaller than distances to vertices in U_{y_j}"),
so the only out-of-order settlements are the line-22 W-additions ("Finally,
before we end the sub-routine, we update U to include every vertex x in the
set W returned by FindPivots with d̂[x] < B′", p. 9), and those exist only
when FindPivots takes the pivot branch (line 16) rather than the early exit
(lines 12–14, "if |W| > k|S| then P ← S") — which, at k = 2 on a
degree-≤2 transformed graph, one Bellman–Ford round nearly always triggers.
A Dijkstra-degenerate implementation settles sorted under *all* parameters,
so the (2,2) regime strictly strengthens the discriminating power of the
test. On the open question: yes — recommend the spec authors cap `t` for
*verification* purposes only (e.g. a separate verification-params hook or
`t ≤ max(2, ⌈log2(n)/3⌉)` in tests); `compute_params` itself must stay as
specified, since it is what Theorem 1.1's bound (and the S7.a empirical
check, which passes: r(n) ≈ 2.54, 2.53, 2.31, 2.24 for n = 2^10..2^16,
flat-to-decreasing) is about.
