# AUDIT.md — implementation audit against the paper

Audit of `python/logtwothirds/_reference.py` against **the paper**
(`paper.pdf`, Duan–Mao–Mao–Shu–Yin, arXiv:2504.17033v2, "Breaking the
Sorting Barrier for Directed Single-Source Shortest Paths"), treating
`ALGORITHM.md` as a secondary source that was itself audited. Date:
2026-06-12.

**Verdict: zero blocker findings.** Two minor findings fixed (F4 code, F14
documentation), the rest are confirmed-correct or documented engineering
decisions. All 136 tests pass after the fixes; all four QUESTIONS.md items
are answered with paper citations (answers appended in `QUESTIONS.md`).

Method:

- Full read of the paper (text extraction `paper_text.txt`, all 17 pages,
  spot-checked against `paper.pdf`; pseudocode **indentation** verified from
  PDF glyph x-coordinates where text extraction loses it).
- Line-by-line cross-reference of `find_pivots` / `base_case` / `bmssp` /
  `BlockDS` against Algorithms 1–3 and Lemma 3.3.
- Invariant stress-check (`audit_stress.py`): `bmssp` wrapped with
  pre/postcondition asserts checked against a Dijkstra oracle on the
  transformed graph; 330 random graphs (n ≤ 900, zero-weight edges and
  forced ties, (k,t) ∈ {(1,1),(1,2),(2,1),(2,2),(3,2)} and defaults);
  77,396 checked calls, **no assertion failed**.
- Counterfactual experiment (`audit_nofilter.py`): paper-literal Algorithm 3
  (QUESTIONS.md item-3 filter removed) on the same corpus, to classify that
  deviation (see F3).
- Full test suite + empirical complexity test (see "Test results").

---

## Findings

| # | Severity | Component | Finding | Action |
|---|----------|-----------|---------|--------|
| F1 | ok | `BlockDS.pull` / `bmssp` L10 | Pull return order `(S′, x)` vs pseudocode `B_i, S_i ←` — paper uses both (Lemma 3.3 vs Alg. 3 L10); cosmetic | none (QUESTIONS.md #1 answered) |
| F2 | ok | all three bound gates | Bound comparisons re-keyed by target vertex (`vkey`), relax test keyed by predecessor — exactly Assumption 2.1's reversed tuple ⟨l, α, v_α, …, v_1⟩ (p. 4) | none (QUESTIONS.md #2 answered) |
| F3 | ok | `bmssp` settled-filter | Deviation from literal Alg. 3, compensating a real gap in the paper's presentation (stored-value-vs-current-d̂ in Lemmas 3.6/3.7/3.10); verified both ways empirically | none (QUESTIONS.md #3 answered; ALGORITHM.md corrected, see F14) |
| F4 | **minor — fixed** | `BlockDS.pull` | Emptied D1 blocks (and bounds) were never removed; paper's Delete step removes the bound when a block empties (Lemma 3.3 proof), keeping Pull's prefix scan O(M) amortized. Complexity-only; correctness unaffected | fixed: prune leading empty D1 blocks in `pull()` |
| F5 | ok | `BlockDS` bounds | Sorted list + `bisect` instead of red-black tree — SPEC.md §4 sanctions it explicitly (constants/worst-case only) | none |
| F6 | ok | `BlockDS._split_d1` | Split halves of an (M+1)-element block are ⌊(M+1)/2⌋ / ⌈(M+1)/2⌉; the paper's "at most ⌈M/2⌉" is itself unsatisfiable for even M (its own prose says "about ⌈M/2⌉"). Both halves ≤ M; Θ(M)-per-split preserved | none |
| F7 | ok | `bmssp` | `M = 2^{(l−1)t}` capped at n — SPEC.md §5 sanctions; any M ≥ n behaves identically | none |
| F8 | ok | transform | One cycle vertex per incident **edge occurrence** vs the paper's per-**neighbor** x_vw (p. 3) — SPEC.md §2 sanctions; in/out-degree ≤ 2 and distances preserved (verified by `test_transform.py`) | none |
| F9 | ok | `base_case` | Lazy-deletion heap instead of Insert/DecreaseKey — SPEC.md §6 sanctions; stale pops are skipped and do not count toward \|U₀\| | none |
| F10 | minor (documented) | `sssp` | Raises `IndexError` for out-of-range source; SPEC.md §9 checklist says `ValueError`. Pinned by `test_sssp.py` / `test_edge_cases.py` and consistent with the package API docstring (`__init__.py`). Spec-vs-impl mismatch only, not a paper matter — flag for the spec authors | none (deliberate; do not break the pinned tests) |
| F11 | ok | `find_pivots` L12 | Early-exit threshold `k·len(S)` assumes S duplicate-free; holds at every call site (asserted over all 77,396 stress calls) | none |
| F12 | ok | `bmssp` L22 | Lemma 3.9's lower bound (B′ < B ⟹ \|U\| ≥ k·2^{lt}) has the known corner where a **partial child empties D** (ALGORITHM.md §9 uncertainty): observed 2/77,396 calls in stress runs. Correctness unaffected (U = T_{<B′}(S) still holds); the paper's Lemma 3.9 proof ("If D is empty, the algorithm succeeds") glosses it. Note: `tests/test_bmssp.py::check_bmssp` asserts the lower bound strictly — safe with its fixed seeds, but would mis-fire on a fixture hitting this corner | none (documented) |
| F13 | ok | `find_pivots` L11–14 | Early exit **inside** the k-round loop — confirmed from PDF glyph x-coordinates (lines 11–12 at the `for`-body indent, 13–14 one level deeper), matching ALGORITHM.md's note and the implementation | none |
| F14 | **minor — fixed** | ALGORITHM.md §3.2 | The "derived invariant" (stored value in D always equals current d̂[v]) is **false** in the F3/Q3 scenario; this is the very assumption whose failure motivates the settled-filter | fixed: §3.2 bullet rewritten as a near-invariant with the correction and pointer here |

No other discrepancies: Algorithms 1–3 are otherwise implemented line-for-line
(paper line numbers annotated in the code), including every load-bearing
subtlety — the `≤` relaxation tests (Remark 3.4, p. 8); relaxation **un**gated
by B in Algorithms 1 and 3 but gated in Algorithm 2 line 8; Alg. 1 L9 gating
only W-membership; Alg. 3 L7's `P = ∅ ⟹ B′₀ = B` footnote; L21 re-prepending
the unfinished part of S_i at **current** d̂; L22's `min{B′_i, B}` and the
W-completion sweep; Alg. 2's truncation excluding exactly the unique maximum.

### F3 in brief (full analysis in QUESTIONS.md #3)

The paper's Lemma 3.7 proof applies Lemma 3.6 with X := S_i split by
*current* d̂, while Pull separates by *stored* values; Lemma 3.10's proof
likewise reads "min_{v∈D} d̂[v] ≥ B′_{i−1}" off the construction. Both
implicitly assume stored = current. That fails when a vertex v sitting in an
ancestor's D (stale value ≥ B_i) has its d̂ improved deep inside the i-th
child call and is returned in U_i; the Remark-3.4 re-relaxation then lands
below B′_i, failing both bucket windows (lines 17/19), so the stale entry
survives and is pulled later as an already-settled vertex. Counterfactual
run (`audit_nofilter.py`, 320 graphs): paper-literal code suffers 235
U_i-disjointness violations and 1 B′-monotonicity regression — but **0
output mismatches** (answers are read from d̂, §4.4). The settled-filter
restores Remark 3.8's disjointness, which Remark 3.5 explicitly uses to
bound the insertion count N = O(k·2^{lt}). Classification: **acceptable
engineering decision** repairing a paper-presentation gap; not an
implementation bug.

---

## "What this is NOT" (ALGORITHM.md §7) — regression sweep

1. **Pull degenerating to a heap / M = 1 everywhere** — absent.
   `BlockDS.pull` returns an unsorted batch of up to `M = 2^{(l−1)t}` keys
   plus only a separating bound; M = 1 occurs only where the paper says so
   (l = 1). Guarded by `test_not_globally_sorted` / `test_not_dijkstra`
   (pass) and the pull unit tests.
2. **FindPivots skipped / P ≡ S** — absent. Tight-forest construction and
   the ≥ k subtree-size pivot rule are implemented (L15–16); `P = S` only on
   the L12 early exit, as in the paper. Guarded by
   `test_find_pivots.py` branch-exact size assertions.
3. **k-round loop run to exhaustion** — absent. Exactly `range(k)` rounds
   with the in-loop `|W| > k|S|` bail-out (placement verified against the
   PDF, F13).
4. **BatchPrepend replaced by repeated Insert** — absent. L21 uses
   `D.batch_prepend` with the two-sequence D0/D1 design;
   `ds_prepend_items > 0` on large graphs (verified by the counter).
5. **Partial-execution cutoff removed** — absent. `while len(U) <
   k·2^{lt}` guard and the `k+1` BaseCase cap are both present; Lemma 3.9
   size assertions in `test_bmssp.py` pass.

Also checked the two extra watch items: relaxation is *not* gated by B
outside BaseCase, and no relaxation test uses strict `<` where the paper has
`≤`.

---

## Recursion invariants (assert-checked at call boundaries)

`audit_stress.py` wraps every `bmssp` call (including recursive ones) and
asserts, against a Dijkstra oracle on the transformed graph:

- preconditions: `|S| ≤ 2^{lt}`, S duplicate-free, `key(x) < B` for x ∈ S,
  d̂ sound at entry;
- postconditions: `B′ ≤ B`; `|U| ≤ 4k·2^{lt}` (Lemma 3.9); U duplicate-free;
  every u ∈ U complete (`d̂[u] = d(u)`) with `key(u) < B′` (Lemma 3.7);
  d̂ non-increasing and sound across the call;
- per run: final d̂ equals the oracle on the transformed graph; settle log
  values equal true distances, one event per vertex; mapped-back answers
  equal the oracle on the original graph.

Built-in production asserts exercised throughout (`__debug__` on): U_i
pairwise disjointness, BatchPrepend's strict-precondition check, Pull
non-emptiness, `B′ ≤ B`.

Result: **330/330 runs, 77,396 calls, zero failures.** Statistics: 387
pulled batches contained an already-settled vertex (the F3 filter is
load-bearing, not dead code); 2 occurrences of the F12 Lemma-3.9 corner.

---

## Test results

- `pytest -q tests` → **136 passed** (after fixes F4/F14; also green before
  them).
- Empirical complexity (`test_verification.py::test_empirical_complexity`,
  m ≈ 2n, fixed seed): normalized cost
  `r(n) = ops_total / (m′·log2^{2/3} n′)` =
  2.540 (n=2¹⁰), 2.529 (2¹²), 2.311 (2¹⁴), 2.242 (2¹⁶) — flat-to-decreasing,
  consistent with O(m log^{2/3} n); spread 1.13 ≪ threshold 4.
- Non-sorting checks pass under the (k,t) = (2,2) verification regime
  (justified in QUESTIONS.md #4: the paper's parameter choice is a
  complexity matter only; a Dijkstra-degenerate implementation fails the
  check under *any* parameters).

## Changes applied by this audit

1. `python/logtwothirds/_reference.py` — `BlockDS.pull`: prune leading
   emptied D1 blocks and their bounds (F4; paper Lemma 3.3 "Delete").
2. `ALGORITHM.md` §3.2 — corrected the overstated "derived invariant" (F14).
3. `QUESTIONS.md` — all four items answered with paper citations; file
   marked closed.
4. New: `AUDIT.md` (this file), `audit_stress.py`, `audit_nofilter.py`
   (reproducible evidence for F3/F12 and the invariant checks).
