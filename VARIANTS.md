# VARIANTS — algorithm-level BMSSP variants

Research track: mathematical/structural variants of the BMSSP implementation
(Duan–Mao–Mao–Shu–Yin, arXiv:2504.17033v2) that are faster in practice while
provably preserving distance correctness. Companion to BENCHMARKS.md (which
benchmarks the *faithful* mainline) and ALGORITHM.md (the paper distillation
all lemma references below point into).

**Ground rules followed:**

- The mainline (`src/bmssp.rs`, `src/block_queue.rs`) is untouched; the
  differential gate still passes (`cargo test --test differential`, re-run
  after this change set: green).
- Every variant lives in `src/variants/` and is exposed as
  `shortest_paths(..., method="bmssp-<name>")`.
- Every variant passes `cargo test --release --test variants_correctness`:
  **520 property graphs** per variant (4 weight regimes × 130 rounds: floats
  with zeros, small-integer weights, all-unit weights, ~50%-zero weights;
  random sources; with/without connectivity backbone; n ≤ 250, self-loops and
  parallel edges included) with **bit-exact distances vs the production
  Dijkstra** and predecessor-consistency checks, plus a **10⁶-edge stress
  graph** (n = 250 000) per variant, also bit-exact.
- Settlement-order fidelity vs `_reference.py` is *not* required of variants
  (they legitimately change the order); it remains in force for the mainline.

All variants share a parameterized engine (`src/variants/engine.rs`), a fork
of the mainline recursion that keeps: the total order on labels
(Assumption 2.1), the `≤` relaxation rule (Remark 3.4), the settled-vertex
filter on pulled batches (AUDIT.md F3), and the Algorithm 3 loop structure.
Each variant is one `Config` away from the faithful algorithm, so the deltas
below are exactly what was measured.

---

## The variants

### 1. `bmssp-tuned` — (k, t) as free parameters

**What changed vs the paper.** The paper fixes `k = ⌊log^(1/3) n⌋`,
`t = ⌊log^(2/3) n⌋` to optimize the worst-case bound. This variant treats
them as free and ships the empirical optimum from a grid search
(`benchmarks/grid_kt.py`; raw surfaces in `benchmarks/results/grid_*.json`).

**Why correctness holds.** Verified against the proofs: no correctness lemma
constrains the *values* of k and t.

- Lemma 3.2 (FindPivots) is proved for arbitrary `k ≥ 1`: "if there are no
  more than k−1 edges from y to x on the path, x is complete after k
  relaxations; otherwise the tree rooted at y contains at least k vertices" —
  both halves hold for every positive k.
- Lemma 3.7's induction over Algorithm 3 uses t only through
  `M = 2^((l−1)t)` and the requirement `|S| ≤ 2^(lt)`, both of which hold for
  any `t ≥ 1` by construction (Pull returns ≤ M keys; the top level uses
  `L = ⌈log n / t⌉` so `2^(Lt) ≥ n`).
- k and t appear with specific values **only** in Lemma 3.12 / Remark 3.5 /
  Lemma 3.1's cost expression — the time bound, not correctness.

**Worst-case complexity.** With `k, t = Θ(1)` constants the bound from
Lemma 3.12 becomes `O(m·(k + t/k)·log n / t) = O(m log n)`-grade; the
`O(m log^(2/3) n)` form is recovered only by the paper's growing parameters.
Accepted: practice improves (below).

**Measured (k, t) surface.** Grid over k ∈ {1..8}, t ∈ {2..24}, time in
seconds, best run, transform engine:

| graph | paper (k,t) | paper time | best (k,t) | best time | gain |
|---|---|---:|---|---:|---:|
| random n=10⁵ | (2, 7) | 1.55 s | (8, 12) | 1.08 s | −31% |
| random n=10⁶ | (2, 8) | ~25 s | (8, 12) | 13.9 s | −45% |
| USA-road-d.NY | (3, 7) | 1.66 s | (8, 8)≈(8,12) | 1.08 s | −35% |

Shape of the surface: time falls monotonically in k up to ~8 (larger k means
FindPivots completes more vertices per call and certifies fewer pivots,
`|P| ≤ |W|/k`, so far fewer expensive `D.Insert`s); moderate t (8–12) beats
the paper's t; **t ≥ 16 is catastrophic** (5–40× slowdowns: `L` collapses to
2 with `M = 1` at the child level, so the structure degenerates into millions
of singleton pulls). The shipped table is `(8, 12)` for `n₂ ≥ 2^14`, paper
formula below that (sub-100-ms regime either way).

### 2. `bmssp-hybrid` — Dijkstra as the base-case oracle

**What changed vs the paper.** Algorithm 2 (BaseCase: singleton source,
truncation after k+1 settled vertices) is replaced by a **bounded
multi-source Dijkstra run to exhaustion** whenever `l ≤ D` or the pulled
frontier has `|S| ≤ B` vertices. Tunables D and B.

**Why correctness holds.** A bounded multi-source Dijkstra is a valid BMSSP
oracle: given Algorithm 3's requirement 2 (every incomplete v with d(v) < B
has its shortest path through a complete y ∈ S), initializing a heap with
`⟨x, d̂[x]⟩` for x ∈ S and relaxing under bound B settles exactly
`T_<B(S)` — the standard Dijkstra induction goes through because the first
incomplete-but-minimal vertex would need a strictly smaller complete ancestor
in the heap, which the precondition supplies. It returns `B' = B` and the
complete `U = T_<B(S)`, i.e. Lemma 3.1's "successful execution" branch, so
the parent's invariants (a)/(b) of Lemma 3.7 are maintained verbatim. What is
given up is Lemma 3.9's size control `|U| ≤ 4k·2^(lt)` at the swallowed
levels — a *complexity* device (it keeps subproblems balanced), not a
correctness one; the parent's own `|U| < k·2^(lt)` loop guard still bounds
its accumulation and termination is unaffected.

**Worst-case complexity.** The oracle sorts its subproblem:
`O(|U| log |U|)` per call, so the bound regresses to `O(m log n)` whenever
the oracle handles Θ(m)-sized subproblems (which the tuning *prefers* — see
below). 

**Measured (D, B) sweep** (`benchmarks/sweep_hybrid.py`, paper (k,t),
transform engine, best of 2):

| config | random n=10⁵ | USA-road-d.NY |
|---|---:|---:|
| D=0, B=0 (Dijkstra replaces only Algorithm 2) | 1.61 s | 1.17 s |
| D=1, B=0 | 1.07 s | 1.02 s |
| D=2, B=0 | 0.77 s | 0.89 s |
| D=0, B=32 | 0.59 s | 0.66 s |
| D=1, B=64 | 0.55 s | 0.70 s |
| **D=1, B=1024** (shipped default) | **0.54 s** | **0.64 s** |
| D=2, B=1024 | 0.59 s | 0.68 s |

The sweep is essentially monotone: the more of the recursion the Dijkstra
oracle swallows, the faster the run. That is the honest headline — the
recursion's per-vertex machinery never beats a heap at these sizes.

### 3. `bmssp-simpleq` — flat lazy-deletion heap instead of Lemma 3.3

**What changed vs the paper.** The block-based structure D (two block
sequences, per-block bound BST, median splits, quickselect pulls) is replaced
by a single binary heap with a `key → smallest live value` map and lazy
deletion (`src/variants/simple_queue.rs`).

**Why correctness holds.** Algorithm 3 only consumes the *semantic* contract
of Lemma 3.3, which the heap satisfies exactly: Insert keeps the smaller
value per key (the map is authoritative; stale heap entries are skipped on
pop); BatchPrepend is semantically Insert (its precondition — every value
smaller than current contents — needs no structural support when Pull pops in
global order); Pull returns the ≤ M smallest values and a separating bound x
with `max(S') < x ≤ min(remaining)` (the next live minimum; values are
pairwise distinct under Assumption 2.1's total order). Lemma 3.7 never looks
inside D.

**Which amortized bound is lost.** Lemma 3.3's `O(max{1, log(N/M)})` Insert,
`O(L·max{1, log(L/M)})` BatchPrepend and `O(|S'|)` Pull all become
`O(log N)` per element. Exactly regression #4 of ALGORITHM.md §7: a K-set
element re-prepended on every level of an ancestor chain now costs
`O(log N)` each time, so the worst case returns to **O(m log n)** — the
sorting barrier reappears inside the data structure. Accepted and measured.

### 4. `bmssp-lazypiv` — early-terminating FindPivots

**What changed vs the paper.** Algorithm 1 runs its Bellman-Ford loop for a
fixed k rounds. This variant stops after round `j < k` when the frontier
stops shrinking (`|W_j| ≥ |W_{j−1}|`, j ≥ 2), and lowers the pivot tree-size
threshold from k to j (the rounds actually run). Part (b) of the task —
relaxing only edges whose tail changed in the previous round — is **already
how Algorithm 1 is stated** (line 6 scans edges out of `W_{i−1}` only) and
how the mainline implements it; only the adaptive round count is new.

**Why correctness holds (Lemma 3.2).** The covering property is what the
caller needs: every x ∈ Ũ is either complete in W or has its shortest path
through a complete pivot y ∈ P. The paper's proof parameterizes verbatim by
the number of rounds: after j rounds, every x whose path has ≤ j−1 tight
edges below its last complete-in-S ancestor is complete and in W; otherwise
that ancestor roots a tight tree with ≥ j vertices and is kept (threshold j).
Early termination therefore only weakens the *size* bound `|P| ≤ |W|/k` to
`|P| ≤ |W|/j` — more pivots, more Insert traffic, never a lost vertex. The
`|W| > k|S|` early exit (P ← S) is kept unchanged.

**Worst-case complexity.** Unchanged in form; the `|P| ≤ |W|/k` term of
Lemma 3.12 degrades to `|W|/j` for calls that stop early.

**Measured.** Marginal: at matched (k,t) = (8,12) it gains 7% on random 10⁵
and ~1% on the road graph; at (2,7) (paper parameters, where there are no
rounds to cut — k=2 means the stop rule can never fire before round 2) and at
(16,12) the difference is inside run-to-run noise. An honest near-null
result: the interesting regime (large k) is exactly where `bmssp-tuned`
already sits, and there the saving is single-digit percent.

### 5. `bmssp-notransform` — skip the constant-degree transform (own idea)

**The proof-forced cost identified.** Section 2 of the paper transforms the
input so every vertex has in/out-degree ≤ 2: each vertex becomes a zero-weight
cycle with one node per incident edge endpoint. For the benchmark family
(m = 4n) this turns an (n, m) problem into (n₂ ≈ 2m = 8n, m₂ = 3m = 12n) —
**8× the vertices and 3× the edges before the algorithm starts**, plus
zero-weight cycle edges that every traversal must walk and that force the
hop-count tie-breaking into every comparison. BENCHMARKS.md's profile shows
the cost is spread across all phases (state arrays, sets, queue traffic scale
with n₂), making the transform the single largest constant-factor lever in
the implementation.

**Justification before implementing (which lemmas need the degree bound).**
Audit of every correctness lemma:

- Lemma 3.6, Lemma 3.7 (the main induction), Lemma 3.10, and the Algorithm 2
  Dijkstra argument never mention degree.
- Lemma 3.2's *covering* property is degree-free; the degree bound enters
  only its early-exit size claim "|W| = O(k|S|) since out-degrees are
  constant" and its O(k|W|) running time.
- Remark 3.5 (insertion count `N = O(k·2^(lt))` into D) uses constant degree
  — again a time bound.

So the transform is consumed exclusively by the **time analysis**; running
the identical recursion on the raw graph preserves correctness with k, t, L
computed from the original n. (The engine's W-set early exit still fires at
`|W| > k|S|` — on an unbounded-degree graph one round can overshoot that
threshold by up to the max degree before the check, which costs time, never
correctness.)

**Worst-case complexity.** `O(m log^(2/3) n)` is no longer claimable for
unbounded-degree inputs: a round of FindPivots costs the out-volume of W (up
to `Θ(m)` instead of `O(k|S|)`), and the per-level Insert accounting loses
the `N = O(k·2^(lt))` cap. For inputs with `m = O(n)` and bounded degree
skew it stays `O(m log^(2/3) n)`; in general it degrades toward
`O(k·m·log n / t + m·t)` — i.e. Dijkstra-grade with the paper's parameters.
For m = Θ(n) sparse benchmark graphs, the practical effect is purely the 8×
state shrink.

### `bmssp-fast` — the winning combination

`no-transform` ∘ `hybrid (D=1, B=1024)` ∘ `simple-queue` ∘ `tuned (k=1, t=12)`.
The four deltas are independent — each replaces a different component against
the same Algorithm 3 contracts (graph representation / base-case oracle /
queue / parameters) — so their correctness arguments compose. The whole
combination passes the same 520-graph + 10⁶-edge gate as every other variant.

Structurally, the phase profile (`examples/profile_fast.rs`) shows what the
tuned knobs actually do at run time: for a **single-source** run the root
call has |S| = 1 ≤ B = 1024, so the hybrid frontier rule fires *at the root*
and the entire run is **one bounded multi-source Dijkstra call** (b = ∞)
executing BMSSP's machinery-free path — zero FindPivots calls, zero queue
pulls, one oracle call covering ~98% of vertices. The recursion, the flat
heap, and FindPivots exist and are exercised by multi-pivot subproblems
(and by the correctness suite via small-n configurations), but the measured
single-source optimum is the configuration in which the Lemma 3.1 oracle
swallows everything. bmssp-fast is therefore *literally* the variant
ladder's endpoint made explicit: a Dijkstra run that carries BMSSP's
lexicographic `(len, hops, id)` labels and bound checks — the data, asked
for the fastest correct BMSSP instantiation, answered "Dijkstra in BMSSP
clothing". Every knob turned one notch further (t ≥ 17 at 10⁵, D = L) just
removes the clothing.

---

## Results matrix

> **Research-phase numbers — superseded for `bmssp-fast`.** This matrix is
> the variant *study*'s internal comparison, measured in one session before
> the low-level consolidation pass (see [the Consolidation
> section](#consolidation-low-level-engineering-pass-final) and
> **OPTIMIZATION.md**) and on a different harness than the final report. It
> is kept because the *relative* variant ranking it establishes is the
> study's actual finding. The **authoritative final wall-clock numbers** for
> `lt-dijkstra` / `lt-bmssp` / `bmssp-fast` are the median-of-5 matrix in
> **BENCHMARKS.md** (2026-06-13): there `bmssp-fast` is **1.4–1.9×** of
> Dijkstra on random graphs, 1.4–2.4× on Barabási–Albert, and 5.0× on the NY
> road graph — i.e. faster, and with a different ratio, than the
> pre-consolidation cells below. Where the two disagree, BENCHMARKS.md wins.

Same machine, build, and graph generation as BENCHMARKS.md (portable release
build, mimalloc; random family seeds `0xC0FFEE+i`, m = 4n, source 0; DIMACS
NY road graph). All implementations were re-timed in this session so ratios
are internally consistent; n = 10⁵/10⁶/DIMACS rows are medians of 3 runs
after 1 warmup, n = 10⁷ rows are single runs (run-to-run spread at that size
is ~2–3%). Distances of every cell were verified bit-exact against
lt-dijkstra (in-harness check, plus the 520-graph/10⁶-edge gate), except the
10⁷ cells of the four transform-based variants where only the gate applies
(the in-harness check would have doubled multi-minute runs). The mainline
10⁷ cell is BENCHMARKS.md's measurement with the same harness.

Time (× lt-dijkstra):

| impl | random 10⁵ | random 10⁶ | random 10⁷ | USA-road-d.NY |
|---|---:|---:|---:|---:|
| lt-dijkstra | 35.8 ms (1×) | 778 ms (1×) | 14.0 s (1×) | 32.8 ms (1×) |
| lt-bmssp (mainline) | 1.52 s (42×) | 21.8 s (28×) | 405 s (27×) | 1.59 s (48×) |
| bmssp-tuned | 1.05 s (29×) | 13.1 s (17×) | 243 s (17×) | 1.07 s (33×) |
| bmssp-hybrid | 579 ms (16×) | 9.0 s (12×) | 159 s (11×) | 700 ms (21×) |
| bmssp-simpleq | 1.42 s (40×) | 17.6 s (23×) | 337 s (23×) | 1.20 s (37×) |
| bmssp-lazypiv | 1.54 s (43×) | 19.8 s (25×) | 372 s (26×) | 1.68 s (51×) |
| bmssp-notransform | 133 ms (3.7×) | 1.99 s (2.6×) | 52.6 s (3.7×) | 311 ms (9.5×) |
| **bmssp-fast** | **108 ms (3.0×)** | **1.90 s (2.4×)** | **35.2 s (2.5×)** | **210 ms (6.4×)** |

Speedup over the mainline lt-bmssp:

| variant | random 10⁵ | random 10⁶ | random 10⁷ | NY road |
|---|---:|---:|---:|---:|
| tuned | 1.4× | 1.7× | 1.7× | 1.5× |
| hybrid | 2.6× | 2.4× | 2.6× | 2.3× |
| simpleq | 1.1× | 1.2× | 1.2× | 1.3× |
| lazypiv | **1.0× (none)** | 1.1× | 1.1× | **0.9× (slower)** |
| notransform | 11.4× | 11.0× | 7.7× | 5.1× |
| **fast (combination)** | **14.1×** | **11.5×** | **11.5×** | **7.6×** |

Negative results, stated plainly:

- **lazy-pivots is a wash.** At the paper's k = 2–3 there are no rounds to
  cut, and at the tuned k = 8 it saves 0–7% (inside noise on the road
  graph, where it is marginally *slower*). Not worth its complexity alone;
  excluded from the combination because the combination's optimum sits at
  k = 1, where it is inert by construction.
- **simple-queue alone is small.** 10–25% — the block structure's overheads
  are real but are not the leading constant; it earns its place only inside
  the combination, where the queue sees far less traffic and simplicity wins.
- **No variant changes BENCHMARKS.md's verdict.** Even bmssp-fast loses to
  this crate's Dijkstra everywhere measured — by 2.4–6.4× in these
  research-phase cells, and by **1.4–5.0×** in the final post-consolidation
  matrix (BENCHMARKS.md). The variants cut the faithful constant by more than
  an order of magnitude (29× → 1.6× at 10⁶, final); they do not move the
  asymptotic crossover into reach. And the tuning gradients all point the
  same way: every knob that makes a variant faster makes it more
  Dijkstra-like (more oracle, fewer rounds, flatter queue, fewer levels).
  The measured optimum is the minimal BMSSP instantiation that is not
  literally Dijkstra — and the profile (OPTIMIZATION.md) shows that for a
  single-source run even *that* reduces to one bounded Dijkstra call.

## Consolidation: low-level engineering pass (final)

After the variant study froze, a low-level engineering pass (flat arenas,
epoch stamps instead of clearing, no hot-loop allocations, feature-gated
instrumentation) was applied to the shared engine, profile-driven via
`examples/profile_fast.rs` (`--features phase-timer`). Gate for every step:
the full `variants_correctness` suite (520 graphs + 10⁶-edge stress per
variant, bit-exact distances), one commit per accepted change. **The full
record — per-change rationale, the rejected/skipped list, the profile, and
the distinction from the mainline optimization pass — is OPTIMIZATION.md;
this section summarises it.**

Applied (bmssp-fast, n = 10⁶ random m = 4n, phase-timer build):

| # | change | total after | Δ |
|---|---|---:|---:|
| 0 | baseline (engine as of the variant study) | 2.13 s | — |
| 1 | oracles drop hash bookkeeping: the `best: FxHashMap<u32, Key>` was provably `key(v)` of the current labels (recompute on pop instead); the per-call popped set becomes an epoch-stamped `Vec<u32>` (no clearing) | 1.60 s | −25% |
| 2 | SoA 4-ary `KeyHeap` replaces `BinaryHeap<Reverse<(Key, u32)>>` (heap entries always have `Key.id == vertex`, so `(len, hops, vertex)` in three parallel arrays reproduces `Key` order exactly) | 1.32 s | −18% |
| 3 | `(dhat, hops)` fused into one 16-byte `Label` array (one cache line per relax-target read; `pred` separate, touched only on ties/successes) | 1.21 s | −2% (7-run medians) |

Rejected after measurement (both within run-to-run noise): software
prefetch of `lab[v]` in the oracle relax loop; iterator-zip edge scan to
elide bounds checks. Skipped with justification: scratch pools for the
Algorithm-3 body buffers and a FlatHeap arena (the profile shows
`q_pulls = 0` for single-source bmssp-fast — that code never runs);
de-hashing FindPivots (0.0% of time); an explicit recursion stack (the
"recursion" is one root call). Instrumentation (phase timers, counters) is
compiled out unless `--features phase-timer`; engine invariant checks are
`debug_assert!` unless `--features verify`.

Net: **2.13 s → 1.21 s (−43%)** for bmssp-fast; the same engine serves all
variants, so the hybrid/notransform rows above also improved (their table
entries predate this pass — the final authoritative matrix is in
BENCHMARKS.md). The remaining gap to `lt-dijkstra` (~1.5× at 10⁶) is the
contract itself: 16-byte lexicographic labels vs 8-byte distances, i64 hop
arithmetic in every comparison, 20-byte vs 12-byte heap entries, and the
`vkey < B` bound test — the price of being a BMSSP oracle rather than plain
Dijkstra.

## Recommendation

**Ship `bmssp-fast`** (implemented as `method="bmssp-fast"`,
`src/variants/fast.rs`): no-transform + hybrid Dijkstra oracle (D=1, B=1024)
+ flat lazy-deletion heap + (k=1, t=12). It passes the full correctness
suite (520 property graphs + 10⁶-edge stress, bit-exact distances and
consistent predecessors). After the consolidation pass below, the final
matrix (BENCHMARKS.md, 2026-06-13) has it 12.6–53× faster than the mainline
across every graph family and size measured, narrowing the gap to Dijkstra
from 26–128× to 1.4–5.0×.

If a single delta must be chosen instead, choose **`bmssp-notransform`**:
one conceptual change, 5–11× of the win, and the clearest correctness story
(the degree bound is provably absent from every correctness lemma).

Keep the mainline `method="bmssp"` as the faithful, settlement-order-pinned
reference implementation; the variants are not substitutes for it in the
differential-gate role.

---

## Reproducing

```bash
# correctness gate (all variants, 520 graphs each + 1e6-edge stress)
cargo test --release --test variants_correctness
# mainline gates, unchanged
cargo test --release --test differential

# benchmarks
.venv/Scripts/maturin develop --release
.venv/Scripts/python benchmarks/variants_bench.py --sizes 100000,1000000 \
    --include-mainline --dimacs --runs 3 --tag final_small
.venv/Scripts/python benchmarks/variants_bench.py --sizes 10000000 \
    --include-mainline --runs 1 --warmup 0 --tag final_1e7
# parameter surfaces
.venv/Scripts/python benchmarks/grid_kt.py --graph random:100000 --variant tuned
.venv/Scripts/python benchmarks/sweep_hybrid.py 100000 dimacs
.venv/Scripts/python benchmarks/sweep_lazypiv.py 100000 dimacs
```
