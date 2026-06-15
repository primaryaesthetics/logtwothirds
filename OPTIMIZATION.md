# OPTIMIZATION.md — low-level engineering pass on `bmssp-fast`

Research record of the consolidation pass that follows the VARIANTS.md study:
the algorithm-level winner `bmssp-fast` (and the shared variants engine it
rides on, `src/variants/engine.rs`) tightened with low-level, behaviour-
neutral engineering. Companion to:

- **VARIANTS.md** — the algorithm-level variant study that produced
  `bmssp-fast` (no constant-degree transform + bounded multi-source Dijkstra
  oracle (D=1, B=1024) + flat lazy-deletion heap + tuned (k=1, t=12)). Has a
  short "Consolidation" section that summarises this file.
- **BENCHMARKS.md** — the final cross-implementation matrix (median of 5,
  fixed seeds) and the verdict. Its own "Optimizations: proposed, applied,
  rejected" section is a *different* pass on a *different* engine: the
  **mainline** faithful `bmssp` (`src/bmssp.rs`, 49.8 s → 27.2 s at n=10⁶,
  gated by the Step E differential test). Do not conflate the two — see
  [Scope](#scope-two-distinct-optimization-passes).

## Scope: two distinct optimization passes

| | mainline pass (BENCHMARKS.md) | this pass (OPTIMIZATION.md) |
|---|---|---|
| engine | `src/bmssp.rs` + `src/block_queue.rs` (faithful) | `src/variants/engine.rs` (variants), config = `bmssp-fast` |
| gate | Step E differential (200 graphs, bit-exact distances **and** settlement order vs the Python reference) | `variants_correctness` (≥520 property graphs + 10⁶-edge stress per variant, **bit-exact distances** + predecessor consistency; settlement order deliberately not pinned) |
| changes | FxHash, base-case scratch reuse, pull union buffer, mimalloc | hash-free oracles + epoch stamps, SoA `KeyHeap`, fused `Label` |
| result | 49.8 s → 27.2 s (−45%) at n=10⁶ | 2.13 s → 1.21 s (−43%) at n=10⁶ |

The mainline stays frozen as the faithful, settlement-order-pinned reference
engine; this pass never touches it. The two passes are complementary, not
alternatives: they optimise different code under different correctness bars.

## Rules followed

- **One change at a time, each its own commit.** After every change the full
  `cargo test --release --test variants_correctness` suite (13 tests: ≥520
  property graphs over four weight regimes + a 10⁶-edge stress graph per
  variant, distances bit-exact vs the production Dijkstra, predecessors
  consistent) must stay green. It did, at every step.
- **Behaviour-neutral only.** No change may alter a distance, a predecessor,
  an RNG draw, or the set of vertices returned. Every change below is a
  representation or bookkeeping change that provably preserves the values and
  the operation order the engine already produced.
- **Instrumentation stays out of the default build.** Phase timers and
  counters compile to nothing unless `--features phase-timer`; engine
  invariant checks are `debug_assert!` unless `--features verify` upgrades
  them to hard asserts. Release hot loops carry zero instrumentation.

## Per-change results

n = 10⁶, m = 4n uniform random (seed `0xC0FFEE`), portable release build with
`phase-timer`, via `examples/profile_fast.rs`:

| # | commit | change | total | Δ |
|---|---|---|---:|---:|
| 0 | (49.8 s baseline of the *mainline* pass is unrelated) | engine as of the variant study | 2.13 s | — |
| 1 | `22c88e4` | oracles drop hash bookkeeping; epoch-stamped popped-set | 1.60 s | −25% |
| 2 | `aabf7cd` | SoA 4-ary `KeyHeap` replaces `BinaryHeap<Reverse<(Key,u32)>>` | 1.32 s | −18% |
| 3 | `19ebabe` | fuse `(dhat, hops)` into one 16-byte `Label` array | 1.21 s | −2% |

(Commit `6ade296` precedes these: it adds the feature-gated timers/counters,
the `verify` flag, and `examples/profile_fast.rs` itself — measurement
infrastructure, no hot-path change.)

**Net: 2.13 s → 1.21 s, −43%** at n=10⁶, the suite green throughout.

### 1. Hash-free oracles + epoch-stamped membership (`22c88e4`)

The two heap oracles (`dijkstra_base`, the paper `base_case`) each kept a
`best: FxHashMap<u32, Key>` of the smallest live value per vertex and an
`in_u0: FxHashSet<u32>` of popped vertices, both cleared per call.

- **`best` was redundant.** Every value it ever held was, by construction,
  `key(v)` for the *current* `dhat`/`hops` — both writers (`relax_bounded`
  and the seed loop) store exactly `key(v)`. So the pop-time staleness check
  `best[u] == popped_key` is identical to recomputing `key(u)` from the live
  label arrays and comparing. The map is gone; the check recomputes.
- **`in_u0` became an epoch-stamped `Vec<u32>` on the engine** (`pop_stamp`,
  `pop_epoch`): "popped during this oracle call" is `pop_stamp[v] ==
  pop_epoch`; a new call just bumps the epoch (no clearing, no hashing, no
  per-call allocation).

U-set semantics are unchanged, including the subtlety that a vertex re-settled
on a tie stays in U. Biggest single win (−25%): the oracle is the entire hot
path (see the profile), and it was spending its time in hash lifecycles.

### 2. Structure-of-arrays 4-ary `KeyHeap` (`aabf7cd`)

The oracles used `std::collections::BinaryHeap<Reverse<(Key, u32)>>`, i.e. a
binary heap of 20-byte `(Key{f64,i64,i64}, u32)` entries. Replaced with a
purpose-built 4-ary min-heap storing `(len: f64, hops: i64, vertex: u32)` in
three parallel `Vec`s. Two facts make this exact:

- **Heap entries always satisfy `Key.id == vertex`.** Both push sites use the
  target vertex's own key, so the third tuple field is the vertex id and need
  not be stored separately.
- **The comparison is byte-for-byte `Key`'s order.** `KeyHeap::less` does
  `total_cmp` on `len`, then `hops`, then `vertex` — exactly `Key`'s
  `Ord`. Pop order is therefore identical to the old heap's, and equal entries
  are bit-identical and interchangeable.

Same layout idea as `src/dijkstra.rs`'s heap: 4-ary fan-out shortens sift
chains, and sifting reads the dense `len` array first, touching `hops`/`vertex`
only on float ties. −18%.

### 3. Fused 16-byte `Label` (`19ebabe`)

`dhat: Vec<f64>` and `hops: Vec<i64>` became one `Vec<Label{len: f64, hops:
i64}>`. The oracle relax loop reads *both* fields for every scanned edge; one
16-byte-aligned slot now serves both reads from a single cache line (a label
never straddles a line). `pred` stays a separate array — it is read only on
full `(len, hops)` ties and written only on successful relaxations, so fusing
it in would pollute the line for no gain. Same values, same operations, same
order. −2% (within run-to-run noise at one notch, kept because it is free and
strictly reduces memory traffic; 7-run medians confirm the direction).

## Rejected after measurement

Both implemented, A/B-benchmarked over 7 runs each, and reverted as
within-noise:

- **Software prefetch of `lab[v]`** in the oracle relax loop (the
  `dijkstra::prefetch_neighbors` trick). The random `lab[v]` reads are the
  loop's bottleneck, but at m=4n the out-degree is ~4, too few edges per
  vertex for the prefetch distance to hide latency; medians moved within
  ±2%. Reverted.
- **Iterator-zip edge scan** (`indices[start..end].iter().zip(weights…)`) to
  elide bounds checks. LLVM already removes the checks in the indexed form;
  no measurable change. Reverted.

## Skipped, with justification

Not attempted, because the profile shows the work they target does not exist
in `bmssp-fast`:

- **Scratch pools for the Algorithm-3 body buffers** (`si_fresh`/`kk`/
  `prepend`) and a **FlatHeap arena**: the profile shows `q_pulls = 0` — for
  a single-source `bmssp-fast` run this code never executes (see below).
- **De-hashing FindPivots' `w_set`/`children`**: `find_pivots` is 0.0% of the
  run (zero calls).
- **An explicit recursion stack**: the recursion never recurses (depth 1).
- **Epoch-stamping the variants' remaining `HashSet`s** (`u_set`, etc.) — the
  same idea the mainline pass rejected for its O(n₂) memory cost — would only
  touch code paths that, again, do not run in the tuned `bmssp-fast`
  configuration.

## The load-bearing finding: `bmssp-fast` is one bounded Dijkstra call

The profile is not "Dijkstra-oracle-dominated"; it is *Dijkstra-oracle-only*.
At the tuned parameters (k=1, t=12, so L=2) the hybrid frontier rule
(`|S| ≤ B = 1024`) is already satisfied by the **root** call's singleton
frontier `{source}`, so the very first thing the engine does is take the
bounded multi-source Dijkstra branch with `b = ∞` and run it to exhaustion.
There is no second iteration.

`examples/profile_fast.rs` at n=10⁶ (after this pass):

```
n=1000000 m=4000000 | n_inner=1000000 k=1 t=12 L=2
total     1.240 s
  dijkstra oracle (hybrid base)              1.231 s   99.3%
  finalize (dist + pred recovery)            0.003 s    0.2%
  bmssp body (unattributed)                  0.007 s    0.5%
  find_pivots / base_case / pull / prepend   0.000 s    0.0%
counters: edge_scans=3921495 relaxations=0 q_inserts=0 q_pulls=0
          q_pulled_keys=0 q_prepend_items=0 oracle_calls=1
          oracle_settled=980297 findpivots_calls=0 basecase_calls=0
          bmssp_calls=1
```

`bmssp_calls=1`, `oracle_calls=1`, `findpivots_calls=0`, `q_pulls=0`: the
entire run is **one** bounded multi-source Dijkstra over the original graph,
carrying BMSSP's lexicographic `(len, hops, id)` labels, its `≤` relaxation
rule (Remark 3.4), and its `vkey < B` bound test. The FindPivots pass, the
flat heap, and the recursion exist and are exercised by multi-pivot
subproblems and by the small-n configurations in the correctness suite — but
the measured single-source optimum is the configuration in which the
Lemma 3.1 oracle swallows everything.

This is the sharp end of VARIANTS.md's observation that every tuning gradient
points toward Dijkstra: the data, asked for the fastest correct BMSSP
instantiation, returned **Dijkstra in BMSSP clothing**.

### What the residual ~1.4× to `lt-dijkstra` is

Since the two are doing the same graph traversal, the remaining gap (1.4× at
n=10⁶ on random graphs; up to 5.0× on the tie-heavy DIMACS road graph — see
BENCHMARKS.md) is exactly the cost of the BMSSP *contract* over plain
Dijkstra, and it does not vanish with n:

- **16-byte lexicographic labels** `(len, hops)` vs Dijkstra's 8-byte `f64`
  distance — double the per-vertex state traffic;
- **i64 hop arithmetic and comparison** in every relaxation and every heap
  comparison (Dijkstra compares one `f64`);
- **20-byte heap entries** `(f64, i64, u32)` vs Dijkstra's 12-byte
  `(f64, u32)` — bigger sift payloads;
- **the `≤` (not `<`) relaxation rule**, which admits equal-key candidates
  and so performs strictly more heap pushes on tie-heavy inputs (why the road
  graph is the worst case).

These are constant-factor per-edge/per-vertex costs, not a vanishing
asymptotic term — which is the structural reason BENCHMARKS.md concludes
there is no crossover for `bmssp-fast` either.

## Reproducing

```bash
# the gate (must stay green after every change)
cargo test --release --test variants_correctness

# before/after profile (zero overhead unless this feature is on)
cargo run --release --features phase-timer --example profile_fast -- 1000000
# DIMACS road graph instead:
cargo run --release --features phase-timer --example profile_fast -- 0 benchmarks/data/USA-road-d.NY.gr

# upgrade engine invariant checks to hard asserts in a release build
cargo test --release --features verify --test variants_correctness
```
