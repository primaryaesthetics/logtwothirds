# logtwothirds

A verified, honestly benchmarked Rust implementation of the 2025 algorithm that
broke the **sorting barrier** for shortest paths — packaged as a small Python
library (Rust core via PyO3 + maturin), and measured against plain Dijkstra to
answer the question the theory leaves open: is it actually faster?

## The sorting barrier, and the algorithm that broke it

Dijkstra's algorithm settles vertices in order of increasing distance from the
source. That ordering is the whole trick — and also a tax. Producing `n` numbers
in sorted order costs `Ω(n log n)` comparisons, so any shortest-path algorithm
that hands you vertices in distance order inherits that `log n` factor. On a
sparse graph — `m` edges with `m` close to `n` — that sorting term is what
dominates the clock. For sixty years this looked fundamental: to find shortest
paths you seemed to have to sort, and sorting has a floor. Call it the *sorting
barrier*.

In 2025, Duan, Mao, Mao, Shu, and Yin broke it. Their algorithm — *Breaking the
Sorting Barrier for Directed Single-Source Shortest Paths*
([arXiv:2504.17033](https://arxiv.org/abs/2504.17033)) — finds the same shortest
paths in `O(m log^(2/3) n)` time, strictly below Dijkstra's `O(m + n log n)` on
sparse graphs. The idea is to stop fully sorting. Rather than pulling vertices
one at a time in distance order, it recursively shrinks the frontier: a
`FindPivots` step picks a small set of vertices whose settlement unlocks
everything behind them, and a divide-and-conquer recursion (`BMSSP`) settles
whole blocks of vertices without ever materializing the complete sorted
sequence. The `log^(2/3)` is what the bookkeeping costs once you no longer pay
for the sort. The "two-thirds" in the exponent is where this project's name
comes from.

This repository implements that algorithm, checks it against the paper line by
line, and then asks the question the asymptotics don't: does breaking the
barrier make anything *run* faster? The honest answer — measured, not asserted —
is **no, not at any size you can actually run.** The faithful implementation is
26–128× slower than a good Dijkstra; the most aggressively engineered variant
closes that to 1.4–5×, but never crosses over. The asymptotic advantage is real
and it does narrow with `n` exactly as `log^(2/3) n` vs `log n` predicts — it
just doesn't pay off until somewhere around `n ≈ 2^400000` — for scale, the
observable universe holds roughly `2^266` atoms. That gap between what the
theory promises and what
the hardware delivers is the actual subject of this repo, documented honestly
below and in [BENCHMARKS.md](BENCHMARKS.md).

So the library ships **Dijkstra** as the thing you should use, and keeps the
BMSSP engines as instrumented, verified research objects — the sharpest way to
state precisely *where* the algorithm's constant factor lives.

## Install (from source)

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows; use bin/activate on POSIX
pip install maturin numpy scipy pytest
maturin develop --release
```

## API

```python
from logtwothirds import shortest_paths, shortest_paths_multi

dist, pred = shortest_paths(graph, source)                     # method="auto"
dist, pred = shortest_paths(graph, source, method="bmssp")     # research
dists, preds = shortest_paths_multi(graph, [s0, s1, s2])       # parallel (rayon)
```

- `graph`: a `scipy.sparse` matrix (any format) **or** a CSR triple
  `(indptr: int64, indices: int32, weights: float64)`. The CSR arrays are
  borrowed into Rust **zero-copy** via `rust-numpy`.
- `source`: source vertex index. Out of range raises `IndexError`.
- Returns `(distances: float64[n], predecessors: int32[n])` — for
  `shortest_paths_multi`, shape `(k, n)`, row `i` bit-identical to the
  single-source call for `sources[i]`. Unreachable vertices have `inf`
  distance; the source and unreachable vertices have predecessor `-1`. A
  negative edge weight raises `ValueError`.

### Methods

| method | what it is | when to use it |
|---|---|---|
| `"auto"` (default) | selects `"dijkstra"`, always | always |
| `"dijkstra"` | 4-ary SoA heap Dijkstra | the fastest method at every measured size |
| `"bmssp"` | BMSSP, faithful to the paper | studying the algorithm (settlement logs, counters, differential-tested vs the Python reference) |
| `"bmssp-fast"` | the fastest BMSSP instantiation found (VARIANTS.md) | research: the sharpest statement of the BMSSP-vs-Dijkstra verdict |
| `"bmssp-<name>"` | single-delta research variants (`tuned`, `hybrid`, `simpleq`, `lazypiv`, `notransform`) | research: isolating where BMSSP's constant factor lives |

`"auto"` selecting Dijkstra unconditionally is a measured verdict, not a
stub. Median-of-5 benchmarks (BENCHMARKS.md, fixed seeds, distances
cross-checked across five implementations):

| graph | lt-dijkstra | bmssp (faithful) | bmssp-fast | scipy | rustworkx |
|---|---:|---:|---:|---:|---:|
| random m=4n, n=10⁶ | 854 ms | 24.6 s (29×) | 1.34 s (1.6×) | 806 ms | 1.59 s |
| random m=4n, n=10⁷ | 13.3 s | 345 s (26×) | 18.4 s (1.4×) | 10.7 s | — |
| Barabási–Albert, n=10⁶ | 1.28 s | 43.9 s (34×) | 1.79 s (1.4×) | 1.06 s | 1.86 s |
| USA-road-d.NY | 25.9 ms | 1.65 s (64×) | 130 ms (5.0×) | 40.7 ms | 126 ms |

The faithful gap narrows with `n` exactly as `O(m log^(2/3) n)` vs `O(m log n)`
predicts, but extrapolates to a crossover near `n ≈ 2^400000`; the
maximally-engineered `bmssp-fast` is structurally a Dijkstra run carrying
BMSSP's heavier labels, so its remaining 1.4–5× gap is a constant factor,
not a vanishing one. There is no practical size at which any BMSSP engine
wins — `bmssp` and `bmssp-fast` are provided for research and verification.
Full story, methodology, and the variant ladder: **BENCHMARKS.md** (final
matrix and verdict), **VARIANTS.md** (the algorithm-level variant study),
and **OPTIMIZATION.md** (the low-level pass that produced the shipping
`bmssp-fast`).

## Implementation

`src/dijkstra.rs` implements Dijkstra with an implicit **4-ary** min-heap using
**lazy deletion** (stale entries are skipped on pop). The heap is
structure-of-arrays (keys / vertex-ids in parallel arrays) and pre-reserved, so
the relaxation loop performs **no allocations**. Neighbor `dist[v]` slots are
software-prefetched to overlap the random-access cache misses that dominate the
runtime.

`src/bmssp.rs` + `src/block_queue.rs` implement BMSSP as a **semantically 1:1
port** of the pure-Python reference `python/logtwothirds/_reference.py` (see
`ALGORITHM.md` / `SPEC.md`): the constant-degree transform, the path-key total
order, FindPivots / BaseCase / BMSSP, and the block data structure D,
reproducing the reference's observable orders exactly (Python-dict insertion
order in D's blocks, insertion-ordered result sets, an explicit SplitMix64 for
the quickselect pivots). The differential test `tests/differential.rs` checks
distances **and settlement order** bit-for-bit against the reference on 200
random graphs via `tests/diff_driver.py`; `tests/property_vs_dijkstra.rs`
checks distances against Rust Dijkstra up to 10^6 edges;
`tests/not_dijkstra.rs` ports the suite's non-sorted-settlement acceptance
check. This mainline is frozen as the reference engine.

`src/variants/` holds the research variants behind `method="bmssp-<name>"`: a
shared engine (`engine.rs`) that keeps the paper's correctness contracts
(label total order, `<=` relaxation, the Lemma 3.1 oracle contract) while
making the transform, queue, base-case oracle, and (k, t) configurable. Every
variant is gated on `tests/variants_correctness.rs`: ≥520 property graphs
(zero weights, ties, integer weights, self-loops, parallel edges) plus a
10⁶-edge stress graph, **bit-exact distances vs Dijkstra** plus predecessor
consistency. Engine instrumentation is compiled out unless built with
`--features phase-timer`; invariant checks upgrade from `debug_assert!` to
hard asserts with `--features verify`.

## Tests & benchmark

```bash
pytest -q                         # Python API tests (vs scipy + edge cases)
cargo test                        # Rust unit + differential + property tests
cargo test --release --test variants_correctness   # variant distance gate
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features python -- -D warnings
python benchmarks/run.py          # full benchmark matrix (~1.5 h)
```

The differential test needs a Python interpreter to run the reference; it uses
`.venv` next to `Cargo.toml` (or `LOGTWOTHIRDS_PYTHON`) and skips with a notice
if neither exists.

## Documents

The research record, in reading order:

| file | what it is |
|---|---|
| `ALGORITHM.md` | self-contained distillation of the Duan–Mao–Mao–Shu–Yin paper (the source of truth all lemma references point into) |
| `SPEC.md` | engineering spec for the pure-Python reference (`python/logtwothirds/_reference.py`); defers to ALGORITHM.md on any conflict |
| `AUDIT.md` | line-by-line audit of the Python reference against the paper (zero blockers; findings F1–F14) |
| `QUESTIONS.md` / `FAILCASE.md` | the four resolved paper-interpretation questions, and the worked failure case that motivated the settled-vertex filter |
| `VARIANTS.md` | algorithm-level variant study (`src/variants/`) that produced `bmssp-fast`; ranks the variants and proves each delta correctness-preserving |
| `OPTIMIZATION.md` | two low-level engineering passes that tightened `bmssp-fast` (2.13 s → 1.21 s at n=10⁶, then ~1.49× → ~1.24× of Dijkstra by same-process ratio, ~1.12× at 10⁷); distinct from the mainline pass in BENCHMARKS.md |
| `BENCHMARKS.md` | final cross-implementation matrix and the honest verdict (Dijkstra wins everywhere; no crossover) — the authoritative wall-clock numbers |

Numbers across these are consistent as of 2026-06-13; where a research-phase
table (VARIANTS.md) and the final matrix (BENCHMARKS.md) differ for
`bmssp-fast`, BENCHMARKS.md is authoritative and the older table is marked as
superseded in place.
