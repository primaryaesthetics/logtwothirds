# logtwothirds

Fast single-source shortest paths with a Rust core (PyO3 + maturin) and a thin
Python wrapper — plus a verified, instrumented implementation of the BMSSP
O(m log^(2/3) n) algorithm of Duan–Mao–Mao–Shu–Yin ("Breaking the Sorting
Barrier...", arXiv:2504.17033), benchmarked honestly against it.

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

The faithful gap narrows with n exactly as O(m log^(2/3) n) vs O(m log n)
predicts, but extrapolates to a crossover near n ≈ 2^400,000; the
maximally-engineered `bmssp-fast` is structurally a Dijkstra run carrying
BMSSP's heavier labels, so its remaining 1.4–5× gap is a constant factor,
not a vanishing one. There is no practical size at which any BMSSP engine
wins — `bmssp` and `bmssp-fast` are provided for research and verification.
Full story, methodology, and the variant ladder: **BENCHMARKS.md** and
**VARIANTS.md**.

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
