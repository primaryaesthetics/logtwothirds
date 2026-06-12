# logtwothirds

Fast single-source shortest paths with a Rust core (PyO3 + maturin) and a thin
Python wrapper.

## Install (from source)

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows; use bin/activate on POSIX
pip install maturin numpy scipy pytest
maturin develop --release
```

## API

```python
from logtwothirds import shortest_paths

distances, predecessors = shortest_paths(graph, source, *, method="dijkstra")
distances, predecessors = shortest_paths(graph, source, *, method="bmssp")
```

- `graph`: a `scipy.sparse` matrix (any format) **or** a CSR triple
  `(indptr: int64, indices: int32, weights: float64)`. The CSR arrays are
  borrowed into Rust **zero-copy** via `rust-numpy`.
- `source`: source vertex index. Out of range raises `IndexError`.
- Returns `(distances: float64[n], predecessors: int32[n])`. Unreachable
  vertices have `inf` distance; the source and unreachable vertices have
  predecessor `-1`. A negative edge weight raises `ValueError`.

## Implementation

`src/dijkstra.rs` implements Dijkstra with an implicit **4-ary** min-heap using
**lazy deletion** (stale entries are skipped on pop). The heap is
structure-of-arrays (keys / vertex-ids in parallel arrays) and pre-reserved, so
the relaxation loop performs **no allocations**. Neighbor `dist[v]` slots are
software-prefetched to overlap the random-access cache misses that dominate the
runtime.

`src/bmssp.rs` + `src/block_queue.rs` implement the BMSSP algorithm of
Duan–Mao–Mao–Shu–Yin ("Breaking the Sorting Barrier...", arXiv:2504.17033) as a
**semantically 1:1 port** of the pure-Python reference
`python/logtwothirds/_reference.py` (see `ALGORITHM.md` / `SPEC.md`): the
constant-degree transform, the path-key total order, FindPivots / BaseCase /
BMSSP, and the block data structure D, reproducing the reference's observable
orders exactly (Python-dict insertion order in D's blocks, insertion-ordered
result sets, an explicit SplitMix64 for the quickselect pivots). The
differential test `tests/differential.rs` checks distances **and settlement
order** bit-for-bit against the reference on 200 random graphs via
`tests/diff_driver.py`; `tests/property_vs_dijkstra.rs` checks distances
against Rust Dijkstra up to 10^6 edges; `tests/not_dijkstra.rs` ports the
suite's non-sorted-settlement acceptance check.

## Tests & benchmark

```bash
pytest -q                         # comparison vs scipy + edge cases
python benchmarks/baseline.py     # n=1e6, m=4e6 vs scipy
cargo test                        # Rust unit + differential + property tests
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features python -- -D warnings
```

The differential test needs a Python interpreter to run the reference; it uses
`.venv` next to `Cargo.toml` (or `LOGTWOTHIRDS_PYTHON`) and skips with a notice
if neither exists.
