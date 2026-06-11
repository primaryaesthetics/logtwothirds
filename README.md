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

## Tests & benchmark

```bash
pytest -q                         # comparison vs scipy + edge cases
python benchmarks/baseline.py     # n=1e6, m=4e6 vs scipy
cargo clippy --all-targets -- -D warnings
```
