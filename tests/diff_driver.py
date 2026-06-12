"""Differential-test driver: Python `_reference.py` vs the Rust BMSSP port.

Invoked by the Rust integration test `tests/differential.rs` as

    python diff_driver.py <results_file> <num_graphs>

The results file holds, per seed, the Rust port's distance array and
settlement log (f64s as raw bit patterns in hex). This driver regenerates the
identical graphs from the seeds (the generator below is a draw-for-draw mirror
of ``gen_diff_graph`` in ``tests/common/mod.rs``), runs the pure-Python
reference, and compares bit-for-bit. The first divergence in the settlement
log is located and printed with context.

Pinning unspecified behavior
----------------------------
The reference is deterministic except for two things that its *semantics*
leave open:

* ``random.randint`` pivot draws in ``_select_smallest`` (any RNG yields a
  correct run). We patch ``ref.random`` to the same SplitMix64 stream the
  Rust port uses, seeded identically per graph.
* The iteration order of the builtin ``set`` objects ``U`` / ``result_U`` in
  ``bmssp`` (sets are unordered; the order leaks into the result via
  ``list(result_U)``). We patch ``ref.set`` to an insertion-ordered set,
  which is the order the Rust port implements.

Both patches select one *valid execution* of the reference; no algorithmic
behavior is altered. With them in place, distances AND settlement order are
required to match the Rust port exactly.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

MASK64 = (1 << 64) - 1


class SplitMix64:
    """Mirror of `block_queue::SplitMix64` (including `randint`)."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive, like random.randint; realized as lo + next % span."""
        return lo + self.next64() % (hi - lo + 1)


class OSet:
    """Insertion-ordered stand-in for builtin ``set`` (the operations the
    reference performs), pinning iteration order to insertion order."""

    __slots__ = ("_d",)

    def __init__(self, iterable=()):
        d = {}
        for x in iterable:
            d[x] = None
        self._d = d

    def add(self, x):
        self._d[x] = None

    def __contains__(self, x):
        return x in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __ior__(self, other):
        for x in other:
            self._d[x] = None
        return self

    def isdisjoint(self, other):
        return all(x not in self._d for x in other)


def gen_diff_graph(seed: int):
    """Draw-for-draw mirror of tests/common/mod.rs::gen_diff_graph."""
    r = SplitMix64(seed ^ 0xD1FFE12E5EED5EED)
    cls = seed % 4
    if cls == 0:
        n = 1 + r.next64() % 40
    elif cls == 1:
        n = 2 + r.next64() % 459
    else:
        n = 500 + r.next64() % 4501
    m = r.next64() % (3 * n + 1)
    edges = []
    for _ in range(m):
        u = r.next64() % n
        v = r.next64() % n
        if r.next64() % 20 == 0:
            w = 0.0
        else:
            w = ((r.next64() % 1_000_000) + 1) / 1e6
        edges.append((u, v, w))
    source = r.next64() % n
    algo_seed = r.next64()
    return n, edges, source, algo_seed


def f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def load_reference():
    here = Path(__file__).resolve()
    ref_path = here.parents[1] / "python" / "logtwothirds" / "_reference.py"
    spec = importlib.util.spec_from_file_location("_reference_under_test", ref_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolves cls.__module__ here
    spec.loader.exec_module(mod)
    return mod


def parse_results(path: Path, expected: int):
    """Parse the Rust results file into {seed: (params, dist_bits, settle)}."""
    results = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        assert lines[i].startswith("GRAPH "), lines[i]
        seed = int(lines[i].split()[1])
        params = tuple(int(x) for x in lines[i + 1].split()[1:])
        dist = [int(x, 16) for x in lines[i + 2].split()[1:]]
        settle = []
        for tok in lines[i + 3].split()[1:]:
            v, b = tok.split(":")
            settle.append((int(v), int(b, 16)))
        assert lines[i + 4] == "END", lines[i + 4]
        results[seed] = (params, dist, settle)
        i += 5
    assert len(results) == expected, (len(results), expected)
    return results


def first_mismatch(a, b):
    """Index of the first position where the sequences differ, or None."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def settle_context(label, log, idx, width=3):
    lo = max(0, idx - width)
    hi = min(len(log), idx + width + 1)
    rows = []
    for j in range(lo, hi):
        v, bits = log[j]
        marker = " <-- first divergence" if j == idx else ""
        rows.append(
            f"    {label}[{j}] = (v={v}, dhat={struct.unpack('<d', struct.pack('<Q', bits))[0]!r}, bits={bits:#x}){marker}"
        )
    if idx >= len(log):
        rows.append(f"    {label}[{idx}] = <log ended at {len(log)} events>")
    return "\n".join(rows)


def main() -> int:
    results_path = Path(sys.argv[1])
    num_graphs = int(sys.argv[2])

    ref = load_reference()
    ref.set = OSet  # pin set iteration order (see module docstring)

    rust = parse_results(results_path, num_graphs)
    failures = 0

    for seed in range(num_graphs):
        n, edges, source, algo_seed = gen_diff_graph(seed)
        rust_params, rust_dist, rust_settle = rust[seed]

        g = ref.build_graph(n, edges)
        g2, _src2, _rep = ref.transform_to_constant_degree(g, source)
        k, t, L = ref.compute_params(g2.n)
        py_params = (k, t, L, g2.n)

        ref.random = SplitMix64(algo_seed)  # pin quickselect pivots
        dist, _counter, log = ref.sssp_instrumented(g, source)

        py_dist = [f64_bits(d) for d in dist]
        py_settle = [(v, f64_bits(d)) for (v, d) in log.events]

        problems = []
        if py_params != rust_params:
            problems.append(
                f"  params differ: python (k,t,L,n2)={py_params} rust={rust_params}"
            )
        di = first_mismatch(py_dist, rust_dist)
        if di is not None:
            pd = dist[di] if di < len(dist) else None
            rd = (
                struct.unpack("<d", struct.pack("<Q", rust_dist[di]))[0]
                if di < len(rust_dist)
                else None
            )
            problems.append(
                f"  distances differ first at vertex {di}: python={pd!r} rust={rd!r}"
            )
        si = first_mismatch(py_settle, rust_settle)
        if si is not None:
            problems.append(
                f"  settlement logs differ first at index {si} "
                f"(python has {len(py_settle)} events, rust {len(rust_settle)}):\n"
                + settle_context("python", py_settle, si)
                + "\n"
                + settle_context("rust  ", rust_settle, si)
            )

        if problems:
            failures += 1
            print(
                f"MISMATCH seed={seed} n={n} m={len(edges)} source={source} "
                f"algo_seed={algo_seed:#x}"
            )
            for p in problems:
                print(p)
        else:
            print(f"OK seed={seed} n={n} m={len(edges)} settle_events={len(py_settle)}")

    if failures:
        print(f"FAILED {failures}/{num_graphs} graphs diverged")
        return 1
    print(f"ALL OK {num_graphs}/{num_graphs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
