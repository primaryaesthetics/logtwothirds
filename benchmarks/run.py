"""SSSP benchmark: logtwothirds (dijkstra, bmssp) vs scipy vs rustworkx.

Implementations
---------------
* ``logtwothirds._logtwothirds.dijkstra``  (this crate, binary-heap Dijkstra)
* ``logtwothirds._logtwothirds.bmssp``     (this crate, Duan-Mao-Mao-Shu-Yin,
  faithful to the paper)
* ``logtwothirds._logtwothirds.bmssp_variant("fast")``  (this crate,
  ``method="bmssp-fast"``: the minimal BMSSP instantiation from VARIANTS.md)
* ``scipy.sparse.csgraph.dijkstra``
* ``rustworkx.dijkstra_shortest_path_lengths``

Graph families
--------------
1. ``random``: uniform random directed graphs, m = 4n, n = 10^4 .. 10^7.
2. ``ba``:     Barabasi-Albert preferential-attachment graphs (attachment
               m = 4), symmetrized to a directed graph (each undirected edge
               becomes two arcs), n = 10^4 .. 10^6.
3. ``dimacs``: the 9th DIMACS Challenge distance graph
               ``benchmarks/data/USA-road-d.NY.gr`` (place the file there
               yourself; the family is skipped with a notice if absent).

Methodology
-----------
* Median of ``--runs`` (default 5) timed runs after ``--warmup`` (default 1)
  warmup run(s); ``time.perf_counter``; GC disabled inside the timed region.
* Fixed seeds everywhere (graph generation and the bmssp pivot RNG).
* Only the algorithm call is timed. Graph construction and per-library
  format conversion (CSR triple, ``scipy.sparse.csr_array``, ``PyDiGraph``)
  happen before timing. Note: the ``bmssp`` call itself internally performs
  its constant-degree transform - that is part of the algorithm's pipeline,
  not a format conversion, so it is (honestly) included in its time.
* Each library computes what its natural single-call API returns: both
  logtwothirds methods return (dist, pred); scipy is called with
  ``return_predecessors=False``; rustworkx returns a dist mapping for
  reachable vertices only. This is noted in BENCHMARKS.md.
* Distances are cross-checked between implementations (outside timing);
  a mismatch marks the row and is reported loudly.

Output
------
``benchmarks/results/results.md`` (markdown tables),
``benchmarks/results/results.json`` (raw numbers),
``benchmarks/results/benchmark_loglog.png`` (log-log time-vs-n plot).

Usage
-----
    python benchmarks/run.py                  # full suite
    python benchmarks/run.py --quick          # small sizes, 3 runs (smoke)
    python benchmarks/run.py --families random,dimacs --max-n 1000000
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

import rustworkx as rx

from logtwothirds import _logtwothirds as lt

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
DIMACS_FILE = DATA_DIR / "USA-road-d.NY.gr"

# Fixed seeds (graph generation); the bmssp pivot RNG seed is fixed too.
SEED_RANDOM = 0xC0FFEE
SEED_BA = 0xBA0BAB
BMSSP_SEED = 0

# rustworkx stores one Python object per node and per edge; above this many
# arcs the PyDiGraph build needs more memory/time than this benchmark's
# budget allows on a 16 GB machine. Skipped entries are footnoted, not hidden.
RUSTWORKX_MAX_EDGES_DEFAULT = 10_000_000


# --------------------------------------------------------------------------
# Graph construction (NOT timed)
# --------------------------------------------------------------------------

@dataclass
class Graph:
    """One benchmark instance, pre-converted for every library."""

    family: str
    label: str
    n: int
    m: int  # arcs actually stored in the CSR (after duplicate-sum)
    indptr: np.ndarray  # int64
    indices: np.ndarray  # int32
    weights: np.ndarray  # float64
    source: int
    csr: sp.csr_array = field(repr=False, default=None)
    rxg: "rx.PyDiGraph | None" = field(repr=False, default=None)


def _finalize(family: str, label: str, coo: sp.coo_array, source: int,
              build_rx: bool) -> Graph:
    """COO -> canonical CSR triple (+ scipy and rustworkx forms)."""
    csr = coo.tocsr()  # duplicate (u, v) entries are summed; same graph for all
    csr.sort_indices()
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    weights = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = csr.shape[0]

    rxg = None
    if build_rx:
        rxg = rx.PyDiGraph()
        rxg.add_nodes_from(range(n))
        # Chunked so we never materialize one giant Python tuple list.
        srcs = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        chunk = 2_000_000
        for lo in range(0, len(indices), chunk):
            hi = min(lo + chunk, len(indices))
            rxg.extend_from_weighted_edge_list(
                list(zip(srcs[lo:hi].tolist(),
                         indices[lo:hi].tolist(),
                         weights[lo:hi].tolist()))
            )
        del srcs

    return Graph(family, label, n, len(indices), indptr, indices, weights,
                 source, csr=csr, rxg=rxg)


def gen_random(n: int, seed: int, build_rx: bool) -> Graph:
    """Uniform random directed graph, m = 4n arcs, weights U[0.01, 1)."""
    m = 4 * n
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n, size=m, dtype=np.int64).astype(np.int32)
    cols = rng.integers(0, n, size=m, dtype=np.int64).astype(np.int32)
    data = rng.uniform(0.01, 1.0, size=m)
    coo = sp.coo_array((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    del rows, cols, data
    return _finalize("random", f"n=10^{round(math.log10(n))}", coo, 0, build_rx)


def gen_ba(n: int, seed: int, build_rx: bool) -> Graph:
    """Barabasi-Albert graph (attachment 4) symmetrized to a digraph.

    Topology from rustworkx's seeded generator; each undirected edge {u, v}
    becomes arcs u->v and v->u with the same U[0.01, 1) weight.
    """
    ug = rx.barabasi_albert_graph(n, 4, seed=seed)
    und = np.array(ug.edge_list(), dtype=np.int64)
    del ug
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.01, 1.0, size=len(und))
    rows = np.concatenate([und[:, 0], und[:, 1]]).astype(np.int32)
    cols = np.concatenate([und[:, 1], und[:, 0]]).astype(np.int32)
    data = np.concatenate([w, w])
    del und, w
    coo = sp.coo_array((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    del rows, cols, data
    return _finalize("ba", f"n=10^{round(math.log10(n))}", coo, 0, build_rx)


def parse_dimacs_gr(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a 9th-DIMACS-Challenge ``.gr`` file.

    Format: ``c`` comment lines; one ``p sp <n> <m>`` problem line; ``m``
    arc lines ``a <u> <v> <w>`` with 1-based endpoints and integer weights.
    Returns (n, src, dst, weight) with 0-based int32 endpoints.
    """
    n = m = None
    # Token walk over the whole file: ~3x faster than np.loadtxt and no
    # per-line Python objects beyond the token list.
    toks = path.read_text().split()
    src = dst = wgt = None
    i, k = 0, 0
    while i < len(toks):
        t = toks[i]
        if t == "a":
            src[k] = int(toks[i + 1])
            dst[k] = int(toks[i + 2])
            wgt[k] = float(toks[i + 3])
            k += 1
            i += 4
        elif t == "p":
            if toks[i + 1] != "sp":
                raise ValueError(f"not a shortest-path .gr file: p {toks[i+1]}")
            n, m = int(toks[i + 2]), int(toks[i + 3])
            src = np.empty(m, dtype=np.int32)
            dst = np.empty(m, dtype=np.int32)
            wgt = np.empty(m, dtype=np.float64)
            i += 4
        elif t == "c":
            # Comment: skip to end of line. Token-walking can't see line
            # breaks, so re-scan: comments only appear in the header of the
            # DIMACS road files, before any 'a' line; skip tokens until the
            # next structural marker.
            i += 1
            while i < len(toks) and toks[i] not in ("a", "p", "c"):
                i += 1
        else:
            i += 1
    if n is None or k != m:
        raise ValueError(f"malformed .gr file: header m={m}, parsed {k} arcs")
    return n, src - 1, dst - 1, wgt


def gen_dimacs(build_rx: bool) -> Graph | None:
    if not DIMACS_FILE.exists():
        print(f"NOTE: {DIMACS_FILE} not found - skipping the DIMACS family. "
              f"Download USA-road-d.NY.gr into benchmarks/data/ to enable it.")
        return None
    n, src, dst, wgt = parse_dimacs_gr(DIMACS_FILE)
    coo = sp.coo_array((wgt, (src, dst)), shape=(n, n), dtype=np.float64)
    return _finalize("dimacs", "USA-road-d.NY", coo, 0, build_rx)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def bench(fn, runs: int, warmup: int) -> dict:
    """Median/min/max of `runs` timed calls after `warmup` warmup calls."""
    for _ in range(warmup):
        fn()
    times = []
    gc.collect()
    gcold = gc.isenabled()
    gc.disable()
    try:
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    finally:
        if gcold:
            gc.enable()
    return {
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "runs": runs,
    }


def make_impls(g: Graph) -> dict[str, "tuple"]:
    """name -> (timed_zero_arg_callable, get_dist_array_or_None)."""
    impls: dict = {}

    def lt_dij():
        return lt.dijkstra(g.indptr, g.indices, g.weights, g.source)

    def lt_bm():
        return lt.bmssp(g.indptr, g.indices, g.weights, g.source, BMSSP_SEED)

    def lt_bm_fast():
        return lt.bmssp_variant(g.indptr, g.indices, g.weights, g.source,
                                "fast", BMSSP_SEED)

    def sc_dij():
        return scipy_dijkstra(g.csr, directed=True, indices=g.source,
                              return_predecessors=False)

    impls["lt-dijkstra"] = (lt_dij, lambda: lt_dij()[0])
    impls["lt-bmssp"] = (lt_bm, lambda: lt_bm()[0])
    impls["lt-bmssp-fast"] = (lt_bm_fast, lambda: lt_bm_fast()[0])
    impls["scipy"] = (sc_dij, lambda: np.asarray(sc_dij(), dtype=np.float64))

    if g.rxg is not None:
        rxg = g.rxg

        def rx_dij():
            return rx.dijkstra_shortest_path_lengths(rxg, g.source,
                                                     edge_cost_fn=float)

        def rx_dist():
            mapping = rx_dij()
            out = np.full(g.n, np.inf)
            for v, d in mapping.items():
                out[v] = d
            out[g.source] = 0.0
            return out

        impls["rustworkx"] = (rx_dij, rx_dist)
    return impls


def check_dists(g: Graph, dists: dict[str, np.ndarray]) -> list[str]:
    """Cross-check every implementation against lt-dijkstra."""
    problems = []
    ref = dists.get("lt-dijkstra")
    if ref is None:
        return ["no reference distances"]
    for name, d in dists.items():
        if name == "lt-dijkstra" or d is None:
            continue
        if not np.allclose(d, ref, rtol=1e-9, atol=1e-9, equal_nan=False):
            bad = int(np.sum(~np.isclose(d, ref, rtol=1e-9, atol=1e-9)))
            problems.append(f"{name} disagrees with lt-dijkstra on {bad} of "
                            f"{g.n} vertices")
    return problems


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_graph(g: Graph, runs: int, warmup: int, results: list[dict]) -> None:
    print(f"\n=== {g.family}/{g.label}: n={g.n:,}  m={g.m:,}  "
          f"source={g.source} ===", flush=True)
    impls = make_impls(g)

    dists = {}
    for name, (_fn, get_dist) in impls.items():
        try:
            dists[name] = get_dist()
        except MemoryError:
            dists[name] = None
    problems = check_dists(g, dists)
    for p in problems:
        print(f"  !! DISTANCE MISMATCH: {p}", flush=True)
    del dists
    gc.collect()

    for name, (fn, _get) in impls.items():
        try:
            r = bench(fn, runs=runs, warmup=warmup)
            print(f"  {name:12s} median {r['median']*1e3:10.1f} ms   "
                  f"(min {r['min']*1e3:.1f}, max {r['max']*1e3:.1f})",
                  flush=True)
        except MemoryError:
            r = None
            print(f"  {name:12s} SKIPPED (MemoryError)", flush=True)
        results.append({
            "family": g.family, "label": g.label, "n": g.n, "m": g.m,
            "impl": name, "timing": r, "problems": problems,
        })
    if g.rxg is None:
        results.append({
            "family": g.family, "label": g.label, "n": g.n, "m": g.m,
            "impl": "rustworkx", "timing": None,
            "problems": ["skipped: edge count over --rustworkx-max-edges "
                         "(PyDiGraph build too large for this machine)"],
        })


def fmt_time(t: float | None) -> str:
    if t is None:
        return "—"
    if t < 1.0:
        return f"{t*1e3:.1f} ms"
    return f"{t:.2f} s"


def write_markdown(results: list[dict], path: Path, meta: dict) -> None:
    impl_order = ["lt-dijkstra", "lt-bmssp", "lt-bmssp-fast", "scipy", "rustworkx"]
    lines = ["# SSSP benchmark results", ""]
    lines += [f"- {k}: {v}" for k, v in meta.items()]
    lines.append("")
    for family, title in [("random", "Random directed graphs (m = 4n)"),
                          ("ba", "Barabási–Albert graphs (attachment 4, symmetrized)"),
                          ("dimacs", "DIMACS USA-road-d.NY")]:
        rows = [r for r in results if r["family"] == family]
        if not rows:
            continue
        lines.append(f"## {title}\n")
        lines.append("| graph | n | m | " + " | ".join(impl_order) + " |")
        lines.append("|---" * (3 + len(impl_order)) + "|")
        labels = list(dict.fromkeys(r["label"] for r in rows))
        for lab in labels:
            cell = {}
            n = m = 0
            notes = []
            for r in rows:
                if r["label"] != lab:
                    continue
                n, m = r["n"], r["m"]
                t = r["timing"]
                cell[r["impl"]] = fmt_time(t["median"] if t else None)
                for p in r["problems"]:
                    if p not in notes:
                        notes.append(p)
            row = [lab, f"{n:,}", f"{m:,}"] + [cell.get(i, "—") for i in impl_order]
            lines.append("| " + " | ".join(row) + " |")
            for p in notes:
                lines.append(f"| | | | *{p}* |" + " |" * (len(impl_order) - 1))
        lines.append("")
    lines.append("Times are the **median of "
                 f"{meta.get('runs', '?')} runs** after warmup "
                 "(`time.perf_counter`, GC off, algorithm call only). "
                 "“—” = skipped (footnoted above).")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(results: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    impl_style = {
        "lt-dijkstra": ("tab:blue", "o"),
        "lt-bmssp": ("tab:red", "s"),
        "lt-bmssp-fast": ("tab:orange", "v"),
        "scipy": ("tab:green", "^"),
        "rustworkx": ("tab:purple", "d"),
    }
    fams = [("random", "Random directed, m = 4n"),
            ("ba", "Barabási–Albert (m_attach = 4)"),
            ("dimacs", "USA-road-d.NY")]
    fams = [(f, t) for f, t in fams if any(r["family"] == f for r in results)]
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 4.4))
    if len(fams) == 1:
        axes = [axes]

    for ax, (family, title) in zip(axes, fams):
        rows = [r for r in results if r["family"] == family and r["timing"]]
        if family == "dimacs":
            names = [r["impl"] for r in rows]
            vals = [r["timing"]["median"] for r in rows]
            cols = [impl_style[i][0] for i in names]
            ax.bar(names, vals, color=cols)
            ax.set_yscale("log")
            ax.set_ylabel("median time, s (log)")
            ax.tick_params(axis="x", rotation=20)
        else:
            for impl, (color, marker) in impl_style.items():
                pts = sorted((r["n"], r["timing"]["median"]) for r in rows
                             if r["impl"] == impl)
                if pts:
                    ax.loglog([p[0] for p in pts], [p[1] for p in pts],
                              color=color, marker=marker, label=impl)
            ax.set_xlabel("n (vertices)")
            ax.set_ylabel("median time, s")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)
        ax.set_title(title)
    fig.suptitle("Single-source shortest paths (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\nwrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", default="random,ba,dimacs",
                    help="comma list of random,ba,dimacs")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-n", type=int, default=10_000_000,
                    help="largest n for the random family")
    ap.add_argument("--ba-max-n", type=int, default=1_000_000,
                    help="largest n for the Barabasi-Albert family")
    ap.add_argument("--rustworkx-max-edges", type=int,
                    default=RUSTWORKX_MAX_EDGES_DEFAULT)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: n up to 10^5, 3 runs")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    args = ap.parse_args()

    if args.quick:
        args.max_n = min(args.max_n, 100_000)
        args.ba_max_n = min(args.ba_max_n, 100_000)
        args.runs = min(args.runs, 3)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t_start = time.perf_counter()

    if "random" in families:
        n = 10_000
        i = 0
        while n <= args.max_n:
            build_rx = 4 * n <= args.rustworkx_max_edges
            g = gen_random(n, SEED_RANDOM + i, build_rx)
            run_graph(g, args.runs, args.warmup, results)
            del g
            gc.collect()
            n *= 10
            i += 1

    if "ba" in families:
        n = 10_000
        i = 0
        while n <= args.ba_max_n:
            build_rx = 8 * n <= args.rustworkx_max_edges
            g = gen_ba(n, SEED_BA + i, build_rx)
            run_graph(g, args.runs, args.warmup, results)
            del g
            gc.collect()
            n *= 10
            i += 1

    if "dimacs" in families:
        g = gen_dimacs(build_rx=True)
        if g is not None:
            run_graph(g, args.runs, args.warmup, results)
            del g
            gc.collect()

    tag = f"_{args.tag}" if args.tag else ""
    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "machine": f"{platform.processor()} / "
                   f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "numpy/scipy/rustworkx": f"{np.__version__} / "
                                 f"{sp.__name__ and __import__('scipy').__version__} / "
                                 f"{rx.__version__}",
        "runs": args.runs,
        "warmup": args.warmup,
        "seeds": f"random={SEED_RANDOM:#x} ba={SEED_BA:#x} bmssp={BMSSP_SEED}",
        "total wall time": f"{time.perf_counter() - t_start:.0f} s",
    }
    (RESULTS_DIR / f"results{tag}.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=1),
        encoding="utf-8")
    write_markdown(results, RESULTS_DIR / f"results{tag}.md", meta)
    write_plot(results, RESULTS_DIR / f"benchmark_loglog{tag}.png")

    mismatches = {p for r in results for p in r["problems"]
                  if "disagrees" in p}
    if mismatches:
        print("\nDISTANCE MISMATCHES DETECTED:")
        for p in sorted(mismatches):
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
