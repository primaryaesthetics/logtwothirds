"""logtwothirds: fast single-source shortest paths with a Rust core.

Public API
----------
shortest_paths(graph, source, *, method="auto")
    -> (distances: np.float64 array, predecessors: np.int32 array)
shortest_paths_multi(graph, sources, *, method="auto")
    -> (distances: np.float64 array (k, n), predecessors: np.int32 array (k, n))
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from . import _logtwothirds  # native extension module (built by maturin)

__all__ = ["shortest_paths", "shortest_paths_multi"]

_CSRTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]
GraphLike = Union["object", _CSRTriple]


def _as_csr(graph: GraphLike) -> _CSRTriple:
    """Normalize ``graph`` to a CSR triple ``(indptr, indices, weights)`` with
    the exact dtypes the Rust core borrows zero-copy: int64 / int32 / float64.

    Accepts either:
      * a SciPy sparse matrix/array (any format; converted to CSR), or
      * a 3-tuple ``(indptr, indices, weights)`` already in CSR layout.
    """
    if isinstance(graph, tuple):
        if len(graph) != 3:
            raise ValueError(
                "CSR triple must be (indptr, indices, weights); "
                f"got a tuple of length {len(graph)}"
            )
        indptr, indices, weights = graph
    else:
        # Duck-type a SciPy sparse matrix. Importing scipy lazily keeps it an
        # optional dependency for callers that only use CSR triples.
        try:
            import scipy.sparse as sp
        except ImportError as exc:  # pragma: no cover
            raise TypeError(
                "graph is not a CSR triple and SciPy is not installed to "
                "interpret a sparse matrix"
            ) from exc

        if not sp.issparse(graph):
            raise TypeError(
                "graph must be a scipy.sparse matrix or a "
                "(indptr, indices, weights) CSR triple"
            )
        csr = graph.tocsr()
        csr.sort_indices()
        indptr, indices, weights = csr.indptr, csr.indices, csr.data

    # Contiguous, correctly-typed views (copies only if dtype/layout differ).
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    return indptr, indices, weights


def _resolve_method(method: str) -> str:
    """Resolve ``"auto"`` and validate the method name.

    ``"auto"`` always selects ``"dijkstra"``. That is the benchmark verdict,
    not a placeholder: across every graph family and size measured (random
    m = 4n up to n = 10^7, Barabasi-Albert, the DIMACS NY road graph), this
    crate's Dijkstra is the fastest method, and the gap *grows* with n — there
    is no crossover at practical sizes (see BENCHMARKS.md). The BMSSP methods
    are provided for research and verification, not speed.
    """
    if method == "auto":
        return "dijkstra"
    if (
        method in ("dijkstra", "bmssp", "bmssp-fast")
        or method.startswith("bmssp-")
    ):
        return method
    raise ValueError(
        f"unknown method {method!r}; supported methods: 'auto', 'dijkstra', "
        "'bmssp', 'bmssp-fast', 'bmssp-<variant>'"
    )


def shortest_paths(
    graph: GraphLike,
    source: int,
    *,
    method: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-source shortest paths from ``source``.

    Parameters
    ----------
    graph:
        A SciPy sparse matrix (any format) or a CSR triple
        ``(indptr: int64, indices: int32, weights: float64)``. The matrix is
        interpreted as a directed graph with ``graph[u, v]`` the weight of edge
        ``u -> v``. Edge weights must be non-negative.
    source:
        Source vertex index.
    method:
        * ``"auto"`` (default): currently always selects ``"dijkstra"`` — the
          benchmark verdict (BENCHMARKS.md): Dijkstra wins at every measured
          size on every graph family, and the gap grows with n.
        * ``"dijkstra"``: binary-heap Dijkstra (4-ary, structure-of-arrays).
          The fastest method everywhere measured.
        * ``"bmssp"``: the Duan–Mao–Mao–Shu–Yin O(m log^(2/3) n) algorithm,
          faithful to the paper (constant-degree transform, paper (k, t),
          block queue, settlement order pinned to the reference). 27–48×
          slower than ``"dijkstra"`` in practice; provided as the reference
          engine for research and verification, not for speed.
        * ``"bmssp-fast"``: the fastest BMSSP instantiation found in the
          variant study (VARIANTS.md): no degree transform, Dijkstra oracle
          for small subproblems, flat heap, tuned (k, t). Distances are
          bit-exact vs Dijkstra; still 1.5–6× slower than ``"dijkstra"``.
        * ``"bmssp-<name>"``: the other research variants from VARIANTS.md
          (``tuned``, ``hybrid``, ``simpleq``, ``lazypiv``, ``notransform``).

    Returns
    -------
    distances:
        ``float64`` array of length ``n``; ``np.inf`` for unreachable vertices.
    predecessors:
        ``int32`` array of length ``n``; the predecessor of each vertex on a
        shortest path, or ``-1`` for the source and for unreachable vertices.

    Raises
    ------
    ValueError
        If ``method`` is unknown or an edge weight is negative.
    IndexError
        If ``source`` is out of range.
    """
    method = _resolve_method(method)
    indptr, indices, weights = _as_csr(graph)
    source = int(source)

    if method == "bmssp":
        return _logtwothirds.bmssp(indptr, indices, weights, source)
    if method.startswith("bmssp-"):
        # BMSSP variants (src/variants/; see VARIANTS.md). Distances are
        # verified bit-exact vs dijkstra; settlement order is not pinned.
        return _logtwothirds.bmssp_variant(
            indptr, indices, weights, source, method[len("bmssp-"):]
        )
    return _logtwothirds.dijkstra(indptr, indices, weights, source)


def shortest_paths_multi(
    graph: GraphLike,
    sources: Sequence[int],
    *,
    method: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Shortest paths from each of ``k`` sources, computed in parallel.

    The sources are fanned out over a Rust (rayon) thread pool; row ``i`` of
    the result is bit-identical to
    ``shortest_paths(graph, sources[i], method=method)``.

    Parameters
    ----------
    graph:
        Same as :func:`shortest_paths`.
    sources:
        Source vertex indices (duplicates allowed).
    method:
        ``"auto"`` (default, always selects ``"dijkstra"`` — see
        :func:`shortest_paths`), ``"dijkstra"``, or ``"bmssp"``. Note that
        each in-flight ``bmssp`` source holds its own transformed graph, so
        peak memory scales with ``min(k, n_threads)``. The research variants
        are single-source only.

    Returns
    -------
    distances:
        ``float64`` array of shape ``(k, n)``.
    predecessors:
        ``int32`` array of shape ``(k, n)``.
    """
    method = _resolve_method(method)
    if method not in ("dijkstra", "bmssp"):
        raise ValueError(
            f"unknown method {method!r}; supported methods for "
            "shortest_paths_multi: 'auto', 'dijkstra', 'bmssp'"
        )

    indptr, indices, weights = _as_csr(graph)
    src = np.ascontiguousarray(sources, dtype=np.int64)
    if src.ndim != 1:
        raise ValueError("sources must be a 1-D sequence of vertex indices")
    n = len(indptr) - 1

    if method == "bmssp":
        dist, pred = _logtwothirds.bmssp_multisource(
            indptr, indices, weights, src
        )
    else:
        dist, pred = _logtwothirds.dijkstra_multisource(
            indptr, indices, weights, src
        )
    return dist.reshape(len(src), n), pred.reshape(len(src), n)
