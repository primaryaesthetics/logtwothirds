"""logtwothirds: fast single-source shortest paths with a Rust core.

Public API
----------
shortest_paths(graph, source, *, method="dijkstra")
    -> (distances: np.float64 array, predecessors: np.int32 array)
multi_source_shortest_paths(graph, sources, *, method="dijkstra")
    -> (distances: np.float64 array (k, n), predecessors: np.int32 array (k, n))
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from . import _logtwothirds  # native extension module (built by maturin)

__all__ = ["shortest_paths", "multi_source_shortest_paths"]

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


def shortest_paths(
    graph: GraphLike,
    source: int,
    *,
    method: str = "dijkstra",
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
        Algorithm to use: ``"dijkstra"`` (default) or ``"bmssp"`` (the
        Duan–Mao–Mao–Shu–Yin O(m log^(2/3) n) algorithm, run on the
        constant-degree transform of the graph).

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
    if method not in ("dijkstra", "bmssp"):
        raise ValueError(
            f"unknown method {method!r}; supported methods: 'dijkstra', 'bmssp'"
        )

    indptr, indices, weights = _as_csr(graph)
    source = int(source)

    if method == "bmssp":
        return _logtwothirds.bmssp(indptr, indices, weights, source)
    return _logtwothirds.dijkstra(indptr, indices, weights, source)


def multi_source_shortest_paths(
    graph: GraphLike,
    sources: Sequence[int],
    *,
    method: str = "dijkstra",
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
        ``"dijkstra"`` (default) or ``"bmssp"``. Note that each in-flight
        ``bmssp`` source holds its own transformed graph, so peak memory
        scales with ``min(k, n_threads)``.

    Returns
    -------
    distances:
        ``float64`` array of shape ``(k, n)``.
    predecessors:
        ``int32`` array of shape ``(k, n)``.
    """
    if method not in ("dijkstra", "bmssp"):
        raise ValueError(
            f"unknown method {method!r}; supported methods: 'dijkstra', 'bmssp'"
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
