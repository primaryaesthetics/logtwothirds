"""logtwothirds: fast single-source shortest paths with a Rust core.

Public API
----------
shortest_paths(graph, source, *, method="dijkstra")
    -> (distances: np.float64 array, predecessors: np.int32 array)
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from . import _logtwothirds  # native extension module (built by maturin)

__all__ = ["shortest_paths"]

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
        Algorithm to use. Currently only ``"dijkstra"`` is supported.

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
    if method != "dijkstra":
        raise ValueError(
            f"unknown method {method!r}; supported methods: 'dijkstra'"
        )

    indptr, indices, weights = _as_csr(graph)
    source = int(source)

    return _logtwothirds.dijkstra(indptr, indices, weights, source)
