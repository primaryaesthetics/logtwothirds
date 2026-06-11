"""Compare logtwothirds against scipy.sparse.csgraph.dijkstra on random graphs."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from .conftest import random_graph
from logtwothirds import shortest_paths


def _make_cases():
    """50 (n, density) cases spanning n in [10, 2000] and varied densities."""
    rng_sizes = np.random.default_rng(0)
    cases = []
    densities = [0.001, 0.005, 0.02, 0.1, 0.5]
    # 10 sizes log-spaced in [10, 2000], each paired with the 5 densities = 50.
    sizes = np.unique(
        np.round(np.geomspace(10, 2000, 10)).astype(int)
    )
    # Ensure exactly 10 sizes (geomspace may collapse near the low end).
    while len(sizes) < 10:
        extra = rng_sizes.integers(10, 2000)
        sizes = np.unique(np.append(sizes, extra))
    sizes = sizes[:10]
    for n in sizes:
        for d in densities:
            cases.append((int(n), float(d)))
    return cases


CASES = _make_cases()


@pytest.mark.parametrize("n,density", CASES)
def test_matches_scipy(n, density):
    # Fixed, case-dependent seed for reproducibility.
    rng = np.random.default_rng(1000 + n * 17 + int(density * 100000))
    g = random_graph(n, density, rng)
    source = int(rng.integers(0, n))

    dist, pred = shortest_paths(g, source)

    sdist, spred = scipy_dijkstra(
        g, directed=True, indices=source, return_predecessors=True
    )

    # Distances match exactly (atol 1e-9, no relative tolerance).
    assert np.allclose(dist, sdist, rtol=0, atol=1e-9, equal_nan=True)

    # Reachability matches exactly.
    ours_reach = np.isfinite(dist)
    scipy_reach = np.isfinite(sdist)
    assert np.array_equal(ours_reach, scipy_reach)

    # Predecessor of source is -1; scipy uses -9999 for "no predecessor".
    assert pred[source] == -1
    # Unreachable vertices have predecessor -1 in our convention.
    assert np.all(pred[~ours_reach] == -1)


def test_total_case_count():
    assert len(CASES) == 50
