"""Property-based tests (hypothesis). SPEC.md S8.7."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings, strategies as st

import logtwothirds._reference as ref
from logtwothirds._reference import (
    INF,
    BlockDS,
    Key,
    State,
    bmssp,
    build_graph,
    compute_params,
    sssp,
    transform_to_constant_degree,
)
from logtwothirds import shortest_paths

from .reference import dijkstra
from .test_block_ds import _NaiveModel, k


# ---------------------------------------------------------------------------
# digraphs() strategy: n in [1, 60], up to 3n random edges, mixed weights
# (forced ties via 0.0 / 0.5 / 1.0).
# ---------------------------------------------------------------------------


@st.composite
def digraphs(draw):
    n = draw(st.integers(min_value=1, max_value=60))
    weight = st.one_of(
        st.just(0.0),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        st.sampled_from([0.5, 1.0]),
    )
    edges = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=n - 1),
                st.integers(min_value=0, max_value=n - 1),
                weight,
            ),
            max_size=3 * n,
        )
    )
    return n, edges


def _assert_matches_dijkstra(g, source=0):
    got = sssp(g, source)
    want = dijkstra(g, source)

    got_reach = [math.isfinite(d) for d in got]
    want_reach = [math.isfinite(d) for d in want]
    assert got_reach == want_reach

    for a, b in zip(got, want):
        if math.isinf(a):
            continue
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (a, b)
    return got, want


# ---------------------------------------------------------------------------
# Acceptance criterion 2: matches the Rust dijkstra (logtwothirds.shortest_paths)
# on >=500 random graphs with non-negative float weights, plus integer-weight
# graphs.
# ---------------------------------------------------------------------------


def _assert_matches_rust(g, source=0):
    indptr = np.array(g.indptr, dtype=np.int64)
    indices = np.array(g.indices, dtype=np.int32)
    weights = np.array(g.weights, dtype=np.float64)

    got = sssp(g, source)
    rust_dist, _pred = shortest_paths((indptr, indices, weights), source)

    for a, b in zip(got, rust_dist):
        b = float(b)
        if math.isinf(a) and math.isinf(b):
            continue
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-9), (a, b)


@settings(max_examples=500, deadline=None)
@given(digraphs())
def test_matches_rust_dijkstra_float_weights(data):
    n, edges = data
    g = build_graph(n, edges)
    _assert_matches_rust(g, 0)


@st.composite
def integer_weight_digraphs(draw):
    n = draw(st.integers(min_value=1, max_value=60))
    edges = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=n - 1),
                st.integers(min_value=0, max_value=n - 1),
                st.integers(min_value=0, max_value=10).map(float),
            ),
            max_size=3 * n,
        )
    )
    return n, edges


@settings(max_examples=200, deadline=None)
@given(integer_weight_digraphs())
def test_matches_rust_dijkstra_integer_weights(data):
    n, edges = data
    g = build_graph(n, edges)
    _assert_matches_rust(g, 0)


# ---------------------------------------------------------------------------
# Property 1: matches reference Dijkstra
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(digraphs())
def test_matches_dijkstra(data):
    n, edges = data
    g = build_graph(n, edges)
    _assert_matches_dijkstra(g, 0)


# ---------------------------------------------------------------------------
# Property 2: idempotence / purity -- running twice gives identical results
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(digraphs())
def test_idempotent(data):
    n, edges = data
    g = build_graph(n, edges)
    first = sssp(g, 0)
    second = sssp(g, 0)
    assert first == second


# ---------------------------------------------------------------------------
# Property 3: settle log distances per vertex equal final distances
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(digraphs())
def test_settle_log_matches_final_dhat(data):
    n, edges = data
    g = build_graph(n, edges)
    g2, source2, _rep = transform_to_constant_degree(g, 0)
    k_, t_, L = compute_params(g2.n)
    st_ = State.new(g2, source2, k_, t_)
    bmssp(st_, L, INF, [source2])

    for (v, d) in st_.settle_log.events:
        assert d == st_.dhat[v], (v, d, st_.dhat[v])


# ---------------------------------------------------------------------------
# Property 4: correctness is independent of (k, t)
# ---------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(digraphs(), st.sampled_from([(1, 1), (1, 2), (2, 1), (3, 2)]))
def test_correctness_independent_of_k_t(data, kt):
    n, edges = data
    g = build_graph(n, edges)
    k_, t_ = kt

    def fixed_params(n2: int, k_=k_, t_=t_) -> tuple[int, int, int]:
        log_n = max(1.0, math.log2(max(2, n2)))
        L = max(1, math.ceil(log_n / t_))
        return k_, t_, L

    orig = ref.compute_params
    try:
        ref.compute_params = fixed_params
        _assert_matches_dijkstra(g, 0)
    finally:
        ref.compute_params = orig


# ---------------------------------------------------------------------------
# Property 5: BlockDS model-based property test (hypothesis-generated
# operation scripts). SPEC.md S8.1 / S8.7.
# ---------------------------------------------------------------------------


_ops = st.lists(
    st.one_of(
        st.tuples(st.just("insert"), st.integers(min_value=0, max_value=29)),
        st.tuples(
            st.just("batch_prepend"),
            st.lists(
                st.integers(min_value=0, max_value=29),
                min_size=1,
                max_size=8,
                unique=True,
            ),
        ),
        st.tuples(st.just("pull")),
    ),
    max_size=200,
)


@settings(max_examples=200, deadline=None)
@given(_ops, st.sampled_from([1, 2, 3, 8]))
def test_blockds_matches_naive_model_hypothesis(ops, M):
    B: Key = (1e9, ref.INF_INT, ref.INF_INT)
    d = BlockDS(M=M, B=B)
    model = _NaiveModel(M=M, B=B)

    next_insert_id = 0
    next_prepend_id = -1

    for op in ops:
        if op[0] == "insert":
            key_id = op[1]
            value = k(float(next_insert_id), vid=next_insert_id)
            next_insert_id += 1
            d.insert(key_id, value)
            model.insert(key_id, value)
        elif op[0] == "batch_prepend":
            keys_chosen = op[1]
            items = []
            for key_id in keys_chosen:
                value = k(float(next_prepend_id), vid=next_prepend_id)
                next_prepend_id -= 1
                items.append((key_id, value))
            d.batch_prepend(items)
            model.batch_prepend(items)
        else:
            d_S, d_x = d.pull()
            m_S, m_x = model.pull()
            assert sorted(d_S) == sorted(m_S)
            assert d_x == m_x

        d._check_invariants()
        assert len(d) == len(model.data)
