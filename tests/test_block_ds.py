"""Tests for BlockDS (Lemma 3.3, ALGORITHM.md S3 / SPEC.md S4 / S8.1)."""

from __future__ import annotations

import math
import random

import pytest

from logtwothirds._reference import INF, INF_INT, BlockDS, Key


B_BIG: Key = (1_000_000.0, INF_INT, INF_INT)


def k(x: float, h: int = 0, vid: int | None = None) -> Key:
    """Build a Key with distinct values for distinct ``x`` (and ``vid``)."""
    return (float(x), h, vid if vid is not None else int(x * 1000))


# ---------------------------------------------------------------------------
# Basic operation contracts
# ---------------------------------------------------------------------------


def test_empty_pull_returns_bound():
    d = BlockDS(M=4, B=B_BIG)
    assert d.pull() == ([], B_BIG)
    assert len(d) == 0


def test_insert_and_pull_partition_and_bounds():
    M = 3
    d = BlockDS(M=M, B=B_BIG)
    values = {i: k(float(i)) for i in range(10)}
    for key_id, val in values.items():
        d.insert(key_id, val)
    assert len(d) == 10

    seen: list[int] = []
    pulls: list[tuple[list[int], Key]] = []
    while len(d) > 0:
        S, x = d.pull()
        assert len(S) <= M
        pulls.append((S, x))
        seen.extend(S)

    # Concatenated pulls partition the inserted keys.
    assert sorted(seen) == list(range(10))

    # Final pull returns bound B.
    assert pulls[-1][1] == B_BIG

    # All non-final pulls returned exactly M elements.
    for S, _x in pulls[:-1]:
        assert len(S) == M

    # max(value over S') < x <= min(value over remaining), for each pull.
    remaining = list(range(10))
    for S, x in pulls:
        s_values = [values[key_id] for key_id in S]
        remaining = [key_id for key_id in remaining if key_id not in S]
        assert max(s_values) < x
        if remaining:
            rem_values = [values[key_id] for key_id in remaining]
            assert x <= min(rem_values)
        else:
            assert x == B_BIG


def test_pull_when_remaining_le_M_returns_all_with_bound_B():
    d = BlockDS(M=5, B=B_BIG)
    for i in range(3):
        d.insert(i, k(float(i)))
    S, x = d.pull()
    assert sorted(S) == [0, 1, 2]
    assert x == B_BIG
    assert len(d) == 0


# ---------------------------------------------------------------------------
# Duplicate-key handling
# ---------------------------------------------------------------------------


def test_insert_duplicate_smaller_value_replaces():
    d = BlockDS(M=4, B=B_BIG)
    d.insert(0, k(5.0))
    d.insert(0, k(2.0))  # smaller -> replace
    S, x = d.pull()
    assert S == [0]
    assert x == B_BIG


def test_insert_duplicate_larger_value_ignored():
    d = BlockDS(M=4, B=B_BIG)
    d.insert(0, k(2.0))
    d.insert(0, k(5.0))  # larger -> no-op
    assert len(d) == 1
    S, x = d.pull()
    assert S == [0]


def test_batch_prepend_duplicate_replaces_existing_with_smaller_value():
    d = BlockDS(M=4, B=B_BIG)
    d.insert(0, k(5.0))
    d.batch_prepend([(0, k(1.0))])  # smaller, precedes everything
    assert len(d) == 1
    S, x = d.pull()
    assert S == [0]


# ---------------------------------------------------------------------------
# BatchPrepend ordering
# ---------------------------------------------------------------------------


def test_batch_prepend_then_pull_comes_out_first():
    M = 4
    d = BlockDS(M=M, B=B_BIG)
    for i in range(10, 14):
        d.insert(i, k(float(i)))

    # Values strictly below the current minimum (10.0).
    prepend_items = [(i, k(float(i))) for i in range(0, 4)]
    d.batch_prepend(prepend_items)

    S, x = d.pull()
    assert sorted(S) == [0, 1, 2, 3]
    # Separating bound must precede the still-present inserted values.
    assert x <= k(10.0)


def test_batch_prepend_larger_than_M_splits_correctly():
    M = 3
    d = BlockDS(M=M, B=B_BIG)
    for i in range(20, 23):
        d.insert(i, k(float(i)))

    # 7 prepended items, all below 20.0.
    prepend_items = [(i, k(float(i))) for i in range(0, 7)]
    d.batch_prepend(prepend_items)
    d._check_invariants()

    seen: list[int] = []
    while len(d) > 0:
        S, x = d.pull()
        seen.extend(S)
    assert sorted(seen) == list(range(0, 7)) + [20, 21, 22]


# ---------------------------------------------------------------------------
# White-box invariants
# ---------------------------------------------------------------------------


def test_invariants_after_many_inserts():
    M = 3
    d = BlockDS(M=M, B=B_BIG)
    rng = random.Random(42)
    ids = list(range(60))
    rng.shuffle(ids)
    for i, key_id in enumerate(ids):
        d.insert(key_id, k(float(i)))
        d._check_invariants()
    assert len(d) == 60


# ---------------------------------------------------------------------------
# Model-based randomized stress test
# ---------------------------------------------------------------------------


class _NaiveModel:
    """A dict-based reference model matching BlockDS's external contract.

    ``pull`` = sort all (key, value) pairs by value, take the ``M`` smallest
    (or everything if <= M), bound = next-smallest remaining value, or ``B``
    if nothing remains. SPEC.md S8.1.
    """

    def __init__(self, M: int, B: Key) -> None:
        self.M = M
        self.B = B
        self.data: dict[int, Key] = {}

    def insert(self, key_id: int, value: Key) -> None:
        if key_id not in self.data or value < self.data[key_id]:
            self.data[key_id] = value

    def batch_prepend(self, items: list[tuple[int, Key]]) -> None:
        dedup: dict[int, Key] = {}
        for key_id, value in items:
            if key_id not in dedup or value < dedup[key_id]:
                dedup[key_id] = value
        for key_id, value in dedup.items():
            self.insert(key_id, value)

    def pull(self) -> tuple[list[int], Key]:
        if not self.data:
            return [], self.B
        items = sorted(self.data.items(), key=lambda kv: kv[1])
        if len(items) <= self.M:
            for key_id, _v in items:
                del self.data[key_id]
            return [key_id for key_id, _v in items], self.B
        chosen = items[: self.M]
        bound = items[self.M][1]
        for key_id, _v in chosen:
            del self.data[key_id]
        return [key_id for key_id, _v in chosen], bound

    def current_min(self) -> Key:
        if not self.data:
            return self.B
        return min(self.data.values())


@pytest.mark.parametrize("seed", range(8))
def test_blockds_matches_naive_model(seed: int):
    rng = random.Random(1000 + seed)
    M = rng.choice([1, 2, 3, 8])
    B: Key = (1e9, INF_INT, INF_INT)
    d = BlockDS(M=M, B=B)
    model = _NaiveModel(M=M, B=B)

    next_insert_id = 0  # increasing -> always the largest values so far
    next_prepend_id = -1  # decreasing -> always smaller than everything in D
    key_space = 30

    for _ in range(300):
        op = rng.choice(["insert", "insert", "batch_prepend", "pull"])
        if op == "insert":
            key_id = rng.randrange(key_space)
            value = k(float(next_insert_id), vid=next_insert_id)
            next_insert_id += 1
            d.insert(key_id, value)
            model.insert(key_id, value)
        elif op == "batch_prepend":
            n_items = rng.randint(1, 2 * M + 1)
            keys_chosen = rng.sample(range(key_space), min(n_items, key_space))
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

    # Drain both and compare fully.
    while len(model.data) > 0:
        d_S, d_x = d.pull()
        m_S, m_x = model.pull()
        assert sorted(d_S) == sorted(m_S)
        assert d_x == m_x
    assert d.pull() == ([], B)
