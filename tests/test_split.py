"""Stage S1: seeded three-way splits."""

import numpy as np
import pytest

from fastforest.split import Splits, make_splits


def labels() -> np.ndarray:
    return np.repeat([0, 1, 2], [50, 30, 20])


def test_stratified_sizes_and_disjoint():
    s = make_splits(labels(), random_state=0)
    assert s.sizes == {"train": 60, "val": 20, "test": 20}
    all_idx = np.concatenate(list(s))
    assert len(np.unique(all_idx)) == 100
    assert np.array_equal(np.sort(all_idx), np.arange(100))


def test_stratified_preserves_class_proportions():
    y = labels()
    s = make_splits(y, random_state=0)
    for part in s:
        counts = np.bincount(y[part], minlength=3) / len(part)
        assert np.allclose(counts, [0.5, 0.3, 0.2], atol=0.05)


def test_random_split_same_sizes_but_may_not_stratify():
    s = make_splits(labels(), method="random", random_state=0)
    assert s.sizes == {"train": 60, "val": 20, "test": 20}


def test_deterministic_with_seed():
    a = make_splits(labels(), random_state=7)
    b = make_splits(labels(), random_state=7)
    c = make_splits(labels(), random_state=8)
    assert np.array_equal(a.test, b.test) and np.array_equal(a.train, b.train)
    assert not np.array_equal(a.test, c.test)


def test_custom_fractions():
    s = make_splits(labels(), val_size=0.1, test_size=0.3, random_state=0)
    assert s.sizes == {"train": 60, "val": 10, "test": 30}


def test_index_method_uses_given_test_rows_and_splits_rest_75_25():
    test_idx = [0, 1, 2, 3, 50, 51, 80, 81]
    s = make_splits(labels(), method="index", test_idx=test_idx, random_state=0)
    assert s.test.tolist() == sorted(test_idx)
    assert s.sizes["train"] == 69 and s.sizes["val"] == 23


def test_index_method_validation():
    with pytest.raises(ValueError, match="requires test_idx"):
        make_splits(labels(), method="index")
    with pytest.raises(ValueError, match="positions"):
        make_splits(labels(), method="index", test_idx=[200])


def test_invalid_sizes_and_method_raise():
    with pytest.raises(ValueError):
        make_splits(labels(), val_size=0.6, test_size=0.5)
    with pytest.raises(ValueError, match="no rows for training"):
        make_splits(labels(), val_size=0.5, test_size=0.499)
    with pytest.raises(ValueError):
        make_splits(labels(), test_size=0.0)
    with pytest.raises(ValueError, match="method"):
        make_splits(labels(), method="bogus")
    with pytest.raises(ValueError):
        make_splits([0, 1])


def test_stratified_fallback_warns_for_singleton_class():
    y = np.array([0] * 30 + [1] * 30 + [2])
    with pytest.warns(UserWarning, match="falling back"):
        s = make_splits(y, random_state=0)
    assert sum(s.sizes.values()) == 61


def test_from_arrays_validation():
    with pytest.raises(ValueError, match="overlap"):
        Splits.from_arrays([0, 1], [1, 2], [3])
    with pytest.raises(ValueError, match="duplicate"):
        Splits.from_arrays([0, 0], [1], [2])
    with pytest.raises(ValueError, match="empty"):
        Splits.from_arrays([0], [], [2])
    with pytest.raises(ValueError, match="range"):
        Splits.from_arrays([0], [1], [5], n_samples=3)
    s = Splits.from_arrays([2, 0], [1], [3])
    assert s.train.tolist() == [0, 2]
    assert s.to_dict() == {"train": [0, 2], "val": [1], "test": [3]}
