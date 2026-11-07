"""Tests for FaissResult and FaissResults: sorting, slicing, and validation."""

import numpy as np
import pytest

from faiss_wrapper import FaissResult, FaissResults


def test_result_sorts_distances_ascending() -> None:
    result = FaissResult(
        values=np.array([3.0, 1.0, 2.0], dtype=np.float32),
        indices=np.array([30, 10, 20], dtype=np.int64),
        is_larger_nearer=False,
    )

    np.testing.assert_array_equal(result.values, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.indices, [10, 20, 30])
    assert result.nearest_index == 10
    assert result.nearest_value == pytest.approx(1.0)


def test_result_sorts_similarities_descending() -> None:
    result = FaissResult(
        values=np.array([0.1, 0.9, 0.5], dtype=np.float32),
        indices=np.array([1, 9, 5], dtype=np.int64),
        is_larger_nearer=True,
    )

    np.testing.assert_array_equal(result.indices, [9, 5, 1])


def test_result_flattens_single_row() -> None:
    result = FaissResult(
        values=np.array([[2.0, 1.0]], dtype=np.float32),
        indices=np.array([[2, 1]], dtype=np.int64),
        is_larger_nearer=False,
    )

    assert result.values.shape == (2,)
    assert len(result) == 2


def test_result_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        _ = FaissResult(
            values=np.array([1.0, 2.0], dtype=np.float32),
            indices=np.array([1], dtype=np.int64),
            is_larger_nearer=False,
        )


def test_result_top_k_and_bounds() -> None:
    result = FaissResult(
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        indices=np.array([1, 2, 3], dtype=np.int64),
        is_larger_nearer=False,
    )

    np.testing.assert_array_equal(result.top_k_neighbors(2).indices, [1, 2])
    with pytest.raises(ValueError):
        _ = result.top_k_neighbors(0)
    with pytest.raises(ValueError):
        _ = result.top_k_neighbors(4)


def test_result_filter_by_value() -> None:
    result = FaissResult(
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        indices=np.array([1, 2, 3], dtype=np.int64),
        is_larger_nearer=False,
    )

    np.testing.assert_array_equal(result.filter_by_value(2.0).indices, [1, 2])
    with pytest.warns(UserWarning, match="No neighbors"):
        empty = result.filter_by_value(0.5)
    assert len(empty) == 0


def test_results_sort_each_row_and_index_rows() -> None:
    results = FaissResults(
        values=np.array([[2.0, 1.0], [5.0, 4.0]], dtype=np.float32),
        indices=np.array([[2, 1], [5, 4]], dtype=np.int64),
        is_larger_nearer=False,
    )

    assert len(results) == 2
    assert results.neighbor_count == 2
    np.testing.assert_array_equal(results.nearest_values, [1.0, 4.0])
    np.testing.assert_array_equal(results[1].indices, [4, 5])
    np.testing.assert_array_equal(results[-1].indices, [4, 5])
    assert results.nearest_neighbors.values.shape == (2, 1)


def test_results_from_result_round_trip() -> None:
    result = FaissResult(
        values=np.array([1.0, 2.0], dtype=np.float32),
        indices=np.array([1, 2], dtype=np.int64),
        is_larger_nearer=False,
    )

    results = FaissResults.from_result(result)

    assert len(results) == 1
    np.testing.assert_array_equal(results[0].indices, result.indices)
