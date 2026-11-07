"""
Smoke tests for the Faiss managers: add, search, save, and load for each
dtype / method / metric combination supported by FaissParameter.
"""

from pathlib import Path

import numpy as np
import pytest

from faiss_wrapper import (
    BinaryVectorArray,
    FaissDType,
    FaissFloatFlatManager,
    FaissMetric,
    FaissParameter,
    FaissSearchMethod,
    FloatVectorArray,
)

DATABASE_SIZE = 10_000
QUERY_COUNT = 2
FLOAT_DIMENSION = 32
BINARY_BIT_SIZE = 256
NEIGHBOR_COUNT = 2


def create_float_vectors(row_count: int, seed: int) -> FloatVectorArray:
    rng = np.random.default_rng(seed)
    return rng.random((row_count, FLOAT_DIMENSION), dtype=np.float32)


def create_binary_vectors(row_count: int, seed: int) -> BinaryVectorArray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(row_count, BINARY_BIT_SIZE // 8), dtype=np.uint8)


def supported_combinations(
    dtype: FaissDType,
) -> list[tuple[FaissSearchMethod, FaissMetric]]:
    return [
        (method, metric)
        for method in dtype.supported_methods
        for metric in dtype.get_supported_metrics(method)
    ]


@pytest.mark.parametrize(
    ("method", "metric"),
    supported_combinations(FaissDType.FLOAT),
    ids=str,
)
def test_float_manager_round_trip(
    method: FaissSearchMethod, metric: FaissMetric, tmp_path: Path
) -> None:
    parameter = FaissParameter(dtype=FaissDType.FLOAT, method=method, metric=metric)
    manager = parameter.float_manager_class(metric=metric, dimension=FLOAT_DIMENSION)
    database = create_float_vectors(DATABASE_SIZE, seed=42)
    queries = create_float_vectors(QUERY_COUNT, seed=43)

    manager.add(database)
    results = manager.search_batch(queries, k=NEIGHBOR_COUNT)
    result = manager.search_single(queries[:1].reshape(-1), k=NEIGHBOR_COUNT)

    assert results.values.shape == (QUERY_COUNT, NEIGHBOR_COUNT)
    assert results.indices.shape == (QUERY_COUNT, NEIGHBOR_COUNT)
    assert len(result) == NEIGHBOR_COUNT

    path = tmp_path / "index.faiss"
    manager.save(path=str(path))
    manager.load(path=str(path))
    reloaded = manager.search_batch(queries, k=NEIGHBOR_COUNT)
    np.testing.assert_array_equal(reloaded.indices, results.indices)


@pytest.mark.parametrize(
    ("method", "metric"),
    supported_combinations(FaissDType.BINARY),
    ids=str,
)
def test_binary_manager_round_trip(
    method: FaissSearchMethod, metric: FaissMetric, tmp_path: Path
) -> None:
    parameter = FaissParameter(dtype=FaissDType.BINARY, method=method, metric=metric)
    manager = parameter.binary_manager_class(metric=metric, dimension=BINARY_BIT_SIZE)
    database = create_binary_vectors(DATABASE_SIZE, seed=42)
    queries = create_binary_vectors(QUERY_COUNT, seed=43)

    manager.add(database)
    results = manager.search_batch(queries, k=NEIGHBOR_COUNT)
    result = manager.search_single(queries[:1].reshape(-1), k=NEIGHBOR_COUNT)

    assert results.values.shape == (QUERY_COUNT, NEIGHBOR_COUNT)
    assert len(result) == NEIGHBOR_COUNT

    path = tmp_path / "index.faiss"
    manager.save(path=str(path))
    manager.load(path=str(path))
    reloaded = manager.search_batch(queries, k=NEIGHBOR_COUNT)
    np.testing.assert_array_equal(reloaded.indices, results.indices)


def test_flat_l2_search_finds_exact_match() -> None:
    manager = FaissFloatFlatManager(metric=FaissMetric.L2, dimension=FLOAT_DIMENSION)
    database = create_float_vectors(100, seed=0)
    manager.add(database)

    result = manager.search_single(database[7:8].reshape(-1), k=3)

    assert result.nearest_index == 7
    assert result.nearest_value == pytest.approx(0.0)


def test_invalid_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="Dimension"):
        _ = FaissFloatFlatManager(metric=FaissMetric.L2, dimension=0)


def test_search_batch_rejects_single_vector() -> None:
    manager = FaissFloatFlatManager(metric=FaissMetric.L2, dimension=FLOAT_DIMENSION)
    with pytest.raises(ValueError, match="2-D"):
        _ = manager.search_batch(create_float_vectors(1, seed=0).reshape(-1))


def test_search_single_rejects_batch() -> None:
    manager = FaissFloatFlatManager(metric=FaissMetric.L2, dimension=FLOAT_DIMENSION)
    with pytest.raises(ValueError, match="1-D"):
        _ = manager.search_single(create_float_vectors(2, seed=0))


def test_save_rejects_path_without_faiss_extension(tmp_path: Path) -> None:
    manager = FaissFloatFlatManager(metric=FaissMetric.L2, dimension=FLOAT_DIMENSION)
    with pytest.raises(ValueError, match=".faiss"):
        manager.save(path=str(tmp_path / "index.bin"))
