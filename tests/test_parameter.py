"""
Tests for FaissParameter: all valid combinations build successfully,
invalid combinations raise ValueError.
"""

import itertools

import pytest

from faiss_wrapper import (
    FaissBinaryManager,
    FaissDType,
    FaissFloatManager,
    FaissMetric,
    FaissParameter,
    FaissSearchMethod,
)

Combination = tuple[FaissDType, FaissSearchMethod, FaissMetric]


def valid_combinations() -> set[Combination]:
    return {
        (dtype, method, metric)
        for dtype in FaissDType
        for method in dtype.supported_methods
        for metric in dtype.get_supported_metrics(method)
    }


def combination_key(combination: Combination) -> tuple[str, str, str]:
    dtype, method, metric = combination
    return dtype.value, method.value, metric.value


VALID_COMBINATIONS: list[Combination] = sorted(
    valid_combinations(), key=combination_key
)
INVALID_COMBINATIONS: list[Combination] = sorted(
    set(itertools.product(FaissDType, FaissSearchMethod, FaissMetric))
    - set(VALID_COMBINATIONS),
    key=combination_key,
)


@pytest.mark.parametrize(
    ("dtype", "method", "metric"),
    VALID_COMBINATIONS,
    ids=str,
)
def test_valid_combination_selects_matching_manager(
    dtype: FaissDType, method: FaissSearchMethod, metric: FaissMetric
) -> None:
    parameter = FaissParameter(dtype=dtype, method=method, metric=metric)
    manager_class = parameter.manager_class
    match dtype:
        case FaissDType.FLOAT:
            assert issubclass(manager_class, FaissFloatManager)
        case FaissDType.BINARY:
            assert issubclass(manager_class, FaissBinaryManager)


@pytest.mark.parametrize(
    ("dtype", "method", "metric"),
    INVALID_COMBINATIONS,
    ids=str,
)
def test_invalid_combination_is_rejected(
    dtype: FaissDType, method: FaissSearchMethod, metric: FaissMetric
) -> None:
    with pytest.raises(ValueError, match="not supported"):
        _ = FaissParameter(dtype=dtype, method=method, metric=metric)
