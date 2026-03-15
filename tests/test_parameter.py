"""
Tests for FaissParameter: all valid combinations build successfully,
invalid combinations raise ValueError.

Run from project root: PYTHONPATH=. python tests/test_parameter.py
"""

import itertools

from src.faiss_wrapper.dtype import FaissDType
from src.faiss_wrapper.method import FaissSearchMethod
from src.faiss_wrapper.metric import FaissMetric
from src.faiss_wrapper.parameter import FaissParameter


def iter_valid_combinations():
    """Yield all (dtype, method, metric) that FaissParameter accepts."""
    for dtype in FaissDType:
        for method in dtype.supported_methods:
            supported = (
                method.float_supported_metrics
                if dtype is FaissDType.FLOAT
                else method.binary_supported_metrics
            )
            for metric in supported:
                yield dtype, method, metric


def test_all_combinations():
    """Valid combinations build; invalid ones raise ValueError (invalid combos are printed)."""
    valid_set = set(iter_valid_combinations())
    invalid_failed = []

    for dtype, method, metric in itertools.product(
        FaissDType, FaissSearchMethod, FaissMetric
    ):
        combo = (dtype, method, metric)
        try:
            param = FaissParameter(dtype=dtype, method=method, metric=metric)
            assert param.manager_class is not None
            assert combo in valid_set, f"Unexpected success: {dtype.value} + {method.value} + {metric.value}"
        except ValueError as e:
            if combo in valid_set:
                raise AssertionError(f"Valid combo failed: {dtype.value} + {method.value} + {metric.value} -> {e}")
            invalid_failed.append((dtype.value, method.value, metric.value))

    if invalid_failed:
        print("Invalid combinations (correctly rejected):")
        for d, m, g in invalid_failed:
            print(f"  {d}, {m}, {g}")


if __name__ == "__main__":
    test_all_combinations()
    print("All parameter tests passed.")
