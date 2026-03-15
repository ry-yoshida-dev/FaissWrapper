"""
Smoke tests for FaissManager: add, search, save, and load for each
dtype / method / metric combination supported by FaissParameter.
"""

import numpy as np

from src.faiss_wrapper.manager import FaissManager
from src.faiss_wrapper.dtype import FaissDType
from src.faiss_wrapper.parameter import FaissParameter


def create_float_test_data(
    n_db: int = 10_000,
    n_query: int = 2,
    dim: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build random float32 database and query matrices for float indices.

    Parameters:
    ----------
    n_db: int
        Number of database vectors (rows).
    n_query: int
        Number of query vectors (rows).
    dim: int
        Vector dimension (columns); same for db and queries.
    seed: int
        RNG seed for reproducibility.

    Returns:
    ----------
    tuple[np.ndarray, np.ndarray, int]
        ``(db, queries, dim)`` where ``db`` and ``queries`` are float32,
        shapes ``(n_db, dim)`` and ``(n_query, dim)``.
    """
    rng = np.random.default_rng(seed)
    db = rng.random((n_db, dim), dtype=np.float32)
    queries = rng.random((n_query, dim), dtype=np.float32)
    return db, queries, dim


def create_binary_test_data(
    n_db: int = 10_000,
    n_query: int = 2,
    bit_size: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build random packed binary rows (uint8) for binary / Hamming indices.

    Parameters:
    ----------
    n_db: int
        Number of database vectors (rows).
    n_query: int
        Number of query vectors (rows).
    bit_size: int
        Hamming dimension in bits; row width in bytes is ``bit_size // 8``.
    seed: int
        RNG seed for reproducibility.

    Returns:
    ----------
    tuple[np.ndarray, np.ndarray, int]
        ``(db, queries, bit_size)`` where ``db`` and ``queries`` are uint8,
        shapes ``(n_db, bit_size // 8)`` and ``(n_query, bit_size // 8)``.
        The third element is the dimension passed to the binary manager.
    """
    rng = np.random.default_rng(seed)
    d_bytes = bit_size // 8
    db = rng.integers(0, 256, size=(n_db, d_bytes), dtype=np.uint8)
    queries = rng.integers(0, 256, size=(n_query, d_bytes), dtype=np.uint8)
    return db, queries, bit_size


if __name__ == "__main__":
    save_path = "tmp.faiss"

    for dtype in FaissDType:
        for method in dtype.supported_methods:
            supported_metrics = (
                method.float_supported_metrics
                if dtype is FaissDType.FLOAT
                else method.binary_supported_metrics
            )
            for metric in supported_metrics:
                parameter = FaissParameter(
                    dtype=dtype,
                    method=method,
                    metric=metric,
                )

                if dtype is FaissDType.FLOAT:
                    db, queries, dimension = create_float_test_data()
                else:
                    db, queries, dimension = create_binary_test_data()

                print(f"{dtype.value} {method.value} {metric.value}")
                manager_class = parameter.manager_class
                manager: FaissManager = manager_class(
                    metric=metric,
                    dimension=dimension,
                )
                manager.add(db)
                result = manager.search(queries, k=2)
                assert result.distances.shape[0] == queries.shape[0]
                manager.save(path=save_path)
                manager.load(path=save_path)
