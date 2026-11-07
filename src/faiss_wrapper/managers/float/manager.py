from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass

import faiss
from typing_extensions import override

from ...dtype import FaissDType
from ...manager import FaissManager
from ...result import FaissResult, FaissResults
from ...types import FloatVectorArray


@dataclass
class FaissFloatManager(FaissManager[FloatVectorArray, faiss.Index], ABC):
    """
    Faiss manager for float search.

    Parameters
    ----------
    dimension: int
        The dimension of the vectors.
    metric: FaissMetric
        The metric to use.
    """

    @override
    def _validate_parameters(self) -> None:
        """
        Validate the parameters of the manager.
        """
        super()._validate_parameters()

    @override
    def add(self, vectors: FloatVectorArray) -> None:
        """
        Add float vectors to the index.

        Parameters
        ----------
        vectors : FloatVectorArray
            Float vectors to add, shape ``(n, dimension)``.
        """
        super().add(vectors)

    @override
    def search_batch(self, vectors: FloatVectorArray, k: int = 10) -> FaissResults:
        """
        Search the index for nearest neighbors among a batch of float query vectors.

        Parameters
        ----------
        vectors : FloatVectorArray
            Query vectors, shape ``(n_query, dimension)``.
        k : int, optional
            Number of nearest neighbors to return per query.

        Returns
        -------
        FaissResults
            Nearest neighbors for each query, shape ``(n_query, k)``.
        """
        return super().search_batch(vectors, k)

    @override
    def search_single(self, vector: FloatVectorArray, k: int = 10) -> FaissResult:
        """
        Search the index for nearest neighbors among a single float query vector.

        Parameters
        ----------
        vector : FloatVectorArray
            Query vector, shape ``(dimension,)``.
        k : int, optional
            Number of nearest neighbors to return.

        Returns
        -------
        FaissResult
            Nearest neighbors for the query, shape ``(k,)``.
        """
        return super().search_single(vector, k)

    @property
    @override
    def dtype(self) -> FaissDType:
        """
        Returns the data type of the vectors.

        Returns
        -------
        FaissDType: The data type of the vectors.
        """
        return FaissDType.FLOAT

    @property
    @override
    def save_function(self) -> Callable[[faiss.Index, str], None]:
        """
        Returns the save function for the float index.

        Returns
        -------
        Callable[[faiss.Index, str], None]: The save function for the float index.
        """
        return faiss.write_index

    @property
    @override
    def load_function(self) -> Callable[[str], faiss.Index]:
        """
        Returns the load function for the float index.

        Returns
        -------
        Callable[[str], faiss.Index]: The load function for the float index.
        """
        return faiss.read_index
