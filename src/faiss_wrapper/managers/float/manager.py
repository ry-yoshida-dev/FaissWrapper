from __future__ import annotations

import faiss
from dataclasses import dataclass
from typing import Callable

from ...types import FloatVectorArray
from ...dtype import FaissDType
from ...manager import FaissManager
from ...method import FaissSearchMethod
from ...result import FaissResult, FaissResults

@dataclass
class FaissFloatManager(FaissManager):
    """
    Faiss manager for float search.
    
    Parameters:
    ----------
    dimension: int
        The dimension of the vectors.
    metric: FaissMetric
        The metric to use.
    """

    def _validate_parameters(self) -> None:
        """
        Validate the parameters of the manager.
        """
        super()._validate_parameters()

    def add(self, vectors: FloatVectorArray) -> None:
        """
        Add float vectors to the index.

        Parameters
        ----------
        vectors : FloatVectorArray
            Float vectors to add, shape ``(n, dimension)``.
        """
        super().add(vectors)

    def search(self, vectors: FloatVectorArray, k: int = 10) -> FaissResult | FaissResults:
        """
        Search the index for nearest neighbors among float query vectors.

        Parameters
        ----------
        vectors : FloatVectorArray
            Query vectors, shape ``(n_query, dimension)`` or ``(dimension,)``.
        k : int, optional
            Number of nearest neighbors to return per query.

        Returns
        -------
        FaissResult | FaissResults
            ``FaissResult`` for a single query vector; ``FaissResults`` for a batch.
        """
        return super().search(vectors, k)

    @property
    def dtype(self) -> FaissDType:
        """
        Returns the data type of the vectors.
        
        Returns:
        ----------
        FaissDType: The data type of the vectors.
        """
        return FaissDType.FLOAT

    @property
    def save_function(self) -> Callable[[faiss.Index, str], None]:
        """
        Returns the save function for the float index.
        
        Returns:
        ----------
        Callable[[faiss.Index, str], None]: The save function for the float index.
        """
        return faiss.write_index

    @property
    def load_function(self) -> Callable[[str], faiss.Index]:
        """
        Returns the load function for the float index.
        
        Returns:
        ----------
        Callable[[str], faiss.Index]: The load function for the float index.
        """
        return faiss.read_index