import numpy as np
import faiss
from dataclasses import dataclass

from ..manager import FaissBinaryManager
from ...metric import FaissMetric
from ...method import FaissSearchMethod


@dataclass
class FaissBinaryHNSWManager(FaissBinaryManager):
    """
    Faiss manager for binary HNSW (IndexBinaryHNSW), Hamming distance.

    Parameters:
    ----------
    dimension: int
        Number of bits per vector (multiple of 8).
    metric: FaissMetric
        Must be HAMMING.
    M: int
        Number of bi-directional links per node (typical 16–64).
    efSearch: int
        Width of the dynamic candidate list at search time (larger -> more accurate).
    """

    M: int = 32
    efSearch: int = 64

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.M <= 0:
            raise ValueError("M must be greater than 0.")
        if self.efSearch <= 0:
            raise ValueError("efSearch must be greater than 0.")

    def _build_index(self):
        return faiss.IndexBinaryHNSW(self.dimension, self.M)

    def add(
        self, 
        vectors: np.ndarray
        ) -> None:
        """
        Add vectors to the index.

        Parameters:
        ----------
        vectors: np.ndarray
            The vectors to add to the index.
        """
        super().add(vectors)
        self.index.hnsw.efSearch = self.efSearch

    @property
    def search_method(self) -> FaissSearchMethod:
        return FaissSearchMethod.HNSW
