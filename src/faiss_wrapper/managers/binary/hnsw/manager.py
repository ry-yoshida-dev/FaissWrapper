import faiss # type: ignore
from dataclasses import dataclass

from ....types import BinaryVectorArray
from ..manager import FaissBinaryManager
from ....method import FaissSearchMethod


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

    def _build_index(self) -> faiss.Index:
        """
        Build the index.
        
        Returns:
        ----------
        faiss.Index: The built index.
        """
        return faiss.IndexBinaryHNSW(self.dimension, self.M) # type: ignore

    def add(
        self, 
        vectors: BinaryVectorArray
        ) -> None:
        """
        Add vectors to the index.

        Parameters:
        ----------
        vectors: BinaryVectorArray
            The vectors to add to the index.
        """
        super().add(vectors)
        self.index.hnsw.efSearch = self.efSearch # type: ignore

    @property
    def search_method(self) -> FaissSearchMethod:
        """
        Returns the search method.
        
        Returns:
        ----------
        FaissSearchMethod: The search method.
        """
        return FaissSearchMethod.HNSW
