import numpy as np
import faiss # type: ignore
from dataclasses import dataclass

from ..manager import FaissBinaryManager
from ...method import FaissSearchMethod


@dataclass
class FaissBinaryIVFManager(FaissBinaryManager):
    """
    Faiss manager for binary IVF (IndexBinaryIVF) with Hamming distance.

    Parameters:
    ----------
    dimension: int
        Number of bits per vector (multiple of 8). Stored as (n, dimension // 8) uint8.
    metric: FaissMetric
        Must be HAMMING (binary IVF always uses Hamming).
    nlist: int
        Number of coarse clusters (Voronoi cells).
    nprobe: int
        Number of clusters to visit at search time (larger -> more accurate, slower).
    """

    nlist: int = 100
    nprobe: int = 1

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.nlist <= 0:
            raise ValueError("nlist must be greater than 0.")
        if self.nprobe <= 0:
            raise ValueError("nprobe must be greater than 0.")

    def _build_index(self) -> faiss.Index:
        """
        Build the index.
        
        Returns:
        ----------
        faiss.Index: The built index.
        """
        quantizer = faiss.IndexBinaryFlat(self.dimension)
        return faiss.IndexBinaryIVF(quantizer, self.dimension, self.nlist) # type: ignore

    def __post_init__(self) -> None:
        super().__post_init__()
        self.index.nprobe = self.nprobe # type: ignore

    def add(self, vectors: np.ndarray) -> None:
        """
        Train the IVF index then add vectors (required for IndexBinaryIVF).
        
        Parameters:
        ----------
        vectors: np.ndarray
            The vectors to train the index on.
        """
        self.index.train(vectors) # type: ignore
        super().add(vectors)

    @property
    def search_method(self) -> FaissSearchMethod:
        """
        Returns the search method.
        
        Returns:
        ----------
        FaissSearchMethod: The search method.
        """
        return FaissSearchMethod.IVF
