import numpy as np
import faiss
from dataclasses import dataclass

from ..manager import FaissBinaryManager
from ...metric import FaissMetric
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

    def _build_index(self):
        quantizer = faiss.IndexBinaryFlat(self.dimension)
        return faiss.IndexBinaryIVF(quantizer, self.dimension, self.nlist)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.index.nprobe = self.nprobe

    def add(self, vectors: np.ndarray) -> None:
        """Train the IVF index then add vectors (required for IndexBinaryIVF)."""
        self.index.train(vectors)
        super().add(vectors)

    @property
    def search_method(self) -> FaissSearchMethod:
        return FaissSearchMethod.IVF
