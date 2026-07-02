from __future__ import annotations

import faiss # type: ignore
from dataclasses import dataclass

from ....types import FloatVectorArray
from ..manager import FaissFloatManager
from ....method import FaissSearchMethod


@dataclass
class FaissFloatIVFPQManager(FaissFloatManager):
    """
    Faiss manager for IVFPQ (inverted file + product quantization) approximate search.

    Parameters:
    ----------
    dimension: int
        The dimension of the vectors.
    metric: FaissMetric
        The metric to use. Only L2 and INNER_PRODUCT are supported.
    nlist: int = 100
        Number of clusters for the coarse quantizer.
    m: int = 8
        Number of subquantizers (dimension must be divisible by m).
    nbits: int = 8
        Number of bits per subquantizer code.
    nprobe: int = 10
        Number of clusters to search at query time.
    """
    nlist: int = 100
    m: int = 8
    nbits: int = 8
    nprobe: int = 10

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.nlist <= 0:
            raise ValueError("nlist must be greater than 0.")
        if self.dimension % self.m != 0:
            raise ValueError(f"dimension ({self.dimension}) must be divisible by m ({self.m}).")

    def _build_index(self) -> faiss.Index:
        metric_obj = self.metric.object
        quantizer = faiss.IndexFlat(self.dimension, metric_obj)
        return faiss.IndexIVFPQ(
            quantizer, self.dimension, self.nlist, self.m, self.nbits, metric_obj
        )

    def add(self, vectors: FloatVectorArray) -> None:
        """
        Train the index, add vectors, and set nprobe (IVFPQ requires training).
        
        Parameters:
        ----------
        vectors: FloatVectorArray
            The vectors to train the index on.
        """
        self.index.train(vectors) # type: ignore
        super().add(vectors)
        self.index.nprobe = self.nprobe # type: ignore

    @property
    def search_method(self) -> FaissSearchMethod:
        """
        Returns the search method.
        
        Returns:
        ----------
        FaissSearchMethod: The search method.
        """
        return FaissSearchMethod.IVFPQ
