from __future__ import annotations

import faiss # type: ignore
from dataclasses import dataclass

from ..manager import FaissFloatManager
from ....method import FaissSearchMethod


@dataclass
class FaissFloatLSHManager(FaissFloatManager):
    """
    Faiss manager for LSH (Locality-Sensitive Hashing) approximate search.
    Uses Hamming distance on the LSH hash.

    Parameters:
    ----------
    dimension: int
        The dimension of the vectors.
    metric: FaissMetric
        Must be FaissMetric.HAMMING (LSH measures Hamming distance on the hash).
    nbits: int = 128
        Number of bits in the LSH hash (number of random projections).
    """
    nbits: int = 128

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.nbits <= 0:
            raise ValueError("nbits must be greater than 0.")

    def _build_index(self) -> faiss.Index:
        """
        Build the index.
        
        Returns:
        ----------
        faiss.Index: The built index.
        """
        return faiss.IndexLSH(self.dimension, self.nbits)

    @property
    def search_method(self) -> FaissSearchMethod:
        """
        Returns the search method.
        
        Returns:
        ----------
        FaissSearchMethod: The search method.
        """
        return FaissSearchMethod.LSH
