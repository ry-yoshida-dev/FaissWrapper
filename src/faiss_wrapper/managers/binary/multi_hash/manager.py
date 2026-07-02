from __future__ import annotations

import faiss # type: ignore
from dataclasses import dataclass

from ..manager import FaissBinaryManager
from ....method import FaissSearchMethod

@dataclass
class FaissBinaryMultiHashManager(FaissBinaryManager):
    """
    Faiss manager for IndexBinaryMultiHash, Hamming distance.

    Corresponds to faiss.IndexBinaryMultiHash(d, nhash, b) in faiss.ipynb:
    - d: bit length of each vector.
    - nhash: number of hash tables (more → better recall, slower build/search).
    - b: bits per hash table.

    Vectors are uint8 arrays of shape (n, d // 8).

    Parameters
    ----------
    dimension : int
        Number of bits per vector (d_bits).
    metric : FaissMetric
        Must be HAMMING.
    nhash : int
        Number of hash tables.
    hash_bits : int
        b — bits per table.
    """

    nhash: int = 4
    hash_bits: int = 12

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.nhash <= 0:
            raise ValueError("nhash must be greater than 0.")
        if self.hash_bits <= 0:
            raise ValueError("hash_bits must be greater than 0.")
        if self.nhash * self.hash_bits > self.dimension:
            raise ValueError(
                "nhash * hash_bits cannot exceed dimension (bit length); "
                "Faiss partitions distinct bit ranges per table."
            )

    def _build_index(self) -> faiss.Index:
        """
        Build the index.
        
        Returns:
        ----------
        faiss.Index: The built index.
        """
        return faiss.IndexBinaryMultiHash(self.dimension, self.nhash, self.hash_bits) # type: ignore

    @property
    def search_method(self) -> FaissSearchMethod:
        """
        Returns the search method.
        
        Returns:
        ----------
        FaissSearchMethod: The search method.
        """
        return FaissSearchMethod.MULTI_HASH
