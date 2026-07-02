from __future__ import annotations

import faiss # type: ignore
from dataclasses import dataclass

from ..manager import FaissBinaryManager
from ....method import FaissSearchMethod


@dataclass
class FaissBinaryHashManager(FaissBinaryManager):
    """
    Faiss manager for IndexBinaryHash (LSH-style single hash table), Hamming distance.

    Corresponds to faiss.IndexBinaryHash(d, b) in faiss.ipynb:
    - d: bit length of each vector (must be multiple of 8).
    - b: number of leading bits used as bucket key; larger b narrows
      candidates but vectors differing in those bits may not co-bucket.

    Vectors are uint8 arrays of shape (n, d // 8).

    Parameters
    ----------
    dimension : int
        Number of bits per vector (d_bits).
    metric : FaissMetric
        Must be HAMMING.
    hash_bits : int
        b — bits used for hashing (bucket id).
    """

    hash_bits: int = 12

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.hash_bits <= 0:
            raise ValueError("hash_bits must be greater than 0.")
        if self.hash_bits > self.dimension:
            raise ValueError("hash_bits cannot exceed dimension (bit length).")

    def _build_index(self) -> faiss.Index:
        """
        Build the index.
        
        Returns:
        ----------
        faiss.Index: The built index.
        """
        return faiss.IndexBinaryHash(self.dimension, self.hash_bits) # type: ignore

    @property
    def search_method(self) -> FaissSearchMethod:
        return FaissSearchMethod.HASH
