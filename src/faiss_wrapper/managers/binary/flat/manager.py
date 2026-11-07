from __future__ import annotations

from dataclasses import dataclass

import faiss
from typing_extensions import override

from ....method import FaissSearchMethod
from ..manager import FaissBinaryManager


@dataclass
class FaissBinaryFlatManager(FaissBinaryManager):
    """
    Faiss manager for binary (flat) search with Hamming distance.

    Vectors must be uint8 arrays of shape (n, d_bits // 8).

    Parameters
    ----------
    dimension: int
        Number of bits per vector (d_bits). Stored vectors have dimension // 8 bytes.
    metric: FaissMetric
        Must be FaissMetric.HAMMING (Hamming distance is always used).
    """

    @override
    def _build_index(self) -> faiss.IndexBinary:
        """
        Build the index.

        Returns
        -------
        faiss.IndexBinary: The built index.
        """
        return faiss.IndexBinaryFlat(self.dimension)

    @property
    @override
    def search_method(self) -> FaissSearchMethod:
        return FaissSearchMethod.FLAT
