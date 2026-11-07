from __future__ import annotations

from dataclasses import dataclass

import faiss
from typing_extensions import override

from ....method import FaissSearchMethod
from ..manager import FaissBinaryManager


@dataclass
class FaissBinaryHNSWManager(FaissBinaryManager):
    """
    Faiss manager for binary HNSW (IndexBinaryHNSW), Hamming distance.

    Parameters
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

    @override
    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.M <= 0:
            raise ValueError("M must be greater than 0.")
        if self.efSearch <= 0:
            raise ValueError("efSearch must be greater than 0.")

    @override
    def _build_index(self) -> faiss.IndexBinary:
        """
        Build the index.

        Returns
        -------
        faiss.IndexBinary: The built index.
        """
        index = faiss.IndexBinaryHNSW(self.dimension, self.M)
        # The Faiss stubs type IndexBinaryHNSW.hnsw as Any; it is the same HNSW graph as the float index.
        hnsw: faiss.HNSW = index.hnsw  # pyright: ignore[reportAny]
        hnsw.efSearch = self.efSearch
        return index

    @property
    @override
    def search_method(self) -> FaissSearchMethod:
        """
        Returns the search method.

        Returns
        -------
        FaissSearchMethod: The search method.
        """
        return FaissSearchMethod.HNSW
