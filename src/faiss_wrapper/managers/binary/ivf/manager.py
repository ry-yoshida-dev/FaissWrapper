from __future__ import annotations

from dataclasses import dataclass

import faiss
from typing_extensions import override

from ....method import FaissSearchMethod
from ....types import BinaryVectorArray
from ..manager import FaissBinaryManager


@dataclass
class FaissBinaryIVFManager(FaissBinaryManager):
    """
    Faiss manager for binary IVF (IndexBinaryIVF) with Hamming distance.

    Parameters
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

    @override
    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.nlist <= 0:
            raise ValueError("nlist must be greater than 0.")
        if self.nprobe <= 0:
            raise ValueError("nprobe must be greater than 0.")

    @override
    def _build_index(self) -> faiss.IndexBinary:
        """
        Build the index.

        Returns
        -------
        faiss.IndexBinary: The built index.
        """
        quantizer = faiss.IndexBinaryFlat(self.dimension)
        index = faiss.IndexBinaryIVF(quantizer, self.dimension, self.nlist)
        index.nprobe = self.nprobe
        return index

    @override
    def add(self, vectors: BinaryVectorArray) -> None:
        """
        Train the IVF index then add vectors (required for IndexBinaryIVF).

        Parameters
        ----------
        vectors: BinaryVectorArray
            The vectors to train the index on.
        """
        # The Faiss stubs overload on torch.Tensor, which is unknown without torch installed.
        self.index.train(vectors)  # pyright: ignore[reportUnknownMemberType]
        super().add(vectors)

    @property
    @override
    def search_method(self) -> FaissSearchMethod:
        """
        Returns the search method.

        Returns
        -------
        FaissSearchMethod: The search method.
        """
        return FaissSearchMethod.IVF
