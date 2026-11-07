from __future__ import annotations

from dataclasses import dataclass

import faiss
from typing_extensions import override

from ....method import FaissSearchMethod
from ....types import FloatVectorArray
from ..manager import FaissFloatManager


@dataclass
class FaissFloatIVFPQManager(FaissFloatManager):
    """
    Faiss manager for IVFPQ (inverted file + product quantization) approximate search.

    Parameters
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

    @override
    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.nlist <= 0:
            raise ValueError("nlist must be greater than 0.")
        if self.dimension % self.m != 0:
            raise ValueError(
                f"dimension ({self.dimension}) must be divisible by m ({self.m})."
            )

    @override
    def _build_index(self) -> faiss.Index:
        metric_obj = self.metric.object
        quantizer = faiss.IndexFlat(self.dimension, metric_obj)
        index = faiss.IndexIVFPQ(
            quantizer, self.dimension, self.nlist, self.m, self.nbits, metric_obj
        )
        index.nprobe = self.nprobe
        return index

    @override
    def add(self, vectors: FloatVectorArray) -> None:
        """
        Train the index then add vectors (IVFPQ requires training).

        Parameters
        ----------
        vectors: FloatVectorArray
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
        return FaissSearchMethod.IVFPQ
