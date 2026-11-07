from __future__ import annotations

from dataclasses import dataclass

import faiss
from typing_extensions import override

from ....method import FaissSearchMethod
from ....types import FloatVectorArray
from ..manager import FaissFloatManager


@dataclass
class FaissFloatIVFManager(FaissFloatManager):
    """
    Faiss manager for IVF (inverted file) approximate search.

    Parameters
    ----------
    dimension: int
        The dimension of the vectors.
    metric: FaissMetric
        The metric to use. All metrics are supported.
    nlist: int = 100
        Number of clusters (Voronoi cells) for the coarse quantizer.
    """

    nlist: int = 100

    @override
    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.nlist <= 0:
            raise ValueError("nlist must be greater than 0.")

    @override
    def _build_index(self) -> faiss.Index:
        metric_obj = self.metric.object
        quantizer = faiss.IndexFlat(self.dimension, metric_obj)
        return faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric_obj)

    @override
    def add(self, vectors: FloatVectorArray) -> None:
        """
        Train the index then add vectors (IVF requires training).

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
        return FaissSearchMethod.IVF
