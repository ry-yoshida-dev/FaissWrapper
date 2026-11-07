from __future__ import annotations

from dataclasses import dataclass

import faiss
from typing_extensions import override

from ....method import FaissSearchMethod
from ..manager import FaissFloatManager


@dataclass
class FaissFloatHNSWManager(FaissFloatManager):
    """
    Faiss manager for HNSW (Hierarchical Navigable Small World Graph) approximate search.

    Parameters
    ----------
    efSearch: int = 64
        Number of connections to explore at search time.
    M: int
        Number of connections each node has.
    """

    efSearch: int = 64
    M: int = 32

    @override
    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.M <= 0:
            raise ValueError("M must be greater than 0.")

    @override
    def _build_index(self) -> faiss.Index:
        metric_obj = self.metric.object
        index = faiss.IndexHNSWFlat(self.dimension, self.M, metric_obj)
        index.hnsw.efSearch = self.efSearch
        return index

    @property
    @override
    def search_method(self) -> FaissSearchMethod:
        return FaissSearchMethod.HNSW
