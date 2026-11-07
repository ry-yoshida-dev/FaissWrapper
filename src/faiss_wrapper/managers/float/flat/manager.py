from __future__ import annotations

from dataclasses import dataclass

import faiss
from typing_extensions import override

from ....method import FaissSearchMethod
from ..manager import FaissFloatManager


@dataclass
class FaissFloatFlatManager(FaissFloatManager):
    """
    Faiss manager for flat (exact) search. Uses only dimension and metric.

    Parameters
    ----------
    dimension: int
        The dimension of the vectors.
    metric: FaissMetric
        The metric to use. All metrics are supported.
    """

    @override
    def _build_index(self) -> faiss.Index:
        metric_type: faiss.MetricType = self.metric.object
        return faiss.IndexFlat(self.dimension, metric_type)

    @property
    @override
    def search_method(self) -> FaissSearchMethod:
        return FaissSearchMethod.FLAT
