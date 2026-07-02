from __future__ import annotations

import faiss
from dataclasses import dataclass

from ..manager import FaissFloatManager
from ....method import FaissSearchMethod


@dataclass
class FaissFloatFlatManager(FaissFloatManager):
    """
    Faiss manager for flat (exact) search. Uses only dimension and metric.

    Parameters:
    ----------
    dimension: int
        The dimension of the vectors.
    metric: FaissMetric
        The metric to use. All metrics are supported.
    """

    def _build_index(self) -> faiss.Index:
        metric_type: faiss.MetricType = self.metric.object
        return faiss.IndexFlat(self.dimension, metric_type)

    @property
    def search_method(self) -> FaissSearchMethod:
        return FaissSearchMethod.FLAT
