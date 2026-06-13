import faiss # type: ignore
from dataclasses import dataclass

from ...types import FloatVectorArray
from ..manager import FaissFloatManager
from ...method import FaissSearchMethod


@dataclass
class FaissFloatHNSWManager(FaissFloatManager):
    """
    Faiss manager for HNSW (Hierarchical Navigable Small World Graph) approximate search.

    Parameters:
    ----------
    efSearch: int = 64
        Number of connections to explore at search time.
    M: int
        Number of connections each node has.
    """
    efSearch: int = 64
    M: int = 32

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.M <= 0:
            raise ValueError("M must be greater than 0.")

    def _build_index(self) -> faiss.Index:
        metric_obj = self.metric.object
        return faiss.IndexHNSWFlat(self.dimension, self.M, metric_obj)

    def add(self, vectors: FloatVectorArray) -> None:
        """Add vectors to the index."""
        super().add(vectors)
        self.index.hnsw.efSearch = self.efSearch # type: ignore

    @property
    def search_method(self) -> FaissSearchMethod:
        return FaissSearchMethod.HNSW
