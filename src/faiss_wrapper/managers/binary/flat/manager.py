import faiss # type: ignore
from dataclasses import dataclass

from ..manager import FaissBinaryManager
from ....method import FaissSearchMethod


@dataclass
class FaissBinaryFlatManager(FaissBinaryManager):
    """
    Faiss manager for binary (flat) search with Hamming distance.
    Vectors must be uint8 arrays of shape (n, d_bits // 8).

    Parameters:
    ----------
    dimension: int
        Number of bits per vector (d_bits). Stored vectors have dimension // 8 bytes.
    metric: FaissMetric
        Must be FaissMetric.HAMMING (Hamming distance is always used).
    """

    def _build_index(self) -> faiss.Index:
        """
        Build the index.
        
        Returns:
        ----------
        faiss.Index: The built index.
        """
        return faiss.IndexBinaryFlat(self.dimension) # type: ignore

    @property
    def search_method(self) -> FaissSearchMethod:
        return FaissSearchMethod.FLAT

