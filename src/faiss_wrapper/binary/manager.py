import faiss # type: ignore
from dataclasses import dataclass
from typing import Callable

from ..types import BinaryVectorArray
from ..dtype import FaissDType
from ..manager import FaissManager
from ..result import FaissResult, FaissResults

@dataclass
class FaissBinaryManager(FaissManager):
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
    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.dimension % 8 != 0:
            raise ValueError("dimension (d_bits) must be a multiple of 8.")

    def add(self, vectors: BinaryVectorArray) -> None:
        """
        Add packed binary vectors to the index.

        Parameters
        ----------
        vectors : BinaryVectorArray
            Packed binary vectors (uint8), shape ``(n, dimension // 8)``.
        """
        super().add(vectors)

    def search(self, vectors: BinaryVectorArray, k: int = 10) -> FaissResult | FaissResults:
        """
        Search the index for nearest neighbors among binary query vectors.

        Parameters
        ----------
        vectors : BinaryVectorArray
            Packed binary query vectors (uint8), shape
            ``(n_query, dimension // 8)`` or ``(dimension // 8,)``.
        k : int, optional
            Number of nearest neighbors to return per query.

        Returns
        -------
        FaissResult | FaissResults
            ``FaissResult`` for a single query vector; ``FaissResults`` for a batch.
        """
        return super().search(vectors, k)

    @property
    def dtype(self) -> FaissDType:
        """
        Returns the data type of the vectors.
        
        Returns:
        ----------
        FaissDType: The data type of the vectors.
        """
        return FaissDType.BINARY

    @property
    def save_function(self) -> Callable[[faiss.Index, str], None]:
        """
        Returns the save function for the binary index.
        
        Returns:
        ----------
        Callable[[faiss.Index, str], None]: The save function for the binary index.
        """
        return faiss.write_index_binary # type: ignore

    @property
    def load_function(self) -> Callable[[str], faiss.Index]:
        """
        Returns the load function for the binary index.
        
        Returns:
        ----------
        Callable[[str], faiss.Index]: The load function for the binary index.
        """
        return faiss.read_index_binary # type: ignore
