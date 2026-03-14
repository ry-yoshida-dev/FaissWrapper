import faiss
from dataclasses import dataclass
from typing import Callable

from ..dtype import FaissDType
from ..manager import FaissManager

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
        return faiss.write_index_binary

    @property
    def load_function(self) -> Callable[[str], faiss.Index]:
        """
        Returns the load function for the binary index.
        
        Returns:
        ----------
        Callable[[str], faiss.Index]: The load function for the binary index.
        """
        return faiss.read_index_binary
