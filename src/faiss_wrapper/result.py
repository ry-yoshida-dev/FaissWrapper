import numpy as np
from dataclasses import dataclass

@dataclass
class FaissResult:
    """
    Result for Faiss.

    Parameters:
    ----------
    distances: np.ndarray
        The distances of the nearest neighbors.
    indices: np.ndarray
        The indices of the nearest neighbors.
    """
    distances: np.ndarray
    indices: np.ndarray