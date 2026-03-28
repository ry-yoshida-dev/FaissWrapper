import faiss # type: ignore
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, cast

from .metric import FaissMetric
from .result import FaissResult
from .method import FaissSearchMethod
from .dtype import FaissDType

@dataclass
class FaissManager(ABC):
    """
    Abstract manager for Faiss. Subclass per search method (flat, ivf, ivfpq).

    Parameters:
    ----------
    dimension: int
        The dimension of the vectors.
    metric: FaissMetric
        The metric to use.
    """
    dimension: int
    metric: FaissMetric
    index: faiss.Index = field(init=False)
    is_gpu_enabled: bool | None = field(default=None)

    def __post_init__(self) -> None:
        """
        Validate the parameters and build the index.
        """
        self._validate_parameters()
        self._validate_gpu_enabled()
        self.index = self._build_index()

    def _validate_gpu_enabled(self) -> None:
        """
        Validate that the GPU is available.
        
        Raises:
        ----------
        ValueError: If the GPU is not available.
        """
        num_gpus = faiss.get_num_gpus() # type: ignore
        if self.is_gpu_enabled and num_gpus == 0:
            raise ValueError("GPU is not available. Please set is_gpu_enabled to False or None.")
        if self.is_gpu_enabled is None:
            self.is_gpu_enabled = num_gpus > 0            

    def _validate_parameters(self) -> None:
        """
        Validate the parameters of the manager (dimension, metric vs search_method).
        
        Raises:
        ----------
        ValueError: If the dimension is not greater than 0, or metric is unsupported.
        """
        if self.dimension <= 0:
            raise ValueError("Dimension must be greater than 0.")
        self.dtype.validate_metric(self.search_method, self.metric)

    @abstractmethod
    def _build_index(self) -> faiss.Index:
        """
        Build and return the Faiss index. 
        Override in each concrete manager.
        
        Returns:
        ----------
        faiss.Index: The built Faiss index.
        """

    def add(
        self, 
        vectors: np.ndarray
        ) -> None:
        """
        Add vectors to the index.

        Parameters:
        ----------
        vectors: np.ndarray
            The vectors to add to the index.
        """
        self.index.add(vectors) # type: ignore

    def search(
        self, 
        vectors: np.ndarray, 
        k: int = 10
        ) -> FaissResult:
        """
        Search the index for the nearest neighbors.

        Parameters:
        ----------
        vectors: np.ndarray
            The vectors to search the index for.
        k: int = 10
            The number of nearest neighbors to return.

        Returns:
        ----------
        FaissResult: The result of the search.
        """
        distances, indices = cast(tuple[np.ndarray, np.ndarray], self.index.search(vectors, k)) # type: ignore
        return FaissResult(
            distances=distances, 
            indices=indices
            )

    def save(
        self, 
        path: str
        ) -> None:
        """
        Save the index.

        Parameters:
        ----------
        path: str
            The path of the file to save the index to.
            Must end with .faiss.
        """
        self._valid_faiss_path(path)
        self.save_function(self.index, path)

    def load(
        self, 
        path: str
        ) -> None:
        """
        Load the index.

        Parameters:
        ----------
        path: str
            The path of the file to load the index from.
            Must end with .faiss.
        """
        self._valid_faiss_path(path)
        self.index = self.load_function(path)

    def _valid_faiss_path(
        self, 
        path: str
        ) -> None:
        """
        Validate that the path ends with .faiss.
        
        Parameters:
        ----------
        path: str
            The path to validate.
        """
        if not path.endswith(".faiss"):
            raise ValueError(f"Path must end with .faiss. Got {path}.")

    @property
    @abstractmethod
    def search_method(self) -> FaissSearchMethod:
        """
        Returns the search method.
        
        Returns:
        ----------
        FaissSearchMethod: The search method.
        """

    @property
    @abstractmethod
    def dtype(self) -> FaissDType:
        """
        Returns the data type of the vectors.
        
        Returns:
        ----------
        FaissDType: The data type of the vectors.
        """

    @property
    @abstractmethod
    def save_function(self) -> Callable[[faiss.Index, str], None]:
        """
        Returns the save function for the index.
        
        Returns:
        ----------
        Callable[[faiss.Index, str], None]: The save function for the index.
        """

    @property
    @abstractmethod
    def load_function(self) -> Callable[[str], faiss.Index]:
        """
        Returns the load function for the index.
        
        Returns:
        ----------
        Callable[[str], faiss.Index]: The load function for the index.
        """


