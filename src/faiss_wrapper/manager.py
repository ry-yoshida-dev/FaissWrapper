import faiss # type: ignore
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, cast

from .types import SearchResultArrays, VectorArray
from .dtype import FaissDType
from .metric import FaissMetric
from .method import FaissSearchMethod
from .result import FaissResult, FaissResults

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
        vectors: VectorArray
        ) -> None:
        """
        Add vectors to the index.

        Parameters:
        ----------
        vectors: VectorArray
            The vectors to add to the index.
        """
        self.index.add(vectors) # type: ignore

    def search(
        self, 
        vectors: VectorArray, 
        k: int = 10
        ) -> FaissResult | FaissResults:
        """
        Search the index for the nearest neighbors.

        Parameters:
        ----------
        vectors: VectorArray
            Query vectors with shape ``(n_query, dimension)`` or a single
            vector with shape ``(dimension,)``.
        k: int = 10
            The number of nearest neighbors to return.

        Returns:
        ----------
        FaissResult | FaissResults
            ``FaissResult`` when ``vectors`` is 1-D; otherwise ``FaissResults``
            with shape ``(n_query, k)``.
        """
        is_single_query: bool = vectors.ndim == 1
        query_vectors: VectorArray = self._normalize_query_vectors(vectors)
        distances, indices = cast(
            SearchResultArrays,
            self.index.search(query_vectors, k), # type: ignore
        )
        if is_single_query:
            return FaissResult(
                distances=distances[0],
                indices=indices[0],
            )
        return FaissResults(
            distances=distances,
            indices=indices,
        )

    @staticmethod
    def _normalize_query_vectors(vectors: VectorArray) -> VectorArray:
        """
        Ensure query vectors are a 2-D batch matrix for Faiss search.

        Parameters
        ----------
        vectors : VectorArray
            Query vectors with shape ``(n_query, dimension)`` or ``(dimension,)``.

        Returns
        -------
        VectorArray
            Query matrix with shape ``(n_query, dimension)``.

        Raises
        ------
        ValueError
            If ``vectors`` is not 1-D or 2-D.
        """
        if vectors.ndim == 1:
            return vectors.reshape(1, -1)
        if vectors.ndim == 2:
            return vectors
        raise ValueError(
            "Query vectors must be 1-D or 2-D. "
            f"Got ndim={vectors.ndim}."
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


