from __future__ import annotations

import faiss # type: ignore
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar, cast

from .types import SearchResultArrays, VectorArray
from .dtype import FaissDType
from .metric import FaissMetric
from .method import FaissSearchMethod
from .result import FaissResult, FaissResults

VectorArrayT = TypeVar("VectorArrayT", bound=VectorArray)

@dataclass
class FaissManager(ABC, Generic[VectorArrayT]):
    """
    Abstract manager for Faiss. Subclass per search method (flat, ivf, ivfpq).

    Generic over ``VectorArrayT`` so each dtype family (float, binary) narrows
    ``add``/``search_batch``/``search_single`` to its own vector array type
    without violating the Liskov substitution principle.

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
        vectors: VectorArrayT
        ) -> None:
        """
        Add vectors to the index.

        Parameters:
        ----------
        vectors: VectorArrayT
            The vectors to add to the index.
        """
        self.index.add(vectors) # type: ignore

    def search_batch(
        self,
        vectors: VectorArrayT,
        k: int = 10
        ) -> FaissResults:
        """
        Search the index for the nearest neighbors of a batch of query vectors.

        Parameters:
        ----------
        vectors: VectorArrayT
            Query vectors with shape ``(n_query, dimension)``. Use
            ``search_single`` for a single query vector with shape ``(dimension,)``.
        k: int = 10
            The number of nearest neighbors to return per query.

        Returns:
        ----------
        FaissResults
            Nearest neighbors for each query, shape ``(n_query, k)``.

        Raises
        ------
        ValueError
            If ``vectors`` is not 2-D.
        """
        if vectors.ndim != 2:
            raise ValueError(
                "vectors must be 2-D with shape (n_query, dimension). "
                f"Got ndim={vectors.ndim}. Use search_single for a single query vector."
            )
        values, indices = cast(
            SearchResultArrays,
            self.index.search(vectors, k), # type: ignore
        )
        return FaissResults(
            values=values,
            indices=indices,
            is_larger_nearer=self.metric.is_larger_nearer,
        )

    def search_single(
        self,
        vector: VectorArrayT,
        k: int = 10
        ) -> FaissResult:
        """
        Search the index for the nearest neighbors of a single query vector.

        Parameters:
        ----------
        vector: VectorArrayT
            Query vector with shape ``(dimension,)``.
        k: int = 10
            The number of nearest neighbors to return.

        Returns:
        ----------
        FaissResult
            Nearest neighbors for the query, shape ``(k,)``.

        Raises
        ------
        ValueError
            If ``vector`` is not 1-D.
        """
        if vector.ndim != 1:
            raise ValueError(
                "vector must be 1-D with shape (dimension,). "
                f"Got ndim={vector.ndim}. Use search_batch for a batch of query vectors."
            )
        batch_vector = cast(VectorArrayT, vector.reshape(1, -1))
        return self.search_batch(batch_vector, k)[0]

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


