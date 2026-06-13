"""Batch Faiss search results."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import DistanceArray, IndexArray, SearchResultArrays
from .single import FaissResult
from .utils import NeighborSorter


@dataclass
class FaissResults:
    """
    Search results for one or more query vectors.

    Distances are sorted in ascending order in ``__post_init__`` so that
    nearest neighbors appear at the front of each result row.

    Parameters
    ----------
    distances : DistanceArray
        The distances of the nearest neighbors. Shape ``(n_queries, k)``.
    indices : IndexArray
        The indices of the nearest neighbors. Shape ``(n_queries, k)``.
    """

    distances: DistanceArray
    indices: IndexArray

    def __post_init__(self) -> None:
        """
        Normalize to 2-D, validate arrays, and sort neighbors by distance.
        """
        self.distances, self.indices = self._normalize_to_batch(
            self.distances,
            self.indices,
        )
        sorted_distances, sorted_indices = NeighborSorter.sort_batch(
            self.distances,
            self.indices,
        )
        self.distances = sorted_distances
        self.indices = sorted_indices

    @staticmethod
    def _normalize_to_batch(
        distances: DistanceArray,
        indices: IndexArray,
    ) -> SearchResultArrays:
        """
        Ensure distances and indices are 2-D batch arrays.

        Parameters
        ----------
        distances : DistanceArray
            Neighbor distances with shape ``(k,)`` or ``(n_queries, k)``.
        indices : IndexArray
            Neighbor indices aligned with ``distances``.

        Returns
        -------
        SearchResultArrays
            Batch arrays with shape ``(n_queries, k)``.

        Raises
        ------
        ValueError
            If shapes do not match or ndim is invalid.
        """
        if distances.shape != indices.shape:
            raise ValueError(
                "distances and indices must have the same shape. "
                f"Got {distances.shape} and {indices.shape}."
            )
        if distances.ndim == 1:
            return distances.reshape(1, -1), indices.reshape(1, -1)
        if distances.ndim == 2:
            return distances, indices
        raise ValueError(
            "distances and indices must be 1-D or 2-D. "
            f"Got ndim={distances.ndim}."
        )

    def __len__(self) -> int:
        """
        Return the number of queries.

        Returns
        -------
        int
            Query count.
        """
        return int(self.distances.shape[0])

    @property
    def neighbor_count(self) -> int:
        """
        Return the number of neighbors per query.

        Returns
        -------
        int
            Neighbor count (``k``).
        """
        return int(self.distances.shape[1])

    @property
    def nearest_neighbors(self) -> FaissResults:
        """
        Return the closest neighbor for each query.

        Returns
        -------
        FaissResults
            Distances and indices with shape ``(n_queries, 1)``.
        """
        return FaissResults(
            distances=self.distances[:, :1],
            indices=self.indices[:, :1],
        )

    @property
    def min_distances(self) -> DistanceArray:
        """
        Return the minimum distance for each query.

        Returns
        -------
        DistanceArray
            Per-query minimum distances. Shape ``(n_queries,)``.
        """
        return self.distances[:, 0]

    def top_k_neighbors(self, k: int) -> SearchResultArrays:
        """
        Return the first ``k`` neighbors from the sorted result.

        Parameters
        ----------
        k : int
            The number of neighbors to return.

        Returns
        -------
        SearchResultArrays
            Distances and indices with shape ``(n_queries, k)``.

        Raises
        ------
        ValueError
            If ``k`` is not in ``(0, neighbor_count]``.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        neighbor_count: int = self.neighbor_count
        if k > neighbor_count:
            raise ValueError(
                "k must be less than or equal to the number of neighbors. "
                f"Got {k} and {neighbor_count}."
            )
        return self.distances[:, :k], self.indices[:, :k]

    def __getitem__(self, index: int) -> FaissResult:
        """
        Return the search result for a single query row.

        Parameters
        ----------
        index : int
            Zero-based query index.

        Returns
        -------
        FaissResult
            Distances and indices with shape ``(k,)``.
        """
        return FaissResult(
            distances=self.distances[index],
            indices=self.indices[index],
        )

    @classmethod
    def from_result(cls, result: FaissResult) -> FaissResults:
        """
        Wrap a single-query result as a batch of size one.

        Parameters
        ----------
        result : FaissResult
            Single-query search result.

        Returns
        -------
        FaissResults
            Batch result with shape ``(1, k)``.
        """
        return cls(
            distances=result.distances.reshape(1, -1),
            indices=result.indices.reshape(1, -1),
        )
