"""Single-query Faiss search result."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..types import DistanceArray, IndexArray, SearchResultArrays
from .utils import NeighborSorter


@dataclass
class FaissResult:
    """
    Search result for a single query vector.

    Distances are sorted in ascending order in ``__post_init__`` so that
    nearest neighbors appear at the front of the result.

    Parameters
    ----------
    distances : DistanceArray
        The distances of the nearest neighbors. Shape ``(k,)``.
    indices : IndexArray
        The indices of the nearest neighbors. Shape ``(k,)``.
    """

    distances: DistanceArray
    indices: IndexArray

    def __post_init__(self) -> None:
        """
        Normalize to 1-D, validate arrays, and sort neighbors by distance.
        """
        self.distances, self.indices = self._normalize_to_row(
            self.distances,
            self.indices,
        )
        sorted_distances, sorted_indices = NeighborSorter.sort_row(
            self.distances,
            self.indices,
        )
        self.distances = sorted_distances
        self.indices = sorted_indices

    @staticmethod
    def _normalize_to_row(
        distances: DistanceArray,
        indices: IndexArray,
    ) -> SearchResultArrays:
        """
        Ensure distances and indices are 1-D arrays.

        Parameters
        ----------
        distances : DistanceArray
            Neighbor distances with shape ``(k,)`` or ``(1, k)``.
        indices : IndexArray
            Neighbor indices aligned with ``distances``.

        Returns
        -------
        SearchResultArrays
            Row arrays with shape ``(k,)``.

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
            return distances, indices
        if distances.ndim == 2 and distances.shape[0] == 1:
            return distances.reshape(-1), indices.reshape(-1)
        raise ValueError(
            "distances and indices must be 1-D or a single row with shape (1, k). "
            f"Got shape={distances.shape}."
        )

    def __len__(self) -> int:
        """
        Return the number of neighbors.

        Returns
        -------
        int
            Neighbor count (``k``).
        """
        return int(self.distances.shape[0])

    @property
    def neighbor_count(self) -> int:
        """
        Return the number of neighbors.

        Returns
        -------
        int
            Neighbor count (``k``).
        """
        return int(self.distances.shape[0])

    @property
    def nearest_neighbor(self) -> FaissResult:
        """
        Return the closest neighbor.

        Returns
        -------
        FaissResult
            Distance and index with shape ``(1,)``.
        """
        return FaissResult(
            distances=self.distances[:1],
            indices=self.indices[:1],
        )

    @property
    def min_distance(self) -> float:
        """
        Return the minimum distance.

        Returns
        -------
        float
            The closest neighbor distance.
        """
        return float(self.distances[0])

    @property
    def min_index(self) -> int:
        """
        Return the index of the closest neighbor.

        Returns
        -------
        int
            The closest neighbor index.
        """
        return int(self.indices[0])

    def top_k_neighbors(self, k: int) -> FaissResult:
        """
        Return the first ``k`` neighbors from the sorted result.

        Parameters
        ----------
        k : int
            The number of neighbors to return.

        Returns
        -------
        FaissResult
            Distances and indices with shape ``(k,)``.

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
        return FaissResult(
            distances=self.distances[:k],
            indices=self.indices[:k],
        )

    def filter_by_distance(self, distance: float) -> FaissResult:
        """
        Filter neighbors whose distance is less than or equal to ``distance``.

        Parameters
        ----------
        distance : float
            Maximum distance to keep.

        Returns
        -------
        FaissResult
            Filtered distances and indices with shape ``(n,)``, where
            ``n <= k``.
        """
        mask: np.ndarray = self.distances <= distance
        output_distances: DistanceArray = self.distances[mask]
        output_indices: IndexArray = self.indices[mask]
        if output_distances.size == 0:
            warnings.warn("No neighbors found within the distance threshold.")
            return FaissResult(
                distances=np.array([], dtype=self.distances.dtype),
                indices=np.array([], dtype=self.indices.dtype),
            )
        return FaissResult(
            distances=output_distances,
            indices=output_indices,
        )
