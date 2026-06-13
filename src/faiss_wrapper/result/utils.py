"""Shared utilities for search result processing."""

from __future__ import annotations

import numpy as np

from ..types import DistanceArray, IndexArray, SearchResultArrays


class NeighborSorter:
    """
    Sort Faiss neighbor distances and indices in ascending distance order.
    """

    @staticmethod
    def sort_row(
        distances: DistanceArray,
        indices: IndexArray,
    ) -> SearchResultArrays:
        """
        Sort neighbors by ascending distance for a single query row.

        Parameters
        ----------
        distances : DistanceArray
            Unsorted neighbor distances with shape ``(k,)``.
        indices : IndexArray
            Neighbor indices aligned with ``distances``.

        Returns
        -------
        SearchResultArrays
            Sorted ``(distances, indices)`` pair.
        """
        order: IndexArray = np.argsort(distances)
        sorted_distances: DistanceArray = distances[order]
        sorted_indices: IndexArray = indices[order]
        return sorted_distances, sorted_indices

    @staticmethod
    def sort_batch(
        distances: DistanceArray,
        indices: IndexArray,
    ) -> SearchResultArrays:
        """
        Sort neighbors by ascending distance for each query row.

        Parameters
        ----------
        distances : DistanceArray
            Unsorted neighbor distances with shape ``(n_queries, k)``.
        indices : IndexArray
            Neighbor indices aligned with ``distances``.

        Returns
        -------
        SearchResultArrays
            Sorted ``(distances, indices)`` pair.
        """
        order: IndexArray = np.argsort(distances, axis=1)
        sorted_distances: DistanceArray = np.take_along_axis(distances, order, axis=1)
        sorted_indices: IndexArray = np.take_along_axis(indices, order, axis=1)
        return sorted_distances, sorted_indices
