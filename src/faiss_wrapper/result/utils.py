"""Shared utilities for search result processing."""

from __future__ import annotations

import numpy as np

from ..types import IndexArray, SearchResultArrays, ValueArray


class NeighborSorter:
    """
    Sort Faiss neighbor values and indices into nearest-first order.

    "Nearest" means the smallest value for distance metrics (e.g. L2) and the
    largest value for similarity metrics (e.g. inner product); see
    ``FaissMetric.is_larger_nearer``.
    """

    @staticmethod
    def sort_row(
        values: ValueArray,
        indices: IndexArray,
        *,
        is_larger_nearer: bool,
    ) -> SearchResultArrays:
        """
        Sort neighbors into nearest-first order for a single query row.

        Parameters
        ----------
        values : ValueArray
            Unsorted neighbor values with shape ``(k,)``.
        indices : IndexArray
            Neighbor indices aligned with ``values``.
        is_larger_nearer : bool
            Whether a larger value means a nearer neighbor.

        Returns
        -------
        SearchResultArrays
            Nearest-first ``(values, indices)`` pair.
        """
        order: IndexArray = np.argsort(-values, kind="stable") if is_larger_nearer else np.argsort(values, kind="stable")
        sorted_values: ValueArray = values[order]
        sorted_indices: IndexArray = indices[order]
        return sorted_values, sorted_indices

    @staticmethod
    def sort_batch(
        values: ValueArray,
        indices: IndexArray,
        *,
        is_larger_nearer: bool,
    ) -> SearchResultArrays:
        """
        Sort neighbors into nearest-first order for each query row.

        Parameters
        ----------
        values : ValueArray
            Unsorted neighbor values with shape ``(n_queries, k)``.
        indices : IndexArray
            Neighbor indices aligned with ``values``.
        is_larger_nearer : bool
            Whether a larger value means a nearer neighbor.

        Returns
        -------
        SearchResultArrays
            Nearest-first ``(values, indices)`` pair.
        """
        order: IndexArray = (
            np.argsort(-values, axis=1, kind="stable") if is_larger_nearer else np.argsort(values, axis=1, kind="stable")
        )
        sorted_values: ValueArray = np.take_along_axis(values, order, axis=1)
        sorted_indices: IndexArray = np.take_along_axis(indices, order, axis=1)
        return sorted_values, sorted_indices
