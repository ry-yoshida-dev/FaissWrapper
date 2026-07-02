"""Batch Faiss search results."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import IndexArray, SearchResultArrays, ValueArray
from .single import FaissResult
from .utils import NeighborSorter


@dataclass
class FaissResults:
    """
    Search results for one or more query vectors.

    Neighbors are sorted nearest-first in ``__post_init__`` so that the
    nearest neighbor appears at the front of each result row (see
    ``FaissMetric.is_larger_nearer``).

    Parameters
    ----------
    values : ValueArray
        The values of the nearest neighbors. Shape ``(n_queries, k)``.
    indices : IndexArray
        The indices of the nearest neighbors. Shape ``(n_queries, k)``.
    is_larger_nearer : bool
        Whether a larger value means a nearer neighbor.
    """

    values: ValueArray
    indices: IndexArray
    is_larger_nearer: bool

    def __post_init__(self) -> None:
        """
        Normalize to 2-D, validate arrays, and sort neighbors nearest-first.
        """
        self.values, self.indices = self._normalize_to_batch(
            self.values,
            self.indices,
        )
        sorted_values, sorted_indices = NeighborSorter.sort_batch(
            self.values,
            self.indices,
            is_larger_nearer=self.is_larger_nearer,
        )
        self.values = sorted_values
        self.indices = sorted_indices

    @staticmethod
    def _normalize_to_batch(
        values: ValueArray,
        indices: IndexArray,
    ) -> SearchResultArrays:
        """
        Ensure values and indices are 2-D batch arrays.

        Parameters
        ----------
        values : ValueArray
            Neighbor values with shape ``(k,)`` or ``(n_queries, k)``.
        indices : IndexArray
            Neighbor indices aligned with ``values``.

        Returns
        -------
        SearchResultArrays
            Batch arrays with shape ``(n_queries, k)``.

        Raises
        ------
        ValueError
            If shapes do not match or ndim is invalid.
        """
        if values.shape != indices.shape:
            raise ValueError(
                "values and indices must have the same shape. "
                f"Got {values.shape} and {indices.shape}."
            )
        if values.ndim == 1:
            return values.reshape(1, -1), indices.reshape(1, -1)
        if values.ndim == 2:
            return values, indices
        raise ValueError(
            "values and indices must be 1-D or 2-D. "
            f"Got ndim={values.ndim}."
        )

    def __len__(self) -> int:
        """
        Return the number of queries.

        Returns
        -------
        int
            Query count.
        """
        return int(self.values.shape[0])

    @property
    def neighbor_count(self) -> int:
        """
        Return the number of neighbors per query.

        Returns
        -------
        int
            Neighbor count (``k``).
        """
        return int(self.values.shape[1])

    @property
    def nearest_neighbors(self) -> FaissResults:
        """
        Return the nearest neighbor for each query.

        Returns
        -------
        FaissResults
            Values and indices with shape ``(n_queries, 1)``.
        """
        return FaissResults(
            values=self.values[:, :1],
            indices=self.indices[:, :1],
            is_larger_nearer=self.is_larger_nearer,
        )

    @property
    def nearest_values(self) -> ValueArray:
        """
        Return the nearest neighbor's value for each query.

        Returns
        -------
        ValueArray
            Per-query nearest values. Shape ``(n_queries,)``.
        """
        return self.values[:, 0]

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
            Values and indices with shape ``(n_queries, k)``.

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
        return self.values[:, :k], self.indices[:, :k]

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
            Values and indices with shape ``(k,)``.
        """
        return FaissResult(
            values=self.values[index],
            indices=self.indices[index],
            is_larger_nearer=self.is_larger_nearer,
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
            values=result.values.reshape(1, -1),
            indices=result.indices.reshape(1, -1),
            is_larger_nearer=result.is_larger_nearer,
        )
