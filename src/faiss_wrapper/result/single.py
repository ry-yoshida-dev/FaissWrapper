"""Single-query Faiss search result."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..types import IndexArray, SearchResultArrays, ValueArray
from .utils import NeighborSorter


@dataclass
class FaissResult:
    """
    Search result for a single query vector.

    Neighbors are sorted nearest-first in ``__post_init__``: ascending by
    value for distance metrics, descending for similarity metrics such as
    inner product (see ``FaissMetric.is_larger_nearer``).

    Parameters
    ----------
    values : ValueArray
        The values of the nearest neighbors - distances for distance metrics,
        similarities for similarity metrics. Shape ``(k,)``.
    indices : IndexArray
        The indices of the nearest neighbors. Shape ``(k,)``.
    is_larger_nearer : bool
        Whether a larger value means a nearer neighbor.
    """

    values: ValueArray
    indices: IndexArray
    is_larger_nearer: bool

    def __post_init__(self) -> None:
        """
        Normalize to 1-D, validate arrays, and sort neighbors nearest-first.
        """
        self.values, self.indices = self._normalize_to_row(
            self.values,
            self.indices,
        )
        sorted_values, sorted_indices = NeighborSorter.sort_row(
            self.values,
            self.indices,
            is_larger_nearer=self.is_larger_nearer,
        )
        self.values = sorted_values
        self.indices = sorted_indices

    @staticmethod
    def _normalize_to_row(
        values: ValueArray,
        indices: IndexArray,
    ) -> SearchResultArrays:
        """
        Ensure values and indices are 1-D arrays.

        Parameters
        ----------
        values : ValueArray
            Neighbor values with shape ``(k,)`` or ``(1, k)``.
        indices : IndexArray
            Neighbor indices aligned with ``values``.

        Returns
        -------
        SearchResultArrays
            Row arrays with shape ``(k,)``.

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
            return values, indices
        if values.ndim == 2 and values.shape[0] == 1:
            return values.reshape(-1), indices.reshape(-1)
        raise ValueError(
            "values and indices must be 1-D or a single row with shape (1, k). "
            f"Got shape={values.shape}."
        )

    def __len__(self) -> int:
        """
        Return the number of neighbors.

        Returns
        -------
        int
            Neighbor count (``k``).
        """
        return int(self.values.shape[0])

    @property
    def neighbor_count(self) -> int:
        """
        Return the number of neighbors.

        Returns
        -------
        int
            Neighbor count (``k``).
        """
        return int(self.values.shape[0])

    @property
    def nearest_neighbor(self) -> FaissResult:
        """
        Return the nearest neighbor.

        Returns
        -------
        FaissResult
            Value and index with shape ``(1,)``.
        """
        return FaissResult(
            values=self.values[:1],
            indices=self.indices[:1],
            is_larger_nearer=self.is_larger_nearer,
        )

    @property
    def nearest_value(self) -> float:
        """
        Return the value of the nearest neighbor.

        Returns
        -------
        float
            The nearest neighbor's value.
        """
        return float(self.values[0])

    @property
    def nearest_index(self) -> int:
        """
        Return the index of the nearest neighbor.

        Returns
        -------
        int
            The nearest neighbor's index.
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
            Values and indices with shape ``(k,)``.

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
            values=self.values[:k],
            indices=self.indices[:k],
            is_larger_nearer=self.is_larger_nearer,
        )

    def filter_by_value(self, value: float) -> FaissResult:
        """
        Filter neighbors that are at least as near as ``value``.

        Parameters
        ----------
        value : float
            Threshold to keep: neighbors with a value at least as near as
            this are kept (``>= value`` when a larger value is nearer,
            ``<= value`` otherwise).

        Returns
        -------
        FaissResult
            Filtered values and indices with shape ``(n,)``, where
            ``n <= k``.
        """
        mask: np.ndarray = self.values >= value if self.is_larger_nearer else self.values <= value
        output_values: ValueArray = self.values[mask]
        output_indices: IndexArray = self.indices[mask]
        if output_values.size == 0:
            warnings.warn("No neighbors found within the value threshold.")
            return FaissResult(
                values=np.array([], dtype=self.values.dtype),
                indices=np.array([], dtype=self.indices.dtype),
                is_larger_nearer=self.is_larger_nearer,
            )
        return FaissResult(
            values=output_values,
            indices=output_indices,
            is_larger_nearer=self.is_larger_nearer,
        )
