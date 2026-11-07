"""NumPy array type aliases for Faiss vector operations."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

NumericDType: TypeAlias = np.integer | np.unsignedinteger | np.floating
FloatDType: TypeAlias = np.floating
BinaryDType: TypeAlias = np.unsignedinteger
IndexDType: TypeAlias = np.integer

VectorArray: TypeAlias = NDArray[NumericDType]
FloatVectorArray: TypeAlias = NDArray[FloatDType]
BinaryVectorArray: TypeAlias = NDArray[BinaryDType]
ValueArray: TypeAlias = NDArray[FloatDType]
IndexArray: TypeAlias = NDArray[IndexDType]
SearchResultArrays: TypeAlias = tuple[ValueArray, IndexArray]

__all__ = [
    "BinaryDType",
    "BinaryVectorArray",
    "FloatDType",
    "FloatVectorArray",
    "IndexArray",
    "IndexDType",
    "NumericDType",
    "SearchResultArrays",
    "ValueArray",
    "VectorArray",
]
