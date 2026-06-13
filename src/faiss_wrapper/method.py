from __future__ import annotations

import faiss
from enum import Enum
from .metric import FaissMetric

class FaissSearchMethod(Enum):
    """
    Enum class for Faiss search methods.

    Parameters:
    ----------
    FLAT: Flat search method.
    IVF: Inverted file search method.
    HNSW: Hierarchical Navigable Small World graph search method.
    IVFPQ: Inverted file search method with product quantization.
    LSH: Locality-sensitive hashing search method.
    BINARY: Binary (flat) search with Hamming distance.
    """
    FLAT = "flat"
    IVF = "ivf"
    IVFPQ = "ivfpq"
    HNSW = "hnsw"
    LSH = "lsh"
    HASH = "hash"
    MULTI_HASH = "multi_hash"

    @property
    def object(self) -> type[faiss.Index] | type[faiss.IndexBinary]:
        """
        Returns the corresponding Faiss index class for the search method.

        Returns
        -------
        type[faiss.Index] | type[faiss.IndexBinary]
            Faiss index class (not an instance) for the search method.
        """
        match self:
            case FaissSearchMethod.FLAT:
                return faiss.IndexFlat
            case FaissSearchMethod.IVF:
                return faiss.IndexIVFFlat
            case FaissSearchMethod.IVFPQ:
                return faiss.IndexIVFPQ
            case FaissSearchMethod.HNSW:
                return faiss.IndexHNSWFlat
            case FaissSearchMethod.LSH:
                return faiss.IndexLSH
            case FaissSearchMethod.HASH:
                return faiss.IndexBinaryHash
            case FaissSearchMethod.MULTI_HASH:
                return faiss.IndexBinaryMultiHash

    @property
    def float_supported_metrics(self) -> list[FaissMetric]:
        """
        Returns the supported metrics for the search method.
        
        Returns:
        ----------
        list[FaissMetric]: The supported metrics for the search method.
        """
        match self:
            case FaissSearchMethod.FLAT | FaissSearchMethod.IVF | FaissSearchMethod.HNSW:
                return [m for m in FaissMetric if m is not FaissMetric.HAMMING]
            case FaissSearchMethod.LSH | FaissSearchMethod.HASH | FaissSearchMethod.MULTI_HASH:
                return [FaissMetric.HAMMING]
            case FaissSearchMethod.IVFPQ:
                return [FaissMetric.L2, FaissMetric.INNER_PRODUCT]

    @property
    def binary_supported_metrics(self) -> list[FaissMetric]:
        """
        Returns the supported metrics for the search method.
        
        Returns:
        ----------
        list[FaissMetric]: The supported metrics for the search method.
        """
        return [FaissMetric.HAMMING]

