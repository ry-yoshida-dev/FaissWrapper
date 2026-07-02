from __future__ import annotations

import faiss
from enum import Enum

class FaissMetric(Enum):
    """
    Enum class for Faiss metrics.

    Parameters:
    ----------
    L2: L2 distance.
    INNER_PRODUCT: Inner product distance.
    L1: L1 distance.
    LINF: Linf distance.
    Lp: Lp distance.
    CANBERRA: Canberra distance.
    BRAY_CURTIS: Bray-Curtis distance.
    JENSEN_SHANNON: Jensen-Shannon distance.
    HAMMING: Hamming distance.
    """
    L1 = "l1"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"
    LINF = "linf"
    Lp = "lp"
    CANBERRA = "canberra"
    BRAY_CURTIS = "bray_curtis"
    JENSEN_SHANNON = "jensen_shannon"
    HAMMING = "hamming"

    @property
    def object(self) -> faiss.MetricType:
        """
        Returns the corresponding Faiss metric constant.

        Returns
        -------
        faiss.MetricType
            Faiss metric identifier (e.g. ``METRIC_L2``).

        NOTE
        - HAMMING is not used for Index.
        - METRIC_L2 is temporary output.
        """
        match self:
            case FaissMetric.L1:
                return faiss.METRIC_L1
            case FaissMetric.L2:
                return faiss.METRIC_L2
            case FaissMetric.INNER_PRODUCT:
                return faiss.METRIC_INNER_PRODUCT
            case FaissMetric.LINF:
                return faiss.METRIC_Linf
            case FaissMetric.Lp:
                return faiss.METRIC_Lp
            case FaissMetric.CANBERRA:
                return faiss.METRIC_Canberra
            case FaissMetric.BRAY_CURTIS:
                return faiss.METRIC_BrayCurtis
            case FaissMetric.JENSEN_SHANNON:
                return faiss.METRIC_JensenShannon
            case FaissMetric.HAMMING:
                return faiss.METRIC_L2



