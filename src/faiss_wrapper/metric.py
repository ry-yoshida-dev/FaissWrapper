import faiss # type: ignore
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
    def object(self) -> faiss.Index:
        """
        Returns the corresponding Faiss object for the metric.
        
        Returns:
        ----------
        faiss.Index: The corresponding Faiss object for the metric.

        NOTE
        - HAMMING is not used for Index.
        - METRIC_L2 is temporary output.
        """
        match self:
            case FaissMetric.L1:
                return faiss.METRIC_L1 # type: ignore
            case FaissMetric.L2:
                return faiss.METRIC_L2 # type: ignore
            case FaissMetric.INNER_PRODUCT:
                return faiss.METRIC_INNER_PRODUCT # type: ignore
            case FaissMetric.LINF:
                return faiss.METRIC_Linf # type: ignore
            case FaissMetric.Lp:
                return faiss.METRIC_Lp # type: ignore
            case FaissMetric.CANBERRA:
                return faiss.METRIC_Canberra # type: ignore
            case FaissMetric.BRAY_CURTIS:
                return faiss.METRIC_BrayCurtis # type: ignore
            case FaissMetric.JENSEN_SHANNON:
                return faiss.METRIC_JensenShannon # type: ignore
            case FaissMetric.HAMMING:
                return faiss.METRIC_L2 # type: ignore



