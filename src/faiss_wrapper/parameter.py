from dataclasses import dataclass

from typing import TYPE_CHECKING

from .dtype import FaissDType
from .method import FaissSearchMethod
from .metric import FaissMetric
if TYPE_CHECKING:
    from .manager import FaissManager
    from .managers.float import FaissFloatManager
    from .managers.binary import FaissBinaryManager

@dataclass
class FaissParameter:
    """
    Parameters for Faiss.

    Parameters:
    ----------
    dtype: FaissDType
        The data type of the vectors.
    method: FaissSearchMethod
        The search method to use.
    metric: FaissMetric
        The metric to use.
    """
    dtype: FaissDType
    method: FaissSearchMethod
    metric: FaissMetric

    def __post_init__(self) -> None:
        """Validate the parameters."""
        self.dtype.validate_method(self.method)
        self.dtype.validate_metric(self.method, self.metric)

    @property
    def manager_class(self) -> type["FaissManager"]:
        """
        Returns the corresponding manager class for the search method.
        
        Returns:
        ----------
        type[FaissManager]: The corresponding manager class for the search method.
        """
        match self.dtype:
            case FaissDType.FLOAT:
                manager_class = self.float_manager_class
            case FaissDType.BINARY:
                manager_class = self.binary_manager_class
        return manager_class

    @property
    def float_manager_class(self) -> type["FaissFloatManager"]:
        """
        Returns the corresponding float manager class for the search method.
        
        Returns:
        ----------
        type[FaissFloatManager]: The corresponding float manager class for the search method.
        """
        match self.method:
            case FaissSearchMethod.FLAT:
                from .managers.float.flat import FaissFloatFlatManager
                return FaissFloatFlatManager
            case FaissSearchMethod.IVF:
                from .managers.float.ivf import FaissFloatIVFManager
                return FaissFloatIVFManager
            case FaissSearchMethod.IVFPQ:
                from .managers.float.ivfpq import FaissFloatIVFPQManager
                return FaissFloatIVFPQManager
            case FaissSearchMethod.HNSW:
                from .managers.float.hnsw import FaissFloatHNSWManager
                return FaissFloatHNSWManager
            case FaissSearchMethod.LSH:
                from .managers.float.lsh import FaissFloatLSHManager
                return FaissFloatLSHManager
            case _:
                raise ValueError(f"Search method {self.method.value} is not supported for float data type.")

    @property
    def binary_manager_class(self) -> type["FaissBinaryManager"]:
        """
        Returns the corresponding binary manager class for the search method.
        
        Returns:
        ----------
        type[FaissBinaryManager]: The corresponding binary manager class for the search method.
        """
        match self.method:
            case FaissSearchMethod.FLAT:
                from .managers.binary.flat import FaissBinaryFlatManager
                return FaissBinaryFlatManager
            case FaissSearchMethod.IVF:
                from .managers.binary.ivf import FaissBinaryIVFManager
                return FaissBinaryIVFManager
            case FaissSearchMethod.HNSW:
                from .managers.binary.hnsw import FaissBinaryHNSWManager
                return FaissBinaryHNSWManager
            case FaissSearchMethod.HASH:
                from .managers.binary.hash import FaissBinaryHashManager
                return FaissBinaryHashManager
            case FaissSearchMethod.MULTI_HASH:
                from .managers.binary.multi_hash import FaissBinaryMultiHashManager
                return FaissBinaryMultiHashManager
            case _:
                raise ValueError(f"Search method {self.method.value} is not supported for binary data type.")