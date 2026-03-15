from enum import Enum

from .metric import FaissMetric
from .method import FaissSearchMethod


class FaissDType(Enum):
    """
    Enum class for Faiss data types.

    Attributes:
    ----------
    FLOAT: float data type.
    BINARY: binary data type.
    """
    FLOAT = "float"
    BINARY = "binary"

    @property
    def supported_methods(self) -> list[FaissSearchMethod]:
        """
        Returns the supported methods for the data type.
        
        Returns:
        ----------
        list[FaissSearchMethod]: The supported methods for the data type.
        """
        match self:
            case FaissDType.FLOAT:
                return [
                    FaissSearchMethod.FLAT, 
                    FaissSearchMethod.IVF, 
                    FaissSearchMethod.IVFPQ, 
                    FaissSearchMethod.HNSW, 
                    FaissSearchMethod.LSH,
                ]
            case FaissDType.BINARY:
                return [
                    FaissSearchMethod.FLAT, 
                    FaissSearchMethod.IVF,
                    FaissSearchMethod.HNSW,
                    FaissSearchMethod.HASH,
                    FaissSearchMethod.MULTI_HASH
                ]

    def validate_method(
        self, 
        method: FaissSearchMethod
        ) -> None:
        """
        Raise ValueError if method is not supported for this dtype.

        Parameters
        ----------
        method : FaissSearchMethod
            The search method to validate.

        Raises
        ------
        ValueError
            If the method is not supported for this dtype. The message includes
            allowed methods and, when applicable, a hint to use the other dtype.
        """
        if method in self.supported_methods:
            return
        allowed_methods = "\n  - ".join(str(m) for m in self.supported_methods)
        other = FaissDType.BINARY if self is FaissDType.FLOAT else FaissDType.FLOAT
        hint = (
            f"\nOr use {other} ({method} is supported there)."
            if method in other.supported_methods
            else ""
            )
        raise ValueError(
            f"{method} is not supported for {self}.\n"
            f"Use one of:\n  - {allowed_methods}{hint}"
            )

    def validate_metric(
        self, 
        method: FaissSearchMethod, 
        metric: FaissMetric
        ) -> None:
        """
        Raise ValueError if metric is not supported for (dtype, method).

        Parameters
        ----------
        method : FaissSearchMethod
            The search method to validate against.
        metric : FaissMetric
            The metric to validate.

        Raises
        ------
        ValueError
            If the metric is not supported for the given method and this dtype.
            The message includes allowed metrics and, when applicable, methods
            that do support the requested metric.
        """
        supported = self.get_supported_metrics(method)
        if metric in supported:
            return
        allowed = "\n  - ".join(str(m) for m in supported)
        alternatives = self.get_methods_for_metric(metric)
        alt_list = "\n  - ".join(str(m) for m in alternatives)
        hint = (
            f"\nOr use:\n  - {alt_list}\n({metric} is supported there)."
            if alternatives
            else ""
        )
        raise ValueError(
            f"{metric} is not supported for {method}.\n"
            f"Use one of:\n  - {allowed}{hint}"
        )

    def get_supported_metrics(
        self, 
        method: FaissSearchMethod
        ) -> list[FaissMetric]:
        """
        Return metrics supported by the given method for this dtype.

        Parameters
        ----------
        method : FaissSearchMethod
            The search method to query.

        Returns
        -------
        list[FaissMetric]
            Metrics that the method supports for this dtype (float or binary).
        """
        return (
            method.float_supported_metrics
            if self is FaissDType.FLOAT
            else method.binary_supported_metrics
        )

    def get_methods_for_metric(
        self, 
        metric: FaissMetric
        ) -> list[FaissSearchMethod]:
        """
        Return methods that support the given metric for this dtype.

        Parameters
        ----------
        metric : FaissMetric
            The metric to check.

        Returns
        -------
        list[FaissSearchMethod]
            Methods that support the given metric for this dtype.
        """
        return [m for m in self.supported_methods if metric in self.get_supported_metrics(m)]