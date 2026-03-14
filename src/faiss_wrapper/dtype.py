from enum import Enum

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
