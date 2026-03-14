# FaissWrapper

## Overview

FAISSWRAPPER is a Python package that provides a simplified interface for using FAISS, a library for efficient similarity search and clustering of dense vectors.

NOTE: 
GPU debugging is not yet supported.

## Installation

To install the required dependencies, use pip with the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Example

### Float + Flat via `FaissParameter`

Run from the package root (the directory that contains `src/`), with that directory on `PYTHONPATH` (e.g. `export PYTHONPATH=.`) or install the package in editable mode.

```python
import numpy as np

from src.dtype import FaissDType
from src.float.manager import FaissFloatManager
from src.method import FaissSearchMethod
from src.metric import FaissMetric
from src.parameter import FaissParameter
from src.result import FaissResult

dim: int = 64
rng = np.random.default_rng(42)
db_vectors: np.ndarray = rng.random((1_000, dim), dtype=np.float32)
query_vectors: np.ndarray = rng.random((5, dim), dtype=np.float32)

param = FaissParameter(
    dtype=FaissDType.FLOAT,
    method=FaissSearchMethod.FLAT,
    metric=FaissMetric.L2,
)
manager: FaissFloatManager = param.manager_class(dimension=dim, metric=FaissMetric.L2)
manager.add(db_vectors)  # np.ndarray, shape (n, dim), float32

result: FaissResult = manager.search(query_vectors, k=10)  # k: int
# result.distances[i, j], result.indices[i, j]

# Optional: manager.save("my_index.faiss") / manager.load("my_index.faiss")
```

## Test

To run the example, execute the following command:

```bash
python -m tests.test
```
