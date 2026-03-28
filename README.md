# FaissWrapper

## Overview

FaissWrapper is a Python package that provides a simplified interface for FAISS, a library for efficient similarity search and clustering of dense vectors.  
For more details, see [src/faiss_wrapper/README.md](src/faiss_wrapper/README.md).

**Note:** GPU debugging is not yet supported.

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development, install in editable mode so changes to the source take effect immediately:

```bash
pip install -e .
```

Dependencies (numpy, faiss-cpu) are installed automatically.  
To install only the dependencies without the package, use:

```bash
pip install -r requirements.txt
```

## Example

### Float + Flat via `FaissParameter`

After installing the package, import it from any directory:

```python
import numpy as np

from faiss_wrapper import (
    FaissDType,
    FaissManager,
    FaissSearchMethod,
    FaissMetric,
    FaissParameter,
    FaissResult,
)

dim: int = 64
rng = np.random.default_rng(42)
db_vectors: np.ndarray = rng.random((1_000, dim), dtype=np.float32)
query_vectors: np.ndarray = rng.random((5, dim), dtype=np.float32)

param = FaissParameter(
    dtype=FaissDType.FLOAT,
    method=FaissSearchMethod.FLAT,
    metric=FaissMetric.L2,
)
manager: FaissManager = param.manager_class(dimension=dim, metric=FaissMetric.L2)
manager.add(db_vectors)  # np.ndarray, shape (n, dim), float32

result: FaissResult = manager.search(query_vectors, k=10)  # k: int
# result.distances[i, j], result.indices[i, j]

# Optional: manager.save("my_index.faiss") / manager.load("my_index.faiss")
```

## Test

From the package root, run:

```bash
python -m tests.test
```
