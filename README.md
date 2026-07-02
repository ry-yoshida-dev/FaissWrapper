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
    FaissResults,
    FloatVectorArray,
)

dim: int = 64
rng = np.random.default_rng(42)
db_vectors: FloatVectorArray = rng.random((1_000, dim), dtype=np.float32)
query_vectors: FloatVectorArray = rng.random((5, dim), dtype=np.float32)

param = FaissParameter(
    dtype=FaissDType.FLOAT,
    method=FaissSearchMethod.FLAT,
    metric=FaissMetric.L2,
)
manager: FaissManager = param.manager_class(dimension=dim, metric=FaissMetric.L2)
manager.add(db_vectors)  # FloatVectorArray, shape (n, dim), float32

results: FaissResults = manager.search_batch(query_vectors, k=10)  # k: int
# results.values[i, j], results.indices[i, j]

single_query: FloatVectorArray = query_vectors[0]
result: FaissResult = manager.search_single(single_query, k=10)
# result.values[j], result.indices[j]

# Optional: manager.save("my_index.faiss") / manager.load("my_index.faiss")
```

## Test

From the package root, run:

```bash
python -m tests.test
```
