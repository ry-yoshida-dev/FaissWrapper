# FaissWrapper

## Overview

FaissWrapper provides a simplified, typed interface for [FAISS](https://github.com/facebookresearch/faiss) similarity search. A `FaissParameter` validates a data type, search method, and metric and selects the matching index manager. GPU debugging is not yet supported.

| Method | Float metrics | Binary metrics | Description |
| ------ | ------------- | -------------- | ----------- |
| `FLAT` | All except `HAMMING` | `HAMMING` | Exact (brute-force) search. |
| `IVF` | All except `HAMMING` | `HAMMING` | Inverted file over clusters; requires training. |
| `HNSW` | All except `HAMMING` | `HAMMING` | Graph-based approximate search. |
| `IVFPQ` | `L2`, `INNER_PRODUCT` | | IVF + product quantization. |
| `LSH` | `HAMMING` | | Locality-sensitive hashing on float vectors. |
| `HASH` | | `HAMMING` | Single hash table on binary vectors. |
| `MULTI_HASH` | | `HAMMING` | Multiple hash tables on binary vectors. |

## Installation

```bash
pip install .
```

## Usage

```python
import numpy as np

from faiss_wrapper import FaissDType, FaissMetric, FaissParameter, FaissSearchMethod

dim = 64
rng = np.random.default_rng(42)
db_vectors = rng.random((1_000, dim), dtype=np.float32)
query_vectors = rng.random((5, dim), dtype=np.float32)

param = FaissParameter(
    dtype=FaissDType.FLOAT,
    method=FaissSearchMethod.FLAT,
    metric=FaissMetric.L2,
)
manager = param.manager_class(dimension=dim, metric=param.metric)
manager.add(db_vectors)

results = manager.search_batch(query_vectors, k=10)  # results.values, results.indices
result = manager.search_single(query_vectors[0], k=10)
manager.save("my_index.faiss")
```

## Development

The tests are scripts run as modules from the project root:

```bash
python -m tests.test
PYTHONPATH=. python tests/test_parameter.py
```
