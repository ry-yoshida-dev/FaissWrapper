# result

## Overview

Dataclasses that wrap Faiss ``search_batch``/``search_single`` output. Neighbors are sorted nearest-first
in ``__post_init__``: ascending by value for distance metrics, descending for
similarity metrics such as inner product (see ``FaissMetric.is_larger_nearer``).

## Components

| Component | Description |
| --------- | ----------- |
| [`single.py`](single.py) | ``FaissResult`` for one query vector; arrays have shape ``(k,)``. Supports ``filter_by_value``. |
| [`batch.py`](batch.py) | ``FaissResults`` for multiple queries; arrays have shape ``(n_queries, k)``. |
| [`utils.py`](utils.py) | ``NeighborSorter`` sorts value/index pairs for single rows and batches. |

## Examples

```python
from faiss_wrapper import FaissResult, FaissResults

# Single query
result: FaissResult = manager.search_single(single_vector, k=10)
result.nearest_value
result.values[0]
filtered: FaissResult = result.filter_by_value(0.5)

# Batch
results: FaissResults = manager.search_batch(query_vectors, k=10)
row: FaissResult = results[0]
results.neighbor_count
```
