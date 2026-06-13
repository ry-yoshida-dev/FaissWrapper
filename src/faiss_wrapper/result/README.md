# result

## Overview

Dataclasses that wrap Faiss ``search`` output. Neighbors are sorted by ascending
distance in ``__post_init__``.

## Components

| Component | Description |
| --------- | ----------- |
| [`single.py`](single.py) | ``FaissResult`` for one query vector; arrays have shape ``(k,)``. Supports ``filter_by_distance``. |
| [`batch.py`](batch.py) | ``FaissResults`` for multiple queries; arrays have shape ``(n_queries, k)``. |
| [`utils.py`](utils.py) | ``NeighborSorter`` sorts distance/index pairs for single rows and batches. |

## Examples

```python
from faiss_wrapper import FaissResult, FaissResults

# Single query
result: FaissResult = manager.search(single_vector, k=10)
result.min_distance
result.distances[0]
filtered: FaissResult = result.filter_by_distance(0.5)

# Batch
results: FaissResults = manager.search(query_vectors, k=10)
row: FaissResult = results[0]
results.neighbor_count
```
