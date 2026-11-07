# result

## Overview

Dataclasses that wrap `search_single` / `search_batch` output, with neighbors sorted nearest-first.

## Components

| Component | Description |
| --------- | ----------- |
| [`single.py`](single.py) | `FaissResult` for one query vector. |
| [`batch.py`](batch.py) | `FaissResults` for multiple queries. |
| [`utils.py`](utils.py) | `NeighborSorter` sorts value/index pairs nearest-first. |

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
