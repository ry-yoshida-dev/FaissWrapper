# src

## Overview

Python wrapper around [FAISS](https://github.com/facebookresearch/faiss).
See [float/README.md](float/README.md) and [binary/README.md](binary/README.md) for index types and metrics.

## Data types

| Data Type | Description |
| --------- | ----------- |
| **Float** | Vectors are represented as float32. |
| **Binary** | Vectors are represented as uint8. |

## Method

| Method         | Float | Binary | Description |
| -------------- | ----- | ------ | ----------- |
| **FLAT**       | ✔     | ✔      | Exact (brute-force) search. |
| **IVF**        | ✔     | ✔      | Inverted file over clusters; requires training. |
| **HNSW**       | ✔     | ✔      | Graph-based approximate search. |
| **IVFPQ**      | ✔     | ✗      | IVF + product quantization. |
| **LSH**        | ✔     | ✗      | LSH on float vectors. |
| **HASH**       | ✗     | ✔      | Single hash table on binary vectors. |
| **MULTI_HASH** | ✗     | ✔      | Multiple hash tables on binary vectors. |

✔ = supported in this wrapper for that data type; ✗ = not exposed.

## Metrics

| Metric             | Description                                            |
| ------------------ | ------------------------------------------------------ |
| **L2**             | Squared L2 (Euclidean) distance (FAISS convention).    |
| **INNER_PRODUCT**  | Negative inner product (maximum inner product search). |
| **L1**             | L1 (Manhattan) distance.                               |
| **LINF**           | L-infinity (max) distance.                             |
| **Lp**             | Lp distance (configurable **p**).                      |
| **CANBERRA**       | Canberra distance.                                     |
| **BRAY_CURTIS**    | Bray–Curtis distance.                                  |
| **JENSEN_SHANNON** | Jensen–Shannon divergence.                             |
| **HAMMING**        | Used with **LSH** (Hamming on the LSH code).           |

## Components

| Component | Description |
| --------- | ----------- |
| [`types.py`](types.py) | NumPy array type aliases for vectors, distances, and indices. |
| [`manager.py`](manager.py) | Common index API. Concrete indexes inherit from this. |
| [`method.py`](method.py) | Search methods and supported metrics. |
| [`metric.py`](metric.py) | Distance/similarity kinds. |
| [result/](result/README.md) | ``FaissResult`` (shape ``(k,)``) and ``FaissResults`` (shape ``(n_queries, k)``); neighbors sorted in ``__post_init__``. |
| [float/](float/README.md) | Float index API. Concrete float indexes inherit from the common manager. |
| [binary/](binary/README.md) | Binary index API. Concrete binary indexes inherit from the common manager. |
