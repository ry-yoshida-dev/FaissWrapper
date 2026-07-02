# src

## Overview

Python wrapper around [FAISS](https://github.com/facebookresearch/faiss).
See [managers/float/README.md](managers/float/README.md) and [managers/binary/README.md](managers/binary/README.md) for index types and metrics.

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
| [`types.py`](types.py) | NumPy array type aliases for vectors, values, and indices. |
| [`manager.py`](manager.py) | Abstract index API shared by all concrete managers. |
| [`method.py`](method.py) | Search methods and supported metrics. |
| [`metric.py`](metric.py) | Distance/similarity kinds; exposes ``is_larger_nearer`` per metric. |
| [result/](result/README.md) | ``FaissResult`` (shape ``(k,)``) and ``FaissResults`` (shape ``(n_queries, k)``); neighbors sorted nearest-first in ``__post_init__``. |
| [managers/](managers/README.md) | Concrete manager implementations grouped by data type. |
| [managers/float/](managers/float/README.md) | Float32 vector managers. |
| [managers/binary/](managers/binary/README.md) | Packed binary vector managers. |
