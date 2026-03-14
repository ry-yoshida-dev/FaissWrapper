# binary

## Overview

This subtree wraps FAISS **binary** indices (`uint8` packed bit vectors; Hamming distance).

## Method–metric compatibility

Aligned with **`FaissSearchMethod.supported_metrics`** in [`method.py`](../method.py). Hamming only (float metrics N/A).

| Method     | L1  | L2  | INNER_PRODUCT | Linf | Lp  | CANBERRA | BRAY_CURTIS | JENSEN_SHANNON | HAMMING |
| ---------- | --- | --- | ------------- | ---- | --- | -------- | ----------- | -------------- | ------- |
| FLAT       |     |     |               |      |     |          |             |                | ✔       |
| IVF        |     |     |               |      |     |          |             |                | ✔       |
| HNSW       |     |     |               |      |     |          |             |                | ✔       |
| HASH       |     |     |               |      |     |          |             |                | ✔       |
| MULTI_HASH |     |     |               |      |     |          |             |                | ✔       |


Unsupported or untested combinations may change with FAISS releases; confirm against the [FAISS documentation](https://github.com/facebookresearch/faiss/wiki) when in doubt.

## Components

| Component | Description |
| --------- | ----------- |
| [`manager.py`](manager.py) | Base for binary indexes; save/load via FAISS. |
| [flat/](flat/) | Exact (brute-force) search. |
| [ivf/](ivf/) | Inverted file over clusters; requires training. |
| [hnsw/](hnsw/) | Graph-based approximate search. |
| [hash/](hash/) | Single hash table on binary vectors. |
| [multi_hash/](multi_hash/) | Multiple hash tables on binary vectors. |

