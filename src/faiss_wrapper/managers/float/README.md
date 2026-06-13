# float

## Overview

This subtree wraps FAISS **dense float** indices (`float32` vectors). 

## Method–metric compatibility

Aligned with **`FaissSearchMethod.float_supported_metrics`** in [`method.py`](../../method.py).

| Method | L1  | L2  | INNER_PRODUCT | Linf | Lp  | CANBERRA | BRAY_CURTIS | JENSEN_SHANNON | HAMMING |
| ------ | --- | --- | ------------- | ---- | --- | -------- | ----------- | -------------- | ------- |
| FLAT   | ✔   | ✔   | ✔             | ✔    | ✔   | ✔        | ✔           | ✔              |         |
| IVF    | ✔   | ✔   | ✔             | ✔    | ✔   | ✔        | ✔           | ✔              |         |
| HNSW   | ✔   | ✔   | ✔             | ✔    | ✔   | ✔        | ✔           | ✔              |         |
| IVFPQ  |     | ✔   | ✔             |      |     |          |             |                |         |
| LSH    |     |     |               |      |     |          |             |                |         |

Unsupported or untested combinations may change with FAISS releases; confirm against the [FAISS documentation](https://github.com/facebookresearch/faiss/wiki) when in doubt.

## Components

| Component | Description |
| --------- | ----------- |
| [`manager.py`](manager.py) | Base for float indexes; save/load via FAISS. |
| [flat/](flat/) | Exact (brute-force) search. |
| [ivf/](ivf/) | Inverted file over clusters; requires training. |
| [ivfpq/](ivfpq/) | IVF + product quantization. |
| [hnsw/](hnsw/) | Graph-based approximate search. |
| [lsh/](lsh/) | LSH on float vectors. |
