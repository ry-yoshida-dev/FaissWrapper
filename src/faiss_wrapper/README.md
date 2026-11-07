# faiss_wrapper

## Overview

Python wrapper around [FAISS](https://github.com/facebookresearch/faiss).

## Components

| Component | Description |
| --------- | ----------- |
| [`types.py`](types.py) | NumPy array type aliases for vectors, values, and indices. |
| [`manager.py`](manager.py) | Abstract index API shared by all concrete managers. |
| [`dtype.py`](dtype.py) | Vector data types (float, binary) and the methods each supports. |
| [`parameter.py`](parameter.py) | `FaissParameter`: validated dtype / method / metric combination that selects a manager class. |
| [`method.py`](method.py) | Search methods and supported metrics. |
| [`metric.py`](metric.py) | Distance and similarity metrics. |
| [result/](result/README.md) | `FaissResult` and `FaissResults` search results, sorted nearest-first. |
| [managers/](managers/README.md) | Concrete manager implementations grouped by data type. |
