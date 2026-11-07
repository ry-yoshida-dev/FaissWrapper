# float

## Overview

FAISS dense float indices (`float32` vectors).

## Components

| Component | Description |
| --------- | ----------- |
| [`manager.py`](manager.py) | Base for float indexes; save/load via FAISS. |
| [flat/](flat/README.md) | Exact (brute-force) search. |
| [ivf/](ivf/README.md) | Inverted file over clusters; requires training. |
| [ivfpq/](ivfpq/README.md) | IVF + product quantization. |
| [hnsw/](hnsw/README.md) | Graph-based approximate search. |
| [lsh/](lsh/README.md) | LSH on float vectors. |
