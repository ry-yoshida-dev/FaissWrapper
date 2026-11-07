# binary

## Overview

FAISS binary indices (`uint8` packed bit vectors, Hamming distance).

## Components

| Component | Description |
| --------- | ----------- |
| [`manager.py`](manager.py) | Base for binary indexes; save/load via FAISS. |
| [flat/](flat/README.md) | Exact (brute-force) search. |
| [ivf/](ivf/README.md) | Inverted file over clusters; requires training. |
| [hnsw/](hnsw/README.md) | Graph-based approximate search. |
| [hash/](hash/README.md) | Single hash table on binary vectors. |
| [multi_hash/](multi_hash/README.md) | Multiple hash tables on binary vectors. |
