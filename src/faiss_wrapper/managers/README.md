# managers

## Overview

Concrete FAISS index managers grouped by vector data type (`FaissDType`).
Each dtype subtree exposes one manager class per supported search method.

The abstract base [`FaissManager`](../manager.py) lives at the package root; this
directory holds dtype-specific subclasses and their method implementations.

## Components

| Component | Description |
| --------- | ----------- |
| [float/](float/README.md) | Float32 vector managers (`FaissFloat*` classes). |
| [binary/](binary/README.md) | Packed binary vector managers (`FaissBinary*` classes). |
