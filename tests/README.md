# tests

## Overview

Smoke and validation tests for `faiss_wrapper`.

## Components

| Component | Description |
| --------- | ----------- |
| [`test.py`](test.py) | Add, search, save, and load for every supported dtype / method / metric combination. |
| [`test_parameter.py`](test_parameter.py) | `FaissParameter` accepts valid combinations and rejects invalid ones. |
