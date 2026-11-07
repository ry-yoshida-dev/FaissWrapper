# tests

## Overview

Smoke and validation tests for `faiss_wrapper`.

## Components

| Component | Description |
| --------- | ----------- |
| [`test_manager.py`](test_manager.py) | Add, search, save, and load for every supported dtype / method / metric combination, plus input validation. |
| [`test_parameter.py`](test_parameter.py) | `FaissParameter` accepts valid combinations and rejects invalid ones. |
| [`test_result.py`](test_result.py) | `FaissResult` and `FaissResults` sorting, slicing, and validation. |
