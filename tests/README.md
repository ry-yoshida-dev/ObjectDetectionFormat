# tests

## Overview

Unit tests of the annotation model and every writer.

## Components

| Component | Description |
| --------- | ----------- |
| [conftest.py](./conftest.py) | Shared dataset fixture with a labeled and an empty image. |
| [boxes.py](./boxes.py) | Box construction helper shared by tests. |
| [test_annotation.py](./test_annotation.py) | Validation of boxes, objects, images and datasets. |
| [test_writers.py](./test_writers.py) | Output of each format writer. |

## Examples

```bash
pytest
```
