# annotation_writers

## Overview

`AnnotationWriter` implementations writing the whole dataset to one JSON file.

## Components

| Component | Description |
| --------- | ----------- |
| [coco.py](./coco.py) | `CocoWriter`: MS COCO JSON with optional `score`. |
| [createml.py](./createml.py) | `CreateMlWriter`: Apple Create ML JSON. |

## Examples

```python
CocoWriter(is_confidence_included=False).write(dataset, Path("export/coco"))
```
