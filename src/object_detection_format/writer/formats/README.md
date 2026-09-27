# formats

## Overview

Concrete writers, one per detection format. Single-file formats extend `AnnotationWriter`; formats with one label
file per image extend `PerImageAnnotationWriter`. JSON files are saved with
[FileHandler](https://github.com/ry-yoshida-dev/FileHandler)'s `FileHandler` and YOLO text files with its `TxtWriter`.

## Components

| Component | Description |
| --------- | ----------- |
| [coco.py](./coco.py) | `CocoWriter`: MS COCO JSON with optional `score`. |
| [yolo.py](./yolo.py) | `YoloWriter`: normalized YOLO text labels and `classes.txt`. |
| [pascal_voc.py](./pascal_voc.py) | `PascalVocWriter`: Pascal VOC XML. |
| [labelme.py](./labelme.py) | `LabelMeWriter`: LabelMe JSON with rectangle shapes. |
| [createml.py](./createml.py) | `CreateMlWriter`: Apple Create ML JSON. |

## Examples

```python
YoloWriter(is_confidence_included=True).write(dataset, Path("export/labels"))
```
