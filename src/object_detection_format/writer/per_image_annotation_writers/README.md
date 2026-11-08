# per_image_annotation_writers

## Overview

`PerImageAnnotationWriter` implementations writing one label file per image.

## Components

| Component | Description |
| --------- | ----------- |
| [yolo.py](./yolo.py) | `YoloWriter`: normalized YOLO text labels and `classes.txt`. |
| [pascal_voc.py](./pascal_voc.py) | `PascalVocWriter`: Pascal VOC XML. |
| [labelme.py](./labelme.py) | `LabelMeWriter`: LabelMe JSON with rectangle shapes. |

## Examples

```python
YoloWriter(is_confidence_included=True).write(dataset, Path("export/labels"))
```
