# ObjectDetectionFormat

## Overview

Export bounding-box annotations to common object detection formats. Boxes are
[Geometry](https://github.com/ry-yoshida-dev/Geometry) `Box2D` objects in any format (XYXY, XYWH, ...); the library
does not depend on any detector: build an `AnnotationDataset` from your own boxes and pick a writer. Files are
written with [FileHandler](https://github.com/ry-yoshida-dev/FileHandler).

| Format | Output | Box encoding |
| ------ | ------ | ------------ |
| MS COCO | `annotations.json` | `[x_min, y_min, width, height]` pixels, ids from 1, optional `score` |
| YOLO | `<stem>.txt` per image + `classes.txt` | `class_id cx cy w h` normalized, optional confidence column |
| Pascal VOC | `<stem>.xml` per image | integer `xmin ymin xmax ymax` pixels |
| LabelMe | `<stem>.json` per image | rectangle `[[x_min, y_min], [x_max, y_max]]` pixels |
| Create ML | `annotations.json` | center `x, y, width, height` pixels |

See [src/object_detection_format](src/object_detection_format/README.md) for the package layout.

## Installation

```bash
pip install git+https://github.com/ry-yoshida-dev/ObjectDetectionFormat.git
```

## Usage

```python
from pathlib import Path

import numpy as np
from geometry import Box2D, Box2DFormat
from object_detection_format import (
    AnnotationDataset,
    AnnotationFormat,
    ImageAnnotation,
    ImageSize,
    ObjectAnnotation,
    YoloWriter,
)

image = ImageAnnotation(
    image_path=Path("images/street.jpg"),
    image_size=ImageSize(width=1920, height=1080),
    objects=(
        ObjectAnnotation(
            box=Box2D.register(value=np.array([10.0, 20.0, 200.0, 400.0]), box2d_format=Box2DFormat.XYXY),
            class_name="person",
            confidence=0.87,
        ),
    ),
)
dataset = AnnotationDataset.from_images([image], class_names=["person", "car"])

AnnotationFormat.COCO.create_writer().write(dataset, Path("export/coco"))
YoloWriter(is_confidence_included=True).write(dataset, Path("export/labels"))
```

Converting an [OpenVocabularyDetector](https://github.com/ry-yoshida-dev/OpenVocabularyDetector) result:

```python
objects = tuple(
    ObjectAnnotation(
        box=detection.box,
        class_name=detection.class_name,
        confidence=detection.confidence,
    )
    for detection in result
)
image = ImageAnnotation(
    image_path=path,
    image_size=ImageSize(width=result.image_size.width, height=result.image_size.height),
    objects=objects,
)
```
