# annotation

## Overview

Immutable, validated annotation model shared by every writer. Boxes are `geometry.Box2D` in absolute pixel
coordinates and any supported layout; each writer converts them to its own encoding. Invalid data (empty boxes,
boxes outside the image, unknown classes) is rejected at construction.

## Components

| Component | Description |
| --------- | ----------- |
| [image_size.py](./image_size.py) | `ImageSize`: image width, height and channel count, and whether a box fits inside. |
| [object_annotation.py](./object_annotation.py) | `ObjectAnnotation`: `Box2D`, class name and optional confidence. |
| [image_annotation.py](./image_annotation.py) | `ImageAnnotation`: image path, size and its objects. |
| [annotation_dataset.py](./annotation_dataset.py) | `AnnotationDataset`: images and the ordered class list defining class ids. |

## Examples

```python
dataset = AnnotationDataset.from_images(images, class_names=["person", "car"])
dataset.class_names
dataset.class_index("car")
```
