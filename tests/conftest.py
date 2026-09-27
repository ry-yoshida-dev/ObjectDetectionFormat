from pathlib import Path

import numpy as np
import pytest
from boxes import xyxy_box
from geometry import Box2D, Box2DFormat

from object_detection_format import AnnotationDataset, ImageAnnotation, ImageSize, ObjectAnnotation


@pytest.fixture
def dataset(tmp_path: Path) -> AnnotationDataset:
    street: ImageAnnotation = ImageAnnotation(
        image_path=tmp_path / "images" / "street.jpg",
        image_size=ImageSize(width=200, height=100),
        objects=(
            ObjectAnnotation(box=xyxy_box(10.0, 20.0, 50.0, 60.0), class_name="person", confidence=0.9),
            ObjectAnnotation(
                box=Box2D.register(value=np.array([100, 0, 100, 100]), box2d_format=Box2DFormat.XYWH),
                class_name="car",
                confidence=0.5,
            ),
        ),
    )
    empty: ImageAnnotation = ImageAnnotation(
        image_path=tmp_path / "images" / "empty.png", image_size=ImageSize(width=64, height=32)
    )
    return AnnotationDataset.from_images([street, empty], class_names=["dog"])
