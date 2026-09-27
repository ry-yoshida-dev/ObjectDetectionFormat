from pathlib import Path

import pytest
from boxes import xyxy_box

from object_detection_format import AnnotationDataset, ImageAnnotation, ImageSize, ObjectAnnotation


def test_non_finite_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        ObjectAnnotation(box=xyxy_box(0.0, 0.0, float("inf"), 1.0), class_name="cat")


def test_confidence_outside_unit_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        ObjectAnnotation(box=xyxy_box(0.0, 0.0, 1.0, 1.0), class_name="cat", confidence=1.5)


def test_class_name_with_line_break_is_rejected() -> None:
    with pytest.raises(ValueError, match="line breaks"):
        ObjectAnnotation(box=xyxy_box(0.0, 0.0, 1.0, 1.0), class_name="cat\ndog")


def test_image_size_contains_boxes_up_to_its_edges() -> None:
    image_size: ImageSize = ImageSize(10, 10)
    assert image_size.contains(xyxy_box(0.0, 0.0, 10.0, 10.0))
    assert not image_size.contains(xyxy_box(-1.0, 0.0, 5.0, 5.0))


def test_box_outside_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds image size"):
        ImageAnnotation(
            image_path=Path("a.jpg"),
            image_size=ImageSize(10, 10),
            objects=(ObjectAnnotation(box=xyxy_box(0.0, 0.0, 11.0, 5.0), class_name="cat"),),
        )


def test_dataset_orders_given_classes_first(dataset: AnnotationDataset) -> None:
    assert dataset.class_names == ("dog", "person", "car")
    assert dataset.class_index("car") == 2


def test_dataset_rejects_unlisted_class(dataset: AnnotationDataset) -> None:
    with pytest.raises(ValueError, match="missing from class_names"):
        AnnotationDataset(images=dataset.images, class_names=("person",))


def test_dataset_rejects_duplicate_image_paths(dataset: AnnotationDataset) -> None:
    with pytest.raises(ValueError, match="image paths"):
        AnnotationDataset(images=(dataset.images[0], dataset.images[0]), class_names=dataset.class_names)
