import json
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import pytest

from object_detection_format import (
    AnnotationDataset,
    AnnotationFormat,
    CocoWriter,
    CreateMlWriter,
    ImageAnnotation,
    LabelMeWriter,
    PascalVocWriter,
    YoloWriter,
)


def test_coco_document_uses_one_based_ids_and_xywh(dataset: AnnotationDataset, tmp_path: Path) -> None:
    written: tuple[Path, ...] = CocoWriter().write(dataset, tmp_path / "coco")
    assert written == (tmp_path / "coco" / "annotations.json",)
    document = json.loads(written[0].read_text(encoding="utf-8"))
    assert [image["file_name"] for image in document["images"]] == ["street.jpg", "empty.png"]
    assert [category["id"] for category in document["categories"]] == [1, 2, 3]
    person, car = document["annotations"]
    assert person == {
        "id": 1,
        "image_id": 1,
        "category_id": 2,
        "bbox": [10.0, 20.0, 40.0, 40.0],
        "area": 1600.0,
        "iscrowd": 0,
        "segmentation": [],
        "score": 0.9,
    }
    assert (car["id"], car["category_id"]) == (2, 3)


def test_coco_can_omit_scores(dataset: AnnotationDataset, tmp_path: Path) -> None:
    document = CocoWriter(is_confidence_included=False).build_document(dataset)
    assert isinstance(document["annotations"], list)
    assert all(isinstance(annotation, dict) and "score" not in annotation for annotation in document["annotations"])


def test_yolo_writes_normalized_labels_and_classes(dataset: AnnotationDataset, tmp_path: Path) -> None:
    output_directory: Path = tmp_path / "yolo"
    written: tuple[Path, ...] = YoloWriter().write(dataset, output_directory)
    assert [path.name for path in written] == ["street.txt", "empty.txt", "classes.txt"]
    assert (output_directory / "street.txt").read_text() == (
        "1 0.150000 0.400000 0.200000 0.400000\n2 0.750000 0.500000 0.500000 1.000000\n"
    )
    assert (output_directory / "empty.txt").read_text() == ""
    assert (output_directory / "classes.txt").read_text() == "dog\nperson\ncar\n"


def test_yolo_can_append_confidence(dataset: AnnotationDataset, tmp_path: Path) -> None:
    lines: list[str] = YoloWriter(is_confidence_included=True, decimal_places=2).build_lines(dataset.images[0], dataset)
    assert lines[0] == "1 0.15 0.40 0.20 0.40 0.90"


def test_yolo_confidence_column_requires_confidence(dataset: AnnotationDataset, tmp_path: Path) -> None:
    street: ImageAnnotation = dataset.images[0]
    unscored: ImageAnnotation = replace(street, objects=(replace(street.objects[0], confidence=None),))
    with pytest.raises(ValueError, match="no confidence"):
        YoloWriter(is_confidence_included=True).build_lines(unscored, dataset)


def test_per_image_writer_rejects_colliding_stems(dataset: AnnotationDataset, tmp_path: Path) -> None:
    street: ImageAnnotation = dataset.images[0]
    twin: ImageAnnotation = replace(street, image_path=street.image_path.with_name("STREET.png"))
    colliding: AnnotationDataset = AnnotationDataset(images=(street, twin), class_names=dataset.class_names)
    with pytest.raises(ValueError, match="collide"):
        PascalVocWriter().write(colliding, tmp_path / "voc")
    assert not (tmp_path / "voc").exists()


def test_yolo_rejects_stem_matching_class_file_ignoring_case(dataset: AnnotationDataset, tmp_path: Path) -> None:
    street: ImageAnnotation = dataset.images[0]
    shadowing: ImageAnnotation = replace(street, image_path=street.image_path.with_name("Classes.jpg"))
    shadowing_dataset: AnnotationDataset = AnnotationDataset(images=(shadowing,), class_names=dataset.class_names)
    with pytest.raises(ValueError, match="collides"):
        YoloWriter().write(shadowing_dataset, tmp_path / "yolo")


def test_pascal_voc_writes_rounded_corners(dataset: AnnotationDataset, tmp_path: Path) -> None:
    PascalVocWriter().write(dataset, tmp_path / "voc")
    root: ElementTree.Element = ElementTree.parse(tmp_path / "voc" / "street.xml").getroot()
    assert root.findtext("filename") == "street.jpg"
    assert root.findtext("size/width") == "200"
    assert root.findtext("size/depth") == "3"
    objects: list[ElementTree.Element] = root.findall("object")
    assert [element.findtext("name") for element in objects] == ["person", "car"]
    assert [objects[0].findtext(f"bndbox/{tag}") for tag in ("xmin", "ymin", "xmax", "ymax")] == [
        "10",
        "20",
        "50",
        "60",
    ]
    assert ElementTree.parse(tmp_path / "voc" / "empty.xml").getroot().findall("object") == []


def test_labelme_points_to_image_relative_to_labels(dataset: AnnotationDataset, tmp_path: Path) -> None:
    LabelMeWriter().write(dataset, tmp_path / "labelme")
    document = json.loads((tmp_path / "labelme" / "street.json").read_text(encoding="utf-8"))
    assert document["imagePath"] == "../images/street.jpg"
    assert (document["imageWidth"], document["imageHeight"]) == (200, 100)
    assert document["shapes"][0]["points"] == [[10.0, 20.0], [50.0, 60.0]]
    assert document["shapes"][0]["shape_type"] == "rectangle"


def test_createml_uses_center_coordinates(dataset: AnnotationDataset, tmp_path: Path) -> None:
    written: tuple[Path, ...] = CreateMlWriter().write(dataset, tmp_path / "createml")
    document = json.loads(written[0].read_text(encoding="utf-8"))
    assert document[0]["annotations"][0] == {
        "label": "person",
        "coordinates": {"x": 30.0, "y": 40.0, "width": 40.0, "height": 40.0},
    }
    assert document[1] == {"image": "empty.png", "annotations": []}


@pytest.mark.parametrize("annotation_format", list(AnnotationFormat))
def test_every_format_writes_files(
    annotation_format: AnnotationFormat, dataset: AnnotationDataset, tmp_path: Path
) -> None:
    written: tuple[Path, ...] = annotation_format.create_writer().write(dataset, tmp_path / annotation_format.value)
    assert written
    assert all(path.is_file() for path in written)
    assert annotation_format.display_name
