from pathlib import Path

from file_handler import FileHandler
from geometry import Box2D, Box2DFormat

from ...annotation import AnnotationDataset, ObjectAnnotation
from ..base import AnnotationWriter
from ..json_value import JsonValue


class CocoWriter(AnnotationWriter):
    """
    Writes MS COCO object detection JSON.

    Image, annotation and category ids start at 1. ``bbox`` is ``[x_min, y_min, width, height]``
    in pixels and ``file_name`` is the image file name without directories.

    Parameters
    ----------
    file_name : str, optional
        Name of the JSON file.
    is_confidence_included : bool, optional
        Whether to add ``score`` to annotations that carry a confidence.
    """

    def __init__(self, file_name: str = "annotations.json", is_confidence_included: bool = True) -> None:
        self.file_name: str = file_name
        self.is_confidence_included: bool = is_confidence_included

    def build_document(self, dataset: AnnotationDataset) -> dict[str, JsonValue]:
        """
        Build the COCO document.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to convert.

        Returns
        -------
        dict[str, JsonValue]
            Document with ``images``, ``annotations`` and ``categories``.
        """
        images: list[JsonValue] = []
        annotations: list[JsonValue] = []
        for image_id, image in enumerate(dataset.images, start=1):
            images.append(
                {
                    "id": image_id,
                    "file_name": image.image_path.name,
                    "width": image.image_size.width,
                    "height": image.image_size.height,
                }
            )
            for annotated_object in image.objects:
                annotations.append(self._build_annotation(annotated_object, len(annotations) + 1, image_id, dataset))
        categories: list[JsonValue] = [
            {"id": category_id, "name": class_name, "supercategory": ""}
            for category_id, class_name in enumerate(dataset.class_names, start=1)
        ]
        return {"images": images, "annotations": annotations, "categories": categories}

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write the COCO JSON file.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to export.
        output_directory : Path
            Directory receiving the file.

        Returns
        -------
        tuple[Path, ...]
            The JSON file.
        """
        output_path: Path = output_directory / self.file_name
        FileHandler.save(self.build_document(dataset), str(output_path))
        return (output_path,)

    def _build_annotation(
        self, annotated_object: ObjectAnnotation, annotation_id: int, image_id: int, dataset: AnnotationDataset
    ) -> dict[str, JsonValue]:
        box: Box2D = annotated_object.box
        bbox: list[JsonValue] = [float(value) for value in box.to_format(Box2DFormat.XYWH)]
        annotation: dict[str, JsonValue] = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": dataset.class_index(annotated_object.class_name) + 1,
            "bbox": bbox,
            "area": float(box.area),
            "iscrowd": 0,
            "segmentation": [],
        }
        if self.is_confidence_included and annotated_object.confidence is not None:
            annotation["score"] = annotated_object.confidence
        return annotation
