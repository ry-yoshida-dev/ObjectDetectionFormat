from pathlib import Path

from file_handler import FileHandler
from geometry import Box2DFormat

from ...annotation import AnnotationDataset, ObjectAnnotation
from ..base import AnnotationWriter
from ..json_value import JsonValue


class CreateMlWriter(AnnotationWriter):
    """
    Writes Apple Create ML object detection JSON.

    Coordinates are the box center and size in pixels, and ``image`` is the image file name.

    Parameters
    ----------
    file_name : str, optional
        Name of the JSON file.
    """

    def __init__(self, file_name: str = "annotations.json") -> None:
        self.file_name: str = file_name

    def build_document(self, dataset: AnnotationDataset) -> list[JsonValue]:
        """
        Build the Create ML document.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to convert.

        Returns
        -------
        list[JsonValue]
            One entry per image.
        """
        return [
            {
                "image": image.image_path.name,
                "annotations": [self._build_annotation(annotated_object) for annotated_object in image.objects],
            }
            for image in dataset.images
        ]

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write the Create ML JSON file.

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

    @staticmethod
    def _build_annotation(annotated_object: ObjectAnnotation) -> dict[str, JsonValue]:
        center_size: list[float] = [float(value) for value in annotated_object.box.to_format(Box2DFormat.CXCYWH)]
        coordinates: dict[str, JsonValue] = dict(zip(("x", "y", "width", "height"), center_size, strict=True))
        return {"label": annotated_object.class_name, "coordinates": coordinates}
