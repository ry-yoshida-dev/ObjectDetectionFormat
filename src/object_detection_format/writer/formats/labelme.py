from pathlib import Path

from file_handler import FileHandler
from geometry import Box2D

from ...annotation import AnnotationDataset, ImageAnnotation, ObjectAnnotation
from ..json_value import JsonValue
from ..per_image import PerImageAnnotationWriter


class LabelMeWriter(PerImageAnnotationWriter):
    """
    Writes LabelMe JSON with rectangle shapes, one file per image.

    ``imagePath`` is relative to the output directory so LabelMe can open the image,
    and ``imageData`` is left empty.
    """

    labelme_version: str = "5.5.0"

    @property
    def file_suffix(self) -> str:
        """
        Extension of the label files.

        Returns
        -------
        str
            ``".json"``.
        """
        return ".json"

    def build_document(self, image: ImageAnnotation, output_directory: Path) -> dict[str, JsonValue]:
        """
        Build the LabelMe document of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to render.
        output_directory : Path
            Directory the JSON file is written to, used to relativize ``imagePath``.

        Returns
        -------
        dict[str, JsonValue]
            LabelMe document.
        """
        shapes: list[JsonValue] = [self._build_shape(annotated_object) for annotated_object in image.objects]
        image_path: Path = image.image_path.resolve().relative_to(output_directory.resolve(), walk_up=True)
        document: dict[str, JsonValue] = {
            "version": self.labelme_version,
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path.as_posix(),
            "imageData": None,
            "imageHeight": image.image_size.height,
            "imageWidth": image.image_size.width,
        }
        return document

    def write_label(self, image: ImageAnnotation, dataset: AnnotationDataset, label_path: Path) -> None:
        """
        Write the JSON file of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to write.
        dataset : AnnotationDataset
            Dataset the image belongs to.
        label_path : Path
            Destination JSON file.
        """
        FileHandler.save(self.build_document(image, label_path.parent), str(label_path))

    @staticmethod
    def _build_shape(annotated_object: ObjectAnnotation) -> dict[str, JsonValue]:
        box: Box2D = annotated_object.box
        return {
            "label": annotated_object.class_name,
            "points": [[float(box.x1), float(box.y1)], [float(box.x2), float(box.y2)]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
        }
