from pathlib import Path

from geometry import Box2DFormat
from txt_writer import TxtWriter

from ...annotation import AnnotationDataset, ImageAnnotation, ObjectAnnotation
from ..per_image import PerImageAnnotationWriter


class YoloWriter(PerImageAnnotationWriter):
    """
    Writes YOLO (Darknet / Ultralytics) text labels.

    Each line is ``class_id center_x center_y width height`` normalized to ``[0, 1]``;
    images without objects get an empty file. Class names are written to ``classes.txt``,
    one per line in id order.

    Parameters
    ----------
    is_confidence_included : bool, optional
        Whether to append the confidence as a sixth column, as Ultralytics ``save_conf`` does.
    decimal_places : int, optional
        Digits after the decimal point of normalized values.

    Raises
    ------
    ValueError
        If ``decimal_places`` is negative.
    """

    class_file_name: str = "classes.txt"

    def __init__(self, is_confidence_included: bool = False, decimal_places: int = 6) -> None:
        if decimal_places < 0:
            raise ValueError(f"decimal_places must not be negative. got {decimal_places}")
        self.is_confidence_included: bool = is_confidence_included
        self.decimal_places: int = decimal_places

    @property
    def file_suffix(self) -> str:
        """
        Extension of the label files.

        Returns
        -------
        str
            ``".txt"``.
        """
        return ".txt"

    def build_lines(self, image: ImageAnnotation, dataset: AnnotationDataset) -> list[str]:
        """
        Render the label lines of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to render.
        dataset : AnnotationDataset
            Dataset providing class ids.

        Raises
        ------
        ValueError
            If confidences are included but an object has none.

        Returns
        -------
        list[str]
            One line per object, without line breaks.
        """
        return [self._format_line(annotated_object, image, dataset) for annotated_object in image.objects]

    def write_label(self, image: ImageAnnotation, dataset: AnnotationDataset, label_path: Path) -> None:
        """
        Write the label file of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to write.
        dataset : AnnotationDataset
            Dataset providing class ids.
        label_path : Path
            Destination text file.

        Raises
        ------
        ValueError
            If confidences are included but an object has none.
        """
        self._write_lines(self.build_lines(image, dataset), label_path)

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write one label file per image and the class list.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to export.
        output_directory : Path
            Directory receiving the files.

        Raises
        ------
        ValueError
            If two images share a stem or an image stem matches the class file name, ignoring case.

        Returns
        -------
        tuple[Path, ...]
            Label files in image order followed by ``classes.txt``.
        """
        class_file_path: Path = output_directory / self.class_file_name
        if any(
            self.label_path(image, output_directory).name.casefold() == self.class_file_name.casefold()
            for image in dataset.images
        ):
            raise ValueError(f"an image stem collides with {self.class_file_name}")
        label_paths: tuple[Path, ...] = super().write(dataset, output_directory)
        self._write_lines(list(dataset.class_names), class_file_path)
        return (*label_paths, class_file_path)

    @staticmethod
    def _write_lines(lines: list[str], output_path: Path) -> None:
        with TxtWriter(output_path=str(output_path), is_reset_enabled=True) as writer:
            for line in lines:
                writer.write(line, is_new_line=True)

    def _format_line(
        self, annotated_object: ObjectAnnotation, image: ImageAnnotation, dataset: AnnotationDataset
    ) -> str:
        width: int = image.image_size.width
        height: int = image.image_size.height
        values: list[float] = [
            float(value) / scale
            for value, scale in zip(
                annotated_object.box.to_format(Box2DFormat.CXCYWH), (width, height, width, height), strict=True
            )
        ]
        if self.is_confidence_included:
            if annotated_object.confidence is None:
                raise ValueError(
                    f"object of class {annotated_object.class_name!r} in {image.image_path} has no confidence"
                )
            values.append(annotated_object.confidence)
        formatted_values: list[str] = [f"{value:.{self.decimal_places}f}" for value in values]
        return " ".join([str(dataset.class_index(annotated_object.class_name)), *formatted_values])
