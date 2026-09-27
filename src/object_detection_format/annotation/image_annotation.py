from dataclasses import dataclass
from pathlib import Path

from .image_size import ImageSize
from .object_annotation import ObjectAnnotation


@dataclass(frozen=True, eq=False)
class ImageAnnotation:
    """
    Objects labeled in one image.

    Attributes
    ----------
    image_path : Path
        Location of the image file.
    image_size : ImageSize
        Size of the image the boxes refer to.
    objects : tuple[ObjectAnnotation, ...], optional
        Labeled objects; empty for a background image.

    Raises
    ------
    ValueError
        If a box extends beyond the image.
    """

    image_path: Path
    image_size: ImageSize
    objects: tuple[ObjectAnnotation, ...] = ()

    def __post_init__(self) -> None:
        for annotated_object in self.objects:
            if not self.image_size.contains(annotated_object.box):
                raise ValueError(
                    f"box {annotated_object.box} of {self.image_path} exceeds image size "
                    + f"({self.image_size.width}, {self.image_size.height})"
                )

    @property
    def class_names(self) -> tuple[str, ...]:
        """
        Classes used in this image.

        Returns
        -------
        tuple[str, ...]
            Distinct class names in order of first appearance.
        """
        return tuple(dict.fromkeys(annotated_object.class_name for annotated_object in self.objects))
