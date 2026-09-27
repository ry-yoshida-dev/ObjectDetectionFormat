from collections.abc import Iterable
from dataclasses import dataclass

from .image_annotation import ImageAnnotation


@dataclass(frozen=True, eq=False)
class AnnotationDataset:
    """
    Annotated images sharing one class list.

    The order of ``class_names`` defines class ids: YOLO uses the index,
    COCO uses the index plus one.

    Attributes
    ----------
    images : tuple[ImageAnnotation, ...]
        Annotated images.
    class_names : tuple[str, ...]
        Every class that may appear, in id order.

    Raises
    ------
    ValueError
        If class names repeat, an image path repeats, or an object uses an unlisted class.
    """

    images: tuple[ImageAnnotation, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError(f"class_names must be unique. got {self.class_names}")
        image_paths: list[str] = [str(image.image_path) for image in self.images]
        if len(set(image_paths)) != len(image_paths):
            raise ValueError("image paths must be unique")
        known_class_names: set[str] = set(self.class_names)
        for image in self.images:
            unknown_class_names: list[str] = [name for name in image.class_names if name not in known_class_names]
            if unknown_class_names:
                raise ValueError(f"{image.image_path} uses classes missing from class_names: {unknown_class_names}")

    @classmethod
    def from_images(cls, images: Iterable[ImageAnnotation], class_names: Iterable[str] = ()) -> "AnnotationDataset":
        """
        Build a dataset whose class list covers every object.

        Parameters
        ----------
        images : Iterable[ImageAnnotation]
            Annotated images.
        class_names : Iterable[str], optional
            Classes placed first, in this order, even if no object uses them.

        Returns
        -------
        AnnotationDataset
            Dataset with ``class_names`` followed by the remaining classes in order of first appearance.
        """
        image_tuple: tuple[ImageAnnotation, ...] = tuple(images)
        ordered_class_names: dict[str, None] = dict.fromkeys(class_names)
        for image in image_tuple:
            ordered_class_names.update(dict.fromkeys(image.class_names))
        return cls(images=image_tuple, class_names=tuple(ordered_class_names))

    def class_index(self, class_name: str) -> int:
        """
        Zero-based position of a class.

        Parameters
        ----------
        class_name : str
            Listed class.

        Raises
        ------
        KeyError
            If the class is not listed.

        Returns
        -------
        int
            Index into ``class_names``.
        """
        if class_name not in self.class_names:
            raise KeyError(f"unknown class: {class_name!r}")
        return self.class_names.index(class_name)
