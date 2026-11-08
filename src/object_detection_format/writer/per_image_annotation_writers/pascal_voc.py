from pathlib import Path
from xml.etree import ElementTree

from geometry import Box2D

from ...annotation import AnnotationDataset, ImageAnnotation, ObjectAnnotation
from ..per_image_annotation_writer import PerImageAnnotationWriter


class PascalVocWriter(PerImageAnnotationWriter):
    """
    Writes Pascal VOC XML, one file per image.

    Box corners are rounded to integer pixels as in LabelImg. ``pose``, ``truncated``
    and ``difficult`` are not tracked and are written as ``Unspecified`` and ``0``.
    """

    @property
    def file_suffix(self) -> str:
        """
        Extension of the label files.

        Returns
        -------
        str
            ``".xml"``.
        """
        return ".xml"

    def build_element(self, image: ImageAnnotation) -> ElementTree.Element:
        """
        Build the XML document of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to render.

        Returns
        -------
        ElementTree.Element
            Indented ``annotation`` root element.
        """
        root: ElementTree.Element = ElementTree.Element("annotation")
        self._add_text(root, "folder", image.image_path.parent.name)
        self._add_text(root, "filename", image.image_path.name)
        self._add_text(root, "path", str(image.image_path))
        source: ElementTree.Element = ElementTree.SubElement(root, "source")
        self._add_text(source, "database", "Unknown")
        size: ElementTree.Element = ElementTree.SubElement(root, "size")
        self._add_text(size, "width", str(image.image_size.width))
        self._add_text(size, "height", str(image.image_size.height))
        self._add_text(size, "depth", str(image.image_size.channel_count))
        self._add_text(root, "segmented", "0")
        for annotated_object in image.objects:
            self._add_object(root, annotated_object)
        ElementTree.indent(root)
        return root

    def write_label(self, image: ImageAnnotation, dataset: AnnotationDataset, label_path: Path) -> None:
        """
        Write the XML file of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to write.
        dataset : AnnotationDataset
            Dataset the image belongs to.
        label_path : Path
            Destination XML file.
        """
        ElementTree.ElementTree(self.build_element(image)).write(label_path, encoding="utf-8", xml_declaration=True)

    def _add_object(self, root: ElementTree.Element, annotated_object: ObjectAnnotation) -> None:
        element: ElementTree.Element = ElementTree.SubElement(root, "object")
        self._add_text(element, "name", annotated_object.class_name)
        self._add_text(element, "pose", "Unspecified")
        self._add_text(element, "truncated", "0")
        self._add_text(element, "difficult", "0")
        bounding_box: ElementTree.Element = ElementTree.SubElement(element, "bndbox")
        box: Box2D = annotated_object.box
        self._add_text(bounding_box, "xmin", str(round(box.x1)))
        self._add_text(bounding_box, "ymin", str(round(box.y1)))
        self._add_text(bounding_box, "xmax", str(round(box.x2)))
        self._add_text(bounding_box, "ymax", str(round(box.y2)))

    @staticmethod
    def _add_text(parent: ElementTree.Element, tag: str, text: str) -> None:
        ElementTree.SubElement(parent, tag).text = text
