from enum import Enum

from .base import AnnotationWriter
from .formats import CocoWriter, CreateMlWriter, LabelMeWriter, PascalVocWriter, YoloWriter


class AnnotationFormat(Enum):
    """
    Supported export formats.
    """

    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"
    LABELME = "labelme"
    CREATEML = "createml"

    @property
    def display_name(self) -> str:
        """
        Human-readable format name.

        Returns
        -------
        str
            Name suitable for menus.
        """
        match self:
            case AnnotationFormat.COCO:
                return "MS COCO (JSON)"
            case AnnotationFormat.YOLO:
                return "YOLO (TXT)"
            case AnnotationFormat.PASCAL_VOC:
                return "Pascal VOC (XML)"
            case AnnotationFormat.LABELME:
                return "LabelMe (JSON)"
            case AnnotationFormat.CREATEML:
                return "Create ML (JSON)"

    def create_writer(self) -> AnnotationWriter:
        """
        Build a writer of this format with default options.

        Returns
        -------
        AnnotationWriter
            Writer producing this format.
        """
        match self:
            case AnnotationFormat.COCO:
                return CocoWriter()
            case AnnotationFormat.YOLO:
                return YoloWriter()
            case AnnotationFormat.PASCAL_VOC:
                return PascalVocWriter()
            case AnnotationFormat.LABELME:
                return LabelMeWriter()
            case AnnotationFormat.CREATEML:
                return CreateMlWriter()
