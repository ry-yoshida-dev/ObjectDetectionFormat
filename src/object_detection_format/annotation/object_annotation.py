import math
from dataclasses import dataclass

import numpy as np
from geometry import Box2D


@dataclass(frozen=True, eq=False)
class ObjectAnnotation:
    """
    One labeled object in an image.

    Attributes
    ----------
    box : Box2D
        Object extent in absolute pixel coordinates, in any ``geometry`` box format.
    class_name : str
        Category of the object.
    confidence : float | None, optional
        Detector confidence in ``[0, 1]``; ``None`` for ground-truth labels.

    Raises
    ------
    ValueError
        If a box coordinate is not finite, the class name is blank or contains a line break,
        or the confidence is outside ``[0, 1]``.
    """

    box: Box2D
    class_name: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not bool(np.isfinite(self.box.value).all()):
            raise ValueError(f"box coordinates must be finite. got {self.box}")
        if not self.class_name.strip():
            raise ValueError("class_name must not be blank")
        if "\n" in self.class_name or "\r" in self.class_name:
            raise ValueError(f"class_name must not contain line breaks. got {self.class_name!r}")
        if self.confidence is not None and not (math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1]. got {self.confidence}")
