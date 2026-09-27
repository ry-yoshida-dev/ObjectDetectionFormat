from dataclasses import dataclass

from geometry import Box2D


@dataclass(frozen=True)
class ImageSize:
    """
    Pixel size of an annotated image.

    Attributes
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    channel_count : int, optional
        Number of color channels, written as ``depth`` by Pascal VOC.

    Raises
    ------
    ValueError
        If width, height or channel count is not positive.
    """

    width: int
    height: int
    channel_count: int = 3

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width and height must be positive. got ({self.width}, {self.height})")
        if self.channel_count <= 0:
            raise ValueError(f"channel_count must be positive. got {self.channel_count}")

    def contains(self, box: Box2D) -> bool:
        """
        Whether a box lies within the image bounds.

        Parameters
        ----------
        box : Box2D
            Box in absolute pixel coordinates.

        Returns
        -------
        bool
            True if every edge is within ``[0, width]`` and ``[0, height]``.
        """
        return box.x1 >= 0 and box.y1 >= 0 and box.x2 <= self.width and box.y2 <= self.height
