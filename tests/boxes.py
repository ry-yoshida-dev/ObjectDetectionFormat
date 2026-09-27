import numpy as np
from geometry import Box2D, Box2DFormat


def xyxy_box(x1: float, y1: float, x2: float, y2: float) -> Box2D:
    """
    Build a pixel box from its corners.

    Parameters
    ----------
    x1 : float
        Left edge.
    y1 : float
        Top edge.
    x2 : float
        Right edge.
    y2 : float
        Bottom edge.

    Returns
    -------
    Box2D
        Box in ``XYXY`` format.
    """
    return Box2D.register(value=np.array([x1, y1, x2, y2], dtype=np.float64), box2d_format=Box2DFormat.XYXY)
