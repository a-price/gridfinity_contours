"""Putting a Part somewhere in a bin.

Orientation is restricted to quarter turns (D1), which makes every
rotation here exact - an axis swap and a sign flip, with no interpolation
and no accumulated error.
"""

from dataclasses import dataclass

import numpy as np

from pipeline.layout.part import Part


def RotatePoints(points: np.ndarray, orientation: int, size: np.ndarray) -> np.ndarray:
    """Rotate local-frame points by `orientation` quarter turns
    counterclockwise, keeping the bounding box's minimum corner at the
    origin. `size` is the part's unrotated (width, height).

    Quarter turns are exact - no interpolation, no accumulated error - which
    is the whole reason D1 restricts orientation to these four.
    """
    x, y = points[:, 0], points[:, 1]
    width, height = float(size[0]), float(size[1])

    match orientation % 4:
        case 0:
            rotated = (x, y)
        case 1:
            rotated = (height - y, x)
        case 2:
            rotated = (width - x, height - y)
        case _:
            rotated = (y, width - x)
    return np.stack(rotated, axis=-1)


def RotateVectors(vectors: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate direction vectors by `orientation` quarter turns
    counterclockwise.

    RotatePoints also shifts the bounding box back to the origin, which is
    right for positions and wrong for directions - a gradient has no
    location to translate. Forces cross between frames constantly, so the
    two operations are kept separate rather than sharing an argument.
    """
    x, y = vectors[..., 0], vectors[..., 1]

    match orientation % 4:
        case 0:
            rotated = (x, y)
        case 1:
            rotated = (-y, x)
        case 2:
            rotated = (-x, -y)
        case _:
            rotated = (y, -x)
    return np.stack(rotated, axis=-1)


def RotatedSize(size: np.ndarray, orientation: int) -> np.ndarray:
    """A part's (width, height) after `orientation` quarter turns."""
    return size[::-1].copy() if orientation % 2 else np.asarray(size, dtype=np.float64).copy()


@dataclass(frozen=True)
class Placement:
    """One part positioned in a bin: `orientation` quarter turns
    counterclockwise about its local origin, then translated by `position`
    into bin-local millimeters (origin at the interior's minimum corner).
    """

    part_id: int
    position: np.ndarray  # (2,) bin-local mm
    orientation: int = 0

    def ToWorld(self, part: Part) -> np.ndarray:
        """The part's contour placed into bin coordinates."""
        return RotatePoints(part.contour, self.orientation, part.size) + self.position

    def SamplesToWorld(self, part: Part) -> np.ndarray:
        """The part's boundary sample points placed into bin coordinates."""
        return RotatePoints(part.samples, self.orientation, part.size) + self.position

    def ToLocal(self, part: Part, points: np.ndarray) -> np.ndarray:
        """Bin coordinates back into this part's own local frame, ready to
        query against its distance field. The inverse of ToWorld.
        """
        centered = np.asarray(points, dtype=np.float64).reshape(-1, 2) - self.position
        return RotatePoints(centered, -self.orientation, RotatedSize(part.size, self.orientation))
