"""The Gridfinity bin a layout has to fit inside.

Spec constants from the vendored gridfinity-rebuilt-openscad submodule,
and the usable interior they imply.

The interior is evaluated analytically rather than rasterized like a Part:
a rounded rectangle has a closed-form distance function, so it costs no
memory, has no resolution to tune, and stays meaningful arbitrarily far
outside the bin - which matters because a part that escapes during
relaxation has to be pulled back from wherever it went, and a raster would
simply end.
"""

from dataclasses import dataclass

import numpy as np

# Gridfinity spec constants, from the vendored
# gridfinity-rebuilt-openscad/src/core/standard.scad. Named here rather
# than re-derived at each call site.
GRID_PITCH_MM = 42.0  # GRID_DIMENSIONS_MM
BASE_GAP_MM = 0.5  # GRID_DIMENSIONS_MM - BASE_TOP_DIMENSIONS
OUTER_CORNER_RADIUS_MM = 3.75  # BASE_TOP_RADIUS
STACKING_LIP_INTRUSION_MM = 2.6  # STACKING_LIP_SIZE.x, wall thickness included
MIN_WALL_MM = 0.95  # d_wall
DIVIDER_WIDTH_MM = 1.2  # d_div

# How far each interior wall sits inside the bin's outer face. The stacking
# lip is the binding constraint on a standard bin; a lipless bin gives back
# STACKING_LIP_INTRUSION_MM - MIN_WALL_MM per side, which matters at 1x1,
# hence the parameter rather than a hardcoded constant.
DEFAULT_INTERIOR_INSET_MM = STACKING_LIP_INTRUSION_MM


def InteriorSpan(cells: int, inset: float = DEFAULT_INTERIOR_INSET_MM) -> float:
    """Usable interior length of a run of `cells` grid units, in mm.

    The bin's outer footprint is `42*cells - 0.5` (the 0.5mm gap keeps
    adjacent bins from binding), and each wall eats `inset` from that.
    """
    return GRID_PITCH_MM * cells - BASE_GAP_MM - 2.0 * inset


def InteriorEnvelope(
    n: int,
    m: int,
    inset: float = DEFAULT_INTERIOR_INSET_MM,
    segments_per_corner: int = 8,
) -> np.ndarray:
    """The usable interior of an `n x m` bin as a closed polygon, origin at
    its minimum corner.

    A rounded rectangle: the bin's outer corners carry BASE_TOP_RADIUS, and
    the wall inset shrinks that to `max(0, radius - inset)` - essentially
    square on a standard bin, but the term is kept so a lipless or
    thin-walled bin rounds correctly.
    """
    return BuildContainer(n, m, inset).Polygon(segments_per_corner)


@dataclass(frozen=True)
class Container:
    """The bin interior a layout has to stay inside: a rounded rectangle
    spanning [0, width] x [0, height] with corner radius `radius`.

    Evaluated analytically rather than rasterized like a Part. A rounded
    rectangle has a closed-form distance function, so this is exact, needs
    no memory, has no resolution to tune, and - unlike a raster - stays
    meaningful arbitrarily far outside the bin, which matters because a
    part that escapes during relaxation has to be pulled back from wherever
    it went.
    """

    width: float
    height: float
    radius: float

    @property
    def area(self) -> float:
        """Interior area in mm^2. The rounded corners give back
        `(4 - pi) * r^2` from the enclosing rectangle.
        """
        return self.width * self.height - (4.0 - np.pi) * self.radius**2

    def Polygon(self, segments_per_corner: int = 8) -> np.ndarray:
        """The interior boundary as a closed polygon, for drawing and for
        the independent checks in verify.py.
        """
        radius = self.radius
        if radius <= 0:
            return np.array([[0.0, 0.0], [self.width, 0.0], [self.width, self.height], [0.0, self.height]])

        corners = [
            ((radius, radius), np.pi),
            ((self.width - radius, radius), 1.5 * np.pi),
            ((self.width - radius, self.height - radius), 0.0),
            ((radius, self.height - radius), 0.5 * np.pi),
        ]
        points = []
        for (cx, cy), start in corners:
            angles = start + np.linspace(0.0, 0.5 * np.pi, segments_per_corner + 1)
            points.append(np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=-1))
        return np.concatenate(points)

    def _Offsets(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        half = np.array([self.width, self.height]) / 2.0
        offset = points - half
        # Distance past the straight section of each side, measured in the
        # positive quadrant so the two axes can be handled symmetrically.
        return offset, np.where(offset >= 0, 1.0, -1.0), np.abs(offset) - (half - self.radius)

    def SampleDepth(self, points: np.ndarray) -> np.ndarray:
        """How far inside the interior each point sits, in mm. Negative
        outside, so a part that has escaped reports how far it has to come
        back.
        """
        _, _, q = self._Offsets(points)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        within = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
        return self.radius - (outside + within)

    def SampleDerivative(self, points: np.ndarray) -> np.ndarray:
        """The gradient of SampleDepth: unit vectors pointing inward, toward
        deeper interior.
        """
        _, signs, q = self._Offsets(points)
        positive = np.maximum(q, 0.0)
        magnitude = np.linalg.norm(positive, axis=1, keepdims=True)

        # Beyond a corner both axes contribute; along a side only the
        # nearer wall does.
        rounded = np.divide(positive, magnitude, out=np.zeros_like(positive), where=magnitude > 1e-12)
        straight = np.zeros_like(positive)
        straight[np.arange(len(q)), np.argmax(q, axis=1)] = 1.0

        outward = np.where(magnitude > 0, rounded, straight) * signs
        return -outward


def BuildContainer(n: int, m: int, inset: float = DEFAULT_INTERIOR_INSET_MM) -> Container:
    """The usable interior of an `n x m` Gridfinity bin."""
    if n < 1 or m < 1:
        raise ValueError(f"grid size must be at least 1x1, got {n}x{m}")

    width, height = InteriorSpan(n, inset), InteriorSpan(m, inset)
    if width <= 0 or height <= 0:
        raise ValueError(f"inset {inset}mm leaves no interior in a {n}x{m} bin")

    radius = min(max(0.0, OUTER_CORNER_RADIUS_MM - inset), width / 2, height / 2)
    return Container(width=width, height=height, radius=radius)
