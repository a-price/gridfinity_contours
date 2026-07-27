"""Geometry primitives for packing contours into a Gridfinity bin.

See docs/layout.md for the design and docs/layout_roadmap.md for the
build order. This module is milestone M1: the container envelope, the
per-part raster fields the solver will push around, and rigid placements
of parts into a bin. No solver yet.

Everything here is pure geometry in millimeters, with no Qt import, so it
stays unit-testable without a display.
"""

from dataclasses import dataclass
from typing import Sequence
import xml.etree.ElementTree as ElementTree

import cv2
import numpy as np

from pipeline.contour_extraction import PCABox

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

# Raster resolution for the signed distance fields, in mm per pixel. Fine
# enough that discretization error stays well under the millimeter-scale
# clearances of D5, coarse enough that a 200mm part is only ~800px across.
DEFAULT_RESOLUTION_MM = 0.25

# How far beyond a part's own bounding box its distance field extends.
# Queries outside this margin report DISTANT_MM instead, so it must exceed
# the largest clearance the solver will ask about (c_pair, 3.2mm by D5).
DEFAULT_PAD_MM = 5.0

# Reported for queries that fall outside a part's rasterized field. Any
# value comfortably past every clearance works; this is not an infinity so
# that arithmetic on it stays finite.
DISTANT_MM = 1.0e6


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
    if n < 1 or m < 1:
        raise ValueError(f"grid size must be at least 1x1, got {n}x{m}")

    width, height = InteriorSpan(n, inset), InteriorSpan(m, inset)
    if width <= 0 or height <= 0:
        raise ValueError(f"inset {inset}mm leaves no interior in a {n}x{m} bin")

    radius = min(max(0.0, OUTER_CORNER_RADIUS_MM - inset), width / 2, height / 2)
    if radius <= 0:
        return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])

    # Corner centers, counterclockwise from the bottom left, each paired
    # with the angle its arc starts at.
    corners = [
        ((radius, radius), np.pi),
        ((width - radius, radius), 1.5 * np.pi),
        ((width - radius, height - radius), 0.0),
        ((radius, height - radius), 0.5 * np.pi),
    ]

    points = []
    for (cx, cy), start in corners:
        angles = start + np.linspace(0.0, 0.5 * np.pi, segments_per_corner + 1)
        points.append(np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=-1))
    return np.concatenate(points)


def PolygonArea(points: np.ndarray) -> float:
    """Unsigned area of a closed polygon, by the shoelace formula. Winding
    order is not assumed - contours arrive in either.
    """
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def ResampleBoundary(points: np.ndarray, spacing: float) -> np.ndarray:
    """The polygon's vertices plus points interpolated along every edge at
    no more than `spacing` apart.

    Vertices alone are not enough to collide with: a simplified contour has
    edges tens of millimeters long, and a part could slide clean through a
    thin feature between two of them without any vertex ever registering a
    penetration.
    """
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")

    closed = np.vstack([points, points[:1]])
    segments = []
    for start, end in zip(closed[:-1], closed[1:]):
        length = float(np.linalg.norm(end - start))
        steps = max(1, int(np.ceil(length / spacing)))
        # Drop the endpoint; the next edge contributes it as its start.
        t = np.linspace(0.0, 1.0, steps, endpoint=False).reshape(-1, 1)
        segments.append(start + t * (end - start))
    return np.concatenate(segments)


# Below this normalized third moment a shape is treated as symmetric about
# the axis, and the 180-degree tiebreak in _AlignToLocalFrame falls through
# to the other axis. Symmetric shapes look identical either way up, so
# which branch wins genuinely does not matter for them.
_SKEW_TOLERANCE = 1e-6


def _AlignToLocalFrame(points: np.ndarray) -> np.ndarray:
    """PCA-align a contour into a canonical local frame: principal axis
    along x, origin at the bounding box's minimum corner.

    PCABox.ToLocal does most of the work, but its basis comes from
    cv2.PCACompute2, whose eigenvector signs are arbitrary. Both resulting
    ambiguities have to be pinned down, or the "same" object photographed
    twice yields two different Parts:

    * A sign flip on one axis alone is a *reflection*, which would silently
      mirror the part. D1 rejects mirroring - a flipped tool sits upside
      down in its pocket - so a left-handed basis is corrected by flipping
      y, restoring handedness without disturbing the bounding box.
    * Flipping both axes is a proper 180-degree rotation, so handedness
      cannot detect it. It is resolved by the shape's own skew: the
      contour is oriented so its third central moment about each axis is
      positive, which for an asymmetric object (a spoon's bowl against its
      handle) is a stable property of the object rather than of the photo.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    box = PCABox(points.astype(np.float32))
    local = box.ToLocal(points).astype(np.float64)
    extent = local.max(axis=0)

    # The 2D cross product, spelled out: np.cross dropped 2-vector support
    # in NumPy 2.
    if box.pc1[0] * box.pc2[1] - box.pc1[1] * box.pc2[0] < 0:
        local[:, 1] = extent[1] - local[:, 1]

    if _IsSkewNegative(local):
        local = extent - local  # 180 degrees, preserving handedness
    return local


def _IsSkewNegative(local: np.ndarray) -> bool:
    """Whether a contour leans toward -x (or -y, for a shape symmetric in
    x), by its normalized third central moments.
    """
    # cv2.moments reports signed moments, so winding order would otherwise
    # flip the answer.
    x, y = local[:, 0], local[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) < 0:
        local = local[::-1]

    moments = cv2.moments(local.astype(np.float32))
    if abs(moments["nu30"]) > _SKEW_TOLERANCE:
        return moments["nu30"] < 0
    if abs(moments["nu03"]) > _SKEW_TOLERANCE:
        return moments["nu03"] < 0
    return False


@dataclass(frozen=True)
class Part:
    """A contour in a canonical local frame, plus the raster fields the
    solver reads to measure and resolve overlap.

    `contour` is PCA-aligned with its bounding box's minimum corner at the
    origin, so two photos of the same object produce the same Part
    regardless of how it happened to sit in frame.

    `sdf` is negative inside the part and positive outside, in mm, sampled
    on a grid that extends `pad` beyond the contour's bounding box on every
    side. `gradient` is its spatial derivative, normalized, pointing away
    from the part's interior - the direction to push a colliding sample.
    """

    contour: np.ndarray  # (K, 2) local mm
    samples: np.ndarray  # (S, 2) local mm, boundary points to collide with
    sdf: np.ndarray  # (H, W) float32 mm, negative inside
    gradient: np.ndarray  # (H, W, 2) float32, unit vectors (x, y)
    origin: np.ndarray  # (2,) local mm coordinate of the raster's corner
    resolution: float  # mm per pixel
    area: float  # mm^2

    @property
    def size(self) -> np.ndarray:
        """The contour's bounding box extent (width, height) in mm."""
        return self.contour.max(axis=0) - self.contour.min(axis=0)

    def _PixelCoordinates(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map local-mm points to fractional pixel coordinates, and flag the
        ones the raster covers. Pixel (r, c) is centered at
        `origin + (c + 0.5, r + 0.5) * resolution`.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        pixels = (points - self.origin) / self.resolution - 0.5
        height, width = self.sdf.shape
        inside = (pixels[:, 0] >= 0) & (pixels[:, 0] <= width - 1) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= height - 1)
        return pixels, inside

    def _Bilinear(self, field: np.ndarray, points: np.ndarray, outside_value: float) -> np.ndarray:
        """Bilinearly interpolate `field` at local-mm `points`, substituting
        `outside_value` wherever a point falls off the raster.
        """
        pixels, inside = self._PixelCoordinates(points)
        trailing = field.shape[2:]
        result = np.full((len(pixels),) + trailing, outside_value, dtype=np.float64)
        if not inside.any():
            return result

        x, y = pixels[inside, 0], pixels[inside, 1]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, field.shape[1] - 1)
        y1 = np.minimum(y0 + 1, field.shape[0] - 1)
        fx, fy = (x - x0).reshape((-1,) + (1,) * len(trailing)), (y - y0).reshape((-1,) + (1,) * len(trailing))

        top = field[y0, x0] * (1 - fx) + field[y0, x1] * fx
        bottom = field[y1, x0] * (1 - fx) + field[y1, x1] * fx
        result[inside] = top * (1 - fy) + bottom * fy
        return result

    def SampleSdf(self, points: np.ndarray) -> np.ndarray:
        """Signed distance in mm at each local-mm point: negative inside the
        part, positive outside, DISTANT_MM beyond the rasterized field.
        """
        return self._Bilinear(self.sdf, points, DISTANT_MM)

    def SampleGradient(self, points: np.ndarray) -> np.ndarray:
        """Unit vectors pointing away from the part's interior at each
        local-mm point - the direction that separates a colliding sample.
        Zero beyond the rasterized field, where there is nothing to push
        away from.
        """
        return self._Bilinear(self.gradient, points, 0.0)


def _RasterizePolygon(pixels: np.ndarray, height: int, width: int) -> np.ndarray:
    """A boolean mask of which pixel centers fall inside a polygon given in
    pixel coordinates, where pixel (r, c)'s center is at (c, r).

    This is a crossing-number test rather than cv2.fillPoly, which rounds
    the polygon's coordinates to whole pixels and fills inclusively - so an
    edge landing on a half-pixel (the common case, since a bounding box
    corner maps to one exactly) gains an extra row or column on the high
    side but not the low side. That asymmetric half-pixel is invisible in a
    mask and shows up later as a distance field that disagrees with the
    geometry by more at one end of a part than the other.
    """
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x, y = grid_x.ravel(), grid_y.ravel()

    inside = np.zeros(x.shape, dtype=bool)
    starts = pixels
    ends = np.roll(pixels, -1, axis=0)
    for (x0, y0), (x1, y1) in zip(starts, ends):
        if y0 == y1:
            continue  # horizontal edges cross no horizontal ray
        straddles = (y0 > y) != (y1 > y)
        crossing_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= straddles & (x < crossing_x)

    return inside.reshape(height, width)


def BuildPart(
    contour: np.ndarray,
    resolution: float = DEFAULT_RESOLUTION_MM,
    pad: float = DEFAULT_PAD_MM,
) -> Part:
    """Rasterize a millimeter contour into a Part: PCA-align it, then build
    its signed distance field, that field's gradient, and its boundary
    sample points.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}")

    local = _AlignToLocalFrame(contour)
    if len(local) < 3:
        raise ValueError(f"a contour needs at least 3 points, got {len(local)}")

    extent = local.max(axis=0)
    origin = np.array([-pad, -pad])
    width = int(np.ceil((extent[0] + 2 * pad) / resolution))
    height = int(np.ceil((extent[1] + 2 * pad) / resolution))

    mask = _RasterizePolygon((local - origin) / resolution - 0.5, height, width)

    # distanceTransform measures to the nearest pixel of opposite state, so
    # a pixel one step inside the boundary reports 1.0 where the true
    # distance to the edge between them is 0.5. Subtracting that half pixel
    # from both sides puts the zero level set on the polygon boundary
    # instead of half a pixel outside it.
    filled = mask.astype(np.uint8) * 255
    inside = cv2.distanceTransform(filled, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(255 - filled, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sdf = (np.where(mask, -(inside - 0.5), outside - 0.5) * resolution).astype(np.float32)

    d_dy, d_dx = np.gradient(sdf.astype(np.float64), resolution)
    gradient = np.stack([d_dx, d_dy], axis=-1)
    norms = np.linalg.norm(gradient, axis=-1, keepdims=True)
    gradient = np.divide(gradient, norms, out=np.zeros_like(gradient), where=norms > 1e-9)

    return Part(
        contour=local,
        samples=ResampleBoundary(local, resolution),
        sdf=sdf,
        gradient=gradient.astype(np.float32),
        origin=origin,
        resolution=resolution,
        area=PolygonArea(local),
    )


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


@dataclass(frozen=True)
class Layout:
    """A solved arrangement: the grid size chosen and where every part
    landed inside it.
    """

    grid: tuple[int, int]
    placements: dict[int, Placement]
    inset: float = DEFAULT_INTERIOR_INSET_MM

    @property
    def cells(self) -> int:
        return self.grid[0] * self.grid[1]

    def Envelope(self) -> np.ndarray:
        """The interior envelope this layout was packed into."""
        return InteriorEnvelope(self.grid[0], self.grid[1], self.inset)


def _ParseLength(text: str | None, attribute: str) -> float:
    """An SVG width/height attribute as millimeters."""
    if text is None:
        raise ValueError(f"SVG is missing a {attribute} attribute")
    value = text.strip()
    if value.endswith("mm"):
        return float(value[:-2])
    if value.replace(".", "", 1).replace("-", "", 1).isdigit():
        # Unitless: SVG's own fallback is CSS pixels, but everything this
        # project writes is millimeters, so guessing would be worse than
        # refusing.
        raise ValueError(f"SVG {attribute} '{value}' has no unit; expected millimeters")
    raise ValueError(f"SVG {attribute} '{value}' is not in millimeters")


def LoadSvgContours(path: str) -> list[np.ndarray]:
    """Read the polygons out of an SVG written by this project, in mm.

    The scale is derived as `viewBox width / width in mm` rather than
    assumed. WriteSvg pre-scales its coordinates by 96/25.4 so that
    DPI-assuming importers get the right size (see svg_writer.py), but
    files written before that change are 1:1 with millimeters - and
    test_data/ holds some of each. Hardcoding either constant would import
    one of the two formats 3.78x wrong.
    """
    root = ElementTree.parse(path).getroot()

    width_mm = _ParseLength(root.get("width"), "width")
    view_box = root.get("viewBox")
    if view_box is None:
        raise ValueError("SVG is missing a viewBox attribute")
    view_box_width = float(view_box.split()[2])
    if width_mm <= 0 or view_box_width <= 0:
        raise ValueError(f"SVG has a non-positive size: width={width_mm}mm, viewBox width={view_box_width}")
    units_per_mm = view_box_width / width_mm

    contours = []
    for polygon in root.iter("{http://www.w3.org/2000/svg}polygon"):
        points = polygon.get("points")
        if not points:
            continue
        pairs = [[float(value) for value in pair.split(",")] for pair in points.split()]
        contours.append(np.array(pairs, dtype=np.float64) / units_per_mm)

    if not contours:
        raise ValueError(f"no <polygon> elements found in {path}")
    return contours


def LoadParts(
    paths: Sequence[str],
    resolution: float = DEFAULT_RESOLUTION_MM,
    pad: float = DEFAULT_PAD_MM,
) -> dict[int, Part]:
    """Build a Part per polygon across a set of SVG files, keyed by the
    order encountered - the `dict[int, ...]` shape the rest of the pipeline
    passes contours around in.
    """
    parts = {}
    for path in paths:
        for contour in LoadSvgContours(path):
            parts[len(parts)] = BuildPart(contour, resolution, pad)
    return parts
