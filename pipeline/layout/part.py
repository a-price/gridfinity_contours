"""A contour turned into something the solver can measure overlap with.

A Part is a contour in a canonical local frame plus the signed distance
field used to detect and resolve collisions. See docs/layout.md for why
distance fields rather than no-fit polygons or convex decomposition.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.contour_extraction import PCABox
from pipeline.layout.raster import SignedDistanceField

# Raster resolution for the signed distance fields, in mm per pixel. Fine
# enough that discretization error stays well under the millimeter-scale
# clearances of D5, coarse enough that a 200mm part is only ~800px across.
DEFAULT_RESOLUTION_MM = 0.25

# How far beyond a part's own bounding box its distance field extends,
# when a caller builds a Part without a LayoutParameters to size it from.
# Queries outside this margin report DISTANT_MM instead, so it must exceed
# the largest separation the solver asks about - which is
# LayoutParameters.c_pair_enforced, not c_pair, since the solver drives to
# the wider of the two. Anything packing through the packer gets
# `params.pad` instead and never sees this.
DEFAULT_PAD_MM = 5.0

# Reported for queries that fall outside a part's rasterized field. Any
# value comfortably past every clearance works; this is not an infinity so
# that arithmetic on it stays finite.
DISTANT_MM = 1.0e6


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
    side. Its spatial derivative is computed on demand from the same
    bilinear interpolant SampleSdf reads, rather than stored as a second
    raster - see SampleDerivative for why that consistency matters.
    """

    contour: np.ndarray  # (K, 2) local mm
    samples: np.ndarray  # (S, 2) local mm, boundary points to collide with
    sdf: np.ndarray  # (H, W) float32 mm, negative inside
    origin: np.ndarray  # (2,) local mm coordinate of the raster's corner
    resolution: float  # mm per pixel
    pad: float  # mm the field reaches beyond the contour's bounding box
    area: float  # mm^2

    @property
    def size(self) -> np.ndarray:
        """The contour's bounding box extent (width, height) in mm."""
        return self.contour.max(axis=0) - self.contour.min(axis=0)

    def DilatedArea(self, radius: float) -> float:
        """Area in mm^2 of everything within `radius` of the part, itself
        included - the footprint it claims once its clearance band is
        counted in.

        Measured by counting field samples rather than from the usual
        `area + perimeter*r + pi*r^2`, which assumes convexity. That
        formula double-counts wherever a concave shape's dilation folds
        into itself - a spoon's bowl, say - and the overcount would make
        the packer's area bound unsound, rejecting bins that genuinely fit.
        Counting the field gets self-overlap right for free.
        """
        if radius > self.pad:
            raise ValueError(f"dilating by {radius}mm exceeds this part's {self.pad}mm field")
        return float((self.sdf <= radius).sum()) * self.resolution**2

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

    def _Corners(self, points: np.ndarray) -> tuple:
        """The four surrounding samples and the fractional offsets into
        their cell, for every point the raster covers.

        Gathered from the float32 raster, then converted - not the other
        way round. Converting the whole field before gathering gives the
        same four values at the cost of a copy the width of the raster, up
        to several hundred KB, every call.
        """
        pixels, covered = self._PixelCoordinates(points)
        x, y = pixels[covered, 0], pixels[covered, 1]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, self.sdf.shape[1] - 1)
        y1 = np.minimum(y0 + 1, self.sdf.shape[0] - 1)

        field = self.sdf
        f00 = field[y0, x0].astype(np.float64)
        f10 = field[y0, x1].astype(np.float64)
        f01 = field[y1, x0].astype(np.float64)
        f11 = field[y1, x1].astype(np.float64)
        return covered, f00, f10, f01, f11, x - x0, y - y0

    def SampleSdf(self, points: np.ndarray) -> np.ndarray:
        """Signed distance in mm at each local-mm point: negative inside the
        part, positive outside, DISTANT_MM beyond the rasterized field.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        result = np.full(len(points), DISTANT_MM)
        covered, f00, f10, f01, f11, fx, fy = self._Corners(points)
        if not covered.any():
            return result

        top = f00 * (1 - fx) + f10 * fx
        bottom = f01 * (1 - fx) + f11 * fx
        result[covered] = top * (1 - fy) + bottom * fy
        return result

    def SampleDerivative(self, points: np.ndarray) -> np.ndarray:
        """The gradient of SampleSdf, in mm per mm - the exact derivative of
        the bilinear interpolant, not of the underlying continuous field.

        The distinction is what makes gradient descent sound. The solver's
        energy is a function of SampleSdf's output, so its true gradient is
        the derivative of *that* interpolant. Differencing the raster
        separately and normalizing gives a slightly different vector, and a
        force that is not quite the gradient of the energy it claims to
        minimize can push uphill near a crease and stall the solver. Taking
        both from the same interpolant keeps them consistent by
        construction, which the finite-difference test pins down.

        Magnitude is ~1 wherever the field is a well-resolved distance,
        falling off across creases.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        result = np.zeros((len(points), 2))
        covered, f00, f10, f01, f11, fx, fy = self._Corners(points)
        if not covered.any():
            return result

        result[covered, 0] = ((f10 - f00) * (1 - fy) + (f11 - f01) * fy) / self.resolution
        result[covered, 1] = ((f01 - f00) * (1 - fx) + (f11 - f10) * fx) / self.resolution
        return result


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

    # Stated rather than derived from `local.min(axis=0)`: _AlignToLocalFrame
    # already put the minimum corner at the origin, to within the rounding
    # error PCA's float32 basis leaves behind.
    extent = local.max(axis=0)
    origin = np.array([-pad, -pad])
    width = int(np.ceil((extent[0] + 2 * pad) / resolution))
    height = int(np.ceil((extent[1] + 2 * pad) / resolution))
    sdf = SignedDistanceField(local, origin, height, width, resolution)

    return Part(
        contour=local,
        samples=ResampleBoundary(local, resolution),
        sdf=sdf,
        origin=origin,
        resolution=resolution,
        pad=pad,
        area=PolygonArea(local),
    )


def CanonicalOrder(parts: dict[int, Part]) -> list[int]:
    """Part ids ordered by the parts themselves, not by their labels.

    Ids are handed out in the order contour files were listed, so anything
    that walks a part dict in id order inherits the command line. That is
    fine for reporting and quietly harmful in a search: the solver draws
    one orientation per part from a single seeded stream, so which part
    gets which draw - and therefore which arrangements the restarts
    explore - depended on the order the files happened to be named in.
    Measured on the three spoons, five of the six orderings found the
    10-cell bin and the sixth returned a 12-cell one.

    Largest first, which is also what the constructive initializer wants:
    the big parts are the constrained ones. Extent breaks ties before the
    id does, so the id is reached only by parts identical in area and
    bounding box - and those are interchangeable, so which is which cannot
    matter.
    """
    return sorted(
        parts,
        key=lambda part_id: (
            -parts[part_id].area,
            -float(parts[part_id].size[0]),
            -float(parts[part_id].size[1]),
            part_id,
        ),
    )
