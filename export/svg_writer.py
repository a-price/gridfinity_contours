"""Writing millimeter geometry out as SVG.

Two layers, deliberately separate. `WriteShapesSvg` writes coordinates
that are already in the frame they should be drawn in; `WriteSvg` is the
contour export on top of it, PCA-aligning each contour into its own frame
first.

The split is what lets a *layout* be drawn at all. Aligning each contour
into its own local frame is right for a cut sheet of unrelated outlines
and catastrophic for an arrangement, whose entire content is where the
parts sit relative to each other - align them individually and they all
stack back onto the origin.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geometry.pca_box import PCABox


@dataclass(frozen=True)
class Shape:
    """One outline to draw, in millimeters, already in page coordinates
    (origin top-left, y increasing downward - what both SVG and PDF use).

    `closed` is not only cosmetic. Closed shapes are written as `<polygon>`
    and open ones as `<polyline>`, and layout.loading.LoadSvgContours reads
    only `<polygon>`. Drawing annotations - a bin outline, a cell boundary
    - open therefore means a written file reads back as exactly the object
    contours in it, with nothing to filter out afterwards.
    """

    points: np.ndarray  # (N, 2) mm
    closed: bool = True
    stroke: str = "black"
    stroke_width: float = 0.1  # mm
    dashes: tuple[float, ...] = ()  # mm on/off run lengths; empty for solid


def AlignContoursToPca(contours: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], float, float]:
    """PCA-aligns each contour into its own local frame (principal axis
    along x, origin at its bounding box's corner) - the same alignment
    used for the text preview, so a shape comes out level instead of at
    whatever angle the object happened to sit at in the photo. Returns the
    aligned contours plus the width/height needed to fit the largest one,
    in the same real-world units as the input (e.g. mm).
    """
    if not contours:
        raise ValueError("no contours to export")

    aligned = {}
    width = height = 0.0
    for obj_id, points in contours.items():
        points = np.asarray(points).reshape(-1, 2).astype(np.float32)
        box = PCABox(points)
        aligned[obj_id] = box.ToLocal(points)
        width = max(width, box.max1 - box.min1)
        height = max(height, box.max2 - box.min2)
    return aligned, width, height


def _FormatPoints(points: np.ndarray) -> str:
    return " ".join(f"{x:.4f},{y:.4f}" for x, y in points)


# SVG's own fallback definition of "1 user unit" absent other info is 1 CSS
# pixel = 1/96 inch. Several real-world SVG importers (Fusion 360 among
# them) apply that conversion to the viewBox/path coordinates unconditionally,
# ignoring the width/height attributes' physical-unit suffix entirely - so a
# viewBox scaled 1:1 with mm imports ~3.78x too small there. Pre-scaling the
# viewBox and path coordinates by this factor makes both kinds of consumer
# agree: spec-compliant viewers still derive the correct real-world size from
# width/height (which stay in true, unscaled mm), while DPI-assuming
# importers now get the right size too, since 1 user unit genuinely is
# 1/96in by construction.
_SVG_USER_UNITS_PER_MM = 96.0 / 25.4


def _FormatShape(shape: Shape, scale: float) -> str:
    """One shape as an SVG element, with every length pre-scaled.

    Stroke width and dash lengths are scaled along with the coordinates.
    They are specified in mm like everything else here, and a user unit is
    not a millimeter, so leaving either unscaled would draw a hairline on
    a preview whose geometry is 3.78x larger.
    """
    element = "polygon" if shape.closed else "polyline"
    dasharray = ""
    if shape.dashes:
        pattern = ",".join(f"{length * scale:.4f}" for length in shape.dashes)
        dasharray = f' stroke-dasharray="{pattern}"'
    return (
        f'  <{element} points="{_FormatPoints(np.asarray(shape.points) * scale)}" '
        f'fill="none" stroke="{shape.stroke}" '
        f'stroke-width="{shape.stroke_width * scale:.4f}"{dasharray} />'
    )


def WriteShapesSvg(path: str, shapes: Sequence[Shape], width: float, height: float) -> None:
    """Write shapes already in page coordinates to an SVG `width` x `height`
    millimeters.

    The `width`/`height` attributes are the true physical size in mm; the
    viewBox and all coordinates are scaled by _SVG_USER_UNITS_PER_MM so the
    file also imports correctly in tools that ignore those attributes'
    units (see the comment above). Not every SVG viewer/print path even
    honors physical print size though - see pdf_writer for a print-safe
    alternative.
    """
    if not shapes:
        raise ValueError("no shapes to write")
    if width <= 0 or height <= 0:
        raise ValueError(f"canvas must be positive, got {width}x{height}mm")

    scale = _SVG_USER_UNITS_PER_MM
    body = "\n".join(_FormatShape(shape, scale) for shape in shapes)

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.4f}mm" height="{height:.4f}mm" '
        f'viewBox="0 0 {width * scale:.4f} {height * scale:.4f}">\n'
        f"{body}\n"
        "</svg>\n"
    )

    with open(path, "w") as f:
        f.write(svg)


def WriteSvg(path: str, contours: dict[int, np.ndarray]) -> None:
    """Writes `contours` (real-world mm coordinates, e.g. Rectify.contours)
    to an SVG file: one closed `<polygon>` per contour, PCA-aligned (see
    AlignContoursToPca), on a canvas sized to the largest of them.
    """
    aligned, width, height = AlignContoursToPca(contours)
    shapes = [Shape(points) for points in aligned.values()]
    WriteShapesSvg(path, shapes, width, height)
