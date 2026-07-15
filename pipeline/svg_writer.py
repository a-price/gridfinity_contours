import numpy as np

from pipeline.contour_extraction import PCABox


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


def WriteSvg(path: str, contours: dict[int, np.ndarray]) -> None:
    """Writes `contours` (real-world mm coordinates, e.g. Rectify.contours)
    to an SVG file: one closed <polygon> per contour, PCA-aligned (see
    AlignContoursToPca). The `width`/`height` attributes are the true
    physical size in mm; the viewBox and path coordinates are scaled by
    _SVG_USER_UNITS_PER_MM to also import correctly in tools that ignore
    those attributes' units (see the comment above). Not every SVG
    viewer/print path even honors physical print size though - see
    WritePdf for a print-safe alternative.
    """
    aligned, width, height = AlignContoursToPca(contours)
    scale = _SVG_USER_UNITS_PER_MM

    polygons = "\n".join(
        f'  <polygon points="{_FormatPoints(points * scale)}" '
        f'fill="none" stroke="black" stroke-width="{0.1 * scale:.4f}" />'
        for points in aligned.values()
    )

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.4f}mm" height="{height:.4f}mm" '
        f'viewBox="0 0 {width * scale:.4f} {height * scale:.4f}">\n'
        f"{polygons}\n"
        "</svg>\n"
    )

    with open(path, "w") as f:
        f.write(svg)
