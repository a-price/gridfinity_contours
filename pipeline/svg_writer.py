import numpy as np

from pipeline.contour_extraction import PCABox


def _FormatPoints(points: np.ndarray) -> str:
    return " ".join(f"{x:.4f},{y:.4f}" for x, y in points)


def WriteSvg(path: str, contours: dict[int, np.ndarray]) -> None:
    """Writes `contours` (real-world mm coordinates, e.g. Rectify.contours)
    to an SVG file: one closed <polygon> per contour, each rotated and
    shifted into its own PCA-aligned local frame (principal axis along x,
    origin at its bounding box's corner) - the same alignment already used
    for the text preview, so the shape comes out level instead of at
    whatever angle the object happened to sit at in the photo. 1 SVG user
    unit = 1mm, the scale tools like Fusion 360 expect when importing an
    SVG sketch.
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

    polygons = "\n".join(
        f'  <polygon points="{_FormatPoints(points)}" fill="none" stroke="black" stroke-width="0.1" />'
        for points in aligned.values()
    )

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.4f}mm" height="{height:.4f}mm" '
        f'viewBox="0 0 {width:.4f} {height:.4f}">\n'
        f"{polygons}\n"
        "</svg>\n"
    )

    with open(path, "w") as f:
        f.write(svg)
