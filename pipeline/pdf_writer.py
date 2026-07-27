"""The same geometry as svg_writer, on a page whose size is unambiguous.

Split the same two ways for the same reason: `WriteShapesPdf` draws
coordinates as given, `WritePdf` PCA-aligns contours first. See
svg_writer for why a layout must never take the aligning path.
"""

from typing import Sequence

import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure

from pipeline.svg_writer import AlignContoursToPca, Shape

MM_PER_INCH = 25.4
POINTS_PER_INCH = 72.0


def _ToPoints(mm: float) -> float:
    """Millimeters as typographic points, which is what matplotlib's
    linewidth and dash lengths are in.
    """
    return mm / MM_PER_INCH * POINTS_PER_INCH


def WriteShapesPdf(path: str, shapes: Sequence[Shape], width: float, height: float) -> None:
    """Write shapes already in page coordinates to a PDF page `width` x
    `height` millimeters, at 1:1 scale.

    A PDF's page size is unambiguous, unlike an SVG's - not every viewer or
    print path honors an SVG's embedded physical units, so this is the
    reliable thing to print at "actual size" instead.
    """
    if not shapes:
        raise ValueError("no shapes to write")
    if width <= 0 or height <= 0:
        raise ValueError(f"page must be positive, got {width}x{height}mm")

    # Figure() and an explicit PDF canvas rather than pyplot: pyplot picks
    # a global *interactive* backend on import, so a headless run of
    # layout_cli.py died trying to open a Qt display it had no business
    # needing. Nothing here ever shows a figure - it saves one - so going
    # around pyplot removes the display dependency instead of papering over
    # it with a matplotlib.use() every caller has to remember.
    fig = Figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH))
    FigureCanvasPdf(fig)
    ax = fig.add_axes((0, 0, 1, 1))  # fill the whole page, no margins
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # page coordinates: origin top-left, y down
    ax.axis("off")

    for shape in shapes:
        points = np.asarray(shape.points, dtype=np.float64).reshape(-1, 2)
        if shape.closed:
            points = np.vstack([points, points[:1]])
        style = (0, tuple(_ToPoints(length) for length in shape.dashes)) if shape.dashes else "solid"
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=shape.stroke,
            linewidth=_ToPoints(shape.stroke_width),
            linestyle=style,
        )

    # No close() to match: a bare Figure is not registered in pyplot's
    # global list, so there is nothing holding it alive to release.
    fig.savefig(path, format="pdf")


def WritePdf(path: str, contours: dict[int, np.ndarray]) -> None:
    """Writes `contours` to a PDF at the same PCA-aligned scale as
    WriteSvg (1 unit = 1mm), one outline per contour.
    """
    aligned, width, height = AlignContoursToPca(contours)
    shapes = [Shape(points) for points in aligned.values()]
    WriteShapesPdf(path, shapes, width, height)
