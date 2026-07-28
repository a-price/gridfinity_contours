"""Rasterizing a layout for the screen.

The third output device for one drawing, after SVG and PDF. It renders
`preview.LayoutShapes` rather than walking the layout itself, so what the
GUI shows and what comes out of the printer cannot drift apart - a preview
that disagreed with the sheet would be worse than no preview, since the
sheet is what gets checked against a bin.

Drawn dark-on-white like the printed page rather than restyled for the
dark image view. The point of looking at it on screen is to judge what
will be printed.
"""

from typing import Sequence

import cv2
import numpy as np

from pipeline.layout.part import Part
from pipeline.layout.placement import Layout
from pipeline.layout.preview import LayoutShapes
from pipeline.svg_writer import Shape

# Screen pixels per millimeter. A 5x2 bin is 209.5mm across, so this puts
# it just under 850px - readable in the image view without the render
# costing anything noticeable.
DEFAULT_PIXELS_PER_MM = 4.0

# Blank margin around the page, so the rim outline is not clipped by the
# edge of the image at whatever stroke width it lands on.
MARGIN_MM = 2.0

PAGE_COLOR = (255, 255, 255)


def _ToBgr(color: str) -> tuple[int, int, int]:
    """A `Shape` stroke color as OpenCV BGR.

    Unknown names raise rather than defaulting to black: a color added to
    preview.py that silently rendered as black here would make the screen
    disagree with the print, which is the one thing this module exists to
    prevent.
    """
    if color == "black":
        return (0, 0, 0)
    if len(color) == 7 and color.startswith("#"):
        red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
        return (blue, green, red)
    raise ValueError(f"unrecognized stroke color '{color}'")


def DashRuns(points: np.ndarray, pattern: Sequence[float]) -> list[np.ndarray]:
    """Split a polyline into the drawn runs of an on/off dash pattern,
    measured in the same units as the points.

    OpenCV has no dashed polyline, and drawing the dashed shapes solid
    would erase the distinction between the bin's rim and its interior
    wall - two lines a couple of millimeters apart that mean quite
    different things.
    """
    if not pattern:
        raise ValueError("a dash pattern needs at least one run length")
    if any(length <= 0 for length in pattern):
        raise ValueError(f"dash run lengths must be positive, got {tuple(pattern)}")

    runs: list[np.ndarray] = []
    current = [points[0]]
    index, remaining, drawing = 0, pattern[0], True

    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end - start))
        if length <= 0:
            continue

        travelled = 0.0
        while length - travelled > remaining:
            travelled += remaining
            split = start + (end - start) * (travelled / length)
            if drawing:
                current.append(split)
                runs.append(np.array(current))
                current = []
            else:
                current = [split]
            index = (index + 1) % len(pattern)
            remaining, drawing = pattern[index], not drawing

        remaining -= length - travelled
        if drawing:
            current.append(end)

    if drawing and len(current) > 1:
        runs.append(np.array(current))
    return runs


def _DrawShape(image: np.ndarray, shape: Shape, pixels_per_mm: float, offset: float) -> None:
    """One shape onto the canvas, in page millimeters."""
    points = np.asarray(shape.points, dtype=np.float64).reshape(-1, 2)
    if shape.closed:
        points = np.vstack([points, points[:1]])

    color = _ToBgr(shape.stroke)
    thickness = max(1, int(round(shape.stroke_width * pixels_per_mm)))
    runs = DashRuns(points, shape.dashes) if shape.dashes else [points]

    for run in runs:
        pixels = np.round((run + offset) * pixels_per_mm).astype(np.int32)
        cv2.polylines(image, [pixels], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def RenderShapes(
    shapes: Sequence[Shape],
    width: float,
    height: float,
    pixels_per_mm: float = DEFAULT_PIXELS_PER_MM,
) -> np.ndarray:
    """A page of shapes as a BGR image, at `pixels_per_mm`.

    Takes a page rather than a layout so that anything with a `Page`'s
    shape can be rasterized - a bin sheet, or a whole drawer floorplan -
    without a second copy of the drawing loop.
    """
    if pixels_per_mm <= 0:
        raise ValueError(f"pixels_per_mm must be positive, got {pixels_per_mm}")

    size = (
        int(round((height + 2 * MARGIN_MM) * pixels_per_mm)),
        int(round((width + 2 * MARGIN_MM) * pixels_per_mm)),
        3,
    )
    image = np.full(size, PAGE_COLOR, dtype=np.uint8)

    for shape in shapes:
        _DrawShape(image, shape, pixels_per_mm, MARGIN_MM)
    return image


def RenderLayout(
    layout: Layout,
    parts: dict[int, Part],
    pixels_per_mm: float = DEFAULT_PIXELS_PER_MM,
) -> np.ndarray:
    """A solved layout as a BGR image, at `pixels_per_mm`.

    Same shapes as the printed sheet, at a scale chosen for a screen
    instead of a page.
    """
    return RenderShapes(*LayoutShapes(layout, parts), pixels_per_mm)
