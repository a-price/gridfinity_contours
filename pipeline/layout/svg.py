"""Reading millimeter contours out of the SVGs this project writes.

The scale is derived from the file rather than assumed, because the
project has written two conventions - see LoadSvgContours.
"""

from typing import Sequence
import xml.etree.ElementTree as ElementTree

import numpy as np

from pipeline.layout.energy import LayoutParameters
from pipeline.layout.part import BuildPart, Part


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


def BuildParts(contours: dict[int, np.ndarray], params: LayoutParameters | None = None) -> dict[int, Part]:
    """Rasterize a set of millimeter contours at the resolution and field
    extent the given parameters call for. Sizing the fields from the same
    parameters that set the clearances is what keeps a part's field wide
    enough to feel every clearance it is subject to.
    """
    params = params or LayoutParameters()
    return {part_id: BuildPart(contour, params.resolution, params.pad) for part_id, contour in sorted(contours.items())}


def LoadParts(paths: Sequence[str], params: LayoutParameters | None = None) -> dict[int, Part]:
    """Build a Part per polygon across a set of SVG files, keyed by the
    order encountered - the `dict[int, ...]` shape the rest of the pipeline
    passes contours around in.
    """
    contours: dict[int, np.ndarray] = {}
    for path in paths:
        for contour in LoadSvgContours(path):
            contours[len(contours)] = contour
    return BuildParts(contours, params)
