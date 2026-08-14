"""Getting parts in: contour files on one side, rasterized Parts on the
other.

Reads either format this project writes - a JSON dump from `contour_io`,
or an exported SVG - renumbers their ids into one set, and grows the
contours into the pockets the solver packs. `BuildParts` is the one place
a session's objects become pockets, so a caller reaching past it to
`BuildPart` is choosing to pack something other than what gets printed.

The SVG scale is derived from the file rather than assumed, because the
project has written two conventions - see `LoadSvgContours`.
"""

from typing import Sequence
import xml.etree.ElementTree as ElementTree

import numpy as np

from export.contour_io import LoadContours
from layout.parameters import LayoutParameters
from layout.part import BuildPart, Part


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


def ReadContours(paths: Sequence[str]) -> dict[int, np.ndarray]:
    """Every contour across the given files, renumbered from zero.

    Takes either format this project writes: a JSON contour dump or an
    exported SVG. Which one is decided by extension, since a dump and a
    drawing of the same contours are not interchangeable - the SVG is
    per-shape PCA-aligned and rounded for drawing.

    Ids are assigned by order encountered rather than carried over from the
    inputs, because two files dumped from two sessions both start at 0 and
    silently dropping half the contours to a key collision would look
    exactly like a packing that went well.
    """
    contours: dict[int, np.ndarray] = {}
    for path in paths:
        loaded = LoadContours(path) if path.lower().endswith(".json") else dict(enumerate(LoadSvgContours(path)))
        for _, points in sorted(loaded.items()):
            contours[len(contours)] = points
    if not contours:
        raise ValueError("no contours found in the given files")
    return contours


def BuildParts(contours: dict[int, np.ndarray], params: LayoutParameters | None = None) -> dict[int, Part]:
    """Grow a set of millimeter *object* contours into their pockets and
    rasterize those, at the offset, resolution and field extent the given
    parameters call for.

    Sizing the fields from the same parameters that set the clearances is
    what keeps a part's field wide enough to feel every clearance it is
    subject to. Taking the pocket offset from there too is what keeps the
    layout and the cut solid talking about the same shape - this is the
    one place a session's objects become pockets, so a caller that
    reaches past it to `BuildPart` is choosing to pack something other
    than what will be printed.
    """
    params = params or LayoutParameters()
    return {
        part_id: BuildPart(
            contour,
            params.pocket_offset,
            resolution=params.resolution,
            pad=params.pad,
            pocket_resolution=params.pocket_resolution,
            pocket_simplify=params.pocket_simplify,
        )
        for part_id, contour in sorted(contours.items())
    }


def LoadParts(paths: Sequence[str], params: LayoutParameters | None = None) -> dict[int, Part]:
    """Build a Part per contour across a set of files - the two steps most
    callers want together.

    Reads whatever `ReadContours` reads rather than SVGs alone: it used to
    walk the paths itself, which meant a second copy of the renumbering
    and a loader that silently could not open a contour dump.
    """
    return BuildParts(ReadContours(paths), params)
