"""Tests for the drawer floorplan.

The floorplan is the one drawing nobody can check by eye against a bin, because
it is the *drawer* it has to match. So what is pinned here is placement
arithmetic: that a bin's drawing lands on the cells the assignment gave
it, that a turned bin's drawing turns with it, and that the page is the
size of the drawer rather than of whatever happened to be put in it.
"""

import re
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no window should pop up when saving the PDF

import numpy as np
import pytest

from pipeline.layout.container import GRID_PITCH_MM
from pipeline.layout.drawer import Assign, AssignmentResult, Drawer, Slot
from pipeline.layout.floorplan import FloorplanPages, DrawerPage, PlacedBinShapes, WriteFloorplanPdf
from pipeline.layout.loading import BuildParts
from pipeline.layout.packer import Pack
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.preview import OuterFootprint


def _rectangle(width: float, height: float) -> np.ndarray:
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)


def _quick(**overrides) -> LayoutParameters:
    settings: dict[str, Any] = dict(restarts=3, iterations=120, patience=12)
    settings.update(overrides)
    return LayoutParameters(**settings)


def _one_bin(params: LayoutParameters):
    """A single 30x30 part packed into whatever bin holds it."""
    parts = BuildParts({0: _rectangle(30, 30)}, params)
    layout = Pack(parts, params).layout
    assert layout is not None
    return parts, layout


def _extent(shapes) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack([np.asarray(shape.points, dtype=np.float64).reshape(-1, 2) for shape in shapes])
    return points.min(axis=0), points.max(axis=0)


# ---------------------------------------------------------------- the pages


def test_a_page_per_drawer_in_the_order_given():
    params = _quick()
    parts, layout = _one_bin(params)
    drawers = [Drawer(3, 3), Drawer(2, 2)]
    result = AssignmentResult("placed", {0: Slot(0, 1, (0, 0))})

    pages = FloorplanPages(drawers, {0: layout}, result, parts)

    assert len(pages) == 2
    assert (pages[0].width, pages[0].height) == OuterFootprint(3, 3)
    assert (pages[1].width, pages[1].height) == OuterFootprint(2, 2)


def test_the_page_is_the_drawer_not_its_contents():
    """A half-empty drawer still prints at drawer size, or the sheet would
    not reach the edges it is meant to be aligned against.
    """
    params = _quick()
    parts, layout = _one_bin(params)
    drawers = [Drawer(5, 4)]
    result = AssignmentResult("placed", {0: Slot(0, 0, (0, 0))})

    page = FloorplanPages(drawers, {0: layout}, result, parts)[0]

    assert (page.width, page.height) == OuterFootprint(5, 4)


def test_an_empty_drawer_still_gets_a_page():
    """Dropping it would silently renumber every later page against the
    drawer it describes - and "this one is empty" is worth saying.
    """
    params = _quick()
    parts, layout = _one_bin(params)
    drawers = [Drawer(2, 2), Drawer(3, 3)]
    result = AssignmentResult("placed", {0: Slot(0, 0, (0, 0))})

    pages = FloorplanPages(drawers, {0: layout}, result, parts)

    assert len(pages) == 2
    assert pages[1].shapes, "an empty drawer still has its outline and grid"


def test_the_grid_covers_the_whole_drawer():
    """The free cells are the question a drawer floorplan answers, so the grid
    cannot stop where the bins do.
    """
    params = _quick()
    parts, layout = _one_bin(params)
    drawers = [Drawer(4, 1)]
    result = AssignmentResult("placed", {0: Slot(0, 0, (0, 0))})

    page = DrawerPage(drawers[0], [(layout, result.slots[0])], parts)

    width, _ = OuterFootprint(4, 1)
    _, high = _extent(page.shapes)
    assert high[0] == pytest.approx(width, abs=0.01), "the drawing should reach the far edge"


# ----------------------------------------------------------- where bins land


def test_a_bin_lands_on_the_cells_it_was_assigned():
    params = _quick()
    parts, layout = _one_bin(params)
    n, m = layout.grid

    shapes = PlacedBinShapes(layout, Slot(0, 0, (2, 1)), parts)

    low, high = _extent(shapes)
    assert low[0] == pytest.approx(2 * GRID_PITCH_MM, abs=0.01)
    assert low[1] == pytest.approx(1 * GRID_PITCH_MM, abs=0.01)
    width, height = OuterFootprint(n, m)
    assert high[0] == pytest.approx(2 * GRID_PITCH_MM + width, abs=0.01)
    assert high[1] == pytest.approx(1 * GRID_PITCH_MM + height, abs=0.01)


def test_a_bin_at_the_origin_starts_at_the_origin():
    params = _quick()
    parts, layout = _one_bin(params)

    shapes = PlacedBinShapes(layout, Slot(0, 0, (0, 0)), parts)

    low, _ = _extent(shapes)
    assert low == pytest.approx([0.0, 0.0], abs=0.01)


def test_a_turned_bin_swaps_its_extent():
    """The quarter turn is the same operation a part uses inside a bin, so
    a 4x1 bin drawn turned occupies a 1x4 footprint with its corner still
    where the assignment put it.
    """
    params = _quick()
    parts = BuildParts({0: _rectangle(130, 25)}, params)
    layout = Pack(parts, params).layout
    assert layout is not None and layout.grid == (4, 1)
    width, height = OuterFootprint(4, 1)

    upright = _extent(PlacedBinShapes(layout, Slot(0, 0, (0, 0)), parts))
    turned = _extent(PlacedBinShapes(layout, Slot(0, 0, (0, 0), turned=True), parts))

    assert upright[1] == pytest.approx([width, height], abs=0.01)
    assert turned[1] == pytest.approx([height, width], abs=0.01)
    assert turned[0] == pytest.approx([0.0, 0.0], abs=0.01)


def test_turning_moves_the_objects_too():
    """A bin turned in a drawer takes its contents with it - drawing the
    rim rotated and the objects upright would be a map of nothing.
    """
    params = _quick()
    parts = BuildParts({0: _rectangle(130, 25)}, params)
    layout = Pack(parts, params).layout
    assert layout is not None

    objects = [s for s in PlacedBinShapes(layout, Slot(0, 0, (0, 0), turned=True), parts) if s.closed]

    assert len(objects) == 1
    low, high = _extent(objects)
    assert high[1] - low[1] > high[0] - low[0], "the long part should now run down the page"


def test_the_objects_are_the_only_closed_shapes():
    """The same invariant the per-bin preview keeps: annotations are
    polylines so a written file reads back as exactly the objects in it.
    """
    params = _quick()
    parts = BuildParts({index: _rectangle(30, 30) for index in range(2)}, params)
    layout = Pack(parts, params).layout
    assert layout is not None

    shapes = PlacedBinShapes(layout, Slot(0, 0, (0, 0)), parts)

    assert sum(1 for shape in shapes if shape.closed) == 2


# ------------------------------------------------------------------ refusals


def test_a_bin_without_a_layout_is_refused():
    params = _quick()
    parts, layout = _one_bin(params)
    result = AssignmentResult("placed", {0: Slot(0, 0, (0, 0)), 1: Slot(1, 0, (1, 0))})

    with pytest.raises(ValueError, match=r"bins \[1\]"):
        FloorplanPages([Drawer(3, 3)], {0: layout}, result, parts)


def test_a_bin_in_a_drawer_that_was_not_given_is_refused():
    params = _quick()
    parts, layout = _one_bin(params)
    result = AssignmentResult("placed", {0: Slot(0, 4, (0, 0))})

    with pytest.raises(ValueError, match="drawer 4"):
        FloorplanPages([Drawer(3, 3)], {0: layout}, result, parts)


# ------------------------------------------------------------------ the file


def test_the_map_writes_one_true_scale_page_per_drawer(tmp_path):
    """Checked through the PDF's own page boxes rather than by counting
    pages, since the size is the whole point - a drawer floorplan printed to fit
    a sheet of A4 is a picture, not a template.
    """
    params = _quick()
    parts, layout = _one_bin(params)
    drawers = [Drawer(3, 3), Drawer(2, 2)]
    result = Assign({0: layout.grid}, drawers)
    path = tmp_path / "map.pdf"

    WriteFloorplanPdf(str(path), drawers, {0: layout}, result, parts)

    assert path.read_bytes().startswith(b"%PDF")
    boxes = re.findall(rb"/MediaBox \[ 0 0 ([\d.]+) ([\d.]+) \]", path.read_bytes())
    assert len(boxes) == len(drawers)

    points_per_mm = 72.0 / 25.4
    for box, drawer in zip(boxes, drawers):
        width, height = OuterFootprint(drawer.width, drawer.height)
        assert float(box[0]) == pytest.approx(width * points_per_mm, abs=0.5)
        assert float(box[1]) == pytest.approx(height * points_per_mm, abs=0.5)
