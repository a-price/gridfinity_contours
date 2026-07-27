"""Tests for the true-scale layout preview (M5).

The thing worth pinning here is that the drawing is a faithful picture of
the layout at real size. A preview that is merely plausible is worse than
none: it gets printed, measured against a bin, and believed.
"""

import re

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless: no window should pop up when saving the PDF

from pipeline.layout.container import BASE_GAP_MM, GRID_PITCH_MM, InteriorSpan
from pipeline.layout.energy import LayoutParameters
from pipeline.layout.loading import BuildParts, LoadSvgContours
from pipeline.layout.placement import Layout, Placement
from pipeline.layout.preview import (
    LayoutShapes,
    OuterFootprint,
    WriteLayoutPdf,
    WriteLayoutSvg,
)


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


def _parts(**shapes) -> dict:
    contours = {index: points for index, points in enumerate(shapes.values())}
    return BuildParts(contours, LayoutParameters())


def _simple_layout(grid=(2, 1)):
    """One 20x10mm part sitting 5mm in from the interior's corner."""
    parts = _parts(block=_rectangle(20.0, 10.0))
    placements = {0: Placement(0, np.array([5.0, 5.0]))}
    return Layout(grid=grid, placements=placements), parts


# ------------------------------------------------------------ page framing


def test_the_page_is_the_bins_outer_footprint():
    # Printed and laid under a real bin, the outline should vanish beneath
    # the rim - so the page is the outer size, not the interior.
    layout, parts = _simple_layout(grid=(3, 2))

    _, width, height = LayoutShapes(layout, parts)

    assert width == pytest.approx(GRID_PITCH_MM * 3 - BASE_GAP_MM)
    assert height == pytest.approx(GRID_PITCH_MM * 2 - BASE_GAP_MM)


def test_the_outer_footprint_is_smaller_than_the_nominal_grid():
    # The half-millimeter of gap per side is what keeps neighbouring bins
    # from binding; a preview drawn at 42*n would print a bin that does not
    # exist.
    assert OuterFootprint(1, 1) == (GRID_PITCH_MM - BASE_GAP_MM, GRID_PITCH_MM - BASE_GAP_MM)
    assert OuterFootprint(4, 2)[0] == pytest.approx(167.5)


def test_a_part_is_drawn_inset_from_the_page_edge_by_the_wall_thickness():
    layout, parts = _simple_layout()
    inset = layout.inset

    shapes, _, _ = LayoutShapes(layout, parts)

    outline = shapes[-1].points
    # The part sits 5mm into the interior, and the interior starts `inset`
    # in from the rim the page is drawn to.
    assert outline.min(axis=0)[0] == pytest.approx(5.0 + inset, abs=1e-6)
    assert outline.min(axis=0)[1] == pytest.approx(5.0 + inset, abs=1e-6)


def test_every_drawn_part_lands_inside_the_page():
    layout, parts = _simple_layout(grid=(2, 2))

    shapes, width, height = LayoutShapes(layout, parts)

    for shape in shapes:
        assert shape.points.min() >= -1e-9
        assert shape.points[:, 0].max() <= width + 1e-9
        assert shape.points[:, 1].max() <= height + 1e-9


def test_the_interior_outline_is_the_containers_own_envelope():
    layout, parts = _simple_layout(grid=(2, 1))

    shapes, _, _ = LayoutShapes(layout, parts)

    interior = next(shape for shape in shapes if shape.dashes == (2.0, 1.0))
    span = interior.points.max(axis=0) - interior.points.min(axis=0)
    assert span[0] == pytest.approx(InteriorSpan(2, layout.inset), abs=1e-6)
    assert span[1] == pytest.approx(InteriorSpan(1, layout.inset), abs=1e-6)


# --------------------------------------------------------------- cell grid


def test_cell_boundaries_fall_on_the_grid_pitch():
    # A boundary is a whole pitch into the *grid*, and the footprint starts
    # half a gap into the first cell. Miss that offset and the grid drifts
    # a quarter millimeter per cell against the bin it is drawn on.
    layout, parts = _simple_layout(grid=(3, 1))

    shapes, _, _ = LayoutShapes(layout, parts)

    verticals = sorted(
        shape.points[0][0] for shape in shapes if not shape.closed and shape.points[0][0] == shape.points[1][0]
    )
    assert verticals == pytest.approx([42.0 - 0.25, 84.0 - 0.25])


def test_a_one_by_one_bin_has_no_cell_boundaries():
    layout, parts = _simple_layout(grid=(1, 1))

    shapes, _, _ = LayoutShapes(layout, parts)

    assert not [shape for shape in shapes if shape.dashes == (1.0, 1.0)]


def test_an_n_by_m_bin_draws_the_boundaries_between_its_cells():
    layout, parts = _simple_layout(grid=(3, 2))

    shapes, _, _ = LayoutShapes(layout, parts)

    grid_lines = [shape for shape in shapes if shape.dashes == (1.0, 1.0)]
    assert len(grid_lines) == 3  # two vertical, one horizontal


# ------------------------------------------------------- part vs annotation


def test_only_parts_are_drawn_closed():
    """Annotation is drawn open so that a written preview reads back as
    exactly the parts in it - LoadSvgContours sees only <polygon>.
    """
    layout, parts = _simple_layout(grid=(2, 1))

    shapes, _, _ = LayoutShapes(layout, parts)

    assert sum(1 for shape in shapes if shape.closed) == len(layout.placements)


def test_a_written_preview_reads_back_as_only_its_parts(tmp_path):
    layout, parts = _simple_layout(grid=(2, 1))
    path = str(tmp_path / "preview.svg")

    WriteLayoutSvg(path, layout, parts)
    reloaded = LoadSvgContours(path)

    assert len(reloaded) == 1
    span = reloaded[0].max(axis=0) - reloaded[0].min(axis=0)
    assert span == pytest.approx([20.0, 10.0], abs=1e-3)


def test_a_reloaded_part_keeps_its_position_in_millimeters(tmp_path):
    # Round-tripping through the file must not lose where the part sat: a
    # preview that reads back re-centered would be a different layout.
    layout, parts = _simple_layout(grid=(2, 1))
    path = str(tmp_path / "preview.svg")

    WriteLayoutSvg(path, layout, parts)

    corner = LoadSvgContours(path)[0].min(axis=0)
    assert corner == pytest.approx([5.0 + layout.inset, 5.0 + layout.inset], abs=1e-3)


# ------------------------------------------------------------------ output


def test_the_svg_declares_the_bins_true_size_in_millimeters(tmp_path):
    layout, parts = _simple_layout(grid=(2, 1))
    path = tmp_path / "preview.svg"

    WriteLayoutSvg(str(path), layout, parts)

    svg = path.read_text()
    assert f'width="{GRID_PITCH_MM * 2 - BASE_GAP_MM:.4f}mm"' in svg
    assert f'height="{GRID_PITCH_MM - BASE_GAP_MM:.4f}mm"' in svg


def test_the_pdf_page_measures_the_bin(tmp_path):
    # This is the file that gets printed and held against a bin, so its
    # page size is the whole point.
    layout, parts = _simple_layout(grid=(2, 1))
    path = tmp_path / "preview.pdf"

    WriteLayoutPdf(str(path), layout, parts)

    match = re.search(rb"/MediaBox \[ 0 0 ([\d.]+) ([\d.]+) \]", path.read_bytes())
    assert match is not None, "no MediaBox found in the written PDF"
    assert float(match.group(1)) == pytest.approx(83.5 / 25.4 * 72, abs=0.01)
    assert float(match.group(2)) == pytest.approx(41.5 / 25.4 * 72, abs=0.01)


def test_drawing_a_layout_whose_parts_are_missing_raises():
    layout, parts = _simple_layout()
    layout = Layout(grid=layout.grid, placements={**layout.placements, 9: Placement(9, np.zeros(2))})

    with pytest.raises(ValueError, match=r"\[9\]"):
        LayoutShapes(layout, parts)


# -------------------------------------------------------------- handedness


def test_an_asymmetric_part_is_not_mirrored_by_being_drawn():
    """The one error a printed template cannot survive.

    A mirrored outline measures correctly and still will not fit, because
    a reflected tool sits upside down in its pocket (D1). An L drawn with
    its long arm the other way would pass every dimension check above.
    """
    corner = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 8.0], [8.0, 8.0], [8.0, 20.0], [0.0, 20.0]])
    parts = _parts(ell=corner)
    layout = Layout(grid=(2, 1), placements={0: Placement(0, np.array([5.0, 5.0]))})

    shapes, _, _ = LayoutShapes(layout, parts)

    # The part's own local contour, translated - nothing else. Any flip or
    # transpose would break this equality while preserving the bounding box.
    expected = layout.placements[0].ToWorld(parts[0]) + layout.inset
    np.testing.assert_allclose(shapes[-1].points, expected)
