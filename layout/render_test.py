"""Tests for the on-screen layout raster (M6).

The rendering is checked against the shapes it claims to draw rather than
against a reference image: the value of this module is that the screen and
the printed sheet cannot disagree, and a golden image would pin the pixels
while saying nothing about that.
"""

import numpy as np
import pytest

from layout.loading import BuildParts
from layout.parameters import LayoutParameters
from layout.placement import Layout, Placement
from layout.preview import LayoutShapes
from layout.render import (
    DEFAULT_PIXELS_PER_MM,
    MARGIN_MM,
    PAGE_COLOR,
    Bordered,
    DashRuns,
    InRows,
    RenderLayout,
    SideBySide,
    Stacked,
    _ToBgr,
)
from conftest import Rectangle as _rectangle


def _simple_layout(grid=(2, 1)):
    parts = BuildParts({0: _rectangle(20.0, 10.0)}, LayoutParameters())
    return Layout(grid=grid, placements={0: Placement(0, np.array([5.0, 5.0]))}), parts


# ------------------------------------------------------------------ colors


def test_named_and_hex_colors_convert_to_bgr():
    assert _ToBgr("black") == (0, 0, 0)
    assert _ToBgr("#808080") == (128, 128, 128)
    # Channel order matters: this is red, and BGR puts it last.
    assert _ToBgr("#ff0000") == (0, 0, 255)


def test_an_unknown_color_raises_rather_than_defaulting():
    """A color added to preview.py that silently rendered black here would
    make the screen disagree with the print - exactly what this module
    exists to prevent.
    """
    with pytest.raises(ValueError, match="unrecognized"):
        _ToBgr("rebeccapurple")


def test_every_color_the_preview_uses_can_be_rendered():
    layout, parts = _simple_layout()

    shapes, _, _ = LayoutShapes(layout, parts)

    for shape in shapes:
        _ToBgr(shape.stroke)  # must not raise


# ------------------------------------------------------------------ dashes


def test_dash_runs_split_a_line_into_alternating_pieces():
    line = np.array([[0.0, 0.0], [10.0, 0.0]])

    runs = DashRuns(line, (2.0, 2.0))

    # on 0-2, off 2-4, on 4-6, off 6-8, on 8-10
    assert len(runs) == 3
    assert runs[0][0][0] == pytest.approx(0.0)
    assert runs[0][-1][0] == pytest.approx(2.0)
    assert runs[1][0][0] == pytest.approx(4.0)


def test_dash_runs_cover_the_on_fraction_of_the_length():
    line = np.array([[0.0, 0.0], [30.0, 0.0]])

    runs = DashRuns(line, (1.0, 2.0))

    drawn = sum(float(np.linalg.norm(run[-1] - run[0])) for run in runs)
    assert drawn == pytest.approx(10.0, abs=1.0)  # one third of 30mm


def test_dash_runs_carry_the_pattern_across_a_corner():
    """The pattern is measured along the path, not restarted per segment -
    otherwise every corner of the bin outline would begin with a fresh
    full-length dash and the spacing would visibly stutter.
    """
    corner = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0]])

    # The corner sits 3mm along, inside the first 4mm "on" run, so that run
    # has to bend through it rather than stopping at the segment's end.
    runs = DashRuns(corner, (4.0, 1.0))

    assert len(runs[0]) == 3
    assert runs[0][1] == pytest.approx([3.0, 0.0])
    assert runs[0][2] == pytest.approx([3.0, 1.0])


def test_dash_runs_ignore_repeated_points():
    # A closed ring repeats its first point, and zero-length segments would
    # otherwise divide by zero.
    line = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])

    runs = DashRuns(line, (2.0, 2.0))

    assert len(runs) == 3


def test_a_degenerate_dash_pattern_is_refused():
    line = np.array([[0.0, 0.0], [10.0, 0.0]])

    with pytest.raises(ValueError):
        DashRuns(line, ())
    with pytest.raises(ValueError, match="positive"):
        DashRuns(line, (2.0, 0.0))


# ----------------------------------------------------------------- raster


def test_the_render_is_the_page_plus_its_margin():
    layout, parts = _simple_layout(grid=(2, 1))

    image = RenderLayout(layout, parts, pixels_per_mm=4.0)

    # 2x1 outer footprint is 83.5 x 41.5mm.
    height, width, channels = image.shape
    assert width == round((83.5 + 2 * MARGIN_MM) * 4.0)
    assert height == round((41.5 + 2 * MARGIN_MM) * 4.0)
    assert channels == 3


def test_the_render_scales_with_pixels_per_mm():
    layout, parts = _simple_layout()

    coarse = RenderLayout(layout, parts, pixels_per_mm=2.0)
    fine = RenderLayout(layout, parts, pixels_per_mm=8.0)

    assert fine.shape[1] == pytest.approx(coarse.shape[1] * 4, rel=0.01)


def test_the_render_draws_something_on_a_blank_page():
    layout, parts = _simple_layout()

    image = RenderLayout(layout, parts)

    assert (image == 255).any(), "expected blank page"
    assert (image < 128).any(), "expected drawn geometry"


def test_the_part_lands_where_the_layout_puts_it():
    """Position has to survive into the raster: a render that re-centered
    the part would look plausible and be wrong.

    Only the part is drawn in black - the bin is #808080 and its grid
    lighter still - so thresholding below that isolates the part from
    everything around it.
    """
    layout, parts = _simple_layout(grid=(2, 1))
    pixels_per_mm = 8.0

    image = RenderLayout(layout, parts, pixels_per_mm)

    black = np.argwhere(image.min(axis=2) < 100)
    # argwhere gives (row, column), i.e. (y, x).
    top_mm, left_mm = black.min(axis=0) / pixels_per_mm - MARGIN_MM
    expected = 5.0 + layout.inset  # placed 5mm into an interior inset from the rim

    assert left_mm == pytest.approx(expected, abs=0.5)
    assert top_mm == pytest.approx(expected, abs=0.5)


def test_a_wider_bin_renders_wider():
    parts = BuildParts({0: _rectangle(20.0, 10.0)}, LayoutParameters())
    placements = {0: Placement(0, np.array([5.0, 5.0]))}

    narrow = RenderLayout(Layout(grid=(2, 1), placements=placements), parts)
    wide = RenderLayout(Layout(grid=(5, 1), placements=placements), parts)

    assert wide.shape[1] > narrow.shape[1]
    assert wide.shape[0] == narrow.shape[0]


def test_a_non_positive_scale_is_refused():
    layout, parts = _simple_layout()

    with pytest.raises(ValueError, match="positive"):
        RenderLayout(layout, parts, pixels_per_mm=0.0)


def test_the_default_scale_keeps_a_big_bin_a_reasonable_size():
    # 5x2 is the spoons' answer; it should come out readable but not
    # enormous in the image view.
    layout, parts = _simple_layout(grid=(5, 2))

    image = RenderLayout(layout, parts, DEFAULT_PIXELS_PER_MM)

    assert 600 < image.shape[1] < 1200


def test_pages_compose_left_to_right():
    """Two drawers of a floorplan in one image, so a bin moving between
    them is visible where two separate images could not show it.
    """
    left = np.zeros((10, 20, 3), dtype=np.uint8)
    right = np.zeros((10, 30, 3), dtype=np.uint8)

    composed = SideBySide([left, right], gap=4)

    assert composed.shape == (10, 54, 3)
    assert (composed[:, 20:24] == PAGE_COLOR).all(), "the gap should be page, not ink"


def test_a_shorter_page_is_padded_rather_than_stretched():
    """A page's own contents must not move when a taller one joins it -
    the drawers are drawn at true relative scale.
    """
    short = np.zeros((6, 20, 3), dtype=np.uint8)
    tall = np.zeros((10, 20, 3), dtype=np.uint8)

    composed = SideBySide([short, tall])

    assert composed.shape == (10, 40, 3)
    assert (composed[:6, :20] == 0).all()
    assert (composed[6:, :20] == PAGE_COLOR).all()


def test_composing_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to compose"):
        SideBySide([])


def test_a_marked_page_keeps_its_drawing():
    """The mark is a border rather than a tint, so what it draws attention
    to stays legible underneath it.
    """
    image = np.full((20, 30, 3), 200, dtype=np.uint8)

    marked = Bordered(image, value=96, width=2)

    assert (marked[2:-2, 2:-2] == 200).all()
    assert (marked[:2, :] == 96).all() and (marked[-2:, :] == 96).all()
    assert (marked[:, :2] == 96).all() and (marked[:, -2:] == 96).all()


def test_marking_leaves_the_original_alone():
    image = np.full((20, 30, 3), 200, dtype=np.uint8)
    Bordered(image)

    assert (image == 200).all()


def test_a_mark_stays_neutral():
    """Frames have to stay gray for the GIF writer's exact ramp to apply."""
    marked = Bordered(np.full((20, 30, 3), 200, dtype=np.uint8))

    assert (marked[..., 0] == marked[..., 1]).all() and (marked[..., 1] == marked[..., 2]).all()


def test_a_border_too_big_for_its_page_is_refused():
    with pytest.raises(ValueError, match="does not fit"):
        Bordered(np.zeros((4, 30, 3), dtype=np.uint8), width=3)


def test_pages_wrap_into_rows():
    """A single row of many bins is unreadable once a browser scales it to
    fit, so they wrap.
    """
    pages = [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(5)]

    wrapped = InRows(pages, columns=3)

    # Two rows: three pages then two, so the canvas is the wider row.
    assert wrapped.shape == (20, 60, 3)
    assert (wrapped[10:, 40:] == PAGE_COLOR).all(), "the short row should be padded, not stretched"


def test_the_column_count_is_the_callers():
    """It must not follow the page count: bins vanish as the search merges
    them, and a derived column count would reflow the survivors into
    different rows every time one went.
    """
    pages = [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(3)]

    assert InRows(pages, columns=3).shape == (10, 60, 3)
    assert InRows(pages[:2], columns=3).shape == (10, 40, 3)


def test_pages_compose_top_to_bottom():
    top = np.zeros((6, 20, 3), dtype=np.uint8)
    bottom = np.zeros((10, 30, 3), dtype=np.uint8)

    composed = Stacked([top, bottom], gap=2)

    assert composed.shape == (18, 30, 3)
    assert (composed[6:8, :] == PAGE_COLOR).all()
    assert (composed[:6, 20:] == PAGE_COLOR).all()


def test_a_row_of_no_pages_is_refused():
    with pytest.raises(ValueError, match="a row holds at least one page"):
        InRows([np.zeros((4, 4, 3), dtype=np.uint8)], columns=0)
