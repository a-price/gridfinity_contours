"""Tests for the on-screen layout raster (M6).

The rendering is checked against the shapes it claims to draw rather than
against a reference image: the value of this module is that the screen and
the printed sheet cannot disagree, and a golden image would pin the pixels
while saying nothing about that.
"""

import numpy as np
import pytest

from pipeline.layout.loading import BuildParts
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.placement import Layout, Placement
from pipeline.layout.preview import LayoutShapes
from pipeline.layout.render import (
    DEFAULT_PIXELS_PER_MM,
    MARGIN_MM,
    DashRuns,
    RenderLayout,
    _ToBgr,
)


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


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
