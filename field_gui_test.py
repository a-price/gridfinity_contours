"""Tests for the standalone distance-field window.

Two things worth pinning. Loading is additive, like `layout_gui` and for
the same reason - contours arrive a capture session at a time. And the
selector survives a reload: every parameter edit re-runs the stage, so a
window that reset the selection on each one would snap back to the first
contour every time the resolution was nudged.
"""

import numpy as np
import pytest

from pipeline.contour_io import SaveContours
from field_gui import FieldGui
from conftest import Rectangle as _rectangle, SPOONS


@pytest.fixture
def gui(qapp):
    return FieldGui()


def _dump(tmp_path, name, contours) -> str:
    path = str(tmp_path / name)
    SaveContours(path, contours)
    return path


# ------------------------------------------------------------- accumulating


def test_loading_two_files_keeps_both(gui, tmp_path):
    first = _dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})
    second = _dump(tmp_path, "b.json", {0: _rectangle(18.0, 12.0)})

    gui.load_contours([first])
    gui.load_contours([second])

    assert len(gui.contours) == 2
    assert gui.contour_box.count() == 2


def test_svgs_and_dumps_can_be_mixed(gui, tmp_path):
    dump = _dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})

    gui.load_contours([dump, "test_data/small_spoon.svg"])

    assert len(gui.contours) == 2


def test_clearing_removes_everything(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])

    gui.clear_contours()

    assert gui.contours == {}
    assert gui.contour_box.count() == 0
    assert gui.field_stage.part is None


def test_the_panel_lists_what_is_loaded(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "spoons.json", {0: _rectangle(20.0, 10.0)})])

    text = gui.source_label.text()
    assert "1 contours" in text
    assert "spoons.json" in text


def test_an_unreadable_file_is_reported_not_raised(gui, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")

    gui.load_contours([str(bad)])

    assert "Could not load" in gui.source_label.text()
    assert gui.contours == {}


# --------------------------------------------------------------- selecting


def test_loading_shows_a_field_without_being_asked(gui, tmp_path):
    """Unlike packing, drawing a field is instant - so there is no reason
    to make somebody press a button to see the thing they just loaded.
    """
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])

    pixmap = gui.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()


def test_the_view_prompts_before_anything_is_loaded(gui):
    assert "Load contours" in gui.image_label.text()


def test_the_field_is_shown_pixel_for_pixel(gui, tmp_path):
    """Every annotation this view draws is one pixel wide so that a line
    stays a line where the field goes flat, and scaling to fit would drop
    most of them. The Scale control is the zoom - it resamples the field,
    not a picture of it.
    """
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(120.0, 40.0)})])
    image = gui.field_stage.Render()
    assert image is not None

    pixmap = gui.image_label.pixmap()

    assert pixmap is not None
    assert (pixmap.height(), pixmap.width()) == image.shape[:2]


def test_choosing_a_contour_shows_that_one(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0), 1: _rectangle(40.0, 30.0)})])

    gui.contour_box.setCurrentIndex(1)

    part = gui.field_stage.part
    assert part is not None
    assert part.size == pytest.approx([40.0, 30.0])


def test_adding_a_file_does_not_move_the_selection(gui, tmp_path):
    """Rebuilding the selector clears it first, which emits a change to
    index -1 - so without blocking that signal, loading a second file
    would deselect whatever was being looked at.
    """
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0), 1: _rectangle(40.0, 30.0)})])
    gui.contour_box.setCurrentIndex(1)

    gui.load_contours([_dump(tmp_path, "b.json", {0: _rectangle(18.0, 12.0)})])

    assert gui.field_stage.selected == 1
    assert gui.contour_box.currentData() == 1


# ---------------------------------------------------------------- readout


def test_the_readout_reports_the_field_under_the_pointer(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])
    part = gui.field_stage.part
    assert part is not None

    # Straight through the stage, which is where the coordinate math is -
    # the window only undoes the pixmap's letterboxing.
    low = part.origin + 0.5 * part.resolution
    pixel = tuple((np.array([10.0, 5.0]) - low) * gui.field_stage.view.pixels_per_mm)

    assert "inside the part" in gui.field_stage.Probe(pixel)


def test_hovering_with_nothing_loaded_does_not_raise(gui):
    gui.image_hovered(None)

    assert gui.readout_label.text() == ""


# --------------------------------------------------------- the real thing


@pytest.mark.slow
def test_a_captured_spoon_shows_its_field(qapp):
    """The fixtures are what the solver actually packs: concave, and with
    edges long enough that D2's boundary resampling is load-bearing.
    """
    window = FieldGui()
    window.load_contours(SPOONS)

    assert len(window.contours) == 3

    image = window.field_stage.Render()
    assert image is not None and image.shape[2] == 3
    assert "reaching" in window.field_stage.Summary()
