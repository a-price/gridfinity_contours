"""Tests for the standalone layout window (M6).

The behavior worth pinning is that loading is *additive* across files -
that is the reason this window exists apart from silhouette.py, which can
only ever produce one session's worth of contours.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from pipeline.contour_io import SaveContours
from pipeline.layout.energy import LayoutParameters
from layout_gui import LayoutGui

SPOONS = ["test_data/big_spoon.svg", "test_data/medium_spoon.svg", "test_data/small_spoon.svg"]


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def gui(qapp):
    window = LayoutGui()
    window.layout_stage.parameters = LayoutParameters(restarts=4, iterations=120, patience=25, max_grid=2)
    return window


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


def _dump(tmp_path, name, contours) -> str:
    path = str(tmp_path / name)
    SaveContours(path, contours)
    return path


# ------------------------------------------------------------- accumulating


def test_loading_two_files_keeps_both(gui, tmp_path):
    """The point of this window: a bin's worth of objects arrives a session
    at a time, so a second load must add rather than replace.
    """
    first = _dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})
    second = _dump(tmp_path, "b.json", {0: _rectangle(18.0, 12.0)})

    gui.load_contours([first])
    gui.load_contours([second])

    assert len(gui.contours) == 2
    assert gui.sources == [first, second]


def test_contours_are_renumbered_so_two_sessions_do_not_collide(gui, tmp_path):
    first = _dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})
    second = _dump(tmp_path, "b.json", {0: _rectangle(18.0, 12.0)})

    gui.load_contours([first, second])

    assert sorted(gui.contours) == [0, 1]


def test_svgs_and_dumps_can_be_mixed(gui, tmp_path):
    dump = _dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})

    gui.load_contours([dump, "test_data/small_spoon.svg"])

    assert len(gui.contours) == 2


def test_clearing_removes_everything(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])

    gui.clear_contours()

    assert gui.contours == {}
    assert gui.sources == []


def test_the_panel_lists_what_is_loaded(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "spoons.json", {0: _rectangle(20.0, 10.0)})])

    text = gui.source_label.text()
    assert "1 contours" in text
    assert "spoons.json" in text


def test_an_unreadable_file_is_reported_not_raised(gui, tmp_path):
    # A dialog full of the wrong files should not take the window down.
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")

    gui.load_contours([str(bad)])

    assert "Could not load" in gui.source_label.text()
    assert gui.contours == {}


def test_loading_more_contours_drops_a_stale_layout(gui, tmp_path):
    """A layout describes the set it was packed from; adding to that set
    makes the picture on screen a lie.
    """
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])
    gui.pipeline.RunFrom("layout")
    assert gui.layout_stage.layout is not None

    gui.load_contours([_dump(tmp_path, "b.json", {0: _rectangle(18.0, 12.0)})])

    assert gui.layout_stage.layout is None


# ------------------------------------------------------------------ packing


def test_packing_runs_only_from_the_pipeline_trigger(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])

    assert gui.layout_stage.layout is None, "loading must not pack"

    gui.pipeline.RunFrom("layout")

    assert gui.layout_stage.layout is not None


def test_packing_with_nothing_loaded_says_so(gui):
    gui.pipeline.RunFrom("layout")

    assert gui.layout_stage.layout is None
    assert "no contours" in gui.layout_stage.Summary().lower()


def test_a_packed_layout_reaches_the_image_view(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])

    gui.pipeline.RunFrom("layout")

    pixmap = gui.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()


def test_the_view_prompts_before_anything_is_packed(gui):
    gui.update_display()

    assert "Pack" in gui.image_label.text()


# ------------------------------------------------------------------- export


def test_exporting_writes_both_files(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])
    gui.pipeline.RunFrom("layout")
    gui.export_edit.setText(str(tmp_path / "out"))

    gui.export_layout()

    assert (tmp_path / "out.svg").exists()
    assert (tmp_path / "out.pdf").exists()
    assert "Wrote" in gui.export_label.text()


def test_exporting_before_packing_is_reported_not_raised(gui, tmp_path):
    gui.export_edit.setText(str(tmp_path / "out"))

    gui.export_layout()

    assert "Could not export" in gui.export_label.text()
    assert not (tmp_path / "out.svg").exists()


def test_a_chosen_svg_filename_does_not_become_svg_svg(gui, tmp_path):
    gui.export_edit.setText(str(tmp_path / "layout.svg"))
    base, extension = os.path.splitext(gui.export_edit.text())
    gui.export_edit.setText(base if extension.lower() in (".svg", ".pdf") else gui.export_edit.text())

    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])
    gui.pipeline.RunFrom("layout")
    gui.export_layout()

    assert (tmp_path / "layout.svg").exists()
    assert not (tmp_path / "layout.svg.svg").exists()


# ----------------------------------------------------------------- the real thing


@pytest.mark.slow
def test_the_three_spoon_captures_pack_in_the_window(qapp):
    """M6's done-when, on the real fixtures: three separate capture
    sessions, loaded together and packed into one bin.
    """
    window = LayoutGui()
    window.load_contours(SPOONS)

    assert len(window.contours) == 3

    window.pipeline.RunFrom("layout")

    layout = window.layout_stage.layout
    assert layout is not None
    assert layout.grid == (5, 2)
    assert "3 parts in 5x2" in window.layout_stage.Summary()

    pixmap = window.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()
