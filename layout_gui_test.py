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
from pipeline.layout_stage import EXPORT_EXTENSIONS
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


def _slow() -> LayoutParameters:
    """A budget big enough that a search is certainly still running when
    the call to start it returns.
    """
    return LayoutParameters(restarts=400, iterations=400, patience=200, max_grid=5)


def _unpackable() -> dict:
    """Parts that keep the solver busy: each fits a cell, together they
    defeat every size the search will reach before being stopped.
    """
    return {index: _rectangle(34.0, 34.0) for index in range(6)}


def _pack(gui) -> None:
    """Start a pack and wait for it, since packing is asynchronous now."""
    gui.pack()
    gui.WaitForPack()


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
    _pack(gui)
    assert gui.layout_stage.layout is not None

    gui.load_contours([_dump(tmp_path, "b.json", {0: _rectangle(18.0, 12.0)})])

    assert gui.layout_stage.layout is None


# ------------------------------------------------------------------ packing


def test_packing_runs_only_when_triggered(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])

    assert gui.layout_stage.layout is None, "loading must not pack"

    _pack(gui)

    assert gui.layout_stage.layout is not None


def test_packing_with_nothing_loaded_says_so(gui):
    _pack(gui)

    assert gui.layout_stage.layout is None
    assert "no contours" in gui.layout_stage.Summary().lower()


def test_a_packed_layout_reaches_the_image_view(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])

    _pack(gui)

    pixmap = gui.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()


def test_the_view_prompts_before_anything_is_packed(gui):
    gui.update_display()

    assert "Pack" in gui.image_label.text()


# ------------------------------------------------------------------- export


def test_exporting_writes_the_preview_and_the_bin(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])
    _pack(gui)
    gui.export_edit.setText(str(tmp_path / "out"))

    gui.export_layout()

    assert (tmp_path / "out.svg").exists()
    assert (tmp_path / "out.pdf").exists()
    assert "bin_render(bin)" in (tmp_path / "out.scad").read_text()
    assert "Wrote" in gui.export_label.text()


def test_exporting_before_packing_is_reported_not_raised(gui, tmp_path):
    gui.export_edit.setText(str(tmp_path / "out"))

    gui.export_layout()

    assert "Could not export" in gui.export_label.text()
    assert not (tmp_path / "out.svg").exists()


@pytest.mark.parametrize("extension", EXPORT_EXTENSIONS)
def test_a_chosen_filename_does_not_gain_a_second_extension(gui, tmp_path, extension):
    """The save dialog offers a filename, but the stage appends its own -
    so every extension it writes has to be strippable, not just the two
    it happened to write first.
    """
    gui.export_edit.setText(str(tmp_path / f"layout{extension}"))
    base, chosen = os.path.splitext(gui.export_edit.text())
    gui.export_edit.setText(base if chosen.lower() in EXPORT_EXTENSIONS else gui.export_edit.text())

    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])
    _pack(gui)
    gui.export_layout()

    assert (tmp_path / f"layout{extension}").exists()
    assert not (tmp_path / f"layout{extension}{extension}").exists()


def test_the_export_button_names_every_format_it_writes(gui):
    """The button said "SVG + PDF" for a while after the export started
    writing a .scad as well.
    """
    label = gui.export_button.text()

    for extension in EXPORT_EXTENSIONS:
        assert extension.lstrip(".").upper() in label


def test_exporting_writes_one_file_per_advertised_format(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])
    _pack(gui)
    gui.export_edit.setText(str(tmp_path / "every"))

    gui.export_layout()

    for extension in EXPORT_EXTENSIONS:
        assert (tmp_path / f"every{extension}").exists(), extension


# --------------------------------------------------------- off the ui thread


def test_a_pack_runs_on_a_worker_thread(gui, tmp_path):
    """pack() must come straight back with the search still going, or none
    of the rest of this matters - a thread that is joined immediately is
    just a slow function call.

    Given a search long enough that it cannot plausibly have finished in
    the time it takes to return.
    """
    gui.layout_stage.parameters = _slow()
    gui.load_contours([_dump(tmp_path, "a.json", _unpackable())])

    gui.pack()

    assert gui._worker is not None
    assert gui._worker.isRunning(), "pack() returned only after the search finished"
    assert gui.layout_stage.result is None, "no result should exist yet"

    gui.cancel_pack()
    gui.WaitForPack()
    assert gui._worker is None, "the worker should be cleared once it reports back"


def test_progress_reaches_the_panel_while_packing(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(60.0, 25.0)})])
    seen = []
    gui.layout_stage.SetStatus = lambda text: seen.append(text)

    _pack(gui)

    assert any("packing" in text for text in seen)


def test_cancelling_stops_the_search_without_claiming_failure(gui, tmp_path):
    """A cancelled search says nothing about whether the parts fit, so the
    panel must not read as though the bin were too small.
    """
    gui.layout_stage.parameters = _slow()
    gui.load_contours([_dump(tmp_path, "a.json", _unpackable())])

    gui.pack()
    gui.cancel_pack()
    gui.WaitForPack()

    result = gui.layout_stage.result
    assert result is not None and result.cancelled
    assert result.layout is None
    assert "cancelled" in gui.layout_stage.Summary().lower()
    assert "no fit" not in gui.layout_stage.Summary()


def test_a_second_pack_is_ignored_while_one_is_running(gui, tmp_path):
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])

    gui.pack()
    first = gui._worker
    gui.pack()

    assert gui._worker is first, "a second Pack must not start a competing search"
    gui.WaitForPack()


def test_the_sources_are_frozen_while_a_pack_runs(gui, tmp_path):
    """The search holds a snapshot, but letting the panel be edited would
    still leave the on-screen file list describing a different set than the
    result about to arrive.
    """
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])

    gui.pack()
    assert not gui.source_group.isEnabled()
    assert not gui.export_group.isEnabled()

    gui.WaitForPack()
    assert gui.source_group.isEnabled()
    assert gui.export_group.isEnabled()


def test_closing_the_window_does_not_leave_a_thread_running(gui, tmp_path):
    # A QThread still running when its window is destroyed takes the
    # process down with it.
    gui.load_contours([_dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})])
    gui.pack()

    gui.close()

    assert gui._worker is None or not gui._worker.isRunning()


# ----------------------------------------------------------------- the real thing


@pytest.mark.slow
def test_the_three_spoon_captures_pack_in_the_window(qapp):
    """M6's done-when, on the real fixtures: three separate capture
    sessions, loaded together and packed into one bin.
    """
    window = LayoutGui()
    window.load_contours(SPOONS)

    assert len(window.contours) == 3

    _pack(window)

    layout = window.layout_stage.layout
    assert layout is not None
    assert layout.grid == (5, 2)
    assert "3 parts in 5x2" in window.layout_stage.Summary()

    pixmap = window.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()
