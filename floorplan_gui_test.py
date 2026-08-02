"""Tests for the whole-library floorplan window.

What is worth pinning here is the drawer list - the input this window has
that no other front end does - and that a search which runs for minutes
stays interruptible and keeps showing what it has found.
"""

import pytest

from pipeline.contour_io import SaveContours
from pipeline.layout.drawer import Drawer
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.plan import GROUPING, Progress, SaveDrawers
from floorplan_gui import FloorplanGui
from conftest import QuickParameters as _quick, Rectangle as _rectangle


@pytest.fixture
def gui(qapp):
    window = FloorplanGui()
    window.floorplan_stage.parameters = _quick(max_grid=3)
    return window


def _dump(tmp_path, name, contours) -> str:
    path = str(tmp_path / name)
    SaveContours(path, contours)
    return path


def _library(tmp_path, count: int = 3) -> str:
    return _dump(tmp_path, "library.json", {index: _rectangle(60.0 - 8.0 * index, 30.0) for index in range(count)})


def _loaded(gui, tmp_path, count: int = 3) -> None:
    gui.load_contours([_library(tmp_path, count)])
    gui.drawer_edit.setText("500x400")
    gui.add_drawer()


def _plan(gui) -> None:
    gui.plan()
    gui.WaitForPlan()


# --------------------------------------------------------------- drawers


def test_a_drawer_is_typed_in_millimetres_and_kept_in_cells(gui):
    """Millimetres on the way in because that is what a tape measure
    reads; whole cells everywhere after, because that is what the search
    and the saved list both speak.
    """
    gui.drawer_edit.setText("500x400")

    gui.add_drawer()

    assert gui.floorplan_stage.drawers == [Drawer(11, 9)]
    assert gui.drawer_edit.text() == "", "the box should clear, ready for the next one"


def test_a_drawer_can_be_typed_in_cells_instead(gui):
    """The saved list is in cells, so adding one more drawer to a list you
    just loaded should not mean converting it back to millimetres in your
    head first.
    """
    gui.drawer_edit.setText("11x9 cells")

    gui.add_drawer()

    assert gui.floorplan_stage.drawers == [Drawer(11, 9)]


def test_the_two_units_agree(gui):
    for text in ("500x400", "11x9 cells"):
        gui.drawer_edit.setText(text)
        gui.add_drawer()

    assert gui.floorplan_stage.drawers[0] == gui.floorplan_stage.drawers[1]


def test_the_list_shows_both_units(gui):
    """The mm-to-cells step is lossy, so it is shown rather than implied -
    somebody wondering why 500mm and 504mm behave identically should be
    able to see it.
    """
    gui.drawer_edit.setText("500x400")
    gui.add_drawer()

    text = gui.drawer_list.item(0).text()

    assert "11 x 9 cells" in text
    assert "461.5" in text, "the span the bins actually occupy, not what was typed"


def test_several_drawers_accumulate(gui):
    for size in ("500x400", "300x200"):
        gui.drawer_edit.setText(size)
        gui.add_drawer()

    assert len(gui.floorplan_stage.drawers) == 2
    assert gui.drawer_list.count() == 2
    assert "2 drawer(s)" in gui.drawer_label.text()


def test_a_drawer_that_does_not_parse_is_reported_not_raised(gui):
    gui.drawer_edit.setText("enormous")

    gui.add_drawer()

    assert gui.floorplan_stage.drawers == []
    assert "WIDTHxHEIGHT" in gui.drawer_label.text()


def test_a_drawer_too_small_for_a_cell_is_reported(gui):
    gui.drawer_edit.setText("20x20")

    gui.add_drawer()

    assert gui.floorplan_stage.drawers == []
    assert "no whole" in gui.drawer_label.text()


def test_a_drawer_can_be_removed(gui):
    for size in ("500x400", "300x200"):
        gui.drawer_edit.setText(size)
        gui.add_drawer()

    gui.drawer_list.setCurrentRow(0)
    gui.remove_drawer()

    assert gui.floorplan_stage.drawers == [Drawer(7, 4)]


def test_drawers_round_trip_through_a_file(gui, tmp_path):
    path = str(tmp_path / "drawers.json")
    SaveDrawers(path, [Drawer(11, 9), Drawer(4, 3)])

    gui.load_drawers(path)

    assert gui.floorplan_stage.drawers == [Drawer(11, 9), Drawer(4, 3)]
    assert gui.drawer_list.count() == 2


def test_an_unreadable_drawer_file_is_reported_not_raised(gui, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")

    gui.load_drawers(str(bad))

    assert gui.floorplan_stage.drawers == []
    assert "Could not load" in gui.drawer_label.text()


def test_changing_the_drawers_drops_a_stale_plan(gui, tmp_path):
    """A plan describes the drawers it was built for; adding one makes the
    picture on screen a lie.
    """
    _loaded(gui, tmp_path)
    _plan(gui)
    assert gui.floorplan_stage.plan is not None

    gui.drawer_edit.setText("300x200")
    gui.add_drawer()

    assert gui.floorplan_stage.plan is None


# -------------------------------------------------------------- contours


def test_loading_two_files_keeps_both(gui, tmp_path):
    first = _dump(tmp_path, "a.json", {0: _rectangle(20.0, 10.0)})
    second = _dump(tmp_path, "b.json", {0: _rectangle(18.0, 12.0)})

    gui.load_contours([first])
    gui.load_contours([second])

    assert len(gui.contours) == 2


def test_an_unreadable_contour_file_is_reported_not_raised(gui, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")

    gui.load_contours([str(bad)])

    assert "Could not load" in gui.source_label.text()
    assert gui.contours == {}


def test_loading_more_contours_drops_a_stale_plan(gui, tmp_path):
    _loaded(gui, tmp_path)
    _plan(gui)
    assert gui.floorplan_stage.plan is not None

    gui.load_contours([_dump(tmp_path, "more.json", {0: _rectangle(18.0, 12.0)})])

    assert gui.floorplan_stage.plan is None


# ------------------------------------------------------------ searching


def test_planning_runs_only_when_triggered(gui, tmp_path):
    _loaded(gui, tmp_path)

    assert gui.floorplan_stage.plan is None, "loading must not start a search"

    _plan(gui)

    assert gui.floorplan_stage.plan is not None


def test_planning_runs_on_a_worker_thread(gui, tmp_path):
    """plan() must come straight back with the search still going, or
    none of the progress reporting matters - a thread that is joined
    immediately is just a slow function call.
    """
    gui.floorplan_stage.parameters = LayoutParameters(restarts=200, iterations=400, patience=200, max_grid=5)
    _loaded(gui, tmp_path, count=6)

    gui.plan()

    assert gui._worker is not None
    assert gui._worker.isRunning(), "plan() returned only after the search finished"

    gui.cancel_plan()
    gui.WaitForPlan()
    assert gui._worker is None, "the worker should be cleared once it reports back"


def test_a_second_plan_is_ignored_while_one_is_running(gui, tmp_path):
    _loaded(gui, tmp_path)

    gui.plan()
    first = gui._worker
    gui.plan()

    assert gui._worker is first, "a second press must not start a competing search"
    gui.WaitForPlan()


def test_the_inputs_are_frozen_while_a_search_runs(gui, tmp_path):
    """The search holds a snapshot, but an editable panel would still
    leave the drawer list describing something other than the answer about
    to arrive.
    """
    _loaded(gui, tmp_path)

    gui.plan()
    assert not gui.source_group.isEnabled()
    assert not gui.drawer_group.isEnabled()

    gui.WaitForPlan()
    assert gui.source_group.isEnabled()
    assert gui.drawer_group.isEnabled()


def test_closing_the_window_does_not_leave_a_thread_running(gui, tmp_path):
    _loaded(gui, tmp_path)
    gui.plan()

    gui.close()

    assert gui._worker is None or not gui._worker.isRunning()


# -------------------------------------------------------------- display


def test_the_view_prompts_before_anything_is_found(gui):
    assert "Plan" in gui.image_label.text()


def test_a_finished_plan_reaches_the_image_view(gui, tmp_path):
    _loaded(gui, tmp_path)

    _plan(gui)

    pixmap = gui.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()


def test_progress_reaches_the_panel_while_searching(gui, tmp_path):
    _loaded(gui, tmp_path)
    seen = []
    gui.floorplan_stage.SetStatus = lambda text: seen.append(text)

    _plan(gui)

    assert any("grouping" in text or "assigning" in text for text in seen)


def test_a_report_with_an_answer_draws_it_mid_search(gui, tmp_path):
    """The whole reason this window reports progress: a search that runs
    for minutes has to show what it would settle for.
    """
    _loaded(gui, tmp_path)
    _plan(gui)
    bins = tuple(gui.floorplan_stage.plan.layouts.values())

    # Mid-search is exactly this: the parts are built and reports are
    # arriving, but no finished plan exists yet.
    gui.floorplan_stage.plan = None
    gui.image_label.setPixmap(type(gui.image_label.pixmap())())

    gui._OnProgress(Progress(GROUPING, 5, bins))

    pixmap = gui.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    assert "best so far" in gui.floorplan_stage.Summary()


# --------------------------------------------------------------- export


def test_exporting_writes_the_floorplan(gui, tmp_path):
    _loaded(gui, tmp_path)
    _plan(gui)
    gui.export_edit.setText(str(tmp_path / "plan"))

    gui.export_plan()

    assert (tmp_path / "plan.pdf").exists()
    assert "Wrote" in gui.export_label.text()


def test_exporting_before_planning_is_reported_not_raised(gui, tmp_path):
    gui.export_edit.setText(str(tmp_path / "plan"))

    gui.export_plan()

    assert "Could not export" in gui.export_label.text()
    assert not (tmp_path / "plan.pdf").exists()


# ------------------------------------------------------- the real thing


@pytest.mark.slow
def test_a_real_library_plans_into_a_real_drawer(qapp):
    """M10's done-when, on the fixtures: several capture sessions, grouped
    into bins and laid out in a drawer.
    """
    window = FloorplanGui()
    window.floorplan_stage.parameters = LayoutParameters(restarts=3, iterations=150, patience=20, max_grid=6, seed=0)
    window.drawer_edit.setText("500x400")
    window.add_drawer()
    window.load_contours(
        [
            "test_data/small_spoon.svg",
            "test_data/medium_spoon.svg",
            "test_data/big_spoon.svg",
            "test_data/medium_fork.svg",
            "test_data/spreader.svg",
        ]
    )

    _plan(window)

    plan = window.floorplan_stage.plan
    assert plan is not None and plan.placed
    assert len(plan.layouts) < 5, "grouping should beat one bin per object"
    assert "in one piece" in plan.Report()
