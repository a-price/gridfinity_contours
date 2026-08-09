"""Tests for the floorplan stage.

The property that matters here is that there is always something honest to
draw. This search runs for minutes, so a stage that could only render a
finished plan would leave the window blank for the whole of it. What it
draws is the drawers - empty before anything is planned, then filling as
the three states it passes through (grouping, assigning, done) each learn
a little more about where the bins go.
"""

import json
from dataclasses import replace

import numpy as np
import pytest
from PyQt5.QtWidgets import QComboBox, QLabel, QPushButton
from pipeline.layout.parameters import FREE_ROTATION, ROTATIONS

from pipeline.layout.drawer import PLACED, AssignmentResult, Drawer, Slot
from pipeline.layout.plan import ASSIGNING, FILLING, GROUPING, Progress
from pipeline.floorplan_stage import EXPORT_EXTENSIONS, FloorplanStage
from conftest import QuickParameters as _quick, Rectangle as _rectangle


def _contours(count: int = 3) -> dict[int, np.ndarray]:
    return {index: _rectangle(60.0 - 8.0 * index, 30.0) for index in range(count)}


def _stage(drawers=None, **overrides) -> FloorplanStage:
    # `drawers is None` rather than a falsy check: an empty list is a
    # meaningful argument here, and the case one of these tests is about.
    params = _quick(**{"max_grid": 3, **overrides})
    return FloorplanStage(params, [Drawer(6, 6)] if drawers is None else drawers)


def _widgets(widget, kind):
    return widget.findChildren(kind)


def _labels(widget) -> str:
    return "\n".join(label.text() for label in _widgets(widget, QLabel))


def _button(widget, text: str) -> QPushButton:
    (found,) = [button for button in _widgets(widget, QPushButton) if button.text() == text]
    return found


# ---------------------------------------------------------------- running


def test_planning_groups_and_assigns():
    stage = _stage()

    stage.Run(_contours())

    assert stage.plan is not None
    assert stage.plan.placed
    assert stage.error is None


def test_the_drawers_restrict_the_bins_that_are_proposed():
    """The stage holds the drawers because they are an input to the
    search, not a display option - `BuildPlan` narrows the grid-size
    search to what they can hold before grouping starts.
    """
    stage = _stage(drawers=[Drawer(2, 2)], max_grid=6)

    stage.Run(_contours(2))

    assert stage.plan is not None
    for layout in stage.plan.layouts.values():
        assert Drawer(2, 2).Holds(layout.grid)


def test_running_with_nothing_loaded_says_so_rather_than_raising():
    stage = _stage()

    stage.Run({})

    assert stage.plan is None
    assert "load some contours" in stage.Summary()


def test_running_with_no_drawers_says_so():
    stage = _stage(drawers=[])

    stage.Run(_contours())

    assert stage.plan is None
    assert "drawer" in stage.Summary()


def test_a_library_no_drawer_could_store_is_reported_not_raised():
    """This arrives from a text box, and should not take the window down."""
    stage = _stage(drawers=[Drawer(1, 1)])

    stage.Run(_contours())

    assert stage.plan is None
    assert stage.error is not None
    assert "Floorplan:" in stage.Summary()


# --------------------------------------------------------------- rendering


def test_a_drawer_draws_itself_before_anything_is_planned():
    """The drawer is the subject, so it is on screen from the moment you
    say you own one - which is also when a mistyped size is cheapest to
    notice.
    """
    image = _stage().Render()

    assert image is not None and image.shape[2] == 3


def test_nothing_renders_without_a_drawer():
    """The one remaining blank state, and the only honest one: there is no
    drawer to draw.
    """
    assert _stage(drawers=[]).Render() is None


def test_every_drawer_is_drawn_including_the_empty_ones():
    """All of them, not only the ones something went into. A drawer left
    out of the picture is a drawer nobody remembers they have.
    """
    one = _stage(drawers=[Drawer(6, 6)]).Render()
    two = _stage(drawers=[Drawer(6, 6), Drawer(2, 2)]).Render()

    assert one is not None and two is not None
    assert two.shape[1] > one.shape[1]


def test_planning_fills_the_drawers_rather_than_replacing_them():
    """The picture is the drawers throughout. Finding an answer puts
    objects in them; it does not swap the drawing for a different one.
    """
    stage = _stage()
    empty = stage.Render()

    stage.Run(_contours())

    planned = stage.Render()
    assert empty is not None and planned is not None
    assert planned.shape == empty.shape, "the same drawers at the same scale"
    assert not np.array_equal(planned, empty), "with something in them now"


def test_a_grouping_in_progress_is_drawn_in_the_drawers():
    """The drawer search has not run, so the positions are a provisional
    first fit - but bins lying in drawers is the picture somebody watching
    a five minute search is waiting for, and a strip of loose bins is not.
    """
    stage = _stage()
    stage.Run(_contours())
    bins = tuple(stage.plan.layouts.values()) if stage.plan else ()
    empty = _stage().Render()
    plan, stage.plan = stage.plan, None
    assert plan is not None and empty is not None

    stage.SetProgress(Progress(GROUPING, 5, bins))

    image = stage.Render()
    assert image is not None
    assert image.shape == empty.shape, "drawn into the drawers, not as a strip beside them"
    assert not np.array_equal(image, empty)


def test_an_assignment_in_progress_renders_as_a_partial_floorplan():
    stage = _stage()
    stage.Run(_contours())
    plan = stage.plan
    assert plan is not None
    bins = tuple(plan.layouts.values())
    partial = AssignmentResult(PLACED, {0: Slot(0, 0, (0, 0))})
    stage.plan = None

    stage.SetProgress(Progress(ASSIGNING, 9, bins, partial))

    assert stage.Render() is not None


def test_a_bin_with_no_slot_is_drawn_beside_the_drawers():
    """Nothing the search is holding may vanish from the picture. A bin
    with nowhere to go is exactly the one a person is looking for.
    """
    stage = _stage()
    stage.Run(_contours())
    bins = tuple(stage.plan.layouts.values()) if stage.plan else ()
    stage.plan = None
    stage.drawers = [Drawer(1, 1)]  # nothing here can hold those bins
    bare = _stage(drawers=[Drawer(1, 1)]).Render()

    stage.SetProgress(Progress(GROUPING, 5, bins))

    image = stage.Render()
    assert image is not None and bare is not None
    assert image.shape[1] > bare.shape[1]


def test_first_fits_opening_bins_are_drawn_as_they_are_opened():
    """The earliest picture there is. These bins hold only part of the
    library, but they are real bins and they exist seconds after Plan is
    pressed - where the first complete grouping is minutes away.
    """
    stage = _stage()
    stage.Run(_contours())
    one_bin = tuple(stage.plan.layouts.values())[:1] if stage.plan else ()
    empty = _stage().Render()
    stage.plan = None
    assert one_bin and empty is not None

    stage.SetProgress(Progress(FILLING, 2, one_bin, expected=3))

    image = stage.Render()
    assert image is not None
    assert image.shape == empty.shape, "drawn in the drawer, like everything else"
    assert not np.array_equal(image, empty), "and the bin should be visible in it"
    assert "/3 objects into 1 bin(s)" in stage.Summary(), "a fraction of the library, not an arrangement of it"


def test_a_report_with_no_answer_yet_still_draws_the_drawers():
    stage = _stage()

    stage.SetProgress(Progress(GROUPING, 3))

    assert stage.Render() is not None


# ---------------------------------------------------------------- summary


def test_the_summary_names_the_bins_and_the_drawers():
    stage = _stage()

    stage.Run(_contours())

    summary = stage.Summary()
    assert "bins" in summary and "cells" in summary
    assert "cells free" in summary or "in one piece" in summary


def test_the_summary_follows_progress_while_running():
    stage = _stage()

    stage.SetProgress(Progress(GROUPING, 7))

    assert "building the first" in stage.Summary()


def test_a_stopped_search_does_not_read_as_a_failure():
    """A cancelled search says nothing about whether the bins fit, so the
    panel must not read as though they did not.
    """
    stage = _stage()

    stage.Run(_contours(), cancelled=lambda: True)

    assert stage.plan is not None and stage.plan.cancelled
    summary = stage.Summary()
    assert "stopped" in summary
    assert "not placed" not in summary


# ---------------------------------------------------------------- pinning


def test_pinning_a_bin_keeps_it_across_a_replan():
    stage = _stage()
    stage.Run(_contours(4))
    held = stage.Bins()[0]

    stage.Pin([0])
    stage.Run(_contours(4))

    assert stage.plan is not None
    assert stage.plan.layouts[0] is held
    assert stage.PinnedIds() == frozenset([0])


def test_a_pin_outlives_the_plan_it_was_made_against():
    """Every edit to the drawers or the library drops the plan, and a pin
    that went with it would be unticked by adding a drawer.
    """
    stage = _stage()
    stage.Run(_contours(4))
    held = stage.Bins()[0]
    stage.Pin([0])

    stage.drawers.append(Drawer(2, 2))
    stage.Clear()

    assert stage.pinned == [held], "Clear drops results, and a pin is an input"


def test_pinning_survives_a_session(tmp_path):
    stage = _stage()
    contours = _contours(4)
    stage.Run(contours)
    stage.Pin([0])
    stage.Run(contours)
    path = str(tmp_path / "s.json")
    stage.Save(path, contours)

    reopened = _stage(drawers=[])
    reopened.Load(path)

    assert reopened.PinnedIds() == frozenset([0])
    assert len(reopened.pinned) == 1


def test_pinning_a_bin_that_is_not_there_is_refused():
    stage = _stage()

    with pytest.raises(ValueError, match="no bin 3"):
        stage.Pin([3])


def test_the_summary_says_how_much_was_left_alone():
    stage = _stage()
    stage.Run(_contours(4))
    stage.Pin([0])

    stage.Run(_contours(4))

    assert "1 bin(s) pinned" in stage.Summary()


def test_a_pinned_bin_is_drawn_differently():
    """The picture has to say which bins you already own, or the pin is
    invisible in the one place it matters most.
    """
    stage = _stage()
    stage.Run(_contours(4))
    plain = stage.Render()

    stage.Pin([0])
    marked = stage.Render()

    assert plain is not None and marked is not None
    assert plain.shape == marked.shape
    assert not np.array_equal(plain, marked)


# ----------------------------------------------------------------- export


def test_exporting_writes_the_floorplan_and_every_bin_on_it(tmp_path):
    """The map says where bins go; what you make is a bin. Nothing else in
    the project writes a solid for a bin the grouping search produced, so
    without this a library plan cannot be turned into anything printable.
    """
    stage = _stage()
    stage.Run(_contours(4))
    bins = len(stage.Bins())

    report = stage.Export(str(tmp_path / "plan"))

    assert (tmp_path / "plan.pdf").exists(), "the drawer map"
    for bin_id in range(bins):
        for extension in (".svg", ".pdf", ".scad"):
            assert (tmp_path / f"plan_bin{bin_id}{extension}").exists()
    assert report.bins == bins
    assert report.problems == {}


def test_an_exported_bin_is_the_one_the_plan_chose(tmp_path):
    """Correct by construction, and the reason this belongs here rather
    than being done by hand in layout_gui: re-packing the same objects is
    a different stochastic search, so the bin coming off the printer
    would not match the map printed to lay it out with.
    """
    stage = _stage()
    stage.Run(_contours(4))
    assert stage.plan is not None
    layout = stage.plan.layouts[0]

    stage.Export(str(tmp_path / "plan"))

    scad = (tmp_path / "plan_bin0.scad").read_text()
    n, m = layout.grid
    assert f"{n}" in scad and f"{m}" in scad
    assert scad.count("polygon") == len(layout.placements), "one pocket per object in that bin"


def test_the_bin_files_sort_the_way_the_report_lists_them(tmp_path):
    """A directory of a dozen bins fed to a slicer should not put bin10
    between bin1 and bin2.
    """
    # Eleven objects too fat to share a one-cell bin, so there are eleven
    # bins and the numbering goes into two digits.
    stage = _stage(max_grid=1)
    stage.Run({index: _rectangle(24.0, 24.0) for index in range(11)})
    assert stage.plan is not None and len(stage.plan.layouts) == 11

    stage.Export(str(tmp_path / "plan"))

    scads = sorted(path.stem for path in tmp_path.glob("plan_bin*.scad"))
    assert scads[1] == "plan_bin01", f"got {scads[:3]}"


def test_pinned_bins_are_exported_too(tmp_path):
    """A pin says the search may not change a bin, not that the file is
    already on disk. An export that silently skipped bins would be the
    more expensive mistake.
    """
    stage = _stage()
    stage.Run(_contours(4))
    stage.Pin([0])

    report = stage.Export(str(tmp_path / "plan"))

    assert (tmp_path / "plan_bin0.scad").exists()
    assert report.bins == len(stage.Bins())


def test_a_solid_that_cannot_be_cut_is_reported_not_raised(tmp_path):
    """Wall thickness is a property of one bin. The sheets for it are
    already written and every other bin is fine, so losing the whole
    export over one would be the wrong trade.
    """
    stage = _stage()
    stage.Run(_contours(4))
    bins = len(stage.Bins())
    # Far more than the layout reserved room for, so no pocket in it can
    # be cut without eating the dividers.
    stage.parameters.pocket_offset = 20.0

    report = stage.Export(str(tmp_path / "plan"))

    assert (tmp_path / "plan.pdf").exists()
    assert (tmp_path / "plan_bin0.svg").exists(), "the sheets still land"
    assert not (tmp_path / "plan_bin0.scad").exists()
    assert sorted(report.problems) == list(range(bins)), "named by bin, so a caller can say which"
    assert "mm" in report.problems[0], "the message should name the measurement that failed"
    assert "could not be cut" in str(report)
    assert str(report).count("pocket offset") == 1, "every bin named, the explanation given once"


def test_exporting_before_planning_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nothing planned"):
        _stage().Export(str(tmp_path / "plan"))


def test_exporting_a_search_stopped_before_the_drawers_still_works(tmp_path):
    """The bins are settled and only the drawer search is missing - and
    that half is exact and takes microseconds, against the minutes the
    grouping took. Refusing to draw a map for bins that exist made a
    person redo the expensive half to recover the cheap one.
    """
    stage = _stage()
    stage.Run(_contours(4))
    grouped = stage.plan
    assert grouped is not None
    stage.plan = replace(grouped, assignment=None, cancelled=True)

    report = stage.Export(str(tmp_path / "plan"))

    assert (tmp_path / "plan.pdf").exists()
    assert report.bins == len(grouped.layouts)


def test_exporting_a_search_stopped_before_any_bins_is_refused(tmp_path):
    """Nothing was found, so there is genuinely nothing to write."""
    stage = _stage()
    stage.Run(_contours(), cancelled=lambda: True)
    assert stage.plan is not None and not stage.plan.layouts

    with pytest.raises(ValueError, match="nothing planned"):
        stage.Export(str(tmp_path / "plan"))


# ---------------------------------------------------------------- session


def test_a_session_round_trips_through_the_stage(tmp_path):
    stage = _stage()
    contours = _contours()
    stage.Run(contours)
    path = str(tmp_path / "s.json")

    stage.Save(path, contours)
    reopened = _stage(drawers=[])
    returned = reopened.Load(path)

    assert sorted(returned) == sorted(contours)
    assert reopened.drawers == stage.drawers
    assert reopened.resume is not None


def test_a_loaded_session_draws_without_searching(tmp_path):
    """The floorplan was printed from these placements, so it can be
    looked at - and exported - without running anything.
    """
    stage = _stage()
    contours = _contours()
    stage.Run(contours)
    path = str(tmp_path / "s.json")
    stage.Save(path, contours)

    reopened = _stage(drawers=[])
    reopened.Load(path)

    assert reopened.Render() is not None
    assert (tmp_path / "plan.pdf") not in tmp_path.iterdir()
    assert reopened.Export(str(tmp_path / "plan")).written[0] == str(tmp_path / "plan.pdf")


def test_a_session_saved_from_a_stopped_search_reopens_whole(tmp_path):
    """The bug this came from: Stop after the grouping lands but before
    the drawers are searched, Save, reopen - and the floorplan could be
    looked at but not exported, "nothing planned to export" about five
    perfectly good bins. The drawer search is filled in on the way back
    in, so a reopened floorplan is a whole one however it was saved.
    """
    stage = _stage()
    contours = _contours(4)
    stage.Run(contours)
    assert stage.plan is not None
    stage.plan = replace(stage.plan, assignment=None, cancelled=True)
    path = str(tmp_path / "stopped.json")
    stage.Save(path, contours)
    assert "assignment" not in json.loads(open(path).read()), "the file genuinely has none"

    reopened = _stage(drawers=[])
    reopened.Load(path)

    assert reopened.plan is not None and reopened.plan.assignment is not None
    assert reopened.plan.assignment.placed
    assert "not placed" not in reopened.Summary()
    assert reopened.Export(str(tmp_path / "plan")).bins == len(reopened.Bins())


def test_saving_before_planning_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nothing planned"):
        _stage().Save(str(tmp_path / "s.json"), _contours())


def test_resuming_reports_which_bins_have_to_be_reprinted(tmp_path):
    """The question the whole flow exists to answer."""
    stage = _stage()
    contours = _contours(3)
    stage.Run(contours)
    path = str(tmp_path / "s.json")
    stage.Save(path, contours)

    reopened = _stage(drawers=[])
    grown = dict(reopened.Load(path))
    grown[max(grown) + 1] = _rectangle(50.0, 25.0)
    reopened.Run(grown)

    assert reopened.changes is not None
    assert "to reprint" in reopened.Summary()


def test_a_rerun_without_a_session_reports_no_churn():
    """Nothing to compare against, so the panel must not invent a
    baseline.
    """
    stage = _stage()

    stage.Run(_contours())

    assert stage.changes is None
    assert "reprint" not in stage.Summary()


# ------------------------------------------------------------------ panel


def test_editing_a_parameter_does_not_start_a_search(qapp):
    """This one takes minutes; a panel that ran on a spin box tick would
    be unusable.
    """
    stage = _stage()
    runs = []
    widget = stage.CreateWidget(on_change=lambda: runs.append(1))

    from PyQt5.QtWidgets import QDoubleSpinBox

    for spin_box in _widgets(widget, QDoubleSpinBox):
        spin_box.setValue(spin_box.value() + 1)
        spin_box.editingFinished.emit()

    assert runs == []


def test_the_plan_button_is_what_triggers(qapp):
    stage = _stage()
    runs = []
    widget = stage.CreateWidget(on_change=lambda: runs.append(1))

    _button(widget, "Plan").click()

    assert runs == [1]


def test_stop_is_offered_only_to_a_host_that_can_stop(qapp):
    stage = _stage()

    without = stage.CreateWidget(on_change=lambda: None)
    assert not _button(without, "Stop").isVisible()

    with_cancel = stage.CreateWidget(on_change=lambda: None, on_cancel=lambda: None)
    assert not _button(with_cancel, "Stop").isEnabled(), "nothing to stop yet"

    stage.SetBusy(True)
    assert _button(with_cancel, "Stop").isEnabled()
    assert not _button(with_cancel, "Plan").isEnabled()


def test_the_panel_shows_what_the_offset_buys(qapp):
    stage = _stage()
    widget = stage.CreateWidget(on_change=lambda: None)

    assert "pockets cut 1.00mm oversize" in _labels(widget)


def test_the_export_button_names_every_format_it_writes(qapp):
    stage = _stage()
    stage.CreateWidget(on_change=lambda: None)

    for extension in EXPORT_EXTENSIONS:
        assert extension == ".pdf", "a floorplan is one page per drawer, so it has to be a PDF"


def test_the_rotation_mode_is_a_dropdown_that_reaches_the_parameters(qapp):
    """It changes which arrangements are legal, so it belongs beside the
    other things the answer must satisfy rather than only on the command
    line - a window that could not ask for it would make the mode
    unreachable for everyone not running `layout_cli`.
    """
    stage = FloorplanStage(_quick())
    widget = stage.CreateWidget(on_change=lambda: None)

    combo = _widgets(widget, QComboBox)[0]
    assert [combo.itemText(index) for index in range(combo.count())] == list(ROTATIONS)
    assert combo.currentText() == stage.parameters.rotation

    combo.setCurrentText(FREE_ROTATION)

    assert stage.parameters.rotation == FREE_ROTATION


def test_choosing_a_rotation_mode_does_not_start_a_search(qapp):
    """The same rule every other control in this panel follows: editing a
    parameter is not a request to run.
    """
    stage = FloorplanStage(_quick())
    runs = []
    widget = stage.CreateWidget(on_change=lambda: runs.append(1))

    _widgets(widget, QComboBox)[0].setCurrentText(FREE_ROTATION)

    assert runs == []
