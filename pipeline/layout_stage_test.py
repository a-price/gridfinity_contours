"""Tests for the layout GUI stage (M6).

The load-bearing property here is that packing runs *only* when asked.
Everything else in the panel reruns on a settled edit; this one takes
seconds, so a stage that quietly packed on a parameter change would lock
the window on every tick of a spin box.
"""

import numpy as np
import pytest
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QLabel, QPushButton

from pipeline.layout.loading import BuildParts
from pipeline.layout.packer import NOT_FOUND, PACKED, GridAttempt, PackResult
from pipeline.layout.parameters import FREE_ROTATION, ROTATIONS, LayoutParameters
from pipeline.layout.placement import Layout
from pipeline.layout_stage import LayoutStage
from conftest import QuickParameters, Rectangle as _rectangle


def _quick(**overrides) -> LayoutParameters:
    """The shared budget, capped at a 2x2 bin - these tests are about the
    panel and its wiring, so the smallest search that produces a layout is
    the right one.
    """
    return QuickParameters(**{"restarts": 4, "iterations": 120, "patience": 25, "max_grid": 2, **overrides})


def _widgets(widget, kind):
    return widget.findChildren(kind)


def _labels(widget) -> str:
    return "\n".join(label.text() for label in _widgets(widget, QLabel))


def _button(widget, text: str) -> QPushButton:
    (found,) = [button for button in _widgets(widget, QPushButton) if button.text() == text]
    return found


# --------------------------------------------------------- explicit trigger


def test_editing_a_parameter_does_not_pack(qapp):
    """The whole reason this stage has a button."""
    stage = LayoutStage(_quick())
    packs = []
    widget = stage.CreateWidget(on_change=lambda: packs.append(1))

    for spin_box in _widgets(widget, QDoubleSpinBox):
        spin_box.setValue(spin_box.value() + 1)
        spin_box.editingFinished.emit()

    assert packs == [], "a parameter edit must not trigger a pack"


def test_the_pack_button_is_what_triggers(qapp):
    stage = LayoutStage(_quick())
    packs = []
    widget = stage.CreateWidget(on_change=lambda: packs.append(1))

    _button(widget, "Pack").click()

    assert packs == [1]


def test_parameter_edits_still_reach_the_parameters(qapp):
    stage = LayoutStage(_quick())
    widget = stage.CreateWidget(on_change=lambda: None)
    spin_boxes = _widgets(widget, QDoubleSpinBox)

    spin_boxes[0].setValue(2.5)
    spin_boxes[0].editingFinished.emit()

    assert stage.parameters.pocket_offset == pytest.approx(2.5)


def test_the_panel_shows_what_the_offset_buys(qapp):
    """The offset no longer sets the clearances, so what the panel has to
    show is the offset itself - it is the number that changes, and the one
    that decides how much bigger every pocket is cut.
    """
    stage = LayoutStage(_quick(pocket_offset=1.0))
    widget = stage.CreateWidget(on_change=lambda: None)

    assert "pockets cut 1.00mm oversize" in _labels(widget)

    spin_box = _widgets(widget, QDoubleSpinBox)[0]
    spin_box.setValue(2.0)
    spin_box.editingFinished.emit()

    assert "pockets cut 2.00mm oversize" in _labels(widget)
    assert "1.20mm dividers" in _labels(widget), "which the offset does not move any more"


# ------------------------------------------------------------------ packing


def test_packing_contours_produces_a_layout(qapp):
    stage = LayoutStage(_quick())

    stage.Run({0: _rectangle(20.0, 10.0), 1: _rectangle(18.0, 12.0)})

    assert stage.layout is not None
    assert stage.layout.cells <= 2
    assert len(stage.parts) == 2


def test_packing_nothing_reports_it_rather_than_raising(qapp):
    stage = LayoutStage(_quick())
    widget = stage.CreateWidget(on_change=lambda: None)

    stage.Run({})
    stage.RefreshStatus()

    assert stage.layout is None
    assert "no contours selected" in _labels(widget)


def test_a_successful_pack_reports_the_grid_size(qapp):
    stage = LayoutStage(_quick())
    widget = stage.CreateWidget(on_change=lambda: None)

    stage.Run({0: _rectangle(20.0, 10.0)})
    stage.RefreshStatus()

    assert "1x1" in _labels(widget)
    assert "1 cells" in _labels(widget)


def test_a_failure_reports_why_rather_than_just_failing(qapp):
    """ "Too small" and "the search gave up" call for different responses
    from the user, and the packer already tells them apart.
    """
    stage = LayoutStage(_quick(max_grid=1))
    widget = stage.CreateWidget(on_change=lambda: None)

    stage.Run({0: _rectangle(200.0, 30.0)})
    stage.RefreshStatus()

    text = _labels(widget)
    assert "no fit" in text
    assert "does not fit" in text


def test_a_pack_that_stepped_up_says_so(qapp):
    """An oversized bin has to stay traceable in the panel too, not just in
    the CLI's report.

    The skipped state is built directly rather than coaxed out of the
    solver: whether a given part set defeats the search is stochastic, and
    a test that only sometimes reaches the branch it is checking is a test
    that passes for the wrong reason.
    """
    stage = LayoutStage(_quick())
    stage.parts = BuildParts({0: _rectangle(20.0, 10.0), 1: _rectangle(18.0, 12.0)})  # only counted here
    stage.result = PackResult(
        Layout(grid=(2, 1), placements={}),
        [GridAttempt((1, 1), NOT_FOUND, "no arrangement in 4 attempts"), GridAttempt((2, 1), PACKED)],
    )

    summary = stage.Summary()

    assert "2 parts in 2x1" in summary
    assert "1x1 was not ruled out" in summary


def test_a_pack_that_found_the_first_feasible_size_makes_no_such_claim(qapp):
    stage = LayoutStage(_quick())
    stage.parts = BuildParts({0: _rectangle(20.0, 10.0)})
    stage.result = PackResult(Layout(grid=(1, 1), placements={}), [GridAttempt((1, 1), PACKED)])

    assert "not ruled out" not in stage.Summary()


def test_repacking_replaces_the_previous_result(qapp):
    stage = LayoutStage(_quick())

    stage.Run({0: _rectangle(20.0, 10.0)})
    stage.Run({})

    assert stage.layout is None
    assert stage.parts == {}


# ----------------------------------------------------------------- rendering


def test_there_is_nothing_to_render_before_packing(qapp):
    stage = LayoutStage(_quick())

    assert stage.Render() is None


def test_a_packed_layout_renders_to_an_image(qapp):
    stage = LayoutStage(_quick())

    stage.Run({0: _rectangle(20.0, 10.0)})
    image = stage.Render()

    assert image is not None
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8


def test_clearing_takes_the_view_back_to_the_photo(qapp):
    # "Show Original" is the way out of the layout view, so Clear has to
    # actually leave nothing to render.
    stage = LayoutStage(_quick())
    stage.Run({0: _rectangle(20.0, 10.0)})

    stage.Clear()

    assert stage.Render() is None
    assert stage.layout is None


# ------------------------------------------------------ off the ui thread


def test_running_a_pack_touches_no_widgets(qapp):
    """Run executes on a worker thread, and a widget written from the wrong
    thread is undefined behavior in Qt rather than merely poor style. So
    the label must still say what it said before the pack.
    """
    stage = LayoutStage(_quick())
    widget = stage.CreateWidget(on_change=lambda: None)
    before = _labels(widget)

    stage.Run({0: _rectangle(20.0, 10.0)})

    assert _labels(widget) == before
    assert stage.layout is not None, "the pack itself should still have happened"


def test_busy_swaps_which_button_is_live(qapp):
    stage = LayoutStage(_quick())
    widget = stage.CreateWidget(on_change=lambda: None, on_cancel=lambda: None)
    pack, cancel = _button(widget, "Pack"), _button(widget, "Cancel")

    assert pack.isEnabled() and not cancel.isEnabled()

    stage.SetBusy(True)
    assert not pack.isEnabled() and cancel.isEnabled()

    stage.SetBusy(False)
    assert pack.isEnabled() and not cancel.isEnabled()


def test_cancel_is_hidden_from_a_host_that_cannot_offer_it(qapp):
    """The CLI-ish case: a host running the pack inline has nothing to
    interrupt, and a dead button is worse than no button.
    """
    stage = LayoutStage(_quick())
    widget = stage.CreateWidget(on_change=lambda: None)

    assert not _button(widget, "Cancel").isVisible()


def test_progress_is_forwarded_to_the_caller(qapp):
    stage = LayoutStage(_quick(max_grid=2))
    seen = []

    stage.Run({0: _rectangle(60.0, 25.0)}, progress=seen.append)

    assert seen, "expected the stage to pass the packer's progress on"
    assert all(report.restarts == stage.parameters.restarts for report in seen)


def test_a_cancelled_pack_says_so_rather_than_claiming_failure(qapp):
    """ "I stopped you" is not evidence that the parts do not fit, and the
    panel must not read as though it were.
    """
    stage = LayoutStage(_quick(max_grid=3))

    stage.Run({0: _rectangle(30.0, 25.0)}, cancelled=lambda: True)

    assert stage.layout is None
    assert stage.result is not None and stage.result.cancelled
    assert "cancelled" in stage.Summary().lower()
    assert "no fit" not in stage.Summary()


def test_the_rotation_mode_is_a_dropdown_that_reaches_the_parameters(qapp):
    """It changes which arrangements are legal, so it belongs beside the
    other things the answer must satisfy rather than only on the command
    line - a window that could not ask for it would make the mode
    unreachable for everyone not running `layout_cli`.
    """
    stage = LayoutStage(_quick())
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
    stage = LayoutStage(_quick())
    runs = []
    widget = stage.CreateWidget(on_change=lambda: runs.append(1))

    _widgets(widget, QComboBox)[0].setCurrentText(FREE_ROTATION)

    assert runs == []
