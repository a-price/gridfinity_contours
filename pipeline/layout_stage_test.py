"""Tests for the layout GUI stage (M6).

The load-bearing property here is that packing runs *only* when asked.
Everything else in the panel reruns on a settled edit; this one takes
seconds, so a stage that quietly packed on a parameter change would lock
the window on every tick of a spin box.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QPushButton

from pipeline.layout.energy import LayoutParameters
from pipeline.layout.loading import BuildParts
from pipeline.layout.packer import NOT_FOUND, PACKED, GridAttempt, PackResult
from pipeline.layout.placement import Layout
from pipeline.layout_stage import LayoutStage


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


def _quick(**overrides) -> LayoutParameters:
    settings = dict(restarts=4, iterations=120, patience=25, max_grid=2)
    settings.update(overrides)
    return LayoutParameters(**settings)


def _widgets(widget, kind):
    return widget.findChildren(kind)


def _labels(widget) -> str:
    return "\n".join(label.text() for label in _widgets(widget, QLabel))


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

    (button,) = _widgets(widget, QPushButton)
    button.click()

    assert packs == [1]


def test_parameter_edits_still_reach_the_parameters(qapp):
    stage = LayoutStage(_quick())
    widget = stage.CreateWidget(on_change=lambda: None)
    spin_boxes = _widgets(widget, QDoubleSpinBox)

    spin_boxes[0].setValue(2.5)
    spin_boxes[0].editingFinished.emit()

    assert stage.parameters.pocket_offset == pytest.approx(2.5)


def test_the_panel_shows_the_clearances_the_offset_implies(qapp):
    """Clearances are derived, not typed (D5) - so the panel has to show
    what the offset actually bought, or the number the user cares about is
    invisible.
    """
    stage = LayoutStage(_quick(pocket_offset=1.0))
    widget = stage.CreateWidget(on_change=lambda: None)

    assert "3.20mm between parts" in _labels(widget)

    spin_box = _widgets(widget, QDoubleSpinBox)[0]
    spin_box.setValue(2.0)
    spin_box.editingFinished.emit()

    assert "5.20mm between parts" in _labels(widget)


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

    assert stage.layout is None
    assert "no contours selected" in _labels(widget)


def test_a_successful_pack_reports_the_grid_size(qapp):
    stage = LayoutStage(_quick())
    widget = stage.CreateWidget(on_change=lambda: None)

    stage.Run({0: _rectangle(20.0, 10.0)})

    assert "1x1" in _labels(widget)
    assert "1 cells" in _labels(widget)


def test_a_failure_reports_why_rather_than_just_failing(qapp):
    """"Too small" and "the search gave up" call for different responses
    from the user, and the packer already tells them apart.
    """
    stage = LayoutStage(_quick(max_grid=1))
    widget = stage.CreateWidget(on_change=lambda: None)

    stage.Run({0: _rectangle(200.0, 30.0)})

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


# -------------------------------------------------------------- reentrancy


def test_pack_is_disabled_while_packing(qapp):
    """The status label pumps the event loop so "packing..." actually
    paints, which would otherwise let a second click re-enter Run.
    """
    stage = LayoutStage(_quick())
    widget = stage.CreateWidget(on_change=lambda: None)
    (button,) = _widgets(widget, QPushButton)
    seen = []

    original = stage._SetBusy

    def record(busy: bool) -> None:
        seen.append((busy, button.isEnabled()))
        original(busy)

    stage._SetBusy = record
    stage.Run({0: _rectangle(20.0, 10.0)})

    assert [busy for busy, _ in seen] == [True, False]
    assert button.isEnabled(), "the button has to come back afterwards"
