"""Tests for the distance-field viewer's Qt adapter.

The split this pins is which controls change the field and which change
only the picture of it. Getting that backwards is invisible on screen -
both redraw - and shows up as a viewer that quietly keeps displaying a
raster built at the previous resolution.
"""

import numpy as np
import pytest
from PyQt5.QtWidgets import QCheckBox, QDoubleSpinBox, QLabel

from panels.field_panel import GRADIENT_LABEL, SAMPLES_LABEL, FieldPanel
from layout.field import FieldView
from layout.parameters import LayoutParameters
from conftest import Rectangle as _rectangle


@pytest.fixture
def panel():
    return FieldPanel()


def _contours() -> dict[int, np.ndarray]:
    return {0: _rectangle(20.0, 10.0), 1: _rectangle(40.0, 30.0)}


def _widgets(widget, kind):
    return widget.findChildren(kind)


def _labels(widget) -> str:
    return "\n".join(label.text() for label in _widgets(widget, QLabel))


def _object_span(part) -> np.ndarray:
    """The object a part was built for, as a width and height.

    Which contour is on screen is a question about the shape somebody
    drew. `part.size` answers a different one - a part is its *pocket*, so
    at the default offset it is a millimetre larger on every side, and a
    test that reads it is asserting `pocket`'s arithmetic rather than this
    panel's selection.
    """
    return part.object_contour.max(axis=0) - part.object_contour.min(axis=0)


def _checkbox(widget, text: str) -> QCheckBox:
    (found,) = [box for box in _widgets(widget, QCheckBox) if box.text() == text]
    return found


# ------------------------------------------------------------- what is built


def test_only_the_selected_contour_is_rasterized():
    """Rasterizing all eighteen of somebody's test_data contours on every
    tick of a spin box would make the panel feel broken, and seventeen of
    them are not on screen.
    """
    panel = FieldPanel()

    panel.Run(_contours())

    assert panel.selected == 0
    assert panel.part is not None
    assert _object_span(panel.part) == pytest.approx([20.0, 10.0])


def test_selecting_another_contour_shows_it(panel):
    panel.Run(_contours())

    panel.Select(1)

    assert panel.part is not None
    assert _object_span(panel.part) == pytest.approx([40.0, 30.0])


def test_the_selection_survives_a_reload(panel):
    """Every parameter edit re-runs the panel, so a selection reset by Run
    would snap back to the first contour on each nudge of the resolution.
    """
    panel.Run(_contours())
    panel.Select(1)

    panel.Run(_contours())

    assert panel.selected == 1


def test_a_selection_that_is_gone_falls_back(panel):
    panel.Run(_contours())
    panel.Select(1)

    panel.Run({0: _rectangle(20.0, 10.0)})

    assert panel.selected == 0
    assert panel.part is not None


def test_nothing_loaded_renders_nothing(panel):
    assert panel.Render() is None
    assert "nothing loaded" in panel.Summary()


def test_clearing_drops_the_field(panel):
    panel.Run(_contours())

    panel.Clear()

    assert panel.part is None
    assert panel.Render() is None


def test_a_contour_too_degenerate_to_rasterize_is_reported_not_raised(panel):
    """These arrive through a file dialog - a stray two-point polygon in
    an SVG should not take the window down.
    """
    panel.Run({0: np.array([[0.0, 0.0], [1.0, 0.0]])})

    assert panel.part is None
    assert panel.error is not None
    assert "could not rasterize" in panel.Summary()


# -------------------------------------------------- field versus its picture


def test_the_resolution_rebuilds_the_raster(panel):
    panel.Run(_contours())
    coarse = panel.part
    assert coarse is not None

    panel.parameters.resolution = 0.5
    panel._Build()

    assert panel.part is not None
    assert panel.part.resolution == 0.5
    assert panel.part.sdf.shape[0] < coarse.sdf.shape[0]


def test_the_pocket_offset_rebuilds_the_raster_around_a_bigger_pocket(panel):
    """The offset used to reach the raster the long way round: clearances
    were derived from it, `pad` from those, and the test was that the
    field stayed long enough to reach the rings being drawn. Since D5 it
    reaches by the short road instead - this panel rasterizes the
    *pocket*, so the offset is in the geometry itself and `pad` no longer
    moves with it at all. An offset wired only to the labels would leave
    both of these unchanged.
    """
    panel.Run(_contours())
    assert panel.part is not None
    object_span = _object_span(panel.part)
    pad = panel.part.pad

    panel.parameters.pocket_offset = 4.0
    panel._Build()

    assert panel.part is not None
    assert _object_span(panel.part) == pytest.approx(object_span), "the object is not what grew"
    assert (panel.part.size > object_span + 8.0).all(), "the pocket is, by twice the offset"
    assert panel.part.pad == pad, "pad is derived from the clearances, which no longer carry the offset"


def test_the_view_settings_change_only_the_picture(panel):
    panel.Run(_contours())
    part = panel.part

    panel.view.pixels_per_mm = 8.0
    panel.view.gradient = True

    assert panel.part is part, "a redraw must not rebuild the raster"
    assert panel.Render() is not None


def test_the_scale_changes_the_rendered_size(panel):
    panel.Run(_contours())
    coarse = panel.Render()

    panel.view.pixels_per_mm = 8.0
    fine = panel.Render()

    assert coarse is not None and fine is not None
    assert fine.shape[1] > coarse.shape[1]


# -------------------------------------------------------------- the readout


def test_a_probe_reports_millimeters_of_clearance(panel):
    panel.Run({0: _rectangle(20.0, 10.0)})
    part = panel.part
    assert part is not None

    # Three millimeters above the top edge, in image pixels.
    low = part.origin + 0.5 * part.resolution
    pixel = tuple((np.array([10.0, 13.0]) - low) * panel.view.pixels_per_mm)

    text = panel.Probe(pixel)

    assert "clear" in text
    assert "3.0" in text or "2.9" in text or "3.1" in text


def test_a_probe_inside_the_part_says_so(panel):
    panel.Run({0: _rectangle(20.0, 10.0)})
    part = panel.part
    assert part is not None

    low = part.origin + 0.5 * part.resolution
    pixel = tuple((np.array([10.0, 5.0]) - low) * panel.view.pixels_per_mm)

    assert "inside the part" in panel.Probe(pixel)


def test_the_probe_describes_whichever_view_is_up(panel):
    """A readout in millimeters beside a picture of gradient length would
    be two answers to different questions sitting next to each other.
    """
    panel.Run({0: _rectangle(20.0, 10.0)})
    panel.view.gradient = True

    assert "gradient" in panel.Probe((20.0, 20.0))


def test_probing_with_nothing_loaded_is_empty(panel):
    assert panel.Probe((0.0, 0.0)) == ""


# --------------------------------------------------------------- the summary


def test_the_summary_is_a_key_to_the_colors(panel):
    panel.Run(_contours())

    summary = panel.Summary()

    assert f"{panel.parameters.c_pair_enforced:.2f}mm" in summary
    assert f"{panel.parameters.spacing_pair:.2f}mm" in summary
    assert "0.25mm/px" in summary


def test_the_summary_follows_the_view(panel):
    panel.Run(_contours())
    panel.view.gradient = True

    summary = panel.Summary()

    assert "creases" in summary
    assert "no other outline" not in summary, "the rings are not drawn on this view"


# ----------------------------------------------------------------- the panel


def test_every_control_redraws_on_a_settled_edit(qapp, panel):
    """The opposite of LayoutPanel, where a redraw costs seconds and so
    runs only on a button. Here it costs milliseconds, and watching the
    field move as the resolution does is the point of the tool.
    """
    redraws = []
    panel.Run(_contours())
    widget = panel.CreateWidget(on_change=lambda: redraws.append(1))

    for spin_box in _widgets(widget, QDoubleSpinBox):
        spin_box.setValue(spin_box.value() + 0.5)
        spin_box.editingFinished.emit()
    _checkbox(widget, GRADIENT_LABEL).setChecked(True)
    _checkbox(widget, SAMPLES_LABEL).setChecked(True)

    assert panel.view.gradient
    assert panel.view.samples
    assert len(redraws) == len(_widgets(widget, QDoubleSpinBox)) + 2


def test_a_rebuilding_edit_rebuilds_before_it_redraws(qapp, panel):
    """The redraw reads `panel.part`, so an edit that changed the raster
    without rebuilding it first would draw one frame of the old field.
    """
    panel.Run(_contours())
    seen = []
    widget = panel.CreateWidget(on_change=lambda: seen.append(panel.part))

    resolution = _widgets(widget, QDoubleSpinBox)[0]
    resolution.setValue(0.5)
    resolution.editingFinished.emit()

    assert seen[-1] is not None
    assert seen[-1].resolution == 0.5


def test_the_panel_reports_the_field_it_is_showing(qapp):
    panel = FieldPanel(LayoutParameters(), FieldView(pixels_per_mm=8.0))
    panel.Run(_contours())
    widget = panel.CreateWidget(on_change=lambda: None)

    panel.RefreshStatus()

    assert "contour 0" in _labels(widget)
