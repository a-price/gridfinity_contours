"""Tests for the shared widget helpers.

The first test module these helpers have had. Sliders and spin boxes were
always exercised through whatever builds them - the capture stages build
the sliders, the panels build the spin boxes, and `layout_panel_test`
drives every `QDoubleSpinBox` in its panel - and that stayed adequate
because neither helper has behaviour a caller cannot reach.

`CreateChoice` does: it refuses a default that is not among its options,
and no caller can ask for one. Covering it from here rather than leaving
it untested is the reason this file exists; the untested history of the
other two is noted rather than backfilled, since a real panel driving the
real widget is the better test where it works.
"""

import pytest
from PyQt5.QtWidgets import QComboBox

from qt_utils.widgets import CreateChoice


def test_a_choice_starts_on_the_given_default(qapp):
    control = CreateChoice("Mode:", ("a", "b", "c"), "b")

    assert isinstance(control["combo"], QComboBox)
    assert control["combo"].currentText() == "b"


def test_a_choice_offers_every_option_in_order(qapp):
    combo = CreateChoice("Mode:", ("a", "b", "c"), "a")["combo"]

    assert [combo.itemText(index) for index in range(combo.count())] == ["a", "b", "c"]


def test_choosing_reports_the_text_rather_than_the_index(qapp):
    """An index would make every caller re-look-up what it meant, and a
    reordered option list would silently change what a saved setting
    selects.
    """
    chosen = []
    combo = CreateChoice("Mode:", ("a", "b", "c"), "a", chosen.append)["combo"]

    combo.setCurrentText("c")

    assert chosen == ["c"]


def test_a_default_outside_the_options_is_refused(qapp):
    """Qt would select nothing and display the first option, so the control
    would read as "a" while the parameter it edits still held something
    else - a control silently disagreeing with the thing it controls.
    """
    with pytest.raises(ValueError, match="is not one of"):
        CreateChoice("Mode:", ("a", "b"), "z")


def test_a_choice_without_a_callback_is_still_usable(qapp):
    """The read-only case: some panels want the box to show a setting
    without wiring an edit to it, and connecting nothing must not raise.
    """
    combo = CreateChoice("Mode:", ("a", "b"), "a")["combo"]

    combo.setCurrentText("b")

    assert combo.currentText() == "b"
