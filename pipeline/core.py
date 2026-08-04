"""Framework for systematizing pipeline stages.

Each stage should expose:
  * a `parameters` dataclass holding its user-configurable inputs (click
    points, a selected mask, simplification thresholds, etc.)
  * a `Run` method that (re)computes the stage's output from `parameters`
    and whatever upstream data it's given
  * a `CreateWidget` method building a QWidget that edits `parameters`,
    calling back through `on_change` once an edit has settled (e.g. the
    slider is released), not on every intermediate tick

A Pipeline wires stages into a small named dependency graph and, when a
stage's parameters change, reruns that stage and everything downstream of
it.
"""

import os
from typing import Callable, Sequence

from PyQt5.QtCore import Qt, QLibraryInfo
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def FixQtOpenCvPluginPath() -> None:
    """Point Qt's platform-plugin search path at PyQt5's own copy.

    OpenCV bundles its own (usually older) Qt plugins, and having both
    installed can leave `QT_QPA_PLATFORM_PLUGIN_PATH` pointing at the wrong
    one - which crashes plugin loading at startup with a version mismatch.
    Every Qt entry point in this project hits it, so it is fixed once here
    rather than separately in each.

    Call this before constructing a `QApplication` - Qt reads the
    environment variable when it loads the platform plugin, not later.
    """
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)


class Stage:
    def Run(self, *args, **kwargs):
        raise NotImplementedError

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        raise NotImplementedError


def CreateGroupBox(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    """A titled QGroupBox with an empty QVBoxLayout, ready for a stage's
    CreateWidget to add its controls into - keeps every stage's widget
    grouped and labeled the same way, without SVGGui having to know each
    stage's display title. Returns both, since QWidget.layout() gives back
    an Optional, insufficiently-specific QLayout.
    """
    group = QGroupBox(title)
    layout = QVBoxLayout(group)
    return group, layout


class Pipeline:
    """A small named dependency graph of runner callbacks."""

    def __init__(self) -> None:
        self._runners: dict[str, Callable[[], None]] = {}
        self._downstream: dict[str, list[str]] = {}

    def Register(
        self,
        name: str,
        runner: Callable[[], None],
        downstream: Sequence[str] = (),
    ) -> None:
        self._runners[name] = runner
        self._downstream[name] = list(downstream)

    def RunFrom(self, name: str) -> None:
        self._runners[name]()
        for downstream_name in self._downstream[name]:
            self.RunFrom(downstream_name)


def CreateSlider(
    label_text: str,
    min_val: int,
    max_val: int,
    default_val: int,
    on_settle: Callable[[int], None] | None = None,
) -> dict:
    """A labeled slider whose label tracks every tick, but which only calls
    `on_settle` when the user releases it, not on every intermediate drag
    tick.
    """
    layout = QVBoxLayout()
    label = QLabel(f"{label_text} {default_val}")
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(min_val)
    slider.setMaximum(max_val)
    slider.setValue(default_val)

    def update_label(value):
        label.setText(f"{label_text} {value}")

    slider.valueChanged.connect(update_label)
    if on_settle is not None:
        slider.sliderReleased.connect(lambda: on_settle(slider.value()))

    layout.addWidget(label)
    layout.addWidget(slider)

    return {"layout": layout, "slider": slider, "label": label}


def CreateSpinBox(
    label_text: str,
    min_val: float,
    max_val: float,
    default_val: float,
    on_settle: Callable[[float], None] | None = None,
    decimals: int = 2,
    suffix: str = "",
) -> dict:
    """A labeled double spin box that calls `on_settle` when editing
    finishes (Enter pressed or focus lost), not on every keystroke.
    """
    layout = QVBoxLayout()
    label = QLabel(label_text)
    spin_box = QDoubleSpinBox()
    spin_box.setRange(min_val, max_val)
    spin_box.setDecimals(decimals)
    spin_box.setValue(default_val)
    if suffix:
        spin_box.setSuffix(suffix)

    if on_settle is not None:
        spin_box.editingFinished.connect(lambda: on_settle(spin_box.value()))

    layout.addWidget(label)
    layout.addWidget(spin_box)

    return {"layout": layout, "spin_box": spin_box, "label": label}


def CreateChoice(
    label_text: str,
    options: Sequence[str],
    default: str,
    on_change: Callable[[str], None] | None = None,
) -> dict:
    """A labeled combo box over a fixed set of strings, reporting the one
    chosen.

    No settling to wait for, unlike the two above. A slider passes through
    every value on the way to the one meant and a spin box can be typed
    into half-finished, so both hold their callback until the user is done;
    a combo box has no intermediate state to suppress - a selection is
    already the answer.

    `default` must be one of `options`. A value that is not there would
    otherwise select nothing, and the box would read as the first option
    while the parameter it edits still held something else - a control
    silently disagreeing with the thing it controls.
    """
    if default not in options:
        raise ValueError(f"default {default!r} is not one of {list(options)}")

    layout = QVBoxLayout()
    label = QLabel(label_text)
    combo = QComboBox()
    combo.addItems(list(options))
    combo.setCurrentIndex(list(options).index(default))

    if on_change is not None:
        combo.currentTextChanged.connect(on_change)

    layout.addWidget(label)
    layout.addWidget(combo)

    return {"layout": layout, "combo": combo, "label": label}
