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

from typing import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDoubleSpinBox, QLabel, QSlider, QVBoxLayout, QWidget


class Stage:
    def Run(self, *args, **kwargs):
        raise NotImplementedError

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        raise NotImplementedError


class Pipeline:
    """A small named dependency graph of runner callbacks."""

    def __init__(self) -> None:
        self._runners: dict[str, Callable[[], None]] = {}
        self._downstream: dict[str, list[str]] = {}

    def Register(
        self, name: str, runner: Callable[[], None], downstream: list[str] = ()
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
