"""Framework for systematizing pipeline stages.

Each stage should expose:
  * a `parameters` dataclass holding its user-configurable inputs (click
    points, a selected mask, simplification thresholds, etc.)
  * a `Run` method that (re)computes the stage's output from `parameters`
    and whatever upstream data it's given
  * a `CreateWidget` method building a QWidget that edits `parameters`,
    calling back through `on_change` once an edit has settled (e.g. after a
    slider stops moving for a moment), not on every intermediate tick

A Pipeline wires stages into a small named dependency graph and, when a
stage's parameters change, reruns that stage and everything downstream of
it.
"""

from typing import Callable

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLabel, QSlider, QVBoxLayout, QWidget

DEBOUNCE_MS = 300


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


def Debounce(callback: Callable[[], None], delay_ms: int = DEBOUNCE_MS) -> Callable[[], None]:
    """Wrap `callback` so a burst of calls collapses into a single call,
    `delay_ms` after the last one - e.g. a flurry of clicks or slider ticks
    that should only trigger one (possibly expensive) recompute once
    they've settled.

    Returns a zero-arg trigger function. Keep a reference to it (or to
    whatever holds it) for as long as debouncing should keep working: the
    underlying QTimer is only kept alive by that reference.
    """
    timer = QTimer()
    timer.setSingleShot(True)
    timer.setInterval(delay_ms)
    timer.timeout.connect(callback)

    def trigger():
        timer.start()

    trigger.timer = timer
    return trigger


def CreateSlider(
    label_text: str,
    min_val: int,
    max_val: int,
    default_val: int,
    on_settle: Callable[[int], None] | None = None,
    debounce_ms: int = DEBOUNCE_MS,
) -> dict:
    """A labeled slider whose label tracks every tick, but which only calls
    `on_settle` once the value has held steady for `debounce_ms` - covering
    drags, keyboard nudges, and programmatic changes alike, without
    re-running anything on every intermediate tick.
    """
    layout = QVBoxLayout()
    label = QLabel(f"{label_text} {default_val}")
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(min_val)
    slider.setMaximum(max_val)
    slider.setValue(default_val)

    debounced_settle = (
        Debounce(lambda: on_settle(slider.value()), debounce_ms)
        if on_settle is not None
        else None
    )

    def update_label(value):
        label.setText(f"{label_text} {value}")
        if debounced_settle is not None:
            debounced_settle()

    slider.valueChanged.connect(update_label)

    layout.addWidget(label)
    layout.addWidget(slider)

    return {"layout": layout, "slider": slider, "label": label}
