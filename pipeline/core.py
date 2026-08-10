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

`Pipeline` holds no Qt at all - it is a dict of callbacks and a recursion.
The controls a stage builds its widget from live in `qt_utils.widgets`.

**Only `silhouette_gui` actually runs one of these.** The capture window
registers eight nodes and re-runs downstream synchronously. The three
`*_stage` modules left in this package subclass `Stage` without being
sequenced by a `Pipeline`: two of them run a cancellable search on a
worker thread, for which "re-run everything downstream as soon as the
stage returns" is precisely wrong. See the note in `layout_gui`.
"""

from typing import Callable, Sequence

from PyQt5.QtWidgets import QWidget


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
