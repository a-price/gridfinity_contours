"""What a capture stage is, and what re-runs it.

Each stage exposes:
  * a `Run` method that (re)computes its output from its parameters and
    whatever upstream data it is given
  * a `CreateWidget` method building a QWidget that edits those
    parameters, calling back through `on_change` once an edit has settled
    (e.g. the slider is released), not on every intermediate tick

Where the parameters live is the stage's own business. Four of the five
hold an algorithm object and edit *its* `parameters` dataclass, which is
what keeps `morphology.py` testable without a display; `SvgExportStage`
has no algorithm behind it and owns one directly.

A Pipeline wires stages into a small named dependency graph and, when a
stage's parameters change, reruns that stage and everything downstream of
it.

`Pipeline` holds no Qt at all - it is a dict of callbacks and a recursion.
The controls a stage builds its widget from live in `qt_utils.widgets`.

**Only `silhouette_gui` runs one of these**, over the five stages in
`capture`. The planner windows do not: their work takes minutes on a
worker thread, for which "re-run everything downstream as soon as the
stage returns" is precisely wrong, so a finished signal sequences them
instead. What they own is a `panels.*Panel`, which deliberately does not
subclass `Stage`.
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
