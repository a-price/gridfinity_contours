"""One picture of each window, for the README.

    python3 -m demos.screenshot_demo --out docs/media

The still-image counterpart to `capture_demo.py`. That one animates a
flow; this one photographs the *tools*, one picture per window, so the
front page can show what each of them looks like instead of describing it.

**These are grabs of the real windows.** Every picture here is a
`QWidget.grab()` of an actual `LayoutGui`, `FloorplanGui` or `FieldGui`,
driven offscreen through the same methods a person's clicks reach: load
these files, press Pack, wait. Nothing is mocked up, and nothing is drawn
by hand, so a picture that looks wrong means the window is wrong.

That matters more than it would for a picture of a drawing. `render_demo.py`
already photographs every renderer, and those pictures are the *contents*
of these windows - the sheet, the field, the floorplan. What a screenshot
adds is the half that is not a rendering: the panel. The sizes, the
status lines, the units, the buttons somebody has to find. None of that
can be reconstructed from the pipeline, because none of it is something
the pipeline produces.

Not committed as assertions - see `render_demo.py`'s docstring, which
argues the case at length for exactly these artifacts. A change to how a
window looks arrives here as a diff somebody reads.

Needs Qt, which it drives offscreen so nothing pops open on whoever ran
`make screenshots`. `--windowed` shows the windows instead, for watching
it happen. Unlike `capture_demo.py` it needs no model and no photograph:
every window here starts from contour files, so this is seconds rather
than a download.
"""

import argparse
import os
import sys
from dataclasses import replace

# Before any Qt import, and before the windows pull one in transitively.
# Without a platform these would need a display; with one that is not
# `offscreen` they would pop open on the desktop of whoever ran this.
if "--windowed" not in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Callable

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget

from pipeline.window_capture import Settled

DEFAULT_OUT = "docs/media"

# Prefix, so these sort together beside the animations and a glob can find
# them without knowing their names - the same convention `render_demo.py`
# uses for the drawings.
PREFIX = "window_"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _Path(*parts: str) -> str:
    """A fixture path, resolved against the repository root - one level up
    from `demos/` - rather than the working directory, so the pictures come
    out identical wherever they were generated from.
    """
    return os.path.join(_ROOT, *parts)


# The same four objects `make gif-pack` animates. Deliberately the same:
# the animation shows the search moving them and this shows where it
# stopped, so a reader can put the two together.
PACKED = [
    _Path("test_data", "small_spoon.svg"),
    _Path("test_data", "medium_spoon.svg"),
    _Path("test_data", "big_spoon.svg"),
    _Path("test_data", "medium_fork.svg"),
]

# The six objects `make gif-group` groups, and two drawers with room to
# spare. The same library on purpose: the animation shows the grouping
# search deciding what shares a bin, and this shows where those bins ended
# up, so the two can be read together.
LIBRARY = [
    _Path("test_data", "big_spoon.svg"),
    _Path("test_data", "small_spoon.svg"),
    _Path("test_data", "screwdriver.svg"),
    _Path("test_data", "spreader.svg"),
    _Path("test_data", "big_measure.svg"),
    _Path("test_data", "small_measure.svg"),
]

# Two real drawers, in millimetres, the way somebody would type them. Sized
# with slack rather than to the exact answer: the grouping is stochastic,
# and drawers that only fit one particular grouping would turn any future
# improvement to the search into a picture that says "not placed".
DRAWERS = ("210x170", "210x130")

# How hard the picture's search looks. Lower than the default 24, and
# measured rather than guessed: on this library restarts=8 reaches the
# same 25-cell grouping the full budget does, in about a minute rather
# than several. Restarts are not shown in the panel, so this changes how
# long the picture takes and nothing the picture claims.
SCREENSHOT_RESTARTS = 8

# The capture fixture with the most interesting field, for the same reason
# `render_demo.py` picks it: a spoon's bowl is concave enough that its
# dilation folds into itself.
SPOON = _Path("test_data", "big_spoon.svg")

# Big enough that the control panel reads at README width, small enough
# that it does not need scaling down to get there.
#
# The height is set by the panel rather than by the picture: these windows
# report a size hint that ignores their word-wrapped status labels, so at
# 800px the line that says what the search found was cut off half way
# through - a screenshot advertising a status the window cannot show.
WINDOW = (1200, 900)

# The floorplan window gets more of both. Its panel has three group boxes
# the others do not, and its drawing is every drawer side by side, which
# is the widest thing anything here draws. At this height the cut falls
# below the pinning controls, so the picture ends at a group boundary
# rather than through the middle of a list.
FLOORPLAN_WINDOW = (1400, 1300)

# How many of the plan's bins to tick in the floorplan picture. One, so
# the drawing shows both states at once - a pinned bin beside an unpinned
# one is what makes the green outline mean something.
PINNED_IN_PICTURE = 1


# The one QApplication, and every window taken so far. Both are held at
# module scope on purpose: a QApplication whose last Python reference goes
# out of scope takes its C++ object with it, and the next window built
# against the dangling pointer aborts with "Must construct a QApplication
# before a QWidget" - which is what happened between the first picture and
# the second. The windows are kept for the same reason, since a parentless
# QWidget is owned by whoever holds the reference.
_APPLICATION: "QApplication | None" = None
_WINDOWS: list[QWidget] = []


def _Application() -> QApplication:
    global _APPLICATION

    if _APPLICATION is None:
        existing = QApplication.instance()
        _APPLICATION = existing if isinstance(existing, QApplication) else QApplication(sys.argv[:1])
    return _APPLICATION


def _Shown(window: QWidget, size: tuple[int, int] = WINDOW) -> QWidget:
    _WINDOWS.append(window)
    window.resize(*size)
    window.show()
    return window


def LayoutWindow() -> np.ndarray:
    """`layout_gui.py` with four objects packed into one bin.

    Packed for real rather than loaded from a fixture, so the status line
    says what the search actually found and the picture cannot drift from
    it. The seed is fixed by `LayoutParameters`, so the same code packs
    these the same way every time.
    """
    from layout_gui import LayoutGui

    application = _Application()
    window = _Shown(LayoutGui())

    window.load_contours(PACKED)
    window.pack()
    window.WaitForPack()

    return Settled(window, application)


def PlanFloorplan():
    """Drive `floorplan_gui.py` through a whole plan, returning the window.

    Separate from photographing it for the same reason `capture_demo.Record`
    is separate from writing its GIF: the thing most likely to go wrong
    here is that the bins do not fit the drawers, and a floorplan that
    could not be assigned still *draws* - drawers, bins and all - with only
    the status line saying so. No picture of it can be told from a working
    one, but the window itself can.
    """
    from floorplan_gui import FloorplanGui

    _Application()
    window = _Shown(FloorplanGui(), FLOORPLAN_WINDOW)
    stage = window.floorplan_stage
    stage.parameters = replace(stage.parameters, restarts=SCREENSHOT_RESTARTS)

    window.load_contours(LIBRARY)

    # Typed into the box and added, as a person would, so the list shows
    # each drawer in both units - which is the one thing about entering a
    # drawer that is worth seeing.
    for text in DRAWERS:
        window.drawer_edit.setText(text)
        window.add_drawer()

    window.plan()
    window.WaitForPlan()

    # Through the list widget rather than `stage.Pin`, so the ticks in the
    # panel and the outlines in the drawing come from the same act - a
    # picture showing green bins beside unticked boxes would be evidence
    # of a bug rather than of a feature.
    for row in range(min(PINNED_IN_PICTURE, window.pin_list.count())):
        item = window.pin_list.item(row)
        if item is not None:
            item.setCheckState(Qt.CheckState.Checked)

    return window


def FloorplanWindow() -> np.ndarray:
    """`floorplan_gui.py` with six objects grouped into bins and laid into
    two drawers, one bin pinned.

    Planned for real, which makes this the slowest picture here by a wide
    margin - the search this window is built around is a discrete one whose
    cost function is a stochastic packer. Loading a saved session instead
    would take a second, but it would show a floorplan somebody else had
    found rather than one this code just found.

    One bin pinned, because the heavy green outline is the thing about this
    window that a description does not convey, and a pinned bin beside an
    unpinned one is what makes the outline mean something.
    """
    return Settled(PlanFloorplan(), _Application())


def FieldWindow() -> np.ndarray:
    """`field_gui.py` showing one spoon's distance field.

    At the window's own defaults, which means a crop rather than the whole
    part: this view is shown at 1:1 and scrolled on purpose, because every
    annotation it draws is one pixel wide and scaling the field down to
    fit would drop most of them. The crop is the honest picture of that
    decision.
    """
    from field_gui import FieldGui

    application = _Application()
    window = _Shown(FieldGui())

    window.load_contours([SPOON])

    return Settled(window, application)


# One entry per picture, as a table rather than a sequence of calls, so
# that adding a window to the gallery is a line - and so the test that
# keeps the gallery honest can walk it.
SCREENSHOTS: tuple[tuple[str, Callable[[], np.ndarray]], ...] = (
    ("layout", LayoutWindow),
    ("floorplan", FloorplanWindow),
    ("field", FieldWindow),
)


def Write(out: str, names: "list[str] | None" = None) -> list[str]:
    """Take every screenshot into `out`, returning what was written.

    PNG because these are compared by eye and by `git diff`, and a lossy
    format would show its own compression noise as though it were a change
    to the window.
    """
    wanted = set(names) if names else None

    written = []
    os.makedirs(out, exist_ok=True)
    for name, grab in SCREENSHOTS:
        if wanted is not None and name not in wanted:
            continue
        path = os.path.join(out, f"{PREFIX}{name}.png")
        if not cv2.imwrite(path, grab()):
            raise OSError(f"could not write {path}")
        written.append(path)
    return written


def Main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"directory to write into (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--only",
        action="append",
        metavar="NAME",
        choices=[name for name, _ in SCREENSHOTS],
        help="take just this screenshot; repeat for several",
    )
    parser.add_argument("--windowed", action="store_true", help="show the windows instead of rendering them offscreen")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    for path in Write(args.out, names=args.only):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(Main())
