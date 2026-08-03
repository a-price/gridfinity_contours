"""Tests for the window screenshots.

Split by cost, and most of it is cost. Taking a picture of the floorplan
window means running the search that window is built around, which is a
minute; the layout one packs four real contours. Both are marked slow.

What is checked cheaply is everything that can be got wrong without
running a window: that the fixtures the pictures name are there, that the
table and the gallery agree, and that a picture is not blank.

Deliberately no comparison against the committed screenshots. That would
be the stored-image assertion `render_demo.py` argues against, and it
would fail on every intentional change to how a window looks - which is
the whole reason these pictures exist.
"""

import os

import numpy as np
import pytest

import screenshot_demo
from screenshot_demo import DRAWERS, LIBRARY, PACKED, SCREENSHOTS, SPOON, Main, Write


def _names() -> list[str]:
    return [name for name, _ in SCREENSHOTS]


# ------------------------------------------------------------------ fixtures


def test_every_fixture_it_names_is_there():
    """These are resolved against the module rather than the working
    directory, so a missing one is a typo rather than a wrong `cd`.
    """
    for path in [*PACKED, *LIBRARY, SPOON]:
        assert os.path.exists(path), f"{path} is missing"


def test_the_drawers_parse_as_millimetres():
    """Typed into the box the way a person would, so a size this script
    cannot enter is one the window would reject too.
    """
    from pipeline.layout.drawer import ParseDrawer

    for text in DRAWERS:
        drawer = ParseDrawer(text)
        assert drawer.cells > 0


def test_the_drawers_have_room_for_more_than_the_answer():
    """Sized with slack on purpose. Drawers that only fit one particular
    grouping would turn any future improvement to the search into a
    picture that says "not placed" - a broken screenshot that still looks
    like a screenshot.
    """
    from pipeline.layout.drawer import ParseDrawer

    space = sum(ParseDrawer(text).cells for text in DRAWERS)

    assert space >= 30, f"only {space} cells for a grouping that takes 25"


def test_the_search_budget_is_below_the_default():
    """The one thing this script changes about the windows it photographs,
    and it is a time budget rather than anything the panel reports.
    """
    from pipeline.layout.parameters import LayoutParameters

    assert screenshot_demo.SCREENSHOT_RESTARTS < LayoutParameters().restarts


def test_the_windows_are_tall_enough_for_their_own_panels(qapp):
    """At 800px the layout window's status label - the line saying what
    the search actually found - was cut off half way through. The windows
    under-report their height because a word-wrapped label's height
    depends on a width the size hint does not know yet, so this is checked
    against the real thing rather than against the hint.
    """
    from layout_gui import LayoutGui

    window = LayoutGui()
    window.resize(*screenshot_demo.WINDOW)
    window.show()
    qapp.processEvents()

    panel = window.export_group
    assert panel.mapTo(window, panel.rect().bottomLeft()).y() <= window.height()


# ------------------------------------------------------------------- gallery


def test_the_gallery_covers_every_window(tmp_path):
    """One picture per front end is the point. A window added without one
    is a window nobody can see before installing the project.
    """
    from pathlib import Path

    guis = {path.name.removesuffix("_gui.py") for path in Path(".").glob("*_gui.py")}

    # The capture window is animated instead - see `capture_demo.py`. A
    # single frame of it would show either an empty window or a finished
    # one, and the interesting thing about that flow is the middle.
    assert guis - set(_names()) == {"silhouette"}


def test_an_unknown_screenshot_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        Main(["--out", str(tmp_path), "--only", "nonesuch"])


# ---------------------------------------------------------------- the images


@pytest.mark.slow
@pytest.mark.parametrize("name", ["layout", "field"])
def test_a_window_picture_shows_a_window(name, tmp_path, qapp):
    """Both halves have to be in it: a control panel, which is light, and
    a drawing area, which this project renders on near-black. A picture
    with only one of them is a window that failed to lay out.
    """
    (path,) = Write(str(tmp_path), names=[name])

    image = _read(path)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.max() > 200, "expected the light control panel"
    assert image.min() < 80, "expected the dark drawing area"


@pytest.mark.slow
def test_the_floorplan_picture_actually_placed_its_bins(tmp_path, qapp):
    """The failure this picture is most likely to fail *silently*. A
    floorplan that could not be assigned still draws - drawers, bins,
    everything - and only the status line says the bins are not in them,
    so a broken screenshot would look almost exactly like a working one.

    Asserted on the window rather than on the picture, since no pixel
    count can tell those two apart either.
    """
    stage = screenshot_demo.PlanFloorplan().floorplan_stage

    assignment = stage.plan.assignment if stage.plan is not None else None
    assert assignment is not None, "the search returned no assignment at all"
    assert assignment.placed, stage.Summary()


def _read(path: str) -> np.ndarray:
    import cv2

    image = cv2.imread(path)
    assert image is not None, f"{path} is not a readable image"
    return image
