"""Animate the capture window: photo to real-world-scale contour.

    python3 -m demos.capture_demo --out docs/media/capture.gif

The counterpart to `layout_demo.py`, one stage earlier in the pipeline.
That one animates a search; this one animates *using the tool* - the
window in `silhouette_gui.py`, doing the thing it exists for, with nobody at
the keyboard.

**It records the real window rather than reconstructing it.** The frames
are `QWidget.grab()` of an actual `SVGGui`, running the actual SAM2 model
on an actual photograph, with ArUco calibration actually resolving. What
makes that possible without a person is that every interaction in that
window already funnels through a method - `load_image`, the click
recorder, `pipeline.RunFrom` - because the pipeline was built to be driven
by a GUI rather than to *be* one. So a script clicks where a person would
and reads back what a person would see.

That distinction matters more here than for the layout animations. A
picture of the packer is a picture of an algorithm, and rendering it
directly is honest. A picture of a *user flow* is a picture of an
interface, and one assembled out of separately-rendered panels would be a
mock-up of a window rather than a window - it would keep looking right
after the real one had stopped working.

**The photograph has to carry a calibration sheet**, which is why this
demo names one specific file. `test_data/screwdriver.jpg` has all four
ArUco markers in frame, so the contour that comes out the far end is in
millimetres. Run this on a photo without them and `silhouette_gui.py` falls
back to pixel space by design - the animation would still look complete
while quietly dropping the one thing the whole window is for.

Two things this needs that no other demo does: Qt, and a SAM2 checkpoint
already in the local Hugging Face cache. See the `gif-capture` note in
the Makefile.
"""

import argparse
import os
import sys

# Before any Qt import, and before `silhouette_gui` pulls one in transitively.
# Without a platform the window would need a display; with one that is not
# `offscreen` it would pop open on the desktop of whoever ran `make gifs`.
# `--windowed` clears it again for anyone who wants to watch it happen.
if "--windowed" not in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QCheckBox

from export.gif_writer import WriteGif
from pipeline.morphology_stage import LATERAL_LABEL
from pipeline.window_capture import DEFAULT_PASSES, Settled
from silhouette_gui import SVGGui, _MODE_SEGMENT, _MODE_SELECT_CONTOUR

# The one photograph this demo can use, and where to click on it. Both are
# properties of that image rather than settings, so they are named here
# rather than exposed: a different photo needs a different click, and a
# flag inviting one without the other only produces an empty mask.
#
# Resolved against the repository root - one level up from `demos/` - rather
# than the working directory, so the picture comes out identical wherever it
# was generated from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHOTO = os.path.join(_ROOT, "test_data", "screwdriver.jpg")

# Where to click, in the photo's own pixels. Ints, because that is what a
# real click produces - `ClickRecorder` indexes the mask with them.
#
# `HANDLE` is the red grip: one left-click is enough to segment the whole
# tool. `SHADOW` is the drop shadow just under it, right-clicked to mark
# it exterior - the thing a soft shadow on white paper actually costs you.
# Measured, it takes 31k pixels off the mask and the contour from 163.1mm
# long to 162.5mm, which is the shadow's width and is the point.
HANDLE = (2000, 1450)
SHADOW = (2600, 1850)

# Window size for the recording. Smaller than the 1300x800 the window
# opens at, since a GIF in a README is read at a few hundred pixels wide
# and the panel is the part that has to stay legible.
WINDOW = (1200, 900)

# How long each step of the flow holds, in frames. The work here is not
# animated - SAM2 either has a mask or does not - so the pacing is
# entirely about reading time, and the steps deserve different amounts of
# it. The contour is the answer and holds longest.
HOLD = 8
HOLD_FINAL = 20

DEFAULT_MS_PER_FRAME = 120

# Far more colours than the layout animations use. Those are line drawings
# on white, where 16 is generous and dithering is the enemy; this is a
# photograph of a red screwdriver on graph paper, and at 16 it came out
# uniformly brown with the ArUco markers barely legible - which would make
# the animation evidence for nothing, since whether the markers are
# readable is the question calibration answers.
PHOTO_COLORS = 128


def _Check(window: SVGGui, label: str) -> None:
    """Tick the checkbox with that caption, as a person would.

    Found by caption because the stages build their own controls and hand
    back a group box, so there is no attribute to reach for - the caption
    is the only handle, and it comes from the same constant the widget was
    built from. Toggling rather than setting the parameter behind it is
    what makes the box appear ticked *and* runs the stage, since the
    widget's own handler does both.
    """
    matches = [box for box in window.findChildren(QCheckBox) if box.text() == label]
    if not matches:
        raise ValueError(f"no checkbox captioned {label!r} in the capture window")
    matches[0].setChecked(True)


class Recording:
    """Frames of the window, in the order they were taken."""

    def __init__(self, window: SVGGui, application: QApplication) -> None:
        self._window = window
        self._application = application
        self.frames: list[np.ndarray] = []

    def Hold(self, frames: int = HOLD) -> None:
        """Take one frame, once the window has settled, and repeat it.

        Repeated rather than paused, because `WriteGif` collapses a run of
        identical frames into one long one - so a hold costs a few bytes
        rather than a copy of the window per frame.
        """
        self.frames.extend([self._Settled()] * frames)

    def _Settled(self, passes: int = DEFAULT_PASSES) -> np.ndarray:
        """The window once it has stopped changing - see `window_capture`.

        Worth a wrapper of its own rather than calling through, because
        every frame in this animation goes through it and getting it wrong
        put the whole recording one beat behind itself.
        """
        return Settled(self._window, self._application, passes)


def Record() -> "tuple[SVGGui, Recording]":
    """Drive the whole flow, returning the window and the frames taken.

    The steps are the ones `silhouette_gui.py`'s own docstring lists, in the
    order a person does them, each held long enough to read.

    Separate from writing the GIF so that what the flow *produced* can be
    checked - the animation's whole claim is that it ends in millimetres,
    and a test of the encoded frames cannot see that.
    """
    application = QApplication.instance() or QApplication(sys.argv[:1])
    assert isinstance(application, QApplication)

    window = SVGGui(local_files_only=True)
    window.resize(*WINDOW)
    window.show()

    recording = Recording(window, application)
    recording.Hold()  # an empty window, so the photo arriving is visible

    # Load. Calibration runs off the back of this, which is why the panel
    # reads 4/4 a frame later without anything else being pressed.
    window.load_image(PHOTO)
    recording.Hold()

    # The clicks, drawn but not yet acted on. Left on the handle, right on
    # the shadow under it - the two inputs the whole capture takes.
    #
    # Injected into the recorder rather than posted as QMouseEvents,
    # because the widget-to-image mapping depends on how the pixmap was
    # scaled into a label that has never been laid out on a real screen,
    # and the click's *position in the photo* is what this is
    # demonstrating, not Qt's coordinate arithmetic.
    #
    # A beat of its own, which is only worth having because click markers
    # are now sized in screen pixels: they used to be sized against the
    # photo, landing on four screen pixels for a 5184px image, and this
    # frame showed nothing at all.
    window.interaction_mode_combo.setCurrentText(_MODE_SEGMENT)
    clicks = window.segmenter_stage.click_recorder

    # The recorder only exists once a photo has been loaded, which it has
    # by now - asserted rather than assumed so that loading silently
    # failing shows up here instead of as an empty mask three steps later.
    assert clicks is not None, "no click recorder; the photo did not load"
    clicks.image_points.extend([list(HANDLE), list(SHADOW)])
    clicks.image_labels.extend([1, 0])
    window.pipeline.RunFrom("display")
    recording.Hold()

    # Segment. The mask reaches the screen as a contour drawn round what
    # it found, in yellow because nothing is selected yet.
    window.pipeline.RunFrom("segmentation")
    recording.Hold()

    # Mirror the mask across its long axis and union the two, which is
    # what lateral symmetry with "or" means. A screwdriver is symmetric
    # about that axis and the lighting is not, so the side in shadow
    # segments a little thin; taking the wider of the two sides at every
    # point puts back what the shadow cost. Measured on this photo: 11912
    # pixels of mask returned, and the simplified outline drops from 73
    # points to 67 as one-sided jitter stops being a feature to trace.
    #
    # Ticked through the checkbox rather than by setting the parameter,
    # because the panel is half the picture. An outline that reshaped
    # itself next to an unticked box would read as the tool doing
    # something on its own.
    _Check(window, LATERAL_LABEL)
    recording.Hold()

    # Select it, which fills it green and triggers the rectification that
    # puts millimetres in the text panel. One object in frame, so there is
    # one contour to select.
    window.interaction_mode_combo.setCurrentText(_MODE_SELECT_CONTOUR)
    window.contour_selection_stage.contour_selection.selected.add(0)
    window.pipeline.RunFrom("selection")
    recording.Hold(HOLD_FINAL)

    return window, recording


def Capture(out: str, milliseconds: int = DEFAULT_MS_PER_FRAME, colors: int = PHOTO_COLORS) -> int:
    """Record the flow and write it to `out`. Returns the frame count."""
    _, recording = Record()
    WriteGif(out, recording.frames, milliseconds, colors)
    return len(recording.frames)


def Main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="capture.gif", help="output GIF (default: capture.gif)")
    parser.add_argument(
        "--ms",
        type=int,
        default=DEFAULT_MS_PER_FRAME,
        help=f"milliseconds per frame (default: {DEFAULT_MS_PER_FRAME})",
    )
    parser.add_argument("--colors", type=int, default=PHOTO_COLORS, help=f"palette size (default: {PHOTO_COLORS})")
    parser.add_argument("--windowed", action="store_true", help="show the window instead of rendering it offscreen")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if not os.path.exists(PHOTO):
        print(f"{PHOTO} is missing; it is the only photo here with a calibration sheet in frame", file=sys.stderr)
        return 1

    # Deliberately stopping at selection rather than pressing Export. The
    # contour is already on screen in millimetres by then, so the export
    # would add a frame identical to the one before it - and would write
    # three files into whatever directory `make gifs` was run from.
    frames = Capture(args.out, args.ms, args.colors)

    print(f"wrote {args.out} ({frames} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(Main())
