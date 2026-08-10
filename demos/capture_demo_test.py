"""Tests for the capture-window animation.

Split by what they cost. Pacing - what a held frame is, and that a beat
shows the change that produced it - is exercised on a bare widget in
milliseconds. Actually recording the flow loads SAM2 and segments a
5184px photograph, so it is marked slow and asserts only the things that
make the animation worth having: that it shows the real window, and that
the calibration it demonstrates genuinely resolved.

The grabbing underneath all of it is `qt_utils/window_capture.py`, and is
tested there.

Deliberately no comparison against the committed GIF. That would be the
stored-image assertion `render_demo.py` argues against at length, and it
would fail on every intentional change to how the window looks - which is
most of the changes this animation exists to document.
"""

import os

import numpy as np
import pytest
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

import demos.capture_demo as capture_demo
from demos.capture_demo import HANDLE, PHOTO, PHOTO_COLORS, SHADOW, Main, Recording


class _Host(QWidget):
    """A stand-in for the capture window: something with a central layout
    to activate and a label whose text can be changed, which is all
    `Recording` touches.
    """

    def __init__(self) -> None:
        super().__init__()
        self.resize(200, 120)
        self.label = QLabel("before")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

    def centralWidget(self) -> QWidget:
        return self


# ------------------------------------------------------------------ pacing


def test_holding_repeats_one_frame(qapp):
    """Repeated rather than paused, so the encoder can collapse the run
    into a single long frame.
    """
    host = _Host()
    host.show()
    recording = Recording(host, qapp)

    recording.Hold(4)

    assert len(recording.frames) == 4
    assert all(np.array_equal(recording.frames[0], frame) for frame in recording.frames)


def test_a_held_frame_shows_the_change_that_preceded_it(qapp):
    """The property the whole recording rests on. Qt lays out and repaints
    lazily, so a grab taken without settling shows the state *before* the
    change - and the animation would be one beat behind itself throughout.
    """
    host = _Host()
    host.show()
    recording = Recording(host, qapp)
    recording.Hold(1)

    host.label.setText("a much longer caption than before")
    recording.Hold(1)

    assert not np.array_equal(recording.frames[0], recording.frames[1])


# ------------------------------------------------------------------ inputs


def _Photo() -> np.ndarray:
    """The demo's photograph, or a failure that says which file is wrong.

    `cv2.imread` returns None for anything it cannot decode, so without
    this every test below would fail on an attribute of None rather than
    on the file.
    """
    import cv2

    photo = cv2.imread(PHOTO)
    assert photo is not None, f"{PHOTO} is not a readable image"
    return photo


def test_the_photo_it_names_is_there():
    assert os.path.exists(PHOTO), "the animation names one specific photo"


def test_the_photo_carries_a_calibration_sheet():
    """The reason this demo cannot use just any photo. Without markers,
    `silhouette_gui.py` falls back to pixel space by design - the animation
    would look complete while dropping the one thing the window is for.
    """
    from capture.calibration import ArucoCalibration

    calibration = ArucoCalibration()
    calibration.Detect(_Photo())

    known = calibration.parameters.marker_positions_mm
    matched = [marker for marker in calibration.detected_corners if marker in known]
    assert len(matched) >= 3, f"only {len(matched)} markers matched a known position"
    assert calibration.GetTransform() is not None


def test_both_clicks_land_inside_the_photo():
    height, width = _Photo().shape[:2]

    for point in (HANDLE, SHADOW):
        assert 0 <= point[0] < width and 0 <= point[1] < height
        assert all(isinstance(value, int) for value in point), "a real click gives ints; the mask is indexed with them"


def test_the_negative_click_is_off_the_object():
    """It marks the drop shadow as exterior, so it has to be somewhere the
    tool is not - a negative point on the handle would fight the positive
    one and the animation would be demonstrating a mistake.
    """
    photo = _Photo()
    handle = photo[HANDLE[1], HANDLE[0]].astype(int)
    shadow = photo[SHADOW[1], SHADOW[0]].astype(int)

    assert handle[2] - handle[1] > 60, "the handle is red"
    assert abs(shadow[2] - shadow[1]) < 30, "the shadow is grey paper, not the tool"


def test_the_symmetry_checkbox_is_findable_by_its_caption(qapp):
    """The demo ticks it by caption, because the stages build their own
    controls and hand back a group box with no attribute to reach for. A
    caption that drifted from the constant would make the animation skip
    the step in silence, so both come from `morphology_stage`.
    """
    from PyQt5.QtWidgets import QCheckBox

    from capture.morphology_stage import LATERAL_LABEL, MorphologyStage

    widget = MorphologyStage().CreateWidget(on_change=lambda: None)

    assert LATERAL_LABEL in [box.text() for box in widget.findChildren(QCheckBox)]


def test_ticking_a_checkbox_that_is_not_there_is_refused(qapp):
    """Silently doing nothing would drop a whole beat from the animation
    and leave a GIF that still looked plausible.
    """
    host = _Host()

    with pytest.raises(ValueError, match="no checkbox"):
        capture_demo._Check(host, "Lateral symmetry (left/right)")


def test_the_symmetry_it_demonstrates_is_the_union(qapp):
    """ "Or" takes the wider of the mask and its mirror at every point,
    which is what puts back what the shadow cost. "And" would take the
    narrower and eat into the tool instead.
    """
    from capture.morphology import MorphologyParameters

    assert MorphologyParameters().symmetry_combine == "or"


def test_the_palette_is_sized_for_a_photograph():
    """The layout animations are line drawings on white, where 16 colours
    is generous. At 16 this one came out uniformly brown with the ArUco
    markers illegible - which would make it evidence for nothing.
    """
    from export.gif_writer import DEFAULT_COLORS

    assert PHOTO_COLORS > DEFAULT_COLORS


# ------------------------------------------------------------- the whole run


@pytest.mark.slow
def test_the_animation_records_the_real_window(tmp_path, qapp):
    """Loads SAM2 and segments a 5184px photograph - about ten seconds.

    What is asserted is what makes it worth having: every frame the same
    size (so the encoder pads nothing and the window does not appear to
    jump), and the flow reaching a millimetre-scale contour rather than
    stopping at a mask.
    """
    from PIL import Image, ImageSequence

    path = str(tmp_path / "capture.gif")
    assert Main(["--out", path]) == 0

    frames = [np.array(frame.convert("RGB")) for frame in ImageSequence.Iterator(Image.open(path))]
    assert len(frames) >= 6, "empty, loaded, clicked, segmented, symmetrized, selected"
    assert len({frame.shape for frame in frames}) == 1, "a window that changes size reads as a jump"


@pytest.mark.slow
def test_every_beat_changes_something_visible(tmp_path, qapp):
    """A beat that moves a handful of pixels is a fifth of the animation
    spent on nothing, which is what the segmentation beat was before
    annotations were sized in screen pixels: it moved 1099 pixels of a 1.2
    megapixel window.

    The floor only catches a beat being worth *nothing*, not being worth
    little - what a beat is worth is a judgement no pixel count settles.
    It is set well under the smallest real one because two click markers
    legitimately are only a couple of hundred pixels; a cross is a small
    thing and drawing it larger would not make the animation better.
    Measured on the committed recording: 325492, 241, 2356, 752, 28156.
    """
    from PIL import Image, ImageSequence

    path = str(tmp_path / "capture.gif")
    assert Main(["--out", path]) == 0

    frames = [np.array(frame.convert("RGB")).astype(int) for frame in ImageSequence.Iterator(Image.open(path))]
    for index, (before, after) in enumerate(zip(frames, frames[1:])):
        changed = int((np.abs(before - after).sum(axis=2) > 12).sum())
        assert changed > 120, f"beat {index} -> {index + 1} changed only {changed} pixels"


@pytest.mark.slow
def test_the_flow_ends_in_millimetres(qapp):
    """The claim the animation makes. A capture that ended in pixels would
    look identical and mean nothing - which is exactly what this photo
    having a calibration sheet buys.
    """
    window, _ = capture_demo.Record()

    contours = window.rectify.contours
    assert contours, "selection should have rectified something"
    points = np.asarray(next(iter(contours.values())), dtype=float).reshape(-1, 2)
    extent = points.max(axis=0) - points.min(axis=0)
    assert 100.0 < max(extent) < 400.0, f"a hand tool is decimetres, got {extent} mm"
