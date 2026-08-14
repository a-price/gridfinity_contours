"""Photographing a live Qt window with nobody at the keyboard.

Shared by the two demos that record real windows rather than redrawing
what one would look like: `capture_demo.py`, which animates the capture
flow, and `screenshot_demo.py`, which takes one still of each tool for
the README.

**Why the pictures are grabs and not reconstructions.** Every window here
is a control panel beside a rendered image, and the renderers already have
their own reference pictures (`render_demo.py`). What a grab adds is the
*panel* - the sizes, the statuses, the units, the buttons someone has to
find - and that half cannot be redrawn from the pipeline, because it is
not something the pipeline produces. An assembled picture of a window
would also keep looking right after the real one had stopped working,
which is the failure this whole approach exists to avoid.

Three things that each produced a wrong picture before they produced this
module: Qt pads scanlines, `QImage.bits()` hands back memory Qt reuses,
and layout happens lazily enough that the obvious grab catches the window
one state behind itself. All three are handled here so that neither demo
has to remember them.

Setting `QT_QPA_PLATFORM=offscreen` is deliberately *not* done here. It
has to happen before the first Qt import, and this module imports Qt, so
a caller that imported it early enough for that to work would already
have imported Qt too late. It stays where it can only be written
correctly: the top of a script, above its own imports.
"""

import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QWidget

# How many rounds of layout and paint to allow before giving up on the
# window ever holding still. Five is far more than anything here needs -
# the point is that it is bounded, so a widget that repaints on a timer
# cannot hang a build.
DEFAULT_PASSES = 5


def Grab(widget: QWidget) -> np.ndarray:
    """The widget as it looks right now, as a BGR image.

    Via `QWidget.grab()` rather than a screen capture, so it works with no
    display at all and catches exactly the widget rather than whatever
    happened to be in front of it.
    """
    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)

    # `bytesPerLine` rather than `width * 4`: Qt pads scanlines to a
    # 4-byte boundary, and on an odd width the padding would shear the
    # image progressively down the frame - which reads as a rendering bug
    # rather than an indexing one.
    height, width = image.height(), image.width()
    buffer = image.bits().asstring(height * image.bytesPerLine())  # pyright: ignore[reportOptionalMemberAccess]
    pixels = np.frombuffer(buffer, np.uint8).reshape(height, image.bytesPerLine() // 4, 4)

    # Format_RGB32 is BGRA in memory on a little-endian machine, so the
    # first three channels are already the BGR that cv2 and the gif writer
    # both want. Copied because the buffer above is Qt's, and Qt will
    # write over it on the next grab.
    return pixels[:, :width, :3].copy()


def _Activate(widget: QWidget) -> None:
    """Run the pending layout pass now rather than at the next repaint.

    Both the widget's own layout and, for a `QMainWindow`, its central
    widget's - the panels these demos photograph live in the second one,
    and a main window's layout activating does not always reach them in
    the same pass.
    """
    layouts = [widget.layout()]

    central = getattr(widget, "centralWidget", None)
    if callable(central):
        inner = central()
        if isinstance(inner, QWidget):
            layouts.append(inner.layout())

    for layout in layouts:
        if layout is not None:
            layout.activate()


def Settled(widget: QWidget, application: QApplication, passes: int = DEFAULT_PASSES) -> np.ndarray:
    """The widget once two consecutive grabs agree.

    Qt lays out and repaints lazily, and one round of event processing is
    not always enough: a word-wrapped label needs a second height-for-width
    pass, so the frame after loading a photo caught the calibration label
    on one line and the frame after it showed the same label on two.
    Nothing had changed between them - it read as a flicker the tool does
    not actually have.

    Settling on agreement rather than on a fixed number of passes, since
    what needs how many rounds is a property of the widgets, and a constant
    tuned against today's panel would quietly stop being enough when
    somebody adds to it.
    """
    previous = None
    for _ in range(max(1, passes)):
        _Activate(widget)
        application.processEvents()

        current = Grab(widget)
        if previous is not None and np.array_equal(previous, current):
            return current
        previous = current

    assert previous is not None  # the loop runs at least once
    return previous
