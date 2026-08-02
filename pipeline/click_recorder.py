"""Clicks on a displayed image, and the geometry that relates the two.

A photo is shown by scaling it to fit a widget, so widget coordinates and
image coordinates differ by a factor that depends on the photo's
resolution *and* the window's size. Two things need that factor and must
agree about it: turning a click back into image coordinates, and sizing
the marks drawn over the image so they are legible on screen. Both live
here for that reason.
"""

import math
from dataclasses import dataclass

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QLabel


def ImagePixelsPerScreenPixel(widget: QLabel, image_shape: tuple) -> float:
    """How many image pixels one displayed pixel covers, if `image_shape`
    were fitted into `widget` the way the window fits it.

    The conversion every annotation has to be drawn through. Overlays are
    drawn into the full-resolution photo and only scaled down afterwards,
    so a mark sized in image pixels has no fixed size on screen: a 40px
    cross is a fifth of a 200px thumbnail and four pixels of a 5184px
    photograph. Sizing marks in *screen* pixels and multiplying by this is
    what makes them legible whatever was photographed, at whatever window
    size.

    Derived from the widget rather than read back off the current pixmap,
    so it is right on the first frame too - before anything has been
    displayed there is no pixmap to measure, and a marker drawn then would
    be sized against nothing. The two agree once a pixmap exists, since
    that pixmap was produced by exactly this fit.

    Returns 1.0 when there is nothing to scale against, which leaves sizes
    in screen pixels unchanged rather than collapsing them to zero.
    """
    height, width = image_shape[0], image_shape[1]
    if width <= 0 or height <= 0:
        return 1.0

    # `KeepAspectRatio` fits the larger relative dimension, so the scale is
    # the smaller of the two ratios - the same rule `QPixmap.scaled` uses.
    scale = min(widget.width() / width, widget.height() / height)
    return 1.0 / scale if scale > 0 else 1.0


def WidgetToImageCoords(widget: QLabel, image_shape: tuple, ev: QMouseEvent) -> tuple[int, int] | None:
    """Convert a mouse event's widget-relative position to image pixel
    coordinates, accounting for the displayed pixmap's letterboxing within
    the widget (it's centered and scaled to fit while preserving aspect
    ratio, so it rarely fills the widget exactly).

    Returns None if there's no pixmap yet or the click fell outside it.
    """
    pixmap = widget.pixmap()
    if pixmap is None:
        return None

    widget_width = widget.width()
    widget_height = widget.height()
    pixmap_width = pixmap.width()
    pixmap_height = pixmap.height()

    scale_x = pixmap_width / image_shape[1]
    scale_y = pixmap_height / image_shape[0]

    # Offset to center the image in the widget
    offset_x = (widget_width - pixmap_width) // 2
    offset_y = (widget_height - pixmap_height) // 2

    widget_x = ev.pos().x() - offset_x
    widget_y = ev.pos().y() - offset_y

    if widget_x < 0 or widget_y < 0 or widget_x >= pixmap_width or widget_y >= pixmap_height:
        return None

    img_x = int(widget_x / scale_x)
    img_y = int(widget_y / scale_y)
    return img_x, img_y


@dataclass
class ClickRecorderParameters:
    """User-configurable inputs for ClickRecorder: how close (in image
    pixels) a middle-click must land to an existing point to erase it.
    """

    erase_radius: float = 5.0


class ClickRecorder:
    def __init__(
        self,
        image_widget: QLabel,
        image_shape: tuple,
        parameters: ClickRecorderParameters | None = None,
    ) -> None:
        self.image_widget = image_widget
        self.image_shape = image_shape
        self.parameters = parameters or ClickRecorderParameters()
        self.image_points = []
        self.image_labels = []

    def OnClick(self, ev: QMouseEvent | None):
        if ev is None:
            return

        coords = WidgetToImageCoords(self.image_widget, self.image_shape, ev)
        if coords is None:
            return
        img_x, img_y = coords

        if ev.button() == Qt.MouseButton.MiddleButton:
            erase_radius = self.parameters.erase_radius
            # Erase points within erase_radius of the click, keeping points/labels in sync
            kept = [
                (point, label)
                for point, label in zip(self.image_points, self.image_labels)
                if math.hypot(point[0] - img_x, point[1] - img_y) >= erase_radius
            ]
            self.image_points = [point for point, _ in kept]
            self.image_labels = [label for _, label in kept]
            return
        else:
            self.image_points.append([img_x, img_y])
            label = 1 if ev.button() == Qt.MouseButton.LeftButton else 0
            self.image_labels.append(label)
