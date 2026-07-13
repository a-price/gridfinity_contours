import math

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QLabel


def WidgetToImageCoords(
    widget: QLabel, image_shape: tuple, ev: QMouseEvent
) -> tuple[int, int] | None:
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

    if (
        widget_x < 0
        or widget_y < 0
        or widget_x >= pixmap_width
        or widget_y >= pixmap_height
    ):
        return None

    img_x = int(widget_x / scale_x)
    img_y = int(widget_y / scale_y)
    return img_x, img_y


class ClickRecorder:
    def __init__(self, image_widget: QLabel, image_shape: tuple) -> None:
        self.image_widget = image_widget
        self.image_shape = image_shape
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
            erase_radius = 5
            # Erase points within 5 pixels of the click, keeping points/labels in sync
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

    def DebugLayer(self) -> cv2.Mat:
        raise NotImplementedError
