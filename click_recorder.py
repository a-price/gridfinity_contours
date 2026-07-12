import math

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QLabel


class ClickRecorder:
    def __init__(self, image_widget: QLabel, image_shape: tuple) -> None:
        self.image_widget = image_widget
        self.image_shape = image_shape
        self.image_points = []
        self.image_labels = []

    def OnClick(self, ev: QMouseEvent | None):
        if ev is None:
            return

        # Get the current pixmap and its dimensions
        pixmap = self.image_widget.pixmap()
        if pixmap is None:
            return

        # Get widget and image dimensions
        widget_width = self.image_widget.width()
        widget_height = self.image_widget.height()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Calculate the actual position of the scaled image within the widget
        # The image is centered and scaled to fit while maintaining aspect ratio
        scale_x = pixmap_width / self.image_shape[1]
        scale_y = pixmap_height / self.image_shape[0]

        # Calculate offset to center the image in the widget
        offset_x = (widget_width - pixmap_width) // 2
        offset_y = (widget_height - pixmap_height) // 2

        # Convert widget coordinates to image coordinates
        widget_x = ev.pos().x() - offset_x
        widget_y = ev.pos().y() - offset_y

        # Check if click is within the image bounds
        if (
            widget_x < 0
            or widget_y < 0
            or widget_x >= pixmap_width
            or widget_y >= pixmap_height
        ):
            return

        # Scale back to original image coordinates
        img_x = int(widget_x / scale_x)
        img_y = int(widget_y / scale_y)

        assert 0 <= img_x
        assert img_x < self.image_shape[1]
        assert 0 <= img_y
        assert img_y < self.image_shape[0]

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
