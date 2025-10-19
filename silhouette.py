import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QWidget,
    QSlider,
    QTextEdit,
    QDialog,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
)
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
from skimage import morphology
import scipy.ndimage


class SVGGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SVG Outliner")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.circles = []
        self.selected_circles = set()
        self.polygon_points = []
        self.drawing = False
        self.current_polygon = []
        # Object selection state
        self.object_mask = None
        self.object_labels = None
        self.object_contours = []
        self.selected_objects = set()
        self.simplified_contours = {}

        self.click_point = []

        self.init_ui()

    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel for controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)

        # Add buttons
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        self.show_binary_btn = QPushButton("Show Binary Image")
        self.show_binary_btn.clicked.connect(self.show_binary_image)
        self.show_binary_btn.setEnabled(False)

        self.show_original_btn = QPushButton("Show Original")
        self.show_original_btn.clicked.connect(self.show_original_image)
        self.show_original_btn.setEnabled(False)

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_contours)
        self.export_btn.setEnabled(False)

        # Add controls to layout
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.show_binary_btn)
        control_layout.addWidget(self.show_original_btn)
        control_layout.addWidget(self.export_btn)

        # Add collapsible circle detection parameters
        self.circle_params_toggle = QPushButton("▶ Circle Detection Parameters")
        self.circle_params_toggle.setFlat(True)
        self.circle_params_toggle.clicked.connect(self.toggle_circle_params)
        control_layout.addWidget(self.circle_params_toggle)

        # Create container widget for circle parameters
        self.circle_params_widget = QWidget()
        circle_params_layout = QVBoxLayout(self.circle_params_widget)
        circle_params_layout.setContentsMargins(20, 0, 0, 0)  # Indent the parameters

        # Max circles slider
        self.max_circles_slider = self.create_slider("Top N Circles:", 1, 20, 5)
        circle_params_layout.addLayout(self.max_circles_slider["layout"])

        # Min distance slider
        self.min_dist_slider = self.create_slider("Min Distance:", 10, 200, 50)
        circle_params_layout.addLayout(self.min_dist_slider["layout"])

        # Param1 slider
        self.param1_slider = self.create_slider(
            "Param1 (edge detection):", 10, 200, 100
        )
        circle_params_layout.addLayout(self.param1_slider["layout"])

        # Param2 slider
        self.param2_slider = self.create_slider("Param2 (threshold):", 1, 100, 30)
        circle_params_layout.addLayout(self.param2_slider["layout"])

        # Min radius slider
        self.min_radius_slider = self.create_slider("Min Radius:", 1, 100, 10)
        circle_params_layout.addLayout(self.min_radius_slider["layout"])

        # Max radius slider
        self.max_radius_slider = self.create_slider("Max Radius:", 10, 200, 100)
        circle_params_layout.addLayout(self.max_radius_slider["layout"])

        # Binary threshold slider
        self.threshold_slider = self.create_slider("Binary Threshold:", 1, 255, 127)
        circle_params_layout.addLayout(self.threshold_slider["layout"])

        # Add the parameters widget to the main control layout
        control_layout.addWidget(self.circle_params_widget)

        # Hide parameters by default (collapsed)
        self.circle_params_widget.setVisible(False)

        # Add Transform Parameters section
        control_layout.addWidget(QLabel("\nTransform Parameters:"))

        # Layout dropdown
        layout_label = QLabel("Layout:")
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Isosceles")
        self.layout_combo.setCurrentText("Isosceles")
        self.layout_combo.currentTextChanged.connect(self.on_layout_changed)

        layout_layout = QVBoxLayout()
        layout_layout.addWidget(layout_label)
        layout_layout.addWidget(self.layout_combo)
        control_layout.addLayout(layout_layout)

        # Leg Distance input (shown when Isosceles is selected)
        self.leg_distance_label = QLabel("Leg Distance (mm):")
        self.leg_distance_input = QDoubleSpinBox()
        self.leg_distance_input.setRange(0.1, 1000.0)
        self.leg_distance_input.setValue(80.0)
        self.leg_distance_input.setDecimals(2)
        self.leg_distance_input.setSuffix(" mm")

        leg_distance_layout = QVBoxLayout()
        leg_distance_layout.addWidget(self.leg_distance_label)
        leg_distance_layout.addWidget(self.leg_distance_input)
        control_layout.addLayout(leg_distance_layout)

        # Transform checkboxes
        self.rectify_checkbox = QCheckBox("Rectify")
        self.rectify_checkbox.setChecked(True)  # Toggled on by default
        control_layout.addWidget(self.rectify_checkbox)

        self.scale_checkbox = QCheckBox("Scale")
        self.scale_checkbox.setChecked(True)  # Toggled on by default
        control_layout.addWidget(self.scale_checkbox)

        # Add stretch to push everything to the top
        control_layout.addStretch()

        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b;")
        self.image_label.mousePressEvent = self.image_clicked

        # Add widgets to main layout
        layout.addWidget(control_panel, stretch=1)
        layout.addWidget(self.image_label, stretch=3)

    def create_slider(self, label_text, min_val, max_val, default_val):
        layout = QVBoxLayout()
        label = QLabel(f"{label_text} {default_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)

        def update_label(value):
            label.setText(f"{label_text} {value}")

        slider.valueChanged.connect(update_label)

        layout.addWidget(label)
        layout.addWidget(slider)

        return {"layout": layout, "slider": slider, "label": label}

    def toggle_circle_params(self):
        """Toggle visibility of circle detection parameters."""
        is_visible = self.circle_params_widget.isVisible()
        self.circle_params_widget.setVisible(not is_visible)

        # Update button text
        if is_visible:
            self.circle_params_toggle.setText("▶ Circle Detection Parameters")
        else:
            self.circle_params_toggle.setText("▼ Circle Detection Parameters")

    def on_layout_changed(self, layout_type):
        """Handle layout dropdown changes."""
        if layout_type == "Isosceles":
            self.leg_distance_label.setVisible(True)
            self.leg_distance_input.setVisible(True)
        else:
            self.leg_distance_label.setVisible(False)
            self.leg_distance_input.setVisible(False)

    def load_image(self, file_path: str = None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
            )
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if self.original_image is not None:
                self.processed_image = self.original_image.copy()
                # Configure radius sliders based on image size
                self._configure_radius_sliders_for_image()
                self.update_display()
                self.show_binary_btn.setEnabled(True)
                self.show_original_btn.setEnabled(True)
                self.export_btn.setEnabled(True)

                # Automatically run detect circles and select objects
                self.find_circles()
                self.find_contours()

    def find_circles(self):
        if self.original_image is None:
            return

        # Get parameters from sliders
        min_dist = self.min_dist_slider["slider"].value()
        param1 = self.param1_slider["slider"].value()
        param2 = self.param2_slider["slider"].value()
        min_radius = self.min_radius_slider["slider"].value()
        max_radius = self.max_radius_slider["slider"].value()
        threshold_value = self.threshold_slider["slider"].value()
        max_circles = self.max_circles_slider["slider"].value()

        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise before thresholding
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Convert to binary image (black/white)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Apply median blur to the binary image for better circle detection
        binary = cv2.medianBlur(binary, 5)

        # Detect circles using Hough transform on the binary image
        circles = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        # Reset previous circles and selection
        self.circles = []
        self.selected_circles = set()

        # Process detected circles and limit to max_circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Limit to the specified maximum number of circles
            circles_to_add = circles[0, :max_circles]
            for i in circles_to_add:
                self.circles.append((i[0], i[1], i[2]))

        # Select the first 3 circles by default
        for i in range(min(3, len(self.circles))):
            self.selected_circles.add(i)

        self.update_display()

    def find_contours(self):
        """Segment objects against a black background using thresholding and morphology."""
        if self.original_image is None:
            return

        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Threshold: objects brighter than background (same as makersaturday.py)
        mask_image = gray > 50

        # Morphological cleanup (same as makersaturday.py)
        mask_image = morphology.binary_closing(mask_image)
        mask_image = morphology.remove_small_holes(mask_image, area_threshold=1000)
        mask_image = morphology.remove_small_objects(mask_image, min_size=1000)

        # Label connected components
        labels, _ = scipy.ndimage.label(mask_image)

        # Store results
        self.object_mask = mask_image
        self.object_labels = labels

        # Extract contours for visualization
        self.object_contours = []
        # Use OpenCV findContours on uint8 mask
        mask_u8 = mask_image.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.object_contours = contours

        self.update_display()

    def show_binary_image(self):
        """Display the binary version of the image for debugging purposes."""
        if self.original_image is None:
            return

        # Get threshold value from slider
        threshold_value = self.threshold_slider["slider"].value()

        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise before thresholding
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Convert to binary image (black/white)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Apply median blur to the binary image
        binary = cv2.medianBlur(binary, 5)

        # Convert binary to 3-channel for display
        binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Update processed image and display
        self.processed_image = binary_color
        self.update_display()

    def show_original_image(self):
        """Show the original image."""
        if self.original_image is None:
            return

        self.processed_image = self.original_image.copy()
        self.update_display()

    def image_clicked(self, event: QMouseEvent):
        if self.processed_image is None:
            return

        # Get the current pixmap and its dimensions
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return

        # Get widget and image dimensions
        widget_width = self.image_label.width()
        widget_height = self.image_label.height()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Calculate the actual position of the scaled image within the widget
        # The image is centered and scaled to fit while maintaining aspect ratio
        scale_x = pixmap_width / self.processed_image.shape[1]
        scale_y = pixmap_height / self.processed_image.shape[0]

        # Calculate offset to center the image in the widget
        offset_x = (widget_width - pixmap_width) // 2
        offset_y = (widget_height - pixmap_height) // 2

        # Convert widget coordinates to image coordinates
        widget_x = event.pos().x() - offset_x
        widget_y = event.pos().y() - offset_y

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

        if self.processed_image is not None:
            height, width, channel = self.processed_image.shape
            assert(0 <= img_x)
            assert(img_x < width)
            assert(0 <= img_y)
            assert(img_y < height)

        self.click_point = [img_x, img_y]

        # Check if a circle was clicked
        for i, (cx, cy, r) in enumerate(self.circles):
            if (img_x - int(cx)) ** 2 + (img_y - int(cy)) ** 2 <= r**2:
                if i in self.selected_circles:
                    self.selected_circles.remove(i)
                else:
                    self.selected_circles.add(i)
                self.update_display()
                return

        # Check if an object was clicked
        if (
            self.object_contours
            and 0 <= img_x < self.processed_image.shape[1]
            and 0 <= img_y < self.processed_image.shape[0]
        ):
            for i, contour in enumerate(self.object_contours):
                # Use cv2.pointPolygonTest to check if point is inside contour
                result = cv2.pointPolygonTest(contour, (img_x, img_y), False)
                if result >= 0:  # Point is inside or on the contour
                    if i in self.selected_objects:
                        self.selected_objects.remove(i)
                    else:
                        self.selected_objects.add(i)
                    self.update_display()
                    return

    def update_display(self):
        if self.processed_image is None:
            return

        # Create a copy of the image to draw on
        display_image = self.processed_image.copy()

        # Create overlay for transparent effects
        overlay = display_image.copy()

        # If we have a clicked point, draw it.
        if self.click_point:
            cv2.circle(overlay, (self.click_point[0], self.click_point[1]), 50, (0, 0, 255), -1)

        # Draw detected circles
        i_selected = 0
        for i, (x, y, r) in enumerate(self.circles):
            color = (255, 0, 0) if i in self.selected_circles else (0, 0, 255)
            cv2.circle(display_image, (x, y), r, color, 2)
            cv2.circle(display_image, (x, y), 2, (0, 255, 0), 3)

            if i in self.selected_circles:
                # Add transparent blue fill
                cv2.circle(overlay, (x, y), r, (255, 0, 0), -1)  # Filled circle

                # Label circle with its index
                label = str(i_selected)
                i_selected += 1
                # Position label slightly above and to the right of center
                label_x = x + r // 5
                label_y = y - r // 5
                cv2.putText(
                    display_image,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (255, 0, 0),
                    8,
                    cv2.LINE_AA,
                )

        # Draw current polygon
        if len(self.current_polygon) > 1:
            pts = np.array(self.current_polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display_image, [pts], True, (255, 0, 0), 2)

        # Draw object boundaries if available
        if self.object_contours:
            for i, contour in enumerate(self.object_contours):
                # Selected objects in green, unselected in yellow
                color = (0, 255, 0) if i in self.selected_objects else (0, 255, 255)
                cv2.drawContours(display_image, [contour], -1, color, 2)

                # Add transparent green fill for selected objects
                if i in self.selected_objects:
                    cv2.drawContours(
                        overlay, [contour], -1, (0, 255, 0), -1
                    )  # Filled contour

                    # Polyline simplification using Douglas-Peucker algorithm
                    epsilon = 0.001 * cv2.arcLength(contour, True)  # 2% of perimeter
                    simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

                    # Store simplified contour for export
                    self.simplified_contours[i] = simplified_contour

                    # Draw simplified contour in bright blue over the original
                    cv2.drawContours(
                        display_image, [simplified_contour], -1, (255, 255, 0), 3
                    )

                    # Find center and compute PCA-based bounding box
                    # Get all points from simplified contour
                    points = simplified_contour.reshape(-1, 2).astype(np.float32)

                    # Compute PCA
                    mean, eigenvectors, _ = cv2.PCACompute2(points, np.array([]))
                    center = mean[0]

                    # Get the principal components (eigenvectors)
                    pc1 = eigenvectors[0]  # First principal component
                    pc2 = eigenvectors[1]  # Second principal component

                    # Project points onto principal components to find extents
                    projected1 = np.dot(points - center, pc1)
                    projected2 = np.dot(points - center, pc2)

                    # Find min/max along each principal component
                    min1, max1 = np.min(projected1), np.max(projected1)
                    min2, max2 = np.min(projected2), np.max(projected2)

                    # Create bounding box corners in PCA space
                    corners_pca = np.array(
                        [[min1, min2], [max1, min2], [max1, max2], [min1, max2]]
                    )

                    # Transform back to image coordinates
                    box = np.array(
                        [
                            center + corner[0] * pc1 + corner[1] * pc2
                            for corner in corners_pca
                        ]
                    )
                    box = np.int32(box)

                    # Draw PCA-aligned bounding box
                    cv2.drawContours(
                        display_image, [box], -1, (255, 0, 255), 2
                    )  # Magenta bounding box

                    # Draw center point
                    cv2.circle(
                        display_image, tuple(np.int32(center)), 5, (255, 0, 255), -1
                    )

        # Blend overlay with main image for transparency effect (30% opacity)
        alpha = 0.3
        display_image = cv2.addWeighted(display_image, 1 - alpha, overlay, alpha, 0)

        # Draw dimension rulers along the image borders
        self._draw_rulers(display_image)

        # Convert to QImage and display
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            display_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        self.image_label.setPixmap(
            QPixmap.fromImage(q_image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _configure_radius_sliders_for_image(self) -> None:
        """Set upper limit of min/max radius sliders to 100% of the minimum image dimension.
        Default max radius to 20% and min radius to 10% of that minimum dimension.
        """
        if self.processed_image is None:
            return
        h, w = self.processed_image.shape[:2]
        min_dim = max(1, min(h, w))
        # Set slider maximums
        self.min_radius_slider["slider"].setMaximum(min_dim)
        self.max_radius_slider["slider"].setMaximum(min_dim)
        # Compute defaults
        default_max = max(1, int(0.20 * min_dim))
        default_min = max(1, int(0.02 * min_dim))
        if default_min > default_max:
            default_min = default_max
        # Set values (this will also update labels via valueChanged)
        self.max_radius_slider["slider"].setValue(default_max)
        self.min_radius_slider["slider"].setValue(default_min)

    def _nice_tick_step(self, length_px: int) -> int:
        """Return a 'nice' tick step (in pixels) for the given length in pixels.
        Aims for around 8-12 major ticks across the length.
        """
        if length_px <= 0:
            return 50
        rough = max(20, length_px // 10)
        # Round rough to 1, 2, or 5 times a power of 10
        magnitude = 10 ** int(np.floor(np.log10(rough)))
        residual = rough / magnitude
        if residual < 1.5:
            nice = 1 * magnitude
        elif residual < 3.5:
            nice = 2 * magnitude
        elif residual < 7.5:
            nice = 5 * magnitude
        else:
            nice = 10 * magnitude
        return int(nice)

    def _draw_rulers(self, img: np.ndarray) -> None:
        """Draw simple rulers with tick marks and pixel labels along image borders.
        Draws top and left rulers with labels, and mirrored ticks on bottom and right without labels.
        """
        h, w = img.shape[:2]
        color_major = (220, 220, 220)  # light gray for visibility on dark backgrounds
        color_minor = (160, 160, 160)
        text_color = (255, 255, 255)
        thickness_major = 2
        thickness_minor = 1
        # tick sizes
        tick_major_len = max(10, min(h, w) // 40)
        tick_minor_len = tick_major_len // 2

        step = self._nice_tick_step(max(w, h))
        minor_div = 5
        minor_step = max(1, step // minor_div)

        # Top and bottom rulers (x axis)
        for x in range(0, w, minor_step):
            is_major = (x % step) == 0
            t_len = tick_major_len if is_major else tick_minor_len
            col = color_major if is_major else color_minor
            # top
            cv2.line(
                img,
                (x, 0),
                (x, t_len),
                col,
                thickness_major if is_major else thickness_minor,
            )
            # bottom
            cv2.line(
                img,
                (x, h - t_len),
                (x, h),
                col,
                thickness_major if is_major else thickness_minor,
            )
            if is_major:
                label = str(x)
                # place label slightly below the top ticks
                cv2.putText(
                    img,
                    label,
                    (x + 3, t_len + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.0,
                    text_color,
                    3,
                    cv2.LINE_AA,
                )

        # Left and right rulers (y axis)
        for y in range(0, h, minor_step):
            is_major = (y % step) == 0
            t_len = tick_major_len if is_major else tick_minor_len
            col = color_major if is_major else color_minor
            # left
            cv2.line(
                img,
                (0, y),
                (t_len, y),
                col,
                thickness_major if is_major else thickness_minor,
            )
            # right
            cv2.line(
                img,
                (w - t_len, y),
                (w, y),
                col,
                thickness_major if is_major else thickness_minor,
            )
            if is_major:
                label = str(y)
                # place label to the right of the left ticks
                cv2.putText(
                    img,
                    label,
                    (t_len + 4, y - 4 if y > 10 else y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.0,
                    text_color,
                    3,
                    cv2.LINE_AA,
                )

    def export_contours(self):
        """Export simplified contour points in a new window."""
        if not self.selected_objects or not self.simplified_contours:
            return

        # Create export dialog
        dialog = ContourExportDialog(
            self.selected_objects, self.simplified_contours, self
        )
        dialog.exec_()


class ContourExportDialog(QDialog):
    def __init__(self, selected_objects, simplified_contours, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Simplified Contours")
        self.setGeometry(200, 200, 600, 400)

        # Create layout
        layout = QVBoxLayout(self)

        # Create text edit widget
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        # Format contour data
        contour_text = ""
        for obj_id in selected_objects:
            if obj_id in simplified_contours:
                contour = simplified_contours[obj_id]

                # Transform contour to origin-based coordinate system
                transformed_points = self.transform_to_origin(contour)

                contour_text += f"Object {obj_id} Simplified Contour Points (Transformed to Origin):\n"
                contour_text += "[\n"

                # Format transformed points
                for i, (x, y) in enumerate(transformed_points):
                    contour_text += f"  [{x:.2f}, {y:.2f}]"
                    if i < len(transformed_points) - 1:
                        contour_text += ","
                    contour_text += "\n"

                contour_text += "]\n\n"

        self.text_edit.setPlainText(contour_text)
        layout.addWidget(self.text_edit)

        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def transform_to_origin(self, contour):
        """Transform contour so that one corner of the PCA-aligned bounding box is at origin."""
        # Get all points from contour
        points = contour.reshape(-1, 2).astype(np.float32)

        # Compute PCA to get principal components
        mean, eigenvectors, _ = cv2.PCACompute2(points, np.array([]))
        center = mean[0]

        # Get the principal components (eigenvectors)
        pc1 = eigenvectors[0]  # First principal component
        pc2 = eigenvectors[1]  # Second principal component

        # Project points onto principal components to find extents
        projected1 = np.dot(points - center, pc1)
        projected2 = np.dot(points - center, pc2)

        # Find min/max along each principal component
        min1, max1 = np.min(projected1), np.max(projected1)
        min2, max2 = np.min(projected2), np.max(projected2)

        # Transform all contour points to PCA coordinate system
        # Then translate so that the minimum corner is at origin
        transformed_points = []
        for point in points:
            # Project to PCA space
            pca_coords = np.array(
                [np.dot(point - center, pc1), np.dot(point - center, pc2)]
            )

            # Translate so min corner is at origin (subtract minimums)
            origin_coords = pca_coords - np.array([min1, min2])

            transformed_points.append(origin_coords)

        return np.array(transformed_points)


def main():
    app = QApplication(sys.argv)
    window = SVGGui()
    # Resize window to 75% of available screen size and center it
    screen = app.primaryScreen()
    if screen is not None:
        avail = screen.availableGeometry()
        w = int(avail.width() * 0.75)
        h = int(avail.height() * 0.75)
        window.resize(w, h)
        # center
        x = avail.x() + (avail.width() - w) // 2
        y = avail.y() + (avail.height() - h) // 2
        window.move(x, y)
    window.show()
    # Optional CLI argument: image path to load at launch
    if len(sys.argv) > 1:
        launch_path = sys.argv[1]
        # Normalize path: expand ~ and env vars, resolve relative to CWD
        launch_path = os.path.expanduser(os.path.expandvars(launch_path))
        if not os.path.isabs(launch_path):
            launch_path = os.path.abspath(launch_path)
        window.load_image(launch_path)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
