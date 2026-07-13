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
    QTextEdit,
    QDialog,
    QCheckBox,
)
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QLibraryInfo
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
import scipy.ndimage

from click_recorder import WidgetToImageCoords
from segmenter_stage import SegmenterStage
from morphology_stage import MorphologyStage
from calibration_stage import IdentityCalibrationStage
from contour_extraction import FindContours, PCABox
from contour_selection_stage import ContourSelectionStage
from rectify import Rectify
from pipeline import Pipeline

# Fix PyQt5 / OpenCV collision
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

# Pipeline stages have the following properties:
#  * User Configuration parameters
#  * Input(s)
#  * Outputs(s)
#  * Debugging

# The pipeline has the following stages
#  * Load Image
#  * Segment Object Contour
#  * Calibrate
#  * Select and Export Contour
#
# Calibration currently uses IdentityCalibration, a stub that treats pixels
# as millimeters 1:1 - a placeholder until a real fiducial-based calibration
# (HoughCircleCalibration, PaperCalibration in calibration.py) is wired in.



class SVGGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SVG Outliner")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.original_image = None
        self.processed_image = None
        # Object selection state
        self.object_mask = None
        self.object_labels = None
        self.object_contours = []

        self.segmenter_stage = SegmenterStage()
        self.morphology_stage = MorphologyStage()
        self.calibration_stage = IdentityCalibrationStage()
        self.contour_selection_stage = ContourSelectionStage()
        self.rectify = Rectify()

        self.pipeline = Pipeline()
        self.pipeline.Register("calibration", lambda: self.calibration_stage.Run(self.original_image))
        self.pipeline.Register("contours", self.find_contours, downstream=["selection"])
        self.pipeline.Register(
            "selection",
            lambda: self.contour_selection_stage.Run(self.object_contours),
            downstream=["display"],
        )
        self.pipeline.Register(
            "morphology",
            lambda: self.morphology_stage.Run(self.segmenter_stage.mask),
            downstream=["contours"],
        )
        self.pipeline.Register(
            "segmentation",
            lambda: self.segmenter_stage.Run(self.original_image),
            downstream=["morphology"],
        )
        # Export is manually triggered (it pops up a dialog), so nothing
        # lists it as a downstream - it isn't part of the auto-cascade, just
        # registered for consistency.
        self.pipeline.Register("export", self.export_contours)
        self.pipeline.Register("display", self.update_display)

        self.init_ui()

    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel for controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Add buttons
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        self.show_original_btn = QPushButton("Show Original")
        self.show_original_btn.clicked.connect(self.show_original_image)
        self.show_original_btn.setEnabled(False)

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(lambda: self.pipeline.RunFrom("export"))
        self.export_btn.setEnabled(False)

        # Add controls to layout
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.show_original_btn)
        control_layout.addWidget(self.export_btn)

        # Clicking the image either adds a segmentation point (default) or,
        # in select mode, toggles the object under the click - the two can't
        # be disambiguated from the click alone once a mask already covers
        # part of the image.
        self.select_mode_checkbox = QCheckBox("Select Mode (click to choose an object)")
        control_layout.addWidget(self.select_mode_checkbox)

        # Calibration stage widget: currently just an explanatory label,
        # since IdentityCalibration has no parameters to tune.
        self.calibration_widget = self.calibration_stage.CreateWidget(on_change=lambda: None)
        control_layout.addWidget(self.calibration_widget)

        # Morphology cleanup parameters: settled edits rerun morphology
        # (and contours/display downstream of it) through the pipeline.
        control_layout.addWidget(QLabel("\nSegmentation Cleanup:"))
        self.morphology_widget = self.morphology_stage.CreateWidget(
            on_change=lambda: self.pipeline.RunFrom("morphology")
        )
        control_layout.addWidget(self.morphology_widget)

        # Contour selection parameters: settled edits rerun selection (and
        # display downstream of it) through the pipeline. Selecting an
        # object itself happens by clicking it in the image view.
        control_layout.addWidget(QLabel("\nContour Simplification:"))
        self.contour_selection_widget = self.contour_selection_stage.CreateWidget(
            on_change=lambda: self.pipeline.RunFrom("selection")
        )
        control_layout.addWidget(self.contour_selection_widget)

        # Add stretch to push everything to the top
        control_layout.addStretch()

        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b;")
        self.image_label.mousePressEvent = self.image_clicked

        # Add widgets to main layout
        layout.addWidget(control_panel, stretch=1)
        layout.addWidget(self.image_label, stretch=3)

    def load_image(self, file_path: str | None = None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
            )
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if self.original_image is not None:
                self.processed_image = self.original_image.copy()
                self.show_original_btn.setEnabled(True)
                self.export_btn.setEnabled(True)

                self.segmenter_stage.AttachToImageWidget(
                    self.image_label,
                    self.original_image.shape,
                    on_change=lambda: self.pipeline.RunFrom("segmentation"),
                )

                self.pipeline.RunFrom("calibration")

                # Object contours only become available once the user clicks
                # segmentation points, cascading through segmentation ->
                # morphology -> contours -> display.
                self.pipeline.RunFrom("display")

    def find_contours(self):
        """Extract object contours from the morphology stage's cleaned-up
        segmentation mask (empty until the user has clicked points on an
        object).
        """
        mask_image = self.morphology_stage.mask
        if mask_image is None:
            self.object_mask = None
            self.object_labels = None
            self.object_contours = []
            return

        # Label connected components
        labels, _ = scipy.ndimage.label(mask_image)

        # Store results
        self.object_mask = mask_image
        self.object_labels = labels

        # Extract contours for visualization
        self.object_contours = FindContours(mask_image)

    def show_original_image(self):
        """Show the original image."""
        if self.original_image is None:
            return

        self.processed_image = self.original_image.copy()
        self.update_display()

    def image_clicked(self, ev: QMouseEvent | None):
        if ev is None:
            return
        if self.processed_image is None:
            return

        coords = WidgetToImageCoords(self.image_label, self.processed_image.shape, ev)
        if coords is None:
            return
        img_x, img_y = coords

        if not self.select_mode_checkbox.isChecked():
            self.segmenter_stage.OnClick(ev)
            return

        # Select mode: toggle an object contour under the click. Doesn't add
        # a segmentation point.
        if self.object_contours:
            if self.contour_selection_stage.contour_selection.ToggleSelection(img_x, img_y, self.object_contours):
                self.pipeline.RunFrom("selection")
                return

    def update_display(self):
        if self.processed_image is None:
            return

        # Create a copy of the image to draw on
        display_image = self.processed_image.copy()

        # Create overlay for transparent effects
        overlay = display_image.copy()

        # Draw segmentation click points: a green "+" for positive
        # (label 1) points, a red "-" for negative (label 0) points.
        click_recorder = self.segmenter_stage.click_recorder
        if click_recorder is not None:
            marker_len = max(15, min(display_image.shape[:2]) // 80)
            thickness = max(3, marker_len // 5)
            for (x, y), label in zip(click_recorder.image_points, click_recorder.image_labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.line(display_image, (x - marker_len, y), (x + marker_len, y), color, thickness)
                if label == 1:
                    cv2.line(display_image, (x, y - marker_len), (x, y + marker_len), color, thickness)

        # Draw object boundaries if available
        selected_objects = self.contour_selection_stage.contour_selection.selected
        if self.object_contours:
            for i, contour in enumerate(self.object_contours):
                # Selected objects in green, unselected in yellow
                color = (0, 255, 0) if i in selected_objects else (0, 255, 255)
                cv2.drawContours(display_image, [contour], -1, color, 2)

                # Add transparent green fill for selected objects
                if i in selected_objects:
                    cv2.drawContours(
                        overlay, [contour], -1, (0, 255, 0), -1
                    )  # Filled contour

                    simplified_contour = self.contour_selection_stage.contour_selection.simplified.get(i)
                    pca_box = self.contour_selection_stage.contour_selection.boxes.get(i)
                    if simplified_contour is None or pca_box is None:
                        continue

                    # Draw simplified contour in bright blue over the original
                    cv2.drawContours(
                        display_image, [simplified_contour], -1, (255, 255, 0), 3
                    )

                    # Draw PCA-aligned bounding box
                    cv2.drawContours(
                        display_image, [pca_box.corners], -1, (255, 0, 255), 2
                    )  # Magenta bounding box

                    # Draw center point
                    cv2.circle(
                        display_image,
                        tuple(np.int32(pca_box.center)),
                        5,
                        (255, 0, 255),
                        -1,
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
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


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
        """Export simplified contour points, converted to real-world units
        by the calibration stage's affine transform (currently
        IdentityCalibration: 1px = 1mm, pending a real fiducial-based
        calibration) in a new window.
        """
        contour_selection = self.contour_selection_stage.contour_selection
        if not contour_selection.selected or not contour_selection.simplified:
            return

        affine = self.calibration_stage.calibration.GetTransform()
        self.rectify.Run(self.processed_image, affine, contour_selection.simplified)

        dialog = ContourExportDialog(contour_selection.selected, self.rectify.contours, self)
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

        # add figure()
        plt.figure()
        plt.title("tranformed contour")
        plt.plot(transformed_points[:, 0], transformed_points[:, 1], "-")
        plt.xlabel("mm")
        plt.ylabel("mm")
        plt.show()

        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def transform_to_origin(self, contour):
        """Transform contour so that one corner of the PCA-aligned bounding box is at origin."""
        points = contour.reshape(-1, 2).astype(np.float32)
        return PCABox(points).ToLocal(points)


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
