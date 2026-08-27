"""Interactive capture: photo to rectified, real-world-scale contour.

Each stage owns its inputs, outputs and (where relevant) debug layer, and
edits whatever parameters object stands behind it - see `capture.pipeline`
for where those live. This window wires them into one Qt session:

  * Load Image
  * Segment Object Contour (SAM2, restricted to the clicked connected
    component, then cleaned up and optionally symmetrized)
  * Calibrate
  * Select Contour - also auto-rectifies to real-world units and refreshes
    the text preview
  * Export - writes the rectified contours as SVG, PDF and a JSON dump.
    The dump is what `layout_cli.py` reads, so this button is the whole
    bridge from the capture half of the project to the layout half.

Calibration is ArucoCalibration, the only strategy this window builds:
print generate_aruco_sheet.py's PDF, place it in frame, and its markers
get detected automatically. If no markers are detected (e.g. no sheet in
frame), update_rectified_contours() falls back to pixel-space output
rather than failing.
"""

import argparse
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
    QComboBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent

from qt_utils.click_recorder import ImagePixelsPerScreenPixel, WidgetToImageCoords
from capture.segmenter import Segmenter
from capture.segmenter_stage import SegmenterStage
from capture.morphology_stage import MorphologyStage
from capture.calibration_stage import ArucoCalibrationStage
from geometry.pca_box import PCABox
from capture.contour_extraction import FindContours
from capture.contour_selection_stage import ContourSelectionStage
from capture.rectify import Rectify
from capture.svg_export_stage import SvgExportStage
from qt_utils.widgets import FixQtOpenCvPluginPath
from capture.pipeline import Pipeline

FixQtOpenCvPluginPath()

# Everything the window draws over the photo, in *screen* pixels.
#
# The overlay is drawn into the full-resolution photo and only scaled to
# fit afterwards, so a size fixed in image pixels has no fixed size on
# screen - the old click marker was `min(height, width) // 80`, which is
# 43px on a 5184px photograph and lands on four screen pixels once that
# photo is displayed. These are converted at draw time by
# `_in_image_pixels`, so a mark is the same size on screen whatever was
# photographed and however big the window is.
CLICK_MARKER_PX = 10
CLICK_THICKNESS_PX = 3
CONTOUR_THICKNESS_PX = 2
SIMPLIFIED_THICKNESS_PX = 3
BOX_THICKNESS_PX = 2
CENTER_RADIUS_PX = 4

# What a click on the image view does - one mode is active at a time, since
# a click alone can't otherwise disambiguate "add a segmentation point" from
# "select something that's already there".
#
# Only the first two do anything here. Manual fiducial selection is
# `HoughCircleCalibration`'s feature: `Calibration.ToggleSelection` returns
# False by default and `ArucoCalibration` - the one this window builds -
# does not override it, so the third mode is inert until some application
# constructs a calibration that selects.
_MODE_SEGMENT = "Add Segmentation Points (left = interior, right = exterior)"
_MODE_SELECT_CONTOUR = "Select a Contour"
_MODE_SELECT_FIDUCIAL = "Select a Fiducial"


class SVGGui(QMainWindow):
    def __init__(self, local_files_only: bool = True):
        super().__init__()
        self.setWindowTitle("SVG Outliner")
        self.setGeometry(100, 100, 1200, 800)

        self.original_image = None
        self.processed_image = None
        self.object_contours = []

        self.segmenter_stage = SegmenterStage(Segmenter(local_files_only=local_files_only))
        self.morphology_stage = MorphologyStage()
        self.calibration_stage = ArucoCalibrationStage()
        self.contour_selection_stage = ContourSelectionStage()
        self.rectify = Rectify()
        self.svg_export_stage = SvgExportStage()

        self.pipeline = Pipeline()
        self.pipeline.Register(
            "calibration",
            lambda: self.calibration_stage.Run(self.original_image),
        )
        self.pipeline.Register("contours", self.find_contours, downstream=["selection"])
        self.pipeline.Register(
            "selection",
            lambda: self.contour_selection_stage.Run(self.object_contours),
            downstream=["display", "rectify"],
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
        # Rectifying to real-world units and refreshing the text preview
        # happens automatically whenever the selection changes - Export is
        # only manually triggered to write the already-rectified contours
        # out to an SVG file.
        self.pipeline.Register("rectify", self.update_rectified_contours)
        self.pipeline.Register("export", self.export_contours)
        self.pipeline.Register("display", self.update_display)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        self.show_original_btn = QPushButton("Show Original")
        self.show_original_btn.clicked.connect(self.show_original_image)
        self.show_original_btn.setEnabled(False)

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(lambda: self.pipeline.RunFrom("export"))
        self.export_btn.setEnabled(False)

        # Read-only preview of the current selection's transformed contour
        # points - refreshed automatically whenever it changes, no popup
        # needed.
        self.contour_text_edit = QTextEdit()
        self.contour_text_edit.setReadOnly(True)

        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.show_original_btn)

        # What a click on the image view does - see the _MODE_* constants.
        control_layout.addWidget(QLabel("Click Mode:"))
        self.interaction_mode_combo = QComboBox()
        self.interaction_mode_combo.addItems([_MODE_SEGMENT, _MODE_SELECT_CONTOUR, _MODE_SELECT_FIDUCIAL])
        control_layout.addWidget(self.interaction_mode_combo)

        # Each pipeline stage's CreateWidget returns its own titled
        # QGroupBox, so the control panel visually mirrors the stage graph
        # without SVGGui needing to know each stage's display title.
        control_layout.addWidget(
            self.segmenter_stage.CreateWidget(on_change=lambda: self.pipeline.RunFrom("segmentation"))
        )

        # Calibration stage widget: a status label showing how many ArUco
        # markers were detected/matched - no parameters to tune from the
        # control panel (the marker layout lives in
        # ArucoCalibration.parameters.marker_positions_mm).
        control_layout.addWidget(self.calibration_stage.CreateWidget(on_change=lambda: None))

        # Morphology cleanup parameters: settled edits rerun morphology
        # (and contours/display downstream of it) through the pipeline.
        control_layout.addWidget(
            self.morphology_stage.CreateWidget(on_change=lambda: self.pipeline.RunFrom("morphology"))
        )

        # Contour selection parameters: settled edits rerun selection (and
        # display downstream of it) through the pipeline. Selecting an
        # object itself happens by clicking it in the image view.
        control_layout.addWidget(
            self.contour_selection_stage.CreateWidget(on_change=lambda: self.pipeline.RunFrom("selection"))
        )

        # SVG export parameters: just the output filename - writing the
        # file itself happens when the user clicks Export, not on edit.
        control_layout.addWidget(self.svg_export_stage.CreateWidget(on_change=lambda: None))

        control_layout.addWidget(self.export_btn)

        # Text preview of the current selection's transformed contour
        # points - takes up the rest of the panel instead of a stretch
        # spacer.
        control_layout.addWidget(QLabel("Exported Contour Points:"))
        control_layout.addWidget(self.contour_text_edit, stretch=1)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b;")
        # A QLabel holding a pixmap asks the layout for the pixmap's own
        # size, so without this the window grows the moment a photo is
        # loaded and can never be made smaller again. `_to_pixmap` already
        # scales the image down to whatever room the label was given, so
        # the label needs no size of its own.
        self.image_label.setMinimumSize(1, 1)
        self.image_label.mousePressEvent = self.image_clicked

        layout.addWidget(control_panel, stretch=1)
        layout.addWidget(self.image_label, stretch=3)

    def load_image(self, file_path: str | None = None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if self.original_image is not None:
                self.processed_image = self.original_image.copy()
                self.show_original_btn.setEnabled(True)
                # Not exportable until something is selected - a new image
                # invalidates whatever the last one had. update_rectified_contours
                # turns this back on when there is something to write.
                self.export_btn.setEnabled(False)

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
        self.object_contours = [] if mask_image is None else FindContours(mask_image)

        # Here rather than in the selection stage, because this is the one
        # place the contour list is rebuilt - moving the simplification
        # slider re-runs "selection" against the same list, and must not
        # undo a deliberate deselection.
        self.contour_selection_stage.contour_selection.SelectSoleContour(self.object_contours)

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

        mode = self.interaction_mode_combo.currentText()

        if mode == _MODE_SEGMENT:
            self.segmenter_stage.OnClick(ev)
            return

        # The other modes toggle something under the click; neither adds a
        # segmentation point.
        coords = WidgetToImageCoords(self.image_label, self.processed_image.shape, ev)
        if coords is None:
            return
        img_x, img_y = coords

        if mode == _MODE_SELECT_CONTOUR:
            if self.contour_selection_stage.contour_selection.ToggleSelection(img_x, img_y, self.object_contours):
                self.pipeline.RunFrom("selection")
            return

        if mode == _MODE_SELECT_FIDUCIAL:
            if self.calibration_stage.calibration.ToggleSelection(img_x, img_y):
                self.pipeline.RunFrom("display")
            return

    def _in_image_pixels(self, screen_pixels: float, image: np.ndarray) -> int:
        """A size given in on-screen pixels, in the image pixels an overlay
        has to be drawn at to come out that size.

        At least one, because a line of thickness zero draws nothing and a
        photo scaled far enough down would silently erase every
        annotation on it.
        """
        factor = ImagePixelsPerScreenPixel(self.image_label, image.shape)
        return max(1, round(screen_pixels * factor))

    def _draw_click_markers(self, image: np.ndarray) -> None:
        """A green '+' for each positive (label 1) segmentation click, a
        red '-' for each negative (label 0) one.
        """
        click_recorder = self.segmenter_stage.click_recorder
        if click_recorder is None:
            return

        marker_len = self._in_image_pixels(CLICK_MARKER_PX, image)
        thickness = self._in_image_pixels(CLICK_THICKNESS_PX, image)
        for (x, y), label in zip(click_recorder.image_points, click_recorder.image_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.line(image, (x - marker_len, y), (x + marker_len, y), color, thickness)
            if label == 1:
                cv2.line(image, (x, y - marker_len), (x, y + marker_len), color, thickness)

    def _draw_contour_overlays(self, image: np.ndarray, overlay: np.ndarray) -> None:
        """Every detected object's boundary - green if selected, yellow if
        not - plus, for each selected object, its filled overlay (blended
        in later for a transparency effect), its simplified outline, and
        its PCA-aligned box and center.
        """
        selected_objects = self.contour_selection_stage.contour_selection.selected
        boundary = self._in_image_pixels(CONTOUR_THICKNESS_PX, image)
        simplified_width = self._in_image_pixels(SIMPLIFIED_THICKNESS_PX, image)
        box_width = self._in_image_pixels(BOX_THICKNESS_PX, image)
        center_radius = self._in_image_pixels(CENTER_RADIUS_PX, image)

        for i, contour in enumerate(self.object_contours):
            color = (0, 255, 0) if i in selected_objects else (0, 255, 255)
            cv2.drawContours(image, [contour], -1, color, boundary)
            if i not in selected_objects:
                continue

            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), -1)

            simplified_contour = self.contour_selection_stage.contour_selection.simplified.get(i)
            pca_box = self.contour_selection_stage.contour_selection.boxes.get(i)
            if simplified_contour is None or pca_box is None:
                continue

            cv2.drawContours(image, [simplified_contour], -1, (255, 255, 0), simplified_width)
            cv2.drawContours(image, [pca_box.corners], -1, (255, 0, 255), box_width)
            cv2.circle(image, tuple(pca_box.center.astype(np.int32)), center_radius, (255, 0, 255), -1)

    def _to_pixmap(self, image: np.ndarray) -> QPixmap:
        height, width, _channels = image.shape
        q_image = QImage(image.tobytes(), width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(q_image).scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def update_display(self):
        if self.processed_image is None:
            return

        display_image = self.processed_image.copy()
        overlay = display_image.copy()  # accumulates the selected-object fills, blended in below

        self._draw_click_markers(display_image)
        if self.object_contours:
            self._draw_contour_overlays(display_image, overlay)

        alpha = 0.3  # how strongly a selected object's fill shows through
        display_image = cv2.addWeighted(display_image, 1 - alpha, overlay, alpha, 0)

        self.image_label.setPixmap(self._to_pixmap(display_image))

    def update_rectified_contours(self):
        """Recomputes real-world (mm) contours for the current selection
        and refreshes the text preview. Runs automatically whenever the
        selection changes, not just when Export is clicked.
        """
        contour_selection = self.contour_selection_stage.contour_selection
        if not contour_selection.selected or not contour_selection.simplified:
            self.rectify.contours = {}
            self.contour_text_edit.clear()
            self.export_btn.setEnabled(False)
            return

        try:
            transform = self.calibration_stage.calibration.GetTransform()
        except ValueError:
            # No calibration sheet detected (or not enough of it) - fall
            # back to pixel space rather than blocking the preview entirely.
            transform = np.eye(3, dtype=np.float32)

        self.rectify.Run(transform, contour_selection.simplified)
        self._update_contour_text(contour_selection.selected, self.rectify.contours)

        # Export writes `self.rectify.contours` and returns silently when
        # they are empty, which looked exactly like a broken button. Gate
        # it on the thing it actually needs instead.
        self.export_btn.setEnabled(bool(self.rectify.contours))

    def _update_contour_text(self, selected_objects, contours):
        """Formats each selected object's transformed contour points into
        the text preview box.
        """
        contour_text = ""
        for obj_id in selected_objects:
            if obj_id not in contours:
                continue

            # Transform contour to origin-based coordinate system
            points = contours[obj_id].reshape(-1, 2).astype(np.float32)
            transformed_points = PCABox(points).ToLocal(points)

            contour_text += f"Object {obj_id} Simplified Contour Points (Transformed to Origin):\n"
            contour_text += "[\n"

            # Format transformed points
            for i, (x, y) in enumerate(transformed_points):
                contour_text += f"  [{x:.2f}, {y:.2f}]"
                if i < len(transformed_points) - 1:
                    contour_text += ","
                contour_text += "\n"

            contour_text += "]\n\n"

        self.contour_text_edit.setPlainText(contour_text)

    def export_contours(self):
        """Writes the current selection's real-world contours out - SVG,
        PDF, and the JSON dump `layout_cli` reads. Rectification and the
        text preview already happened automatically when the selection
        last changed (see update_rectified_contours).
        """
        self.svg_export_stage.Run(self.rectify.contours)


def main():
    # QApplication strips any Qt-specific flags (e.g. -style) out of
    # sys.argv in place, so parse our own arguments from what's left.
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(description="Interactive photo-to-Gridfinity-contour capture tool.")
    parser.add_argument("image_path", nargs="?", default=None, help="Image to load at launch (optional).")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help=(
            "Allow downloading the SAM2 model from the Hugging Face Hub "
            "if it isn't already cached locally (default: offline, local "
            "files only)."
        ),
    )
    args = parser.parse_args(sys.argv[1:])

    window = SVGGui(local_files_only=not args.download_model)
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
    if args.image_path:
        # Normalize path: expand ~ and env vars, resolve relative to CWD
        launch_path = os.path.expanduser(os.path.expandvars(args.image_path))
        if not os.path.isabs(launch_path):
            launch_path = os.path.abspath(launch_path)
        window.load_image(launch_path)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
