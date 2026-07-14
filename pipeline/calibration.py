from dataclasses import dataclass, field

import cv2
import numpy as np


def _AffineToHomogeneous(affine: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Pad a 2x3 affine transform into a 3x3 homogeneous matrix (bottom row
    [0, 0, 1]), so every Calibration.GetTransform() shares one contract
    regardless of whether it's a simple affine fit or a full perspective
    homography (e.g. ArucoCalibration's PnP-based one) - the bottom row of
    [0, 0, 1] makes the perspective divide in Rectify.Run a no-op for the
    affine case.
    """
    return np.vstack([affine, [0, 0, 1]]).astype(np.float32)


class Calibration:
    """Base class for fiducial-based image calibration.

    Subclasses locate a known real-world fiducial in an image and produce
    the transform mapping image pixel coordinates to real-world (mm)
    coordinates, so an extracted contour can be rectified/rescaled.
    """

    def Detect(self, image: cv2.typing.MatLike) -> None:
        """Locate fiducial candidates in `image`."""
        raise NotImplementedError

    def GetTransform(self) -> cv2.typing.MatLike:
        """Return the 3x3 homogeneous transform from image pixels to
        real-world mm: apply as (T @ [x, y, 1]), then divide the result's
        first two components by its third (a no-op for calibrations that
        are really just an affine fit, since their bottom row is
        [0, 0, 1] - but required for a true perspective homography).
        """
        raise NotImplementedError

    def DebugLayer(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        raise NotImplementedError

    def ToggleSelection(self, x: int, y: int) -> bool:
        """Toggle selection of the fiducial under image coordinates (x, y).
        Returns True if one was hit. Default: no selectable fiducials (e.g.
        auto-detected ones like PaperCalibration's corners, or a stub like
        IdentityCalibration) - subclasses that support manual selection
        (HoughCircleCalibration) override this.
        """
        return False


class IdentityCalibration(Calibration):
    """Stub calibration with no fiducial to detect: treats image pixels as
    millimeters 1:1. Not wired into the UI by default (ArucoCalibration is),
    but available as a fallback/simple option for callers that don't need
    real-world units.
    """

    def Detect(self, image: cv2.typing.MatLike) -> None:
        pass

    def GetTransform(self) -> cv2.typing.MatLike:
        return np.eye(3, dtype=np.float32)

    def DebugLayer(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        return image


@dataclass
class HoughCircleParameters:
    """User-configurable inputs for HoughCircleCalibration."""

    min_dist: int = 50
    param1: int = 100
    param2: int = 30
    min_radius: int = 10
    max_radius: int = 100
    threshold_value: int = 127
    max_circles: int = 5
    leg_distance_mm: float = 80.0


class HoughCircleCalibration(Calibration):
    """Calibrates against 3 selected circular fiducials arranged in an L
    shape: a right-angle corner circle, with the other two each
    `leg_distance_mm` away along the two legs.
    """

    def __init__(self) -> None:
        self.parameters = HoughCircleParameters()
        self.circles: list[tuple[int, int, int]] = []
        self.selected_circles: set[int] = set()

    def _Preprocess(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Grayscale -> blur -> threshold -> morphological cleanup -> median
        blur: the binary image Hough circle detection runs against.
        """
        p = self.parameters
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, p.threshold_value, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.medianBlur(binary, 5)
        return binary

    def Detect(self, image: cv2.typing.MatLike) -> None:
        p = self.parameters
        binary = self._Preprocess(image)

        circles = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=p.min_dist,
            param1=p.param1,
            param2=p.param2,
            minRadius=p.min_radius,
            maxRadius=p.max_radius,
        )

        self.circles = []
        if circles is not None:
            circles = np.array(np.around(circles), dtype=np.uint16)
            for x, y, r in circles[0, : p.max_circles]:
                self.circles.append((int(x), int(y), int(r)))

        # Select the first 3 circles by default
        self.selected_circles = set(range(min(3, len(self.circles))))

    def DebugLayer(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """The binary image circle detection actually runs against, as a
        3-channel BGR image ready for display.
        """
        return cv2.cvtColor(self._Preprocess(image), cv2.COLOR_GRAY2BGR)

    def ConfigureForImageShape(self, shape: tuple) -> None:
        """Set the min/max radius parameter defaults relative to the loaded
        image's smaller dimension: max radius defaults to 20%, min radius to
        2%.
        """
        h, w = shape[:2]
        min_dim = max(1, min(h, w))
        default_max = max(1, int(0.20 * min_dim))
        default_min = max(1, int(0.02 * min_dim))
        if default_min > default_max:
            default_min = default_max

        self.parameters.max_radius = default_max
        self.parameters.min_radius = default_min

    def ToggleSelection(self, x: int, y: int) -> bool:
        """Toggle selection of the circle under image coordinates (x, y).
        Returns True if a circle was hit.
        """
        for i, (cx, cy, r) in enumerate(self.circles):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r**2:
                if i in self.selected_circles:
                    self.selected_circles.remove(i)
                else:
                    self.selected_circles.add(i)
                return True
        return False

    def GetTransform(self) -> cv2.typing.MatLike:
        if len(self.selected_circles) != 3:
            raise ValueError(
                "HoughCircleCalibration needs exactly 3 selected circles, " f"has {len(self.selected_circles)}"
            )

        centers = np.array(self.circles, dtype=np.float32)[:, :2]
        used = list(self.selected_circles)

        # a: leftmost of the 3; c: topmost (smallest y) of the other two;
        # b: the remaining one, at the right-angle corner.
        a = used[int(np.argmin(centers[used, 0]))]
        remaining = [i for i in used if i != a]
        c = remaining[int(np.argmin(centers[remaining, 1]))]
        b = next(i for i in remaining if i != c)

        image_points = centers[[a, b, c]]
        leg = self.parameters.leg_distance_mm
        target_points = np.array([[leg, 0], [0, 0], [0, leg]], dtype=np.float32)
        return _AffineToHomogeneous(cv2.getAffineTransform(image_points, target_points))


class PaperCalibration(Calibration):
    """Calibrates against a standard 8.5in x 11in sheet of paper visible in
    the frame, using all 4 of its corners as fiducials.
    """

    WIDTH_MM = 215.9  # 8.5 in
    HEIGHT_MM = 279.4  # 11 in

    def __init__(self) -> None:
        self.threshold_value = 200  # paper is bright against most work surfaces
        self.corners: list[tuple[float, float]] | None = None  # TL, TR, BR, BL

    def Detect(self, image: cv2.typing.MatLike) -> None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.corners = None
        if not contours:
            return

        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
        if len(approx) != 4:
            return  # didn't find a clean quadrilateral

        self.corners = self._order_corners(approx.reshape(4, 2).astype(np.float32))

    @staticmethod
    def _order_corners(points: np.ndarray) -> list[tuple[float, float]]:
        """Order 4 corner points as top-left, top-right, bottom-right, bottom-left."""
        total = points.sum(axis=1)
        diff = np.diff(points, axis=1).ravel()
        top_left = points[np.argmin(total)]
        bottom_right = points[np.argmax(total)]
        top_right = points[np.argmin(diff)]
        bottom_left = points[np.argmax(diff)]
        return [
            tuple(top_left),
            tuple(top_right),
            tuple(bottom_right),
            tuple(bottom_left),
        ]

    def GetTransform(self) -> cv2.typing.MatLike:
        if self.corners is None:
            raise ValueError("PaperCalibration has not detected a 4-corner sheet yet")

        # All 4 corners (not just 3), via a full perspective transform: an
        # affine fit can only correct skew/scale/rotation, not the
        # perspective foreshortening a real (non-fronto-parallel) photo of
        # the sheet will have.
        top_left, top_right, bottom_right, bottom_left = self.corners
        image_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        target_points = np.array(
            [
                [0, 0],
                [self.WIDTH_MM, 0],
                [self.WIDTH_MM, self.HEIGHT_MM],
                [0, self.HEIGHT_MM],
            ],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(image_points, target_points).astype(np.float32)


# Defaults for the calibration sheet generate_aruco_sheet.py prints: one
# marker near each corner of a PaperCalibration-sized page. ArucoParameters
# defaults to this same layout, so a freshly printed sheet works with
# ArucoCalibration out of the box.
ARUCO_MARKER_SIZE_MM = 20.0
ARUCO_SHEET_MARGIN_MM = 15.0  # marker-edge-to-page-edge clearance


def DefaultArucoMarkerPositions(
    marker_size_mm: float = ARUCO_MARKER_SIZE_MM,
    page_width_mm: float = PaperCalibration.WIDTH_MM,
    page_height_mm: float = PaperCalibration.HEIGHT_MM,
    margin_mm: float = ARUCO_SHEET_MARGIN_MM,
) -> dict[int, tuple[float, float]]:
    """Marker IDs 0-3, one near each corner (top-left origin, y down),
    inset far enough from the page edge to survive typical printer
    margins. Matches the layout generate_aruco_sheet.py prints.
    """
    inset = margin_mm + marker_size_mm / 2
    return {
        0: (inset, inset),
        1: (page_width_mm - inset, inset),
        2: (page_width_mm - inset, page_height_mm - inset),
        3: (inset, page_height_mm - inset),
    }


@dataclass
class ArucoParameters:
    """User-configurable inputs for ArucoCalibration: the physical size of
    each printed marker, and the known real-world (mm) position of each
    marker's center on the calibration sheet, keyed by ArUco marker ID.
    Defaults to the layout generate_aruco_sheet.py prints; override
    `marker_positions_mm` to match a different printed layout.
    """

    marker_size_mm: float = ARUCO_MARKER_SIZE_MM
    marker_positions_mm: dict[int, tuple[float, float]] = field(default_factory=DefaultArucoMarkerPositions)


class ArucoCalibration(Calibration):
    """Calibrates against ArUco fiducial markers printed at known
    real-world (mm) positions on a sheet.

    Unlike the affine calibrations above, this corrects for real
    perspective distortion rather than just fitting a plane-preserving
    approximation: it detects each marker's 4 corners, solves PnP against
    their known real-world positions to recover the camera's pose relative
    to the sheet, then builds the image-to-sheet homography from that pose.

    Caveats (this hasn't been exercised against a real photo yet):
      * Camera intrinsics aren't actually calibrated - GetTransform() falls
        back to a rough pinhole guess (focal length ~ image width,
        principal point at the image center) via _EstimateCameraMatrix.
        That's good enough for a rough cut, but a real checkerboard-based
        cv2.calibrateCamera pass would meaningfully improve accuracy.
      * A single marker's 4 coplanar corners can leave solvePnP with a
        pose ambiguity at near-fronto-parallel viewing angles; using
        several markers spread across the sheet (as intended) avoids this.
    """

    def __init__(self) -> None:
        self.parameters = ArucoParameters()
        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, cv2.aruco.DetectorParameters())
        self.detected_corners: dict[int, np.ndarray] = {}  # marker id -> (4, 2) pixel corners
        self._image_shape: tuple[int, int] | None = None

    def Detect(self, image: cv2.typing.MatLike) -> None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        self._image_shape = image.shape[:2]
        self.detected_corners = {}
        if ids is None:
            return
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            self.detected_corners[int(marker_id)] = marker_corners.reshape(4, 2).astype(np.float64)

    def DebugLayer(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """The original image with detected marker outlines and IDs drawn
        on top, for checking detection quality.
        """
        debug = image.copy()
        if not self.detected_corners:
            return debug

        corners_list = [corners.reshape(1, 4, 2).astype(np.float32) for corners in self.detected_corners.values()]
        ids_array = np.array(list(self.detected_corners.keys())).reshape(-1, 1)
        cv2.aruco.drawDetectedMarkers(debug, corners_list, ids_array)
        return debug

    def _EstimateCameraMatrix(self) -> np.ndarray:
        """A rough pinhole camera matrix, absent a real calibration: focal
        length approximated as the image width, principal point at the
        image center.
        """
        if self._image_shape is None:
            raise ValueError("ArucoCalibration.Detect() hasn't run yet")
        height, width = self._image_shape
        focal_length = float(width)
        return np.array(
            [
                [focal_length, 0, width / 2],
                [0, focal_length, height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def GetTransform(self) -> cv2.typing.MatLike:
        """Solve PnP against detected markers with known real-world
        positions, then return the 3x3 homography mapping image pixels to
        the calibration sheet's real-world (mm) plane.
        """
        marker_positions = self.parameters.marker_positions_mm
        matched_ids = [marker_id for marker_id in self.detected_corners if marker_id in marker_positions]
        if not matched_ids:
            raise ValueError(
                "ArucoCalibration needs at least one detected marker with a "
                "known position in parameters.marker_positions_mm"
            )

        # cv2.aruco always returns a marker's 4 corners in the same order:
        # top-left, top-right, bottom-right, bottom-left of the marker's
        # own (decoded) orientation.
        half_size = self.parameters.marker_size_mm / 2
        corner_offsets = np.array(
            [
                [-half_size, -half_size],
                [half_size, -half_size],
                [half_size, half_size],
                [-half_size, half_size],
            ]
        )

        object_points = []
        image_points = []
        for marker_id in matched_ids:
            center_x, center_y = marker_positions[marker_id]
            for offset, pixel in zip(corner_offsets, self.detected_corners[marker_id]):
                object_points.append([center_x + offset[0], center_y + offset[1], 0.0])
                image_points.append(pixel)

        camera_matrix = self._EstimateCameraMatrix()
        dist_coeffs = np.zeros(5)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            np.array(object_points, dtype=np.float64),
            np.array(image_points, dtype=np.float64),
            camera_matrix,
            dist_coeffs,
        )
        if not success:
            raise ValueError("ArucoCalibration: solvePnP failed to converge")

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # All object points lie on z=0, so the plane-to-image homography
        # only needs the rotation matrix's x/y columns plus translation -
        # the z column (which would move points off-plane) is dropped.
        plane_to_image = camera_matrix @ np.hstack([rotation_matrix[:, :2], translation_vector])
        return np.linalg.inv(plane_to_image).astype(np.float32)
