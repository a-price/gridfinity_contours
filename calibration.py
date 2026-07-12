from dataclasses import dataclass

import cv2
import numpy as np


class Calibration:
    """Base class for fiducial-based image calibration.

    Subclasses locate a known real-world fiducial in an image and produce
    the affine transform mapping image pixel coordinates to real-world (mm)
    coordinates, so an extracted contour can be rectified/rescaled.
    """

    def Detect(self, image: cv2.typing.MatLike) -> None:
        """Locate fiducial candidates in `image`."""
        raise NotImplementedError

    def GetTransform(self) -> cv2.typing.MatLike:
        """Return the 2x3 affine transform from image pixels to real-world mm."""
        raise NotImplementedError

    def DebugLayer(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        raise NotImplementedError


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

    def Detect(self, image: cv2.typing.MatLike) -> None:
        p = self.parameters
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, p.threshold_value, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.medianBlur(binary, 5)

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
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0, : p.max_circles]:
                self.circles.append((int(x), int(y), int(r)))

        # Select the first 3 circles by default
        self.selected_circles = set(range(min(3, len(self.circles))))

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
                "HoughCircleCalibration needs exactly 3 selected circles, "
                f"has {len(self.selected_circles)}"
            )

        centers = np.float32(self.circles)[:, :2]
        used = list(self.selected_circles)

        # a: leftmost of the 3; c: topmost (smallest y) of the other two;
        # b: the remaining one, at the right-angle corner.
        a = used[int(np.argmin(centers[used, 0]))]
        remaining = [i for i in used if i != a]
        c = remaining[int(np.argmin(centers[remaining, 1]))]
        b = next(i for i in remaining if i != c)

        image_points = centers[[a, b, c]]
        leg = self.parameters.leg_distance_mm
        target_points = np.float32([[leg, 0], [0, 0], [0, leg]])
        return cv2.getAffineTransform(image_points, target_points)


class PaperCalibration(Calibration):
    """Calibrates against a standard 8.5in x 11in sheet of paper visible in
    the frame, using 3 of its corners as fiducials.
    """

    WIDTH_MM = 215.9  # 8.5 in
    HEIGHT_MM = 279.4  # 11 in

    def __init__(self) -> None:
        self.threshold_value = 200  # paper is bright against most work surfaces
        self.corners: list[tuple[float, float]] | None = None  # TL, TR, BR, BL

    def Detect(self, image: cv2.typing.MatLike) -> None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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
        return [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]

    def GetTransform(self) -> cv2.typing.MatLike:
        if self.corners is None:
            raise ValueError("PaperCalibration has not detected a 4-corner sheet yet")

        top_left, top_right, _, bottom_left = self.corners
        image_points = np.float32([top_left, top_right, bottom_left])
        target_points = np.float32([[0, 0], [self.WIDTH_MM, 0], [0, self.HEIGHT_MM]])
        return cv2.getAffineTransform(image_points, target_points)
