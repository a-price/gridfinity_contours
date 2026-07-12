import cv2
import numpy as np


class PCABox:
    """A PCA-aligned bounding box for a set of 2D points: an oriented
    rectangle whose axes are the points' principal components.
    """

    def __init__(self, points: np.ndarray) -> None:
        mean, eigenvectors, _ = cv2.PCACompute2(points, np.array([]))
        self.center = mean[0]
        self.pc1 = eigenvectors[0]
        self.pc2 = eigenvectors[1]

        projected1 = np.dot(points - self.center, self.pc1)
        projected2 = np.dot(points - self.center, self.pc2)
        self.min1, self.max1 = float(projected1.min()), float(projected1.max())
        self.min2, self.max2 = float(projected2.min()), float(projected2.max())

    @property
    def corners(self) -> np.ndarray:
        """The box's 4 corners in the original point space, as int32, ready
        for cv2.drawContours.
        """
        corners_local = np.array(
            [
                [self.min1, self.min2],
                [self.max1, self.min2],
                [self.max1, self.max2],
                [self.min1, self.max2],
            ]
        )
        corners = np.array(
            [self.center + c[0] * self.pc1 + c[1] * self.pc2 for c in corners_local]
        )
        return np.int32(corners)

    def ToLocal(self, points: np.ndarray) -> np.ndarray:
        """Project `points` into the box's local frame: PCA-aligned axes,
        origin at the box's minimum corner.
        """
        projected1 = np.dot(points - self.center, self.pc1)
        projected2 = np.dot(points - self.center, self.pc2)
        return np.stack([projected1 - self.min1, projected2 - self.min2], axis=-1)


def FindContours(mask_image: cv2.typing.MatLike) -> list[cv2.typing.MatLike]:
    """Find external object contours in a cleaned-up mask (e.g. the output
    of Morphology.Apply).
    """
    if mask_image.dtype == bool:
        mask_u8 = mask_image.astype(np.uint8) * 255
    else:
        mask_u8 = mask_image.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def SimplifyContour(
    contour: cv2.typing.MatLike, epsilon_fraction: float = 0.001
) -> cv2.typing.MatLike:
    """Simplify a contour with the Douglas-Peucker algorithm."""
    epsilon = epsilon_fraction * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def ExtractContour(
    contour: cv2.typing.MatLike, epsilon_fraction: float = 0.001
) -> tuple[cv2.typing.MatLike, PCABox]:
    """Simplify a raw contour and compute its PCA-aligned bounding box.

    Returns (simplified_contour, pca_box).
    """
    simplified = SimplifyContour(contour, epsilon_fraction)
    points = simplified.reshape(-1, 2).astype(np.float32)
    return simplified, PCABox(points)
