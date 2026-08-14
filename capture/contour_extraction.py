from dataclasses import dataclass

import cv2
import numpy as np

from geometry.pca_box import PCABox


def FindContours(mask_image: np.ndarray) -> list[cv2.typing.MatLike]:
    """Find external object contours in a cleaned-up mask (e.g. the output
    of Morphology.Apply).
    """
    if mask_image.dtype == bool:
        mask_u8 = mask_image.astype(np.uint8) * 255
    else:
        mask_u8 = mask_image.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def SimplifyContour(contour: cv2.typing.MatLike, epsilon_fraction: float = 0.001) -> cv2.typing.MatLike:
    """Simplify a contour with the Douglas-Peucker algorithm."""
    epsilon = epsilon_fraction * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def ExtractContour(contour: cv2.typing.MatLike, epsilon_fraction: float = 0.001) -> tuple[cv2.typing.MatLike, PCABox]:
    """Simplify a raw contour and compute its PCA-aligned bounding box.

    Returns (simplified_contour, pca_box).
    """
    simplified = SimplifyContour(contour, epsilon_fraction)
    points = simplified.reshape(-1, 2).astype(np.float32)
    return simplified, PCABox(points)


@dataclass
class ContourSelectionParameters:
    """User-configurable inputs for ContourSelection."""

    epsilon_fraction: float = 0.001


class ContourSelection:
    """Tracks which object contours the user has picked (by clicking inside
    them), and the simplified contour + PCA box computed for each one.
    """

    def __init__(self) -> None:
        self.parameters = ContourSelectionParameters()
        self.selected: set[int] = set()
        self.simplified: dict[int, cv2.typing.MatLike] = {}
        self.boxes: dict[int, PCABox] = {}

    def ToggleSelection(self, x: int, y: int, contours: list) -> bool:
        """Toggle selection of the contour under image coordinates (x, y).
        Returns True if a contour was hit.
        """
        for i, contour in enumerate(contours):
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                if i in self.selected:
                    self.selected.remove(i)
                else:
                    self.selected.add(i)
                return True
        return False

    def Run(self, contours: list) -> None:
        """Recompute the simplified contour + PCA box for each selected
        index against the current `contours` list.
        """
        self.simplified = {}
        self.boxes = {}
        for i in self.selected:
            if i >= len(contours):
                continue
            simplified, box = ExtractContour(contours[i], self.parameters.epsilon_fraction)
            self.simplified[i] = simplified
            self.boxes[i] = box
