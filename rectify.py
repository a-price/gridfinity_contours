import cv2
import numpy as np


class Rectify:
    """Applies a calibration's 3x3 homogeneous transform (see
    Calibration.GetTransform) to a set of selected objects' simplified
    contours, producing real-world (mm) coordinates.
    """

    def __init__(self) -> None:
        self.contours: dict[int, np.ndarray] = {}

    def Run(
        self,
        transform: cv2.typing.MatLike,
        simplified_contours: dict[int, cv2.typing.MatLike],
    ) -> None:
        self.contours = {}
        for i, contour in simplified_contours.items():
            points = np.squeeze(contour).astype(np.float64)
            points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
            transformed = (transform @ points.T).T
            # Perspective divide: a no-op for a simple affine fit (whose
            # bottom row is [0, 0, 1], so the third column is always 1),
            # but required for a true perspective homography.
            self.contours[i] = transformed[:, :2] / transformed[:, 2:3]
