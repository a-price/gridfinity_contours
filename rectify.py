import cv2
import numpy as np


class Rectify:
    """Applies a calibration's affine transform to a set of selected
    objects' simplified contours, producing real-world (mm) coordinates.
    """

    def __init__(self) -> None:
        self.contours: dict[int, np.ndarray] = {}

    def Run(
        self,
        affine: cv2.typing.MatLike,
        simplified_contours: dict[int, cv2.typing.MatLike],
    ) -> None:
        self.contours = {}
        for i, contour in simplified_contours.items():
            points = np.squeeze(contour).astype(np.float64)
            points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
            self.contours[i] = (affine @ points.T).T
