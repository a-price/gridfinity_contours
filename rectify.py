import cv2
import numpy as np


class Rectify:
    """Applies a calibration's affine transform to the processed image and
    a set of selected objects' simplified contours, producing real-world
    (mm) coordinates.
    """

    def __init__(self) -> None:
        self.warped_image: np.ndarray | None = None
        self.contours: dict[int, np.ndarray] = {}

    def Run(
        self,
        image: cv2.typing.MatLike,
        affine: cv2.typing.MatLike,
        simplified_contours: dict[int, cv2.typing.MatLike],
    ) -> None:
        warped = np.transpose(np.zeros(image.shape), [1, 0, 2])
        for channel in range(3):
            warped[:, :, channel] = cv2.warpAffine(
                image[:, :, channel], affine, image.shape[:2]
            )
        self.warped_image = warped

        self.contours = {}
        for i, contour in simplified_contours.items():
            points = np.squeeze(contour).astype(np.float64)
            points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
            self.contours[i] = (affine @ points.T).T
