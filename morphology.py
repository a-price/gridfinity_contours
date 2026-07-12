import cv2
from skimage import morphology


class Morphology:
    def __init__(self) -> None:
        self.area = 1000

    def Update(self, area: int) -> None:
        self.area = area

    def Apply(self, mask_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        mask_image = morphology.binary_closing(mask_image)
        mask_image = morphology.remove_small_holes(mask_image, area_threshold=self.area)
        mask_image = morphology.remove_small_objects(mask_image, min_size=self.area)
        return mask_image
