from dataclasses import dataclass

import cv2
from skimage import morphology


@dataclass
class MorphologyParameters:
    """User-configurable inputs for Morphology: the minimum area (in
    pixels) a hole or object must have to survive cleanup, and the radius
    of the closing operation used to bridge small gaps and smooth away
    fine concave detail (e.g. SAM mask jitter along an edge) before that
    area filtering runs.
    """

    area: int = 1000
    closing_radius: int = 5


class Morphology:
    def __init__(self) -> None:
        self.parameters = MorphologyParameters()

    def Apply(self, mask_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        area = self.parameters.area
        radius = self.parameters.closing_radius
        if radius > 0:
            mask_image = morphology.binary_closing(mask_image, morphology.disk(radius))
        mask_image = morphology.remove_small_holes(mask_image, area_threshold=area)
        mask_image = morphology.remove_small_objects(mask_image, min_size=area)
        return mask_image
