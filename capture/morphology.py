from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage import morphology

from geometry.pca_box import PCABox


@dataclass
class MorphologyParameters:
    """User-configurable inputs for Morphology.

    `closing_radius` bridges small gaps and smooths away fine concave
    detail (e.g. SAM mask jitter along an edge); `area` is then the
    smallest hole or object, in pixels, that survives the filtering which
    follows it.

    Manmade objects are often symmetric, so the mask can optionally be
    combined with its own reflection across its PCA axes - lateral
    (left/right, across the major axis) and/or longitudinal (front/back,
    across the minor axis) - to clean up one-sided segmentation errors.
    `symmetry_combine` chooses what "combined" means: "or" fills a gap in
    from its intact mirror, "and" carves the gap out of both sides.
    """

    area: int = 1000
    closing_radius: int = 5
    symmetrize_lateral: bool = False
    symmetrize_longitudinal: bool = False
    symmetry_combine: Literal["and", "or"] = "or"


def _PrincipalBox(mask_u8: np.ndarray) -> PCABox | None:
    """A PCABox over a mask's foreground pixels, or None if the mask is
    empty.
    """
    points = cv2.findNonZero(mask_u8)
    if points is None:
        return None
    return PCABox(points.reshape(-1, 2).astype(np.float32))


def _ReflectAcrossAxis(mask_u8: np.ndarray, center: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Reflects mask_u8 across the line through `center` in direction
    `axis`: points on the line are unchanged, the perpendicular component
    flips sign.
    """
    angle = float(np.arctan2(axis[1], axis[0]))
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    rotation = np.array([[c, s], [s, -c]], dtype=np.float32)
    translation = center - rotation @ center
    transform = np.hstack([rotation, translation.reshape(2, 1)]).astype(np.float32)
    height, width = mask_u8.shape[:2]
    return cv2.warpAffine(mask_u8, transform, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)


class Morphology:
    def __init__(self) -> None:
        self.parameters = MorphologyParameters()

    def Apply(self, mask_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        area = self.parameters.area
        radius = self.parameters.closing_radius
        if radius > 0:
            # OpenCV, given scikit-image's own disk so the two agree bit for
            # bit - checked at radii 2, 5 and 9 on a real SAM2 mask. Only the
            # speed differs: scikit-image is quadratic in the radius, OpenCV
            # nearly flat, so the default radius of 5 costs 3.7ms rather than
            # 263ms. This was three quarters of `Apply`, which runs on every
            # click. The GPU loses here - OpenCL is slower than CPU at this
            # size, and CUDA wins 3.7ms to 2.7ms for a stage that would then
            # need CUDA.
            footprint = np.asarray(morphology.disk(radius), dtype=np.uint8)
            mask_image = cv2.morphologyEx(mask_image.astype(np.uint8), cv2.MORPH_CLOSE, footprint).astype(bool)
        mask_image = morphology.remove_small_holes(mask_image, area_threshold=area)
        mask_image = morphology.remove_small_objects(mask_image, min_size=area)

        if self.parameters.symmetrize_lateral or self.parameters.symmetrize_longitudinal:
            mask_image = self._ApplySymmetry(mask_image)

        return mask_image

    def _ApplySymmetry(self, mask_image: np.ndarray) -> np.ndarray:
        mask_u8 = mask_image.astype(np.uint8) * 255
        box = _PrincipalBox(mask_u8)
        if box is None:
            return mask_image  # empty mask, nothing to reflect against

        # Pivot on the bounding box's center, not the mask's center of mass:
        # for a lopsided object (e.g. a spoon's bowl vs. its handle) those
        # differ substantially along the major axis, and reflecting about
        # the center of mass would mirror the shape well past its own
        # extent instead of across its visual middle.
        bbox_center = box.center + (box.min1 + box.max1) / 2 * box.pc1 + (box.min2 + box.max2) / 2 * box.pc2

        variants = [mask_image]
        if self.parameters.symmetrize_lateral:
            variants.append(_ReflectAcrossAxis(mask_u8, bbox_center, box.pc1) > 0)
        if self.parameters.symmetrize_longitudinal:
            variants.append(_ReflectAcrossAxis(mask_u8, bbox_center, box.pc2) > 0)

        combine = np.logical_and if self.parameters.symmetry_combine == "and" else np.logical_or
        result = variants[0]
        for variant in variants[1:]:
            result = combine(result, variant)
        return result
