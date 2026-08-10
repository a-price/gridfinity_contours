import cv2
import numpy as np

from capture.contour_extraction import ExtractContour, FindContours


def _rectangle_mask(top_left, bottom_right, shape=(300, 300)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x0, y0 = top_left
    x1, y1 = bottom_right
    mask[y0:y1, x0:x1] = True
    return mask


def test_find_contours_on_boolean_mask():
    mask = _rectangle_mask((50, 50), (150, 200))
    contours = FindContours(mask)

    assert len(contours) == 1
    x, y, w, h = cv2.boundingRect(contours[0])
    assert (x, y, w, h) == (50, 50, 100, 150)


def test_find_contours_on_uint8_mask():
    mask = _rectangle_mask((50, 50), (150, 200)).astype(np.uint8) * 255
    contours = FindContours(mask)

    assert len(contours) == 1
    x, y, w, h = cv2.boundingRect(contours[0])
    assert (x, y, w, h) == (50, 50, 100, 150)


def test_find_contours_multiple_objects():
    mask = _rectangle_mask((10, 10), (40, 40)) | _rectangle_mask((100, 100), (140, 140))
    contours = FindContours(mask)
    assert len(contours) == 2


def test_extract_contour_returns_simplified_contour_and_box():
    mask = _rectangle_mask((50, 50), (150, 200))
    (contour,) = FindContours(mask)

    simplified, pca_box = ExtractContour(contour)

    # A rectangle simplifies to (close to) its 4 corners.
    assert 4 <= len(simplified) <= 6

    # The PCA box for an axis-aligned rectangle should reproduce its extents.
    corners = pca_box.corners
    assert cv2.boundingRect(corners) == cv2.boundingRect(contour)
