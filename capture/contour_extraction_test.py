import cv2
import numpy as np

from capture.contour_extraction import ContourSelection, ExtractContour, FindContours


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


# --------------------------------------------------- selecting a lone contour


def _square(size: int = 10):
    """A contour in the shape FindContours returns - Nx1x2, int32."""
    return np.array([[[0, 0]], [[size, 0]], [[size, size]], [[0, size]]], dtype=np.int32)


def test_a_single_contour_selects_itself():
    """The whole point: one object means there is nothing to choose, and
    leaving it unchosen strands the user with an Export that writes
    nothing and says nothing.
    """
    selection = ContourSelection()

    assert selection.SelectSoleContour([_square()]) is True
    assert selection.selected == {0}


def test_several_contours_still_need_a_choice():
    selection = ContourSelection()

    assert selection.SelectSoleContour([_square(), _square(20)]) is False
    assert selection.selected == set()


def test_no_contours_selects_nothing():
    selection = ContourSelection()

    assert selection.SelectSoleContour([]) is False
    assert selection.selected == set()


def test_an_existing_choice_is_not_overridden():
    """Rebuilding the contour list must not move a selection the user made,
    even down to a single contour - index 0 may not be what they picked.
    """
    selection = ContourSelection()
    selection.selected = {3}

    assert selection.SelectSoleContour([_square()]) is False
    assert selection.selected == {3}


def test_a_deliberate_deselection_is_reinstated_only_by_new_contours():
    """Deselecting the lone contour has to stick while the user works the
    simplification slider - that re-runs `Run`, not this. A rebuilt list
    is a different question, and does bring it back.
    """
    selection = ContourSelection()
    selection.SelectSoleContour([_square()])
    selection.selected = set()  # the user clicks it off

    selection.Run([_square()])  # what the slider triggers
    assert selection.selected == set()

    assert selection.SelectSoleContour([_square()]) is True  # what a new mask triggers
    assert selection.selected == {0}
