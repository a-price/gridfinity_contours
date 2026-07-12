import cv2
import numpy as np

from calibration import HoughCircleCalibration, PaperCalibration


def _draw_circle_image(circles: list[tuple[int, int, int]]) -> cv2.typing.MatLike:
    image = np.zeros((900, 900, 3), dtype=np.uint8)
    for x, y, r in circles:
        cv2.circle(image, (x, y), r, (255, 255, 255), -1)
    return image


def test_hough_circle_calibration_detects_and_selects_first_three():
    circles = [(150, 150, 20), (150, 600, 20), (600, 600, 20)]
    image = _draw_circle_image(circles)

    calibration = HoughCircleCalibration()
    calibration.min_dist = 50
    calibration.threshold_value = 100
    calibration.min_radius = 10
    calibration.max_radius = 40
    calibration.param1 = 50
    calibration.param2 = 15
    calibration.Detect(image)

    assert len(calibration.circles) == 3
    assert calibration.selected_circles == {0, 1, 2}


def test_hough_circle_calibration_transform_is_index_order_independent():
    # b is the right-angle corner, a is the leftmost point (offset along the
    # +x leg from b), c is offset along the +y leg from b (smaller pixel y =
    # "up"). Deliberately insert out of a/b/c order and select a
    # non-contiguous, unsorted subset of indices, to exercise the same code
    # path that used to break when selected indices weren't exactly {0, 1, 2}.
    b_point, a_point, c_point = (300.0, 500.0), (100.0, 500.0), (300.0, 200.0)
    distractor = (800.0, 800.0)

    calibration = HoughCircleCalibration()
    calibration.leg_distance_mm = 80.0
    calibration.circles = [
        (*b_point, 20),
        (*distractor, 20),
        (*a_point, 20),
        (*c_point, 20),
    ]
    calibration.selected_circles = {2, 0, 3}  # a, b, c - unsorted, non-contiguous

    affine = calibration.GetTransform()

    def transform(point):
        return affine @ np.array([point[0], point[1], 1.0])

    assert np.allclose(transform(a_point), [80.0, 0.0], atol=1e-3)
    assert np.allclose(transform(b_point), [0.0, 0.0], atol=1e-3)
    assert np.allclose(transform(c_point), [0.0, 80.0], atol=1e-3)


def test_hough_circle_calibration_requires_exactly_three_selected():
    calibration = HoughCircleCalibration()
    calibration.circles = [(0, 0, 5), (1, 1, 5)]
    calibration.selected_circles = {0, 1}

    try:
        calibration.GetTransform()
        assert False, "expected ValueError for fewer than 3 selected circles"
    except ValueError:
        pass


def test_paper_calibration_detects_corners_and_transform():
    image = np.zeros((900, 900, 3), dtype=np.uint8)
    top_left, bottom_right = (100, 150), (500, 750)
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)

    calibration = PaperCalibration()
    calibration.threshold_value = 100
    calibration.Detect(image)

    assert calibration.corners is not None
    detected_top_left = calibration.corners[0]
    detected_bottom_right = calibration.corners[2]
    assert np.allclose(detected_top_left, top_left, atol=2)
    assert np.allclose(detected_bottom_right, bottom_right, atol=2)

    affine = calibration.GetTransform()

    def transform(point):
        return affine @ np.array([point[0], point[1], 1.0])

    assert np.allclose(transform(top_left), [0.0, 0.0], atol=1e-1)
    assert np.allclose(
        transform((bottom_right[0], top_left[1])),
        [PaperCalibration.WIDTH_MM, 0.0],
        atol=1e-1,
    )
    assert np.allclose(
        transform((top_left[0], bottom_right[1])),
        [0.0, PaperCalibration.HEIGHT_MM],
        atol=1e-1,
    )


def test_paper_calibration_no_quadrilateral_found():
    image = np.zeros((200, 200, 3), dtype=np.uint8)  # blank, nothing bright

    calibration = PaperCalibration()
    calibration.Detect(image)

    assert calibration.corners is None
