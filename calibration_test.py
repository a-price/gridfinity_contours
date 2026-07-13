import cv2
import numpy as np

from calibration import HoughCircleCalibration, IdentityCalibration, PaperCalibration


def _draw_circle_image(circles: list[tuple[int, int, int]]) -> cv2.typing.MatLike:
    image = np.zeros((900, 900, 3), dtype=np.uint8)
    for x, y, r in circles:
        cv2.circle(image, (x, y), r, (255, 255, 255), -1)
    return image


def _apply(transform: np.ndarray, point) -> np.ndarray:
    """Apply a 3x3 homogeneous transform (see Calibration.GetTransform) to
    a single (x, y) point, including the perspective divide.
    """
    result = transform @ np.array([point[0], point[1], 1.0])
    return result[:2] / result[2]


def test_hough_circle_calibration_detects_and_selects_first_three():
    circles = [(150, 150, 20), (150, 600, 20), (600, 600, 20)]
    image = _draw_circle_image(circles)

    calibration = HoughCircleCalibration()
    calibration.parameters.min_dist = 50
    calibration.parameters.threshold_value = 100
    calibration.parameters.min_radius = 10
    calibration.parameters.max_radius = 40
    calibration.parameters.param1 = 50
    calibration.parameters.param2 = 15
    calibration.Detect(image)

    assert len(calibration.circles) == 3
    assert calibration.selected_circles == {0, 1, 2}


def test_hough_circle_calibration_debug_layer_matches_detection_input():
    circles = [(150, 150, 20), (150, 600, 20), (600, 600, 20)]
    image = _draw_circle_image(circles)

    calibration = HoughCircleCalibration()
    calibration.parameters.threshold_value = 100

    debug_image = calibration.DebugLayer(image)

    assert debug_image.shape == image.shape
    # The debug layer is the binary image detection runs against: circle
    # interiors should read as bright, the background as dark.
    assert debug_image[150, 150].tolist() == [255, 255, 255]
    assert debug_image[0, 0].tolist() == [0, 0, 0]


def test_hough_circle_calibration_transform_is_index_order_independent():
    # b is the right-angle corner, a is the leftmost point (offset along the
    # +x leg from b), c is offset along the +y leg from b (smaller pixel y =
    # "up"). Deliberately insert out of a/b/c order and select a
    # non-contiguous, unsorted subset of indices, to exercise the same code
    # path that used to break when selected indices weren't exactly {0, 1, 2}.
    b_point, a_point, c_point = (300, 500), (100, 500), (300, 200)
    distractor = (800, 800)

    calibration = HoughCircleCalibration()
    calibration.parameters.leg_distance_mm = 80.0
    calibration.circles = [
        (*b_point, 20),
        (*distractor, 20),
        (*a_point, 20),
        (*c_point, 20),
    ]
    calibration.selected_circles = {2, 0, 3}  # a, b, c - unsorted, non-contiguous

    transform = calibration.GetTransform()

    assert np.allclose(_apply(transform, a_point), [80.0, 0.0], atol=1e-3)
    assert np.allclose(_apply(transform, b_point), [0.0, 0.0], atol=1e-3)
    assert np.allclose(_apply(transform, c_point), [0.0, 80.0], atol=1e-3)


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

    transform = calibration.GetTransform()

    assert np.allclose(_apply(transform, top_left), [0.0, 0.0], atol=1e-1)
    assert np.allclose(
        _apply(transform, (bottom_right[0], top_left[1])),
        [PaperCalibration.WIDTH_MM, 0.0],
        atol=1e-1,
    )
    assert np.allclose(
        _apply(transform, (top_left[0], bottom_right[1])),
        [0.0, PaperCalibration.HEIGHT_MM],
        atol=1e-1,
    )


def test_paper_calibration_transform_corrects_real_perspective_distortion():
    # A deliberately non-affine mapping from the page's real mm-space
    # corners to image pixels, simulating a camera that isn't
    # fronto-parallel to the page (the top edge is foreshortened relative
    # to the bottom edge). A 3-point affine fit can only satisfy 3 of these
    # 4 corners exactly and is off by ~40mm at the page center for this
    # mapping; a true 4-point perspective fit should be exact everywhere.
    world_corners = np.float32(
        [
            [0, 0],
            [PaperCalibration.WIDTH_MM, 0],
            [PaperCalibration.WIDTH_MM, PaperCalibration.HEIGHT_MM],
            [0, PaperCalibration.HEIGHT_MM],
        ]
    )
    image_corners = np.float32([[300, 200], [650, 220], [700, 700], [150, 680]])
    true_homography = cv2.getPerspectiveTransform(world_corners, image_corners)

    calibration = PaperCalibration()
    calibration.corners = [tuple(p) for p in image_corners]  # skip image-based detection

    transform = calibration.GetTransform()

    world_center = (PaperCalibration.WIDTH_MM / 2, PaperCalibration.HEIGHT_MM / 2)
    image_center = _apply(true_homography, world_center)

    assert np.allclose(_apply(transform, image_center), world_center, atol=1e-1)


def test_identity_calibration_is_a_1px_to_1mm_noop():
    calibration = IdentityCalibration()
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    calibration.Detect(image)  # should not raise; there's nothing to detect
    assert calibration.DebugLayer(image) is image

    transform = calibration.GetTransform()

    assert np.allclose(_apply(transform, (0.0, 0.0)), [0.0, 0.0])
    assert np.allclose(_apply(transform, (12.5, 30.0)), [12.5, 30.0])


def test_calibration_base_toggle_selection_is_a_safe_noop():
    # IdentityCalibration and PaperCalibration don't override ToggleSelection
    # (their fiducials aren't manually selectable) - a click should never
    # crash, just report nothing was hit.
    assert IdentityCalibration().ToggleSelection(10, 10) is False
    assert PaperCalibration().ToggleSelection(10, 10) is False


def test_paper_calibration_no_quadrilateral_found():
    image = np.zeros((200, 200, 3), dtype=np.uint8)  # blank, nothing bright

    calibration = PaperCalibration()
    calibration.Detect(image)

    assert calibration.corners is None
