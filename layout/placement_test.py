"""Tests for placing a part in a bin: quarter-turn rotation of points and
of direction vectors, and the round trip between bin and local frames.

Rotation of *points* and of *vectors* is deliberately separate - one
shifts the bounding box back to the origin and the other must not - so
both are exercised here rather than only where they happen to be used.
"""

import numpy as np
import pytest

from layout.part import BuildPart, PolygonArea
from layout.placement import (
    Placement,
    Pose,
    PoseExtent,
    PoseInertia,
    PoseRadius,
    RotatePoints,
    RotatedSize,
    RotateVectors,
    SpinPoints,
    SpinVectors,
)
from conftest import Rectangle as _rectangle


def _l_shape() -> np.ndarray:
    """A 30x30 L with a 20x20 bite taken out of the top right corner."""
    return np.array([[0, 0], [30, 0], [30, 10], [10, 10], [10, 30], [0, 30]], dtype=np.float64)


# ------------------------------------------------------------- orientation


def test_rotate_points_keeps_the_bounding_box_at_the_origin():
    rectangle = _rectangle(20, 10)
    size = np.array([20.0, 10.0])

    for orientation in range(4):
        rotated = RotatePoints(rectangle, orientation, size)

        assert rotated.min(axis=0) == pytest.approx([0.0, 0.0])
        assert rotated.max(axis=0) == pytest.approx(RotatedSize(size, orientation))


def test_rotate_points_preserves_area_rather_than_mirroring():
    shape = _l_shape()
    size = shape.max(axis=0)

    for orientation in range(4):
        assert PolygonArea(RotatePoints(shape, orientation, size)) == pytest.approx(PolygonArea(shape))


def test_four_quarter_turns_return_the_original():
    shape = _l_shape()
    size = shape.max(axis=0)

    points = shape
    for _ in range(4):
        points = RotatePoints(points, 1, RotatedSize(size, 0) if _ % 2 == 0 else RotatedSize(size, 1))
    assert points == pytest.approx(shape)


def test_rotated_size_swaps_axes_on_quarter_turns():
    size = np.array([20.0, 10.0])

    assert RotatedSize(size, 0) == pytest.approx([20.0, 10.0])
    assert RotatedSize(size, 1) == pytest.approx([10.0, 20.0])
    assert RotatedSize(size, 2) == pytest.approx([20.0, 10.0])
    assert RotatedSize(size, 3) == pytest.approx([10.0, 20.0])


# --------------------------------------------------------------- placement


def test_placement_moves_a_part_into_bin_coordinates():
    part = BuildPart(_rectangle(20, 10))
    placement = Placement(part_id=0, position=np.array([5.0, 7.0]))

    placed = placement.ToWorld(part)

    assert placed.min(axis=0) == pytest.approx([5.0, 7.0])
    assert placed.max(axis=0) == pytest.approx([25.0, 17.0])


def test_placement_round_trips_through_the_local_frame():
    part = BuildPart(_l_shape())

    for orientation in range(4):
        placement = Placement(part_id=0, position=np.array([13.0, 29.0]), orientation=orientation)
        world = placement.ToWorld(part)

        assert placement.ToLocal(part, world) == pytest.approx(part.pocket_contour, abs=1e-9)


def test_placement_orientation_rotates_the_footprint():
    part = BuildPart(_rectangle(40, 10))
    upright = Placement(part_id=0, position=np.zeros(2), orientation=1)

    placed = upright.ToWorld(part)

    assert placed.max(axis=0) == pytest.approx([10.0, 40.0], abs=0.01)


# ------------------------------------------------------- direction vectors


def test_rotate_vectors_turns_directions_without_translating():
    right = np.array([[1.0, 0.0]])

    assert RotateVectors(right, 0)[0] == pytest.approx([1.0, 0.0])
    assert RotateVectors(right, 1)[0] == pytest.approx([0.0, 1.0])
    assert RotateVectors(right, 2)[0] == pytest.approx([-1.0, 0.0])
    assert RotateVectors(right, 3)[0] == pytest.approx([0.0, -1.0])


def test_rotate_vectors_preserves_length():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(50, 2))

    for orientation in range(4):
        assert np.linalg.norm(RotateVectors(vectors, orientation), axis=1) == pytest.approx(
            np.linalg.norm(vectors, axis=1)
        )


# -------------------------------------------------------------- free angle


def test_a_pose_with_no_free_angle_leaves_points_untouched():
    """The property the whole 90-degree mode rests on. Not "close to
    unchanged" - identical, so that adding poses cannot move a single
    committed layout by a rounding error.
    """
    points = _l_shape()

    assert SpinPoints(points, 0.0, np.array([5.0, 5.0])) is points
    assert SpinVectors(points, 0.0) is points


def test_spinning_turns_about_the_pivot_and_leaves_it_alone():
    pivot = np.array([3.0, 7.0])
    points = np.array([[3.0, 7.0], [4.0, 7.0]])

    spun = SpinPoints(points, np.pi / 2.0, pivot)

    assert spun[0] == pytest.approx(pivot), "the pivot is the one point that cannot move"
    assert spun[1] == pytest.approx([3.0, 8.0])


def test_a_pose_reports_the_angle_its_halves_come_to():
    assert Pose(1, 0.0).total == pytest.approx(np.pi / 2.0)
    assert Pose(1, np.pi / 4.0).total == pytest.approx(3.0 * np.pi / 4.0)
    assert Pose(2).upright and not Pose(2, 0.1).upright


def test_local_and_world_are_inverses_at_a_free_angle():
    """The round trip every pair term depends on: a sample is taken into
    bin coordinates by one part and read back into another's frame to query
    its field. An inconsistency here is invisible in the geometry and
    surfaces as a clearance that is quietly wrong.
    """
    part = BuildPart(_l_shape(), resolution=0.5, pad=2.0)
    placement = Placement(0, np.array([12.0, -4.0]), orientation=3, angle=0.9)

    world = placement.ToWorld(part)
    back = placement.ToLocal(part, world)

    assert back == pytest.approx(part.pocket_contour, abs=1e-9)


def test_turning_a_part_does_not_change_its_area():
    """A rotation that scaled or sheared would still round-trip and would
    still look like a rotation in a drawing.
    """
    part = BuildPart(_l_shape(), resolution=0.5, pad=2.0)
    before = PolygonArea(part.pocket_contour)

    for angle in (0.3, 1.2, -2.5):
        placed = Placement(0, np.array([1.0, 2.0]), orientation=1, angle=angle)
        assert PolygonArea(placed.ToWorld(part)) == pytest.approx(before)


def test_bounds_measure_the_shape_and_not_its_turned_box():
    """The distinction `PoseBounds` exists for. A rotated box is not the
    box of the rotated shape - it is bigger - and a caller using it would
    reserve room the part does not need and report overlaps that are not
    there.
    """
    part = BuildPart(_rectangle(40, 10), resolution=0.5, pad=2.0)
    placement = Placement(0, np.zeros(2), angle=np.pi / 4.0)

    low, high = placement.Bounds(part)
    world = placement.ToWorld(part)

    assert low == pytest.approx(world.min(axis=0))
    assert high == pytest.approx(world.max(axis=0))
    assert (high - low)[0] < 40.0, "a diagonal 40mm bar spans less than 40mm in x"


def test_an_upright_bounds_is_exactly_the_position_and_size():
    """The fast path has to agree with the measured one, or the 90-degree
    mode and the free mode would disagree about where the same part is.
    """
    part = BuildPart(_rectangle(40, 10), resolution=0.5, pad=2.0)

    for orientation in range(4):
        placement = Placement(0, np.array([3.0, 5.0]), orientation=orientation)
        low, high = placement.Bounds(part)

        assert low is placement.position
        assert high - low == pytest.approx(RotatedSize(part.size, orientation))


def test_pose_extent_shrinks_a_long_bar_turned_diagonally():
    """The measurement the whole experiment turns on: a bar longer than the
    bin can still fit across its diagonal, and only an extent taken from
    the shape says so.
    """
    part = BuildPart(_rectangle(40, 10), resolution=0.5, pad=2.0)

    upright = PoseExtent(part, Pose(0))
    diagonal = PoseExtent(part, Pose(0, np.pi / 4.0))

    assert upright == pytest.approx(part.size)
    assert diagonal[0] < upright[0], "turning shortens the long axis"
    assert diagonal[1] > upright[1], "and pays for it on the short one"


def test_the_normalizers_do_not_depend_on_the_pose():
    """Both are computed once when a descent starts, which is only sound
    because rotation preserves distance from the pivot.
    """
    part = BuildPart(_l_shape(), resolution=0.5, pad=2.0)

    inertia, radius = PoseInertia(part), PoseRadius(part)

    assert inertia > 0 and radius > 0
    assert radius == pytest.approx(np.linalg.norm(part.samples - part.size / 2.0, axis=1).max())


def test_a_bigger_part_turns_less_for_the_same_step_cap():
    """Why the cap is derived from the radius rather than set in degrees. A
    long part's tip travels much further per radian, so a fixed angular cap
    would be safe for a washer and would fling a spoon through its
    neighbour.
    """
    small = BuildPart(_rectangle(10, 10), resolution=0.5, pad=2.0)
    large = BuildPart(_rectangle(200, 10), resolution=0.5, pad=2.0)

    assert PoseRadius(large) > PoseRadius(small)
