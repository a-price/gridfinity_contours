"""Tests for placing a part in a bin: quarter-turn rotation of points and
of direction vectors, and the round trip between bin and local frames.

Rotation of *points* and of *vectors* is deliberately separate - one
shifts the bounding box back to the origin and the other must not - so
both are exercised here rather than only where they happen to be used.
"""

import numpy as np
import pytest

from pipeline.layout.part import BuildPart, PolygonArea
from pipeline.layout.placement import Placement, RotatePoints, RotateVectors, RotatedSize


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


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

        assert placement.ToLocal(part, world) == pytest.approx(part.contour, abs=1e-9)


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
