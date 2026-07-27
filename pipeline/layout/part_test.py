"""Tests for Part: rasterization, distance fields, and the canonical
local frame."""

import numpy as np
import pytest

from pipeline.layout.part import (
    DISTANT_MM,
    BuildPart,
    PolygonArea,
    ResampleBoundary,
)
from pipeline.layout.placement import Placement, RotatePoints, RotatedSize
from pipeline.layout.verify import ExactSignedDistance


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


def _l_shape() -> np.ndarray:
    """A 30x30 L with a 20x20 bite taken out of the top right corner."""
    return np.array([[0, 0], [30, 0], [30, 10], [10, 10], [10, 30], [0, 30]], dtype=np.float64)


def _u_shape() -> np.ndarray:
    """A 30x30 U with a 10-wide, 20-deep notch - the concavity another part
    can nest into, which is the whole reason the packer bothers with
    non-convex shapes.
    """
    return np.array([[0, 0], [30, 0], [30, 30], [20, 30], [20, 10], [10, 10], [10, 30], [0, 30]], dtype=np.float64)


# --------------------------------------------------------------- primitives


def test_polygon_area_ignores_winding_order():
    rectangle = _rectangle(20, 10)

    assert PolygonArea(rectangle) == pytest.approx(200.0)
    assert PolygonArea(rectangle[::-1]) == pytest.approx(200.0)


def test_polygon_area_of_a_concave_shape():
    # 30x30 minus the 20x20 bite.
    assert PolygonArea(_l_shape()) == pytest.approx(900.0 - 400.0)


def test_resample_boundary_keeps_vertices_and_fills_long_edges():
    resampled = ResampleBoundary(_rectangle(20, 10), spacing=1.0)

    for vertex in _rectangle(20, 10):
        assert np.isclose(resampled, vertex).all(axis=1).any(), f"vertex {vertex} was dropped"

    # Consecutive samples, around the closing edge included, stay within
    # the requested spacing.
    steps = np.linalg.norm(np.diff(np.vstack([resampled, resampled[:1]]), axis=0), axis=1)
    assert steps.max() <= 1.0 + 1e-9


def test_resample_boundary_spacing_controls_density():
    coarse = ResampleBoundary(_rectangle(20, 10), spacing=2.0)
    fine = ResampleBoundary(_rectangle(20, 10), spacing=0.5)

    assert len(fine) > len(coarse)


# --------------------------------------------------------- distance fields


def test_sdf_is_negative_inside_and_positive_outside():
    part = BuildPart(_rectangle(20, 10))

    assert part.SampleSdf(np.array([[10.0, 5.0]]))[0] < 0
    assert part.SampleSdf(np.array([[-2.0, 5.0]]))[0] > 0


def test_sdf_magnitude_matches_hand_computed_distances():
    part = BuildPart(_rectangle(20, 10))

    # 3mm beyond the left edge, level with the middle.
    assert part.SampleSdf(np.array([[-3.0, 5.0]]))[0] == pytest.approx(3.0, abs=0.05)
    # 2mm inside the left edge.
    assert part.SampleSdf(np.array([[2.0, 5.0]]))[0] == pytest.approx(-2.0, abs=0.05)
    # On the boundary itself.
    assert part.SampleSdf(np.array([[0.0, 5.0]]))[0] == pytest.approx(0.0, abs=0.05)
    # Diagonally out from a corner: 3-4-5. Corners are the worst case for a
    # grid-sampled distance transform - the nearest boundary is a single
    # point rather than a line the grid can straddle evenly - so this one
    # carries a wider tolerance, still comfortably inside one pixel.
    assert part.SampleSdf(np.array([[-3.0, -4.0]]))[0] == pytest.approx(5.0, abs=0.15)


def test_sdf_is_least_accurate_on_the_interior_medial_axis():
    """A known and harmless limitation, pinned down so it is not mistaken
    for a regression later.

    A distance field creases along the medial axis - the ridge equidistant
    from two edges - and bilinear sampling of a crease always reads low.
    At the center of a 20x10 rectangle that costs half a pixel. It does not
    matter: the solver only ever samples near part boundaries, where the
    field is smooth and exact, and no force depends on how deep the middle
    of a part is.
    """
    part = BuildPart(_rectangle(20, 10))

    center = part.SampleSdf(np.array([[10.0, 5.0]]))[0]

    assert center == pytest.approx(-5.0, abs=0.5 * part.resolution)
    assert center > -5.0, "interpolation across the ridge should read shallow, not deep"
    # A hair off the ridge, the field is accurate again.
    assert part.SampleSdf(np.array([[10.0, 3.0]]))[0] == pytest.approx(-3.0, abs=0.05)


@pytest.mark.parametrize("shape", [_rectangle(20, 10), _l_shape(), _u_shape()], ids=["rectangle", "l", "u"])
def test_sdf_matches_exact_geometry_across_the_whole_field(shape):
    """The strong form of the M1 distance-field criterion: rather than
    spot-checking a few hand-computed points, compare every sample against
    an exact polygon distance computed with no raster involved. A concave
    shape is the interesting case - the notch of a U must read as outside.
    """
    part = BuildPart(shape)
    x = np.arange(-4.0, part.size[0] + 4.0, 0.7)
    y = np.arange(-4.0, part.size[1] + 4.0, 0.7)
    query = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)

    measured = part.SampleSdf(query)
    exact = ExactSignedDistance(query, part.contour)

    assert np.abs(measured - exact).max() < 0.2, "distance field drifts from exact geometry"
    # Sign is what collision detection actually keys on, so it has to be
    # right everywhere except within a pixel of the boundary, where the
    # rasterization genuinely cannot resolve which side a point is on.
    resolvable = np.abs(exact) > part.resolution
    assert (np.sign(measured[resolvable]) == np.sign(exact[resolvable])).all()


def test_sdf_reads_distant_beyond_the_rasterized_field():
    part = BuildPart(_rectangle(20, 10), pad=5.0)

    assert part.SampleSdf(np.array([[500.0, 500.0]]))[0] == DISTANT_MM


def test_gradient_points_away_from_the_part():
    part = BuildPart(_rectangle(20, 10))

    # Just left of the left edge: away is -x.
    assert part.SampleGradient(np.array([[-1.0, 5.0]]))[0] == pytest.approx([-1.0, 0.0], abs=0.05)
    # Just below the bottom edge: away is -y.
    assert part.SampleGradient(np.array([[10.0, -1.0]]))[0] == pytest.approx([0.0, -1.0], abs=0.05)
    # Inside, the gradient still points toward the outside it will escape to.
    assert part.SampleGradient(np.array([[1.0, 5.0]]))[0] == pytest.approx([-1.0, 0.0], abs=0.05)


def test_gradient_is_unit_length_where_the_field_is_defined():
    part = BuildPart(_rectangle(20, 10))
    query = np.array([[-1.0, 5.0], [10.0, -1.0], [2.0, 5.0], [21.0, 11.0]])

    assert np.linalg.norm(part.SampleGradient(query), axis=1) == pytest.approx(1.0, abs=0.05)


def test_area_survives_rasterization():
    part = BuildPart(_l_shape())

    assert part.area == pytest.approx(500.0)


# --------------------------------------------------- canonical local frame


def test_part_frame_is_canonical_under_rotation_and_translation():
    """The M1 rotation criterion: a part is the same part however the object
    happened to sit in the photo, so its field is a property of the shape
    and not of the capture.
    """
    shape = _l_shape()
    angle = np.deg2rad(90.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = shape @ rotation.T + np.array([137.0, -42.0])

    original, rotated = BuildPart(shape), BuildPart(moved)

    assert rotated.sdf.shape == original.sdf.shape
    assert rotated.size == pytest.approx(original.size, abs=1e-6)
    assert np.abs(rotated.sdf - original.sdf).max() < 0.05


def test_part_frame_puts_the_long_axis_on_x():
    part = BuildPart(_rectangle(10, 40))

    assert part.size[0] > part.size[1]
    assert part.size == pytest.approx([40.0, 10.0], abs=0.01)


def test_part_frame_does_not_mirror_the_shape():
    """PCA eigenvector signs are arbitrary, so alignment can hand back a
    left-handed basis; applying it unchecked would silently flip the part
    over, which D1 rules out.
    """
    shape = _l_shape()

    # Every rigid rotation of the same shape must produce the same
    # (unmirrored) local frame, not its reflection.
    reference = BuildPart(shape)
    for degrees in (0.0, 30.0, 90.0, 150.0, 210.0, 270.0, 330.0):
        angle = np.deg2rad(degrees)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        candidate = BuildPart(shape @ rotation.T)

        assert np.abs(candidate.sdf - reference.sdf).max() < 0.15, f"{degrees} degrees produced a different part"


def test_build_part_rejects_a_degenerate_contour():
    with pytest.raises(ValueError):
        BuildPart(np.array([[0.0, 0.0], [1.0, 1.0]]))


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
