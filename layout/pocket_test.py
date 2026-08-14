"""Tests for pocket contours.

Two things are being pinned down. The *bound* - a pocket never cuts
inside the offset it was asked for, and never wanders more than a raster
cell and a simplification tolerance outside it - which is the whole
reason the trace is biased. And the *topology*, which is why this is a
level set rather than a polygon offset: dilation closes notches, folds a
concave shape into itself, and can trap a void, and all three have to
come out right without anything special-casing them.

The bound is measured with `verify.ExactSignedDistance`, which works on
the polygon directly and shares no code with the distance transforms - so
a raster artifact cannot appear in both and confirm itself.
"""

import math

import numpy as np
import pytest

from layout.loading import ReadContours
from layout.part import PolygonArea
from layout.pocket import (
    DEFAULT_RESOLUTION_MM,
    DEFAULT_SIMPLIFY_MM,
    PocketContour,
)
from layout.verify import ExactSignedDistance
from conftest import Rectangle as _rectangle, SPOONS

# How far outside the requested offset a pocket is allowed to sit. Not
# `0.5 * resolution + simplify`, the level the trace is taken at: the
# raster's error is two-sided, so the outline can land half a cell *past*
# that level as easily as half a cell short of it. Measured, a 60-gon
# circle reaches 0.0698mm at the defaults - which is what makes the whole
# cell the honest budget.
_OVERSHOOT_MM = DEFAULT_RESOLUTION_MM + DEFAULT_SIMPLIFY_MM


def _circle(radius: float = 20.0, sides: int = 60) -> np.ndarray:
    """A polygonal circle - the shape whose dilation is hardest to trace,
    since every vertex is a corner the raster has to round.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)


def _l_shape() -> np.ndarray:
    """A 30x30 L with a 20x20 bite out of the top right corner."""
    return np.array([[0, 0], [30, 0], [30, 10], [10, 10], [10, 30], [0, 30]], dtype=np.float64)


def _u_shape() -> np.ndarray:
    """A 30x30 U with a 10-wide, 20-deep notch. Its area is 700 and its
    perimeter 160, both by inspection.
    """
    return np.array(
        [[0, 0], [30, 0], [30, 30], [20, 30], [20, 10], [10, 10], [10, 30], [0, 30]],
        dtype=np.float64,
    )


def _horseshoe(outer: float = 20.0, inner: float = 12.0) -> np.ndarray:
    """A C open across 40 degrees, centered on the origin.

    The interesting fixture: dilate it far enough and the mouth closes
    over, sealing the middle into a void that is no longer connected to
    anywhere. That is the case a polygon offset has to detect and this
    one gets for free.
    """
    angles = np.linspace(math.radians(20.0), math.radians(340.0), 60)
    return np.vstack(
        [
            np.stack([outer * np.cos(angles), outer * np.sin(angles)], axis=-1),
            np.stack([inner * np.cos(angles[::-1]), inner * np.sin(angles[::-1])], axis=-1),
        ]
    )


_SHAPES = {
    "square": lambda: _rectangle(10, 10),
    "sliver": lambda: _rectangle(40, 1.5),
    "circle": _circle,
    "l_shape": _l_shape,
    "u_shape": _u_shape,
    "horseshoe": _horseshoe,
}


def _Centre(pocket: np.ndarray, point: tuple[float, float]) -> float:
    """Signed distance from `point` to the pocket - negative inside."""
    return float(ExactSignedDistance(np.array([point], dtype=np.float64), pocket)[0])


# ------------------------------------------------------- the error bound


@pytest.mark.parametrize("name", sorted(_SHAPES))
@pytest.mark.parametrize("offset", [0.5, 1.0, 2.0])
def test_pocket_stays_within_its_one_sided_error_bound(name, offset):
    shape = _SHAPES[name]()

    pocket = PocketContour(shape, offset)

    distance = ExactSignedDistance(pocket, shape)
    assert distance.min() >= offset, f"{name} pocket cut {offset - distance.min():.4f}mm inside the offset"
    assert distance.max() <= offset + _OVERSHOOT_MM


def test_pocket_area_of_a_convex_shape_brackets_the_analytic_dilation():
    """For a *convex* polygon the dilation's area is exactly
    `A + P*r + pi*r^2`, so the traced pocket has to land between that
    value at the offset asked for and at the bound it may overshoot to.
    An independent check on the same contract the pointwise test makes.
    """

    def Dilated(radius: float) -> float:
        return 100.0 + 40.0 * radius + math.pi * radius**2

    pocket = PocketContour(_rectangle(10, 10), 1.0)

    assert Dilated(1.0) <= PolygonArea(pocket) <= Dilated(1.0 + _OVERSHOOT_MM)


def test_a_finer_raster_tightens_the_bound():
    """The overshoot is half a cell of raster plus the simplification, so
    halving the resolution has to visibly close it - otherwise the trace
    level is being set by something other than what the docstring claims.
    """
    shape = _circle()

    coarse = ExactSignedDistance(PocketContour(shape, 1.0, resolution=0.1), shape).max()
    fine = ExactSignedDistance(PocketContour(shape, 1.0, resolution=0.025), shape).max()

    assert fine < coarse - 0.05


# ----------------------------------------------------------- the topology


def test_a_notch_narrower_than_twice_the_offset_closes_over():
    """The U's notch is 10mm wide, so a 6mm offset grows each wall past
    the middle and the pocket swallows it. Nothing detects this - it is
    what a level set does.
    """
    pocket = PocketContour(_u_shape(), 6.0)

    assert _Centre(pocket, (15.0, 20.0)) < 0.0


def test_a_notch_wider_than_twice_the_offset_stays_open():
    pocket = PocketContour(_u_shape(), 1.0)

    assert _Centre(pocket, (15.0, 20.0)) > 0.0


def test_self_overlap_is_not_double_counted():
    """Where a concavity's walls grow into each other the naive convex
    formula counts the overlap twice, which is the same trap
    `Part.DilatedArea` avoids by counting field samples. The U at 6mm
    overlaps across its whole notch, so the true area must come in under
    it.
    """
    naive = 700.0 + 160.0 * 6.0 + math.pi * 6.0**2

    assert PolygonArea(PocketContour(_u_shape(), 6.0)) < naive


def test_a_trapped_void_is_filled_rather_than_left_as_a_hole():
    """Closing the horseshoe's mouth seals its middle off. `find_contours`
    reports that void as a second loop, and the outer boundary is the one
    kept - so the origin, which is outside the object, ends up inside the
    pocket.
    """
    horseshoe = _horseshoe()
    assert _Centre(horseshoe, (0.0, 0.0)) > 0.0, "the origin should start outside the object"

    pocket = PocketContour(horseshoe, 6.0)

    assert _Centre(pocket, (0.0, 0.0)) < 0.0


def test_a_void_still_open_to_the_outside_is_not_filled():
    """The other half of the pair: at 1mm the mouth is still open, so the
    middle is not a void at all and must stay outside the pocket.
    """
    pocket = PocketContour(_horseshoe(), 1.0)

    assert _Centre(pocket, (0.0, 0.0)) > 0.0


# --------------------------------------------------------- frames and size


def test_a_pocket_follows_its_object():
    """Computed in whatever frame it was given, with no hidden
    normalization - unlike `BuildPart`, which PCA-aligns first.

    Not exact, and cannot be: the raster's origin is the object's own
    minimum corner, so a shift that is not a whole number of cells rounds
    differently. The tolerance is a ten-thousandth of a cell, which is
    float noise rather than a grid the pocket has snapped to.
    """
    shape = _l_shape()
    shift = np.array([0.013, 7.7777])  # deliberately not a multiple of the resolution

    here = PocketContour(shape, 1.0)
    there = PocketContour(shape + shift, 1.0)

    assert len(there) == len(here)
    assert np.abs((there - shift) - here).max() < 1e-4


def test_simplification_cuts_vertices_without_breaking_the_bound():
    shape = _circle()

    traced = PocketContour(shape, 1.0, simplify=0.0)
    simplified = PocketContour(shape, 1.0)

    assert len(simplified) < len(traced) / 10
    assert ExactSignedDistance(simplified, shape).min() >= 1.0


def test_the_spoon_fixtures_get_pockets_at_a_workable_size():
    """The real inputs, at the real default. Marching squares emits a
    vertex per cell crossing - thousands per spoon - and everything
    downstream, `.scad` included, sees whatever survives simplification.
    """
    for part_id, contour in sorted(ReadContours(SPOONS).items()):
        pocket = PocketContour(contour, 1.0)

        assert ExactSignedDistance(pocket, contour).min() >= 1.0, f"spoon {part_id} cut inside its offset"
        assert len(pocket) < 800, f"spoon {part_id} kept {len(pocket)} vertices"


# ------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"offset": -0.1}, "offset must be non-negative"),
        ({"offset": 1.0, "resolution": 0.0}, "resolution must be positive"),
        ({"offset": 1.0, "simplify": -1.0}, "simplify must be non-negative"),
    ],
)
def test_rejects_parameters_it_cannot_honour(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PocketContour(_rectangle(10, 10), **kwargs)


def test_rejects_a_contour_that_is_not_a_polygon():
    with pytest.raises(ValueError, match="at least 3 points"):
        PocketContour(np.array([[0.0, 0.0], [1.0, 1.0]]), 1.0)
