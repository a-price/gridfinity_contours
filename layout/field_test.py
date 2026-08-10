"""Tests for the distance-field viewer.

Checked against the field it claims to be drawing rather than against a
reference image. A golden picture would pin every pixel and still say
nothing about the one property that matters here: that what is on screen
is the field the solver reads, at the place the solver reads it. A viewer
that is merely self-consistent is worse than no viewer, because it is
believed.
"""

import numpy as np
import pytest

from layout.field import (
    CONTOUR_COLOR,
    DEFAULT_BAND_MM,
    PAIR_COLOR,
    SAMPLE_COLOR,
    SPACING_COLOR,
    BandEdges,
    Distances,
    FieldExtent,
    FieldView,
    GradientMagnitudes,
    LevelMask,
    PixelToLocal,
    RenderField,
    ShadeDistance,
    ShadeGradient,
)
from layout.parameters import LayoutParameters
from layout.part import DISTANT_MM, BuildPart
from conftest import Rectangle as _rectangle


def _part(width: float = 20.0, height: float = 10.0, params: LayoutParameters | None = None):
    params = params or LayoutParameters()
    return BuildPart(_rectangle(width, height), resolution=params.resolution, pad=params.pad)


def _pixel_of(part, point, pixels_per_mm: float) -> tuple[int, int]:
    """The (row, column) showing a given local-mm point."""
    low, _ = FieldExtent(part)
    column, row = np.round((np.asarray(point) - low) * pixels_per_mm).astype(int)
    return int(row), int(column)


def _has_color(image: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool((image == np.array(color)).all(axis=2).any())


# ------------------------------------------------------------------ extent


def test_the_extent_covers_the_outline_and_its_pad():
    part = _part(20.0, 10.0)

    low, high = FieldExtent(part)

    # The raster reaches `pad` past the bounding box; the readable part of
    # it stops half a cell short of that on every side.
    assert low == pytest.approx([-part.pad, -part.pad], abs=part.resolution)
    assert high == pytest.approx([20.0 + part.pad, 10.0 + part.pad], abs=part.resolution)


def test_nothing_in_the_extent_reads_as_off_the_field():
    """The extent is the *interpolant's* domain, not the raster's. Sampling
    the half-cell rim between them would ring every picture with a band of
    DISTANT_MM that said nothing about the part.
    """
    part = _part()

    distance = Distances(part, pixels_per_mm=4.0)

    assert distance.max() < DISTANT_MM


def test_a_pixel_maps_back_to_the_point_it_shows():
    """A viewer's coordinate math is the one thing it can get wrong while
    still looking entirely plausible, so the inverse is pinned exactly
    rather than to within a pixel.
    """
    part = _part()
    low, _ = FieldExtent(part)

    assert PixelToLocal(part, (0, 0), 4.0) == pytest.approx(low)
    assert PixelToLocal(part, (8, 12), 4.0) == pytest.approx(low + np.array([2.0, 3.0]))


def test_a_probed_pixel_reports_what_is_drawn_there():
    part = _part()
    pixels_per_mm = 4.0
    distance = Distances(part, pixels_per_mm)
    row, column = 20, 30

    probed = part.SampleSdf(PixelToLocal(part, (column, row), pixels_per_mm))

    assert probed[0] == pytest.approx(distance[row, column])


def test_a_non_positive_scale_is_refused():
    part = _part()

    with pytest.raises(ValueError, match="positive"):
        Distances(part, pixels_per_mm=0.0)
    with pytest.raises(ValueError, match="positive"):
        PixelToLocal(part, (0, 0), pixels_per_mm=-1.0)


# --------------------------------------------------------------- distances


def test_the_sign_is_negative_inside_and_positive_outside():
    part = _part(20.0, 10.0)
    pixels_per_mm = 4.0
    distance = Distances(part, pixels_per_mm)

    inside = distance[_pixel_of(part, (10.0, 5.0), pixels_per_mm)]
    outside = distance[_pixel_of(part, (10.0, 13.0), pixels_per_mm)]

    assert inside < 0
    assert outside > 0


def test_the_distance_is_the_distance():
    """Not merely the right sign: a point 3mm above a rectangle's top edge
    has to read 3mm, or the clearance rings drawn from it are decoration.
    """
    part = _part(20.0, 10.0)
    pixels_per_mm = 4.0
    distance = Distances(part, pixels_per_mm)

    above = distance[_pixel_of(part, (10.0, 13.0), pixels_per_mm)]
    within = distance[_pixel_of(part, (10.0, 8.0), pixels_per_mm)]

    assert above == pytest.approx(3.0, abs=part.resolution)
    assert within == pytest.approx(-2.0, abs=part.resolution)


def test_the_zero_set_sits_on_the_outline():
    part = _part(20.0, 10.0)
    pixels_per_mm = 8.0
    distance = Distances(part, pixels_per_mm)

    assert distance[_pixel_of(part, (10.0, 0.0), pixels_per_mm)] == pytest.approx(0.0, abs=part.resolution)
    assert distance[_pixel_of(part, (0.0, 5.0), pixels_per_mm)] == pytest.approx(0.0, abs=part.resolution)


def test_the_picture_is_of_the_interpolant_not_the_raster():
    """Sampled through Part.SampleSdf, which is what the energy is a
    function of - so past the raster's own resolution the picture keeps
    resolving instead of going blocky. Colorizing part.sdf directly would
    be a picture of something the solver never reads.
    """
    part = _part()

    distance = Distances(part, pixels_per_mm=4.0 / part.resolution)

    # Four screen pixels per raster cell along a straight edge: piecewise
    # constant would repeat each value four times.
    row = distance[distance.shape[0] // 2]
    steps = np.abs(np.diff(row[: row.size // 2]))
    assert steps[steps > 0].min() < part.resolution / 2


def test_the_image_scales_with_pixels_per_mm():
    part = _part()

    coarse = Distances(part, pixels_per_mm=2.0)
    fine = Distances(part, pixels_per_mm=8.0)

    assert fine.shape[0] == pytest.approx(coarse.shape[0] * 4, rel=0.02)
    assert fine.shape[1] == pytest.approx(coarse.shape[1] * 4, rel=0.02)


# ----------------------------------------------------------------- shading


def test_the_two_sides_are_told_apart_by_hue():
    """The sign is the one thing a field viewer must never leave
    ambiguous, so it is carried by hue rather than by brightness - a
    single ramp through white would render a field that is positive inside
    its own outline as a slightly-off shade of the correct answer.
    """
    part = _part(20.0, 10.0)
    pixels_per_mm = 4.0
    image = ShadeDistance(Distances(part, pixels_per_mm))

    inside = image[_pixel_of(part, (10.0, 5.0), pixels_per_mm)]
    outside = image[_pixel_of(part, (10.0, 13.0), pixels_per_mm)]

    # BGR: inside is the blue-dominant tint, outside the red-dominant one.
    assert inside[0] > inside[2]
    assert outside[2] > outside[0]


def test_the_sign_stays_legible_however_far_from_the_outline():
    """Saturation follows |d| so the eye lands on the zero set, but it
    fades to a floor rather than to nothing: deep inside a bowl is still
    unmistakably inside.
    """
    ramp = np.array([[-40.0, 40.0]])

    image = ShadeDistance(ramp)

    assert image[0, 0, 0] > image[0, 0, 2], "deep inside should still read as inside"
    assert image[0, 1, 2] > image[0, 1, 0], "far outside should still read as outside"


def test_contour_lines_land_on_every_multiple_of_the_band():
    ramp = np.arange(0.0, 5.0, 0.05).reshape(1, -1)

    edges = BandEdges(ramp, band_mm=1.0)

    # 1, 2, 3 and 4mm are crossed; 0 is the first value, with nothing
    # before it to change from.
    assert edges.sum() == 4


def test_the_band_spacing_is_the_callers():
    ramp = np.arange(0.0, 5.0, 0.05).reshape(1, -1)

    assert BandEdges(ramp, band_mm=0.5).sum() == 9
    assert BandEdges(ramp, band_mm=2.0).sum() == 2


def test_a_degenerate_band_spacing_is_refused():
    with pytest.raises(ValueError, match="positive"):
        BandEdges(np.zeros((2, 2)), band_mm=0.0)


def test_an_iso_line_marks_only_pixels_at_that_level():
    part = _part(20.0, 10.0)
    pixels_per_mm = 8.0
    distance = Distances(part, pixels_per_mm)

    marked = LevelMask(distance, 2.0)

    assert marked.any()
    # A crossing is bracketed by its two neighbours, so a marked pixel is
    # within one screen pixel's worth of field of the level.
    assert np.abs(distance[marked] - 2.0).max() < 1.0 / pixels_per_mm + part.resolution


def test_an_iso_line_the_field_never_reaches_draws_nothing():
    part = _part()

    assert not LevelMask(Distances(part), part.pad + 10.0).any()


# ------------------------------------------------------------------ render


def test_the_outline_is_drawn_on_both_views():
    """Without it the gradient view is an abstract pattern with no way to
    tell which side of the part any of it is on.
    """
    part = _part()

    for view in (FieldView(), FieldView(gradient=True)):
        assert _has_color(RenderField(part, view), CONTOUR_COLOR), view


def test_the_clearance_rings_appear_only_when_the_parameters_say_what_they_are():
    part = _part()

    assert not _has_color(RenderField(part), PAIR_COLOR)

    annotated = RenderField(part, params=LayoutParameters())

    assert _has_color(annotated, PAIR_COLOR)
    assert _has_color(annotated, SPACING_COLOR)


def test_a_clearance_ring_sits_at_the_clearance():
    """The ring is the reason to look at a part's field at all - it is the
    line no other part's boundary may cross - so it has to be at the level
    it claims and not merely somewhere plausible.
    """
    params = LayoutParameters()
    part = _part(20.0, 10.0, params)
    view = FieldView(pixels_per_mm=8.0)
    distance = Distances(part, view.pixels_per_mm)

    image = RenderField(part, view, params)

    ring = (image == np.array(PAIR_COLOR)).all(axis=2)
    assert np.abs(distance[ring] - params.c_pair_enforced).max() < 1.0 / view.pixels_per_mm + part.resolution


def test_the_gradient_view_carries_no_clearance_rings():
    """They are levels of the distance field, and mean nothing overlaid on
    gradient length.
    """
    part = _part()

    image = RenderField(part, FieldView(gradient=True), LayoutParameters())

    assert not _has_color(image, PAIR_COLOR)
    assert not _has_color(image, SPACING_COLOR)


def test_the_gradient_view_is_gray_apart_from_its_outline():
    part = _part()

    image = RenderField(part, FieldView(gradient=True))

    colored = image[(image != np.array(CONTOUR_COLOR)).any(axis=2)].reshape(-1, 3)
    assert (colored[:, 0] == colored[:, 1]).all() and (colored[:, 1] == colored[:, 2]).all()


def test_boundary_samples_are_drawn_only_when_asked():
    part = _part()

    assert not _has_color(RenderField(part), SAMPLE_COLOR)
    assert _has_color(RenderField(part, FieldView(samples=True)), SAMPLE_COLOR)


def test_samples_cover_a_long_straight_edge():
    """D2's reason for resampling boundaries at all: a simplified contour
    has edges tens of millimeters long, and a part could slide through a
    thin feature between two vertices without either registering. If that
    ever regressed, the drawn samples would be four corner dots.
    """
    part = _part(20.0, 10.0)
    view = FieldView(pixels_per_mm=8.0, samples=True)

    image = RenderField(part, view)

    drawn = (image == np.array(SAMPLE_COLOR)).all(axis=2)
    # The bottom edge alone is 20mm, so a vertices-only sampling could not
    # put anything like this many dots on the picture.
    assert drawn.sum() > 100


def test_a_wider_part_renders_wider():
    narrow, wide = _part(20.0, 10.0), _part(60.0, 10.0)

    assert RenderField(wide).shape[1] > RenderField(narrow).shape[1]
    assert RenderField(wide).shape[0] == RenderField(narrow).shape[0]


def test_the_default_scale_keeps_a_real_part_a_reasonable_size():
    # A big spoon is around 200mm long; it should come out readable in an
    # image view without the render costing anything noticeable.
    part = _part(200.0, 45.0)

    image = RenderField(part)

    assert 600 < image.shape[1] < 1200


# --------------------------------------------------------------- creases


def test_a_well_resolved_interior_reads_as_unit_gradient():
    part = _part(20.0, 10.0)
    pixels_per_mm = 8.0

    magnitude = GradientMagnitudes(part, pixels_per_mm)

    # 2mm above the bottom edge and far from any other wall: one nearest
    # boundary, no ambiguity about which way is out.
    assert magnitude[_pixel_of(part, (10.0, 2.0), pixels_per_mm)] == pytest.approx(1.0, abs=0.05)


def test_a_crease_reads_darker_than_a_resolved_interior():
    """Both branches of a rectangle's medial axis: the spine down its long
    axis, and the 45-degree run into each corner.
    """
    part = _part(20.0, 10.0)
    pixels_per_mm = 8.0
    image = ShadeGradient(GradientMagnitudes(part, pixels_per_mm))

    spine = image[_pixel_of(part, (10.0, 5.0), pixels_per_mm)][0]
    branch = image[_pixel_of(part, (2.5, 2.5), pixels_per_mm)][0]
    resolved = image[_pixel_of(part, (10.0, 2.0), pixels_per_mm)][0]

    assert resolved > 240, "one nearest wall and no ambiguity should read as a clean field"
    assert spine < resolved
    assert branch < resolved


def test_a_gradient_longer_than_unit_is_shaded_as_a_defect():
    """Why this view shades by deviation from unit length rather than by
    length itself.

    Where the gradients of two *opposing* walls meet, the interpolant
    between `(0, 1)` and `(0, -1)` reads near zero. Where two
    *perpendicular* walls meet, it reads up to `sqrt(2)`. Both are the
    same crease and the same untrustworthy force, and pixels of the second
    kind really do occur in the simplest possible part - shading by raw
    magnitude would paint every one of them pure white.
    """
    part = _part(20.0, 10.0)
    magnitude = GradientMagnitudes(part, pixels_per_mm=8.0)

    overshooting = magnitude > 1.1
    assert overshooting.any(), "a rectangle's corner branches should overshoot"
    assert ShadeGradient(magnitude)[overshooting].max() < 240


# -------------------------------------------------------------- real parts


def test_a_captured_contour_renders():
    """The fixtures are concave and self-nesting, which is the case the
    synthetic rectangles above cannot cover: a spoon's bowl folds its own
    dilation into itself.
    """
    from layout.loading import LoadParts

    params = LayoutParameters()
    parts = LoadParts(["test_data/big_spoon.svg"], params)
    part = parts[0]

    image = RenderField(part, FieldView(), params)

    assert image.shape[2] == 3
    assert _has_color(image, CONTOUR_COLOR)
    assert _has_color(image, PAIR_COLOR)
    assert BandEdges(Distances(part), DEFAULT_BAND_MM).any()
