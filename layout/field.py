"""Looking at the distance field a part is packed by.

Both phases of the layout search read this field and neither draws it.
`solver` prices every candidate arrangement by sampling one part's
boundary against another's field, and `spacing` evens the resulting gaps
out against the same numbers - so when a pack behaves oddly, "is the field
what I think it is" is the first question to settle, and there was no way
to ask it.

What is drawn is deliberately the *interpolant*, not the stored raster.
`Part.SampleSdf` interpolates bilinearly between raster cells, and that
interpolant - not the cells - is what the energy is a function of (see
`Part.SampleDerivative` for why the distinction is load-bearing rather
than pedantic). Colorizing `part.sdf` directly would be a picture of
something the solver never reads.

Two views, because the field has two properties worth doubting. The
distance view answers "is the zero set where the outline is, and how far
does the field reach past it". The gradient view answers "where does this
field stop being a distance" - the creases along the medial axis, which
are exactly where `ComputeEnergy` says its forces reverse.

No Qt here and no file I/O: this takes a Part and returns a BGR array,
the same currency `render.py` deals in.
"""

from dataclasses import dataclass

import numpy as np

from layout.parameters import LayoutParameters
from layout.part import Part

# Screen pixels per millimeter. At the default raster resolution of
# 0.25mm this puts exactly one screen pixel on one raster cell, so the
# picture at this scale invents nothing and smooths nothing away.
DEFAULT_PIXELS_PER_MM = 4.0

# Millimeters between contour lines. One millimeter is a legible ruler at
# the scale the clearances live on - `c_pair` is 1.2mm by D5, so the band
# count from the outline to a clearance ring is something the eye can read
# off rather than estimate.
DEFAULT_BAND_MM = 1.0

# BGR, since everything this project puts on a screen is.
#
# Two hues rather than one ramp through white, because the *sign* is the
# one thing that must never be ambiguous: a field reading positive inside
# its own outline is precisely the failure this tool exists to catch, and
# a single ramp would render it as a slightly-off shade of the same color.
INSIDE_COLOR = (196, 122, 48)
OUTSIDE_COLOR = (48, 140, 214)

# How far the tint fades over, and the floor it fades to. A distance field
# is most interesting near its zero set and least interesting deep inside
# a blob, so saturation follows `|d|` - but never all the way out, because
# the hue is what says which side of the boundary is being looked at.
FADE_MM = 4.0
MIN_TINT = 0.35

# Contour lines darken whatever is already there rather than painting a
# color over it, so a line still says which side of the outline it is on.
BAND_SHADE = 0.72

CONTOUR_COLOR = (24, 24, 24)  # the zero level set, which is the outline itself
PAIR_COLOR = (60, 60, 220)  # c_pair_enforced: no other part's boundary may cross
SPACING_COLOR = (90, 170, 90)  # spacing_pair: where the spacing springs aim to hold one
SAMPLE_COLOR = (200, 60, 160)  # the boundary points that get tested against other fields


@dataclass
class FieldView:
    """How to draw a field, as opposed to what the field is.

    Held apart from `LayoutParameters` on purpose. Those decide what the
    field *is* - `resolution` and `pad` change the raster, and changing
    one means rebuilding the part. These change only the picture of it, so
    a front end can redraw on them without recomputing anything.
    """

    pixels_per_mm: float = DEFAULT_PIXELS_PER_MM
    band_mm: float = DEFAULT_BAND_MM

    # The gradient view answers a different question than the distance
    # view and shares no shading with it, so it replaces rather than
    # overlays - two false-color maps on one image read as neither.
    gradient: bool = False

    # Off by default. Samples are resampled at one raster cell apart, so
    # at the default scale there is one per pixel and they do nothing but
    # recolor the outline. They earn their place zoomed in, where the
    # question "is this feature thinner than the gap between two samples"
    # (D2's reason for resampling at all) becomes one you can answer by
    # looking.
    samples: bool = False


def FieldExtent(part: Part) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and highest local-mm points the part's field can be read
    at, as `(low, high)`.

    Half a raster cell inside the raster on every side, because these are
    the corner pixel *centers*: past them the interpolant has nothing on
    one side to interpolate between and `SampleSdf` reports `DISTANT_MM`.
    Sampling that half-cell rim would put a band of "no data" around every
    picture that meant nothing about the part.
    """
    height, width = part.sdf.shape
    low = part.origin + 0.5 * part.resolution
    high = part.origin + (np.array([width, height]) - 0.5) * part.resolution
    return low, high


def PixelToLocal(part: Part, pixel: tuple[float, float], pixels_per_mm: float = DEFAULT_PIXELS_PER_MM) -> np.ndarray:
    """Where an image pixel of this part's field sits in the part's own
    local millimeters - the exact inverse of the grid the views sample on.

    Here rather than in a front end so that a window can turn a mouse
    position into a distance query without owning any of the coordinate
    math, which is the one place a viewer can lie convincingly.
    """
    if pixels_per_mm <= 0:
        raise ValueError(f"pixels_per_mm must be positive, got {pixels_per_mm}")
    return FieldExtent(part)[0] + np.asarray(pixel, dtype=np.float64) / pixels_per_mm


def _ScreenGrid(part: Part, pixels_per_mm: float) -> np.ndarray:
    """An `(H, W, 2)` grid of local-mm points, one per screen pixel,
    covering everything the field can be read at.

    Pixel steps are exactly `1 / pixels_per_mm` rather than a linspace
    across the extent, so `PixelToLocal` inverts this exactly instead of
    to within a rounded pixel count.

    Row index increases with y, matching both the raster's own row order
    and `render.py`. Nothing in this project's drawing path flips a
    coordinate (see preview.py), and a viewer that mirrored the field
    would be worse than none.
    """
    if pixels_per_mm <= 0:
        raise ValueError(f"pixels_per_mm must be positive, got {pixels_per_mm}")

    low, high = FieldExtent(part)
    counts = np.maximum(np.floor((high - low) * pixels_per_mm).astype(int) + 1, 1)
    xs = low[0] + np.arange(counts[0]) / pixels_per_mm
    ys = low[1] + np.arange(counts[1]) / pixels_per_mm
    return np.stack(np.meshgrid(xs, ys), axis=-1)


def Distances(part: Part, pixels_per_mm: float = DEFAULT_PIXELS_PER_MM) -> np.ndarray:
    """The part's signed distance at every screen pixel, in mm, negative
    inside.
    """
    grid = _ScreenGrid(part, pixels_per_mm)
    return part.SampleSdf(grid.reshape(-1, 2)).reshape(grid.shape[:2])


def GradientMagnitudes(part: Part, pixels_per_mm: float = DEFAULT_PIXELS_PER_MM) -> np.ndarray:
    """The length of the field's gradient at every screen pixel.

    One wherever the field is a well-resolved distance, and less wherever
    it is not - which is the whole content of the gradient view.
    """
    grid = _ScreenGrid(part, pixels_per_mm)
    return np.linalg.norm(part.SampleDerivative(grid.reshape(-1, 2)), axis=1).reshape(grid.shape[:2])


def _Changes(values: np.ndarray) -> np.ndarray:
    """Pixels differing from the neighbour above or to the left of them.

    A one-pixel line held consistently to the high side of each crossing
    rather than straddling it, so two crossings one pixel apart stay two
    lines instead of merging into a smear.
    """
    changed = np.zeros(values.shape, dtype=bool)
    changed[:, 1:] |= values[:, 1:] != values[:, :-1]
    changed[1:, :] |= values[1:, :] != values[:-1, :]
    return changed


def LevelMask(distance: np.ndarray, level: float) -> np.ndarray:
    """Pixels the `level` iso-line passes through.

    Found by where the field crosses the level rather than by tracing a
    contour, which keeps the line one pixel wide however steep or flat the
    field is there - a traced polyline would need a width, and a width in
    millimeters would swell into a blob along the medial axis where the
    field goes nearly flat.
    """
    return _Changes(distance > level)


def BandEdges(distance: np.ndarray, band_mm: float = DEFAULT_BAND_MM) -> np.ndarray:
    """Every contour line at once: the pixels where the field crosses any
    multiple of `band_mm`.

    One pass over a band index rather than a `LevelMask` per level, since
    a field reaching 15mm inside a bowl and 6mm outside it would otherwise
    be twenty-one separate sweeps of the whole image.
    """
    if band_mm <= 0:
        raise ValueError(f"band spacing must be positive, got {band_mm}")
    return _Changes(np.floor(distance / band_mm))


def ShadeDistance(distance: np.ndarray, band_mm: float = DEFAULT_BAND_MM) -> np.ndarray:
    """A signed distance field as a BGR image: one hue inside, another
    outside, both fading with distance, contoured every `band_mm`.
    """
    tint = np.where(distance[..., None] < 0, INSIDE_COLOR, OUTSIDE_COLOR).astype(np.float64)
    weight = MIN_TINT + (1.0 - MIN_TINT) * FADE_MM / (FADE_MM + np.abs(distance))

    image = 255.0 + (tint - 255.0) * weight[..., None]
    image[BandEdges(distance, band_mm)] *= BAND_SHADE
    return image.astype(np.uint8)


def ShadeGradient(magnitude: np.ndarray) -> np.ndarray:
    """Gradient length as a BGR image: white where the field is a
    well-resolved distance, dark along the creases where it is not.

    Those creases are the medial axis - points equidistant from two pieces
    of boundary, where "the nearest way out" stops being a single
    direction and the interpolated gradient splits the difference between
    two of them. That set is exactly where `ComputeEnergy` says its forces
    become untrustworthy: a sample that penetrates past one is pushed out
    the far side of the part rather than back the way it came. Seeing
    where they run in a particular part is the difference between "the
    solver wedged" and "the solver wedged *there*".

    **Deviation from unit length, not the length itself**, and the
    difference is not cosmetic. Where two *opposing* walls meet - the
    spine down a rectangle's long axis - the two gradients are `(0, 1)`
    and `(0, -1)` and the interpolant between them reads nearly zero.
    Where two *perpendicular* walls meet - the 45-degree branch running
    into each corner - they are `(1, 0)` and `(0, 1)`, and the interpolant
    reads `sqrt(2)`. Both are the same crease and the same untrustworthy
    force; shading by raw magnitude would paint the second one pure white,
    hiding exactly half the medial axis of the simplest possible part.

    Gray rather than a color ramp: one quantity, with a meaningful zero,
    so brightness alone carries it and there is no key to look up.
    """
    value = ((1.0 - np.clip(np.abs(magnitude - 1.0), 0.0, 1.0)) * 255.0).astype(np.uint8)
    return np.repeat(value[..., None], 3, axis=2)


def _DrawLevel(image: np.ndarray, distance: np.ndarray, level: float, color: tuple[int, int, int]) -> None:
    """One iso-line onto the image, in place."""
    image[LevelMask(distance, level)] = color


def _DrawPoints(image: np.ndarray, points: np.ndarray, low: np.ndarray, pixels_per_mm: float) -> None:
    """The boundary sample points onto the image, in place.

    Clipped rather than scaled to fit: a sample outside the field is not a
    drawing problem to solve but a real one, and dropping it silently is
    the least bad of the options available to a paintbrush.
    """
    pixels = np.round((np.asarray(points, dtype=np.float64) - low) * pixels_per_mm).astype(int)
    height, width = image.shape[:2]
    covered = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    image[pixels[covered, 1], pixels[covered, 0]] = SAMPLE_COLOR


def RenderField(part: Part, view: FieldView | None = None, params: LayoutParameters | None = None) -> np.ndarray:
    """A part's distance field as a BGR image.

    The outline is drawn on both views, since without it the gradient view
    is an abstract pattern. The clearance rings are drawn only on the
    distance view and only when `params` says what they are: they are
    levels of *this* field with a direct reading - `c_pair_enforced` is
    the line no other part's boundary may cross, `spacing_pair` is where
    the spacing springs try to hold one - and both are meaningless
    overlaid on gradient length.

    Note which clearance is absent. `c_wall` is not a level of a part's
    field at all: the wall term measures a part's samples against the
    *container's* depth, which is analytic and has no raster to view.
    Drawing a ring for it here would put a plausible-looking line on the
    picture that stood for nothing the solver computes.
    """
    view = view or FieldView()
    distance = Distances(part, view.pixels_per_mm)

    if view.gradient:
        image = ShadeGradient(GradientMagnitudes(part, view.pixels_per_mm))
    else:
        image = ShadeDistance(distance, view.band_mm)
        if params is not None:
            _DrawLevel(image, distance, params.c_pair_enforced, PAIR_COLOR)
            _DrawLevel(image, distance, params.spacing_pair, SPACING_COLOR)

    _DrawLevel(image, distance, 0.0, CONTOUR_COLOR)
    if view.samples:
        _DrawPoints(image, part.samples, FieldExtent(part)[0], view.pixels_per_mm)
    return image
