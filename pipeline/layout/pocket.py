"""Growing an object's outline into the pocket that gets cut for it.

A pocket is the object dilated by `offset` - every point within that
distance of the outline. Keeping the two words apart matters more than it
sounds: the object is what was photographed, the pocket is what the bin
has removed from it, and every clearance that decides whether a bin
prints is a fact about pockets. This module is the only place the
distinction is made, and everything downstream is handed a pocket.

**What the offset is paying for.** Not one tolerance but a budget of
several, which is why it is a tuned number rather than a printer
specification:

* how accurately the outline was measured - camera resolution, the
  rectifying homography, and Douglas-Peucker on the way in;
* how accurately it gets cut - extrusion width, and the slicer's own
  rounding of a wall to a whole number of perimeters;
* how far off the object sits while it is being put in or taken out.

The last is the one that behaves differently from the others, and the one
that actually bites. The first two are fixed errors in the outline, but
retrieval clearance is a function of `pocket_depth`: lift an object
anywhere other than through its center of mass and it tilts, and a deep
pocket turns that tilt into a jam where a shallow one lets it ride out.
Nothing here models that coupling - `offset` is a single number and
`solid.pocket_depth` is chosen independently - so a deep bin wants a
larger offset, chosen by hand.

**Why a raster level set rather than a polygon offset.** Offsetting a
polygon is not the local operation it looks like. Grow a concave shape
and its walls run into each other: a spoon's bowl folds into itself, a
notch narrower than twice the offset closes over, and the naive
edge-by-edge construction produces a self-intersecting mess that has to
be repaired by a union pass. That repair is the whole difficulty, and it
is what OpenSCAD's `offset()` - Clipper underneath - spends its
complexity on.

A level set cannot self-intersect. Rasterize the object, take a distance
transform, trace the contour at height `offset`, and the topology comes
out right because a scalar field has no way to make it come out wrong.
`Part.DilatedArea` already reached this conclusion for *areas*, where the
usual `area + perimeter*r + pi*r^2` double-counts exactly where the
dilation folds into itself; this is the same argument applied to the
outline.

**Why its own raster rather than `Part.sdf`.** They answer different
questions and can afford different errors. A Part's field is sized for
collision detection at `DEFAULT_RESOLUTION_MM` (0.25mm), where
discretization error is covered by `LayoutParameters.raster_margin` and
costs nothing but a hair of extra clearance. A pocket outline is a *fit*
surface - it is the thing the tool has to drop into - and 0.25mm is a
quarter of a typical offset. So this rasterizes at 0.05mm, an order of
magnitude finer, and pays for it in transient memory rather than in fit.

**The error is deliberately one-sided.** Two approximations sit between
the object and the traced outline: the raster quantizes the boundary and
marching squares interpolates across it, and then Douglas-Peucker moves
the result by up to its own tolerance. Left symmetric, either could cut
*into* the pocket, and a pocket a hair too small is a tool that will not
go in - the one failure a print cannot be talked out of.

So the trace is taken above the requested offset rather than at it, by
enough to cover both. The raster term was measured against exact polygon
geometry rather than assumed: tracing the three spoon fixtures with no
margin at all put the outline at worst 0.023mm inside the requested level
at 0.05mm resolution, and 0.0095mm at 0.025mm - so it scales with the
cell, at just under half of one, and half a cell is the budget. The
simplification term is `simplify` outright, which is what
Douglas-Peucker guarantees.

The result is that the returned pocket always contains the ideal
dilation and overshoots it by at most `resolution + simplify` - 0.07mm
at the defaults, against the 0.29mm *under*shoot of the OpenSCAD
`offset()` this replaces.

Note that the overshoot budget is a *whole* cell while the level only
carries half of one. The half cell is what stops the trace landing
inside; the other half is the same error with the other sign, since
nothing makes the raster round outward in particular. Measured, the
spoon fixtures overshoot by 0.032 - 0.039mm and a 60-gon circle by
0.0698mm, which is what makes the whole cell the honest bound rather
than a doubled one.
"""

import cv2
import numpy as np
from skimage.measure import find_contours

from pipeline.layout.raster import FieldGrid, SignedDistanceField

# How much larger than its object a pocket is cut, in mm, when nobody
# says otherwise. The canonical value: `LayoutParameters.pocket_offset`
# defaults to it, the same way that class takes its raster resolution
# from `part`. See D5 for what a millimetre is buying, and the module
# docstring above for the fact that it is buying several things at once.
DEFAULT_OFFSET_MM = 1.0

# Raster resolution for the dilation, in mm per pixel. An order of
# magnitude finer than a Part's, for the reason in the module docstring.
# The cost is transient and quadratic: a 200mm object needs a raster
# around 4000 cells on its long side, which is tens of megabytes for the
# moment it takes to trace, and nothing afterwards.
DEFAULT_RESOLUTION_MM = 0.05

# Douglas-Peucker tolerance for the traced outline, in mm. Marching
# squares emits a vertex per cell crossing - thousands for a spoon, all
# of them on a grid - which is more than the solver, the PDF or OpenSCAD
# has any use for. This is small enough to be well under the raster's own
# error and still cuts the vertex count by more than an order of
# magnitude.
DEFAULT_SIMPLIFY_MM = 0.02


def PocketContour(
    object_contour: np.ndarray,
    offset: float,
    resolution: float = DEFAULT_RESOLUTION_MM,
    simplify: float = DEFAULT_SIMPLIFY_MM,
) -> np.ndarray:
    """The pocket for `object_contour`: its outline grown by `offset`.

    Returned in the frame it was given, as an open `(K, 2)` polygon - no
    repeated closing vertex, matching how contours are passed everywhere
    else in the pipeline.

    Every point of the result is at least `offset` from the object, and
    at most `offset + resolution + simplify` (0.07mm past it at the
    defaults). The bias is deliberate and outward; see the module
    docstring. An `offset` of zero is the object itself, returned
    unchanged rather than traced.

    **Holes are filled.** Dilating a horseshoe far enough closes its
    mouth and traps a void in the middle, which `find_contours` reports
    as a second loop. Keeping it would put a pillar in the pocket whose
    width nothing downstream measures - `ThinnestWalls` looks at dividers
    between pockets and at bin walls, not at islands inside one - so it
    could easily come out too thin to print. Filling costs a little extra
    material and keeps a pocket a simple polygon, which is what
    `Placement.ToWorld`, `verify` and OpenSCAD's `polygon()` all assume.
    """
    points = np.asarray(object_contour, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError(f"a contour needs at least 3 points, got {len(points)}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if simplify < 0:
        raise ValueError(f"simplify must be non-negative, got {simplify}")

    # A pocket with no offset is the object, and is returned as the object
    # rather than as a trace of it. Not a shortcut for speed: round-tripping
    # a rectangle through a raster hands back a few hundred vertices sitting
    # a simplification tolerance outside the four it started with, which is
    # strictly worse than the answer, and would make every caller that wants
    # a part with no clearance at all pay for the difference.
    if offset == 0.0:
        return points

    # Biased outward, half a cell for the raster and the full tolerance
    # for the simplification - see the module docstring for both numbers.
    level = offset + 0.5 * resolution + simplify

    # Far enough that the level set never reaches the raster's edge, which
    # `find_contours` would report as an open curve rather than a loop.
    origin, height, width = FieldGrid(points, level + 3.0 * resolution, resolution)
    sdf = SignedDistanceField(points, origin, height, width, resolution)

    traced = find_contours(sdf, level=level)
    if not traced:
        raise ValueError(f"no pocket outline at {level:.4f}mm - the contour may be degenerate")

    loops = [_ToMillimeters(loop, origin, resolution) for loop in traced]
    return _Simplify(_Outermost(loops), simplify)


def _ToMillimeters(loop: np.ndarray, origin: np.ndarray, resolution: float) -> np.ndarray:
    """One traced loop mapped from array indices back to millimeters.

    `find_contours` works in array-index space, where an integer index is
    a pixel *center* and fractions interpolate between them - so this is
    the inverse of the convention `raster.FieldGrid` documents, and the
    same one `Part._PixelCoordinates` applies in the other direction.

    The closing vertex is dropped: a loop that does not touch the array
    edge comes back with its first point repeated at the end, and the
    rest of the pipeline stores polygons open.
    """
    rows, cols = loop[:, 0], loop[:, 1]
    points = np.stack(
        [origin[0] + (cols + 0.5) * resolution, origin[1] + (rows + 0.5) * resolution],
        axis=-1,
    )
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    return points


def _Outermost(loops: list[np.ndarray]) -> np.ndarray:
    """The outer boundary among the traced loops.

    Chosen by bounding box area, which is exact for this question rather
    than a heuristic: every other loop `find_contours` can return here is
    a hole, a hole lies inside the outer boundary, and so its box is
    strictly smaller. Comparing polygon areas would work equally well and
    would need a winding convention this does not.
    """
    return max(loops, key=lambda loop: np.prod(loop.max(axis=0) - loop.min(axis=0)))


def _Simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Douglas-Peucker, at a tolerance the trace level has already paid for.

    Returns the input untouched if simplifying would leave too few points
    to be a polygon - which a real pocket never approaches, but a
    millimeter-scale test fixture can.
    """
    if epsilon <= 0:
        return points
    reduced = cv2.approxPolyDP(points.astype(np.float32).reshape(-1, 1, 2), epsilon, True)
    reduced = reduced.reshape(-1, 2).astype(np.float64)
    return reduced if len(reduced) >= 3 else points
