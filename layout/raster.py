"""Turning a millimeter polygon into a signed distance field.

Shared by `part`, which hands the field to the solver to sample, and by
`pocket`, which traces a level set out of one. It lives in its own module
because those two cannot import each other: a Part is built from a pocket
contour, and a pocket contour is traced from a raster.

The half-pixel correction below is the part worth not re-deriving. It is
invisible in a mask and shows up much later as a field that disagrees
with the geometry it came from.
"""

import cv2
import numpy as np


def RasterizePolygon(pixels: np.ndarray, height: int, width: int) -> np.ndarray:
    """A boolean mask of which pixel centers fall inside a polygon given in
    pixel coordinates, where pixel (r, c)'s center is at (c, r).

    This is a crossing-number test rather than cv2.fillPoly, which rounds
    the polygon's coordinates to whole pixels and fills inclusively - so an
    edge landing on a half-pixel (the common case, since a bounding box
    corner maps to one exactly) gains an extra row or column on the high
    side but not the low side. That asymmetric half-pixel is invisible in a
    mask and shows up later as a distance field that disagrees with the
    geometry by more at one end of a part than the other.

    Each edge is cast only across the rows it can actually straddle,
    rather than against the whole raster. That is the same mask - an edge
    contributes no crossing to a row outside its own y-range, so those
    pixels were being XORed with False - but it is what makes the fine
    rasters `pocket` needs affordable. A spoon at 0.05mm is 3.6M pixels
    against 40 edges: 1.75s scanned in full, 0.005s scanned by span, and
    the two masks are bit-identical.
    """
    x = np.arange(width, dtype=np.float64)
    y = np.arange(height, dtype=np.float64)

    inside = np.zeros((height, width), dtype=bool)
    starts = pixels
    ends = np.roll(pixels, -1, axis=0)
    for (x0, y0), (x1, y1) in zip(starts, ends):
        if y0 == y1:
            continue  # horizontal edges cross no horizontal ray

        # `straddles` below is true exactly on min(y0, y1) <= y < max(y0, y1),
        # so no row outside this span can contribute.
        low = max(0, int(np.ceil(min(y0, y1))))
        high = min(height, int(np.ceil(max(y0, y1))))
        if low >= high:
            continue

        rows = y[low:high, np.newaxis]
        straddles = (y0 > rows) != (y1 > rows)
        crossing_x = x0 + (rows - y0) * (x1 - x0) / (y1 - y0)
        inside[low:high] ^= straddles & (x < crossing_x)

    return inside


def FieldGrid(polygon: np.ndarray, pad: float, resolution: float) -> tuple[np.ndarray, int, int]:
    """A grid covering `polygon` plus `pad` on every side, as
    `(origin, height, width)`.

    `origin` is the millimeter coordinate of the raster's corner. Pixel
    (r, c) is centered at `origin + (c + 0.5, r + 0.5) * resolution`, which
    is the convention `Part._PixelCoordinates` inverts and the one
    `pocket` maps traced contours back through.

    For a caller that has no opinion about where its raster sits. `Part`
    does have one - `_AlignToLocalFrame` already put the bounding box's
    minimum corner at the origin, so it states `[-pad, -pad]` directly
    rather than re-deriving it from a minimum that PCA's float32 basis
    leaves a rounding error short of zero.
    """
    low = polygon.min(axis=0)
    extent = polygon.max(axis=0) - low
    origin = low - pad
    width = int(np.ceil((extent[0] + 2 * pad) / resolution))
    height = int(np.ceil((extent[1] + 2 * pad) / resolution))
    return origin, height, width


def SignedDistanceField(
    polygon: np.ndarray,
    origin: np.ndarray,
    height: int,
    width: int,
    resolution: float,
) -> np.ndarray:
    """A polygon's signed distance field on the given grid: negative
    inside, positive outside, in millimeters.

    distanceTransform measures to the nearest pixel of opposite state, so
    a pixel one step inside the boundary reports 1.0 where the true
    distance to the edge between them is 0.5. Subtracting that half pixel
    from both sides puts the zero level set on the polygon boundary
    instead of half a pixel outside it.
    """
    mask = RasterizePolygon((polygon - origin) / resolution - 0.5, height, width)

    filled = mask.astype(np.uint8) * 255
    inside = cv2.distanceTransform(filled, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(255 - filled, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (np.where(mask, -(inside - 0.5), outside - 0.5) * resolution).astype(np.float32)
