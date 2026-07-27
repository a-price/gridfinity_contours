"""Independent geometric checks on a layout.

Deliberately shares no code with the solver. It measures overlap
with rasterized distance fields, which are approximate by construction; if
the same fields were used to confirm the result, a raster artifact would
confirm itself and the first sign of trouble would be a failed print. So
these run on the polygons directly, via matplotlib's exact path
intersection - a library already in the dependency set for plotting, and
one that shares nothing with OpenCV's distance transforms.

Used by the tests, and by the CLI to self-check before writing a preview.
"""

import numpy as np
from matplotlib.path import Path

from pipeline.layout.part import Part
from pipeline.layout.solver import Layout


def _AsPath(points: np.ndarray) -> Path:
    return Path(np.asarray(points, dtype=np.float64).reshape(-1, 2), closed=False)


def PolygonsOverlap(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two closed polygons share any interior area.

    Edge crossings alone miss the case where one polygon sits entirely
    within the other without any edge intersection, so containment is
    tested separately in both directions. Exact edge contact counts as
    overlapping, which is conservative and harmless: the clearances of D5
    are millimeters, so nothing legitimate ever comes that close.
    """
    path_a, path_b = _AsPath(a), _AsPath(b)
    if path_a.intersects_path(path_b, filled=True):
        return True
    return bool(path_a.contains_point(b[0]) or path_b.contains_point(a[0]))


def MinimumSeparation(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between two non-overlapping polygons' boundaries,
    in mm. Negative is not reported - use PolygonsOverlap for that - so
    this is only meaningful once overlap is ruled out.

    Measured point-to-segment in both directions, which is exact for
    polygons.
    """
    return min(_PointsToEdges(a, b), _PointsToEdges(b, a))


def DistanceToBoundary(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Exact distance from each point to a polygon's boundary, in mm.
    Unsigned - inside and outside both read positive.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    starts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    ends = np.roll(starts, -1, axis=0)
    edges = ends - starts

    lengths = np.einsum("ij,ij->i", edges, edges)
    lengths = np.where(lengths > 0, lengths, 1.0)

    # For every (point, edge) pair, project the point onto the edge's
    # infinite line, clamp to the segment, and measure.
    offsets = points[:, None, :] - starts[None, :, :]
    t = np.clip(np.einsum("pej,ej->pe", offsets, edges) / lengths, 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * edges[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)


def ExactSignedDistance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Signed distance from each point to a polygon, negative inside -
    computed from the polygon itself, with no raster anywhere in sight.

    This is the reference Part.SampleSdf is checked against. Because it
    shares no machinery with the distance transforms, a raster artifact
    cannot hide by appearing in both.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    distances = DistanceToBoundary(points, polygon)
    inside = _AsPath(polygon).contains_points(points)
    return np.where(inside, -distances, distances)


def _PointsToEdges(points: np.ndarray, polygon: np.ndarray) -> float:
    """Smallest distance from any of `points` to any edge of `polygon`."""
    return float(DistanceToBoundary(points, polygon).min())


def PolygonInside(polygon: np.ndarray, container: np.ndarray) -> bool:
    """True if every vertex of `polygon` lies within `container` and their
    boundaries do not cross.

    Both halves are needed: vertex containment alone would accept a shape
    that bulges out through a container edge and back in, and the crossing
    test alone would accept a shape entirely outside a container it never
    touches.
    """
    container_path = _AsPath(container)
    if not container_path.contains_points(np.asarray(polygon, dtype=np.float64)).all():
        return False
    return not container_path.intersects_path(_AsPath(polygon), filled=False)


def CheckLayout(
    layout: Layout,
    parts: dict[int, Part],
    pair_clearance: float = 0.0,
    wall_clearance: float = 0.0,
) -> list[str]:
    """Every way `layout` violates its own constraints, as human-readable
    strings. An empty list means the arrangement is sound.

    Returns all violations rather than the first, so a failing test says
    what went wrong everywhere instead of one symptom at a time.
    """
    envelope = layout.Envelope()
    placed = {part_id: placement.ToWorld(parts[part_id]) for part_id, placement in layout.placements.items()}
    problems = []

    for part_id, polygon in sorted(placed.items()):
        if not PolygonInside(polygon, envelope):
            problems.append(f"part {part_id} is not fully inside the {layout.grid[0]}x{layout.grid[1]} interior")
        elif wall_clearance > 0:
            separation = _PointsToEdges(polygon, envelope)
            if separation < wall_clearance:
                problems.append(
                    f"part {part_id} is {separation:.3f}mm from the wall, below the {wall_clearance:.3f}mm clearance"
                )

    for i, (id_a, polygon_a) in enumerate(sorted(placed.items())):
        for id_b, polygon_b in sorted(placed.items())[i + 1 :]:
            if PolygonsOverlap(polygon_a, polygon_b):
                problems.append(f"parts {id_a} and {id_b} overlap")
            elif pair_clearance > 0:
                separation = MinimumSeparation(polygon_a, polygon_b)
                if separation < pair_clearance:
                    problems.append(
                        f"parts {id_a} and {id_b} are {separation:.3f}mm apart, "
                        f"below the {pair_clearance:.3f}mm clearance"
                    )

    return problems
