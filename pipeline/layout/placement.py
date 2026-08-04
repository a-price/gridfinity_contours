"""Putting a Part somewhere in a bin.

How a part is turned is a `Pose`: an exact quarter turn, plus a free angle
on top of it. The split is deliberate and is what lets the three rotation
modes share one transform.

**The quarter turn stays exact.** It is an axis swap and a sign flip, with
no interpolation and no accumulated error, exactly as D1 describes - so
with the free angle at zero, every number this module produces is
bit-identical to what it produced when quarter turns were the only option.
That is what keeps the 90-degree mode a true default rather than a
rounding of the general case.

**The free angle turns about a fixed pivot**, the center of the
quarter-turned bounding box. The pivot has to be independent of the angle
for the angle to be something a descent can move: `position` anchors the
*minimum corner* of the box, and re-deriving that corner after an
arbitrary rotation would make it a minimum over sample points - continuous
in the angle but not differentiable, and wrong in exactly the places a
torque is trying to read. Turning about a constant pivot makes the
derivative of every sample position a clean `z x r`, which is what
`energy.ComputeEnergy` differentiates.

The consequence to know: with a non-zero angle, `position` is no longer
the placed part's bounding-box corner. Ask `Placement.Bounds` for that.
"""

from dataclasses import dataclass

import numpy as np

from pipeline.layout.container import DEFAULT_INTERIOR_INSET_MM, BuildContainer, Container
from pipeline.layout.part import Part

# One quarter turn in radians, so that a Pose can report the single angle
# its two halves add up to.
QUARTER_TURN = np.pi / 2.0


@dataclass(frozen=True)
class Pose:
    """How a part is turned: `orientation` exact quarter turns, then
    `angle` radians more.

    One value would have been simpler and is the wrong shape. The whole
    reason to keep the quarter turn as its own integer is that it can be
    applied exactly; folding it into a single float would put `cos(pi/2)`
    at 6.1e-17 rather than 0 and quietly make every "unrotated" part
    slightly rotated. Splitting it means the discrete modes never touch a
    trigonometric function at all.

    Hashable, because the candidate poses are dictionary keys in
    `orientation._Profiles`.
    """

    orientation: int = 0
    angle: float = 0.0

    @property
    def total(self) -> float:
        """The single angle the two halves come to, in radians. For
        reporting and for tests - never for applying, which is what the
        split exists to avoid.
        """
        return self.orientation * QUARTER_TURN + self.angle

    @property
    def upright(self) -> bool:
        """Whether this is a plain quarter turn, and so exact."""
        return self.angle == 0.0

    def __str__(self) -> str:
        return f"{np.rad2deg(self.total):.1f} deg"


def SpinPoints(points: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    """Rotate points by `angle` radians counterclockwise about `pivot`.

    Short-circuits at zero rather than multiplying by an identity matrix.
    That is not only for speed: it is what makes a pose with no free angle
    return its input untouched, so the 90-degree mode cannot drift by a
    rounding error the quarter-turn path does not have.
    """
    if angle == 0.0:
        return points

    cos, sin = np.cos(angle), np.sin(angle)
    offset = np.asarray(points, dtype=np.float64) - pivot
    x, y = offset[..., 0], offset[..., 1]
    return np.stack([cos * x - sin * y, sin * x + cos * y], axis=-1) + pivot


def SpinVectors(vectors: np.ndarray, angle: float) -> np.ndarray:
    """Rotate direction vectors by `angle` radians counterclockwise.

    The same split `RotateVectors` makes against `RotatePoints`: a
    direction has no location, so there is no pivot to turn about.
    """
    if angle == 0.0:
        return vectors

    cos, sin = np.cos(angle), np.sin(angle)
    vectors = np.asarray(vectors, dtype=np.float64)
    x, y = vectors[..., 0], vectors[..., 1]
    return np.stack([cos * x - sin * y, sin * x + cos * y], axis=-1)


def RotatePoints(points: np.ndarray, orientation: int, size: np.ndarray) -> np.ndarray:
    """Rotate local-frame points by `orientation` quarter turns
    counterclockwise, keeping the bounding box's minimum corner at the
    origin. `size` is the part's unrotated (width, height).

    Quarter turns are exact - no interpolation, no accumulated error - which
    is the whole reason D1 restricts orientation to these four.
    """
    x, y = points[:, 0], points[:, 1]
    width, height = float(size[0]), float(size[1])

    match orientation % 4:
        case 0:
            rotated = (x, y)
        case 1:
            rotated = (height - y, x)
        case 2:
            rotated = (width - x, height - y)
        case _:
            rotated = (y, width - x)
    return np.stack(rotated, axis=-1)


def RotateVectors(vectors: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate direction vectors by `orientation` quarter turns
    counterclockwise.

    RotatePoints also shifts the bounding box back to the origin, which is
    right for positions and wrong for directions - a gradient has no
    location to translate. Forces cross between frames constantly, so the
    two operations are kept separate rather than sharing an argument.
    """
    x, y = vectors[..., 0], vectors[..., 1]

    match orientation % 4:
        case 0:
            rotated = (x, y)
        case 1:
            rotated = (-y, x)
        case 2:
            rotated = (-x, -y)
        case _:
            rotated = (y, -x)
    return np.stack(rotated, axis=-1)


def RotatedSize(size: np.ndarray, orientation: int) -> np.ndarray:
    """A part's (width, height) after `orientation` quarter turns."""
    return size[::-1].copy() if orientation % 2 else np.asarray(size, dtype=np.float64).copy()


def PoseBounds(part: Part, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """Where the part's outline sits at this pose, relative to the anchor
    `Placement.position` names - as (minimum, maximum) corners.

    At a quarter turn this is `(0, 0)` to `RotatedSize`: the anchor *is*
    the corner, which is the convention the whole package was built on. Off
    axis the two come apart, and both directions matter. The shape can
    reach outside the quarter-turned box - a 200x42mm spoon spun 45 degrees
    about its box center sweeps far past the box's short sides - so a
    caller that assumed containment would place parts through the bin wall.
    And it can fall well inside on the other axis, so a caller that assumed
    the box would waste room it did not need to.

    Everything that needs to know where a placed part actually is asks
    this, or asks `Placement.Bounds`, which is this plus the position.
    """
    turned = RotatePoints(part.contour, pose.orientation, part.size)
    spun = SpinPoints(turned, pose.angle, RotatedSize(part.size, pose.orientation) / 2.0)
    return spun.min(axis=0), spun.max(axis=0)


def PoseExtent(part: Part, pose: Pose) -> np.ndarray:
    """The (width, height) of the part's axis-aligned bounding box at this
    pose.

    Measured off the contour rather than by rotating `part.size`, and the
    difference is the point: for any angle that is not a quarter turn, the
    bounding box of a *rotated shape* is smaller than the rotation of its
    bounding box - a 200mm spoon turned 45 degrees needs far less room than
    its tilted 200x42mm box does. `solver.FittingPoses` filters on this and
    `packer.ProvablyTooSmall` reads that filter as proof a bin is too
    small, so overstating a pose's footprint here does not make the search
    safely conservative; it makes the bound wrong.

    Agrees with `RotatedSize` exactly at a quarter turn, where the two
    boxes coincide.
    """
    low, high = PoseBounds(part, pose)
    return high - low


def PoseInertia(part: Part) -> float:
    """The second moment of a part's boundary samples about its own pivot,
    in mm^2 - unit mass per sample, and independent of pose.

    This is what makes one angular step size mean the same thing for a
    200mm spoon and a 15mm washer. A torque is a sum of `r x f` over
    samples, so it grows with both how many samples a part has and how far
    they sit from the pivot; dividing by this takes out both at once and
    leaves `angular_step_scale` meaning "how far the outermost point moves
    per unit of violation", in the same spirit as `step_scale`.

    Pose-independent because rotation preserves distance from the pivot, so
    it can be computed once when a descent starts rather than per step.
    """
    pivot = part.size / 2.0
    return float((np.square(np.asarray(part.samples, dtype=np.float64) - pivot)).sum())


def PoseRadius(part: Part) -> float:
    """How far the outermost boundary sample sits from the pivot, in mm.

    The lever that converts the descent's millimetre-denominated limits
    into radians: `max_step` and `jitter` both say how far a part may move,
    and dividing them by this says how far it may *turn* for the same
    movement at its extremity. That keeps the reason `max_step` exists - a
    part that jumps too far lands past another's medial axis, where the
    forces reverse - applying to rotation without a second number anybody
    would have to tune.
    """
    pivot = part.size / 2.0
    return float(np.linalg.norm(np.asarray(part.samples, dtype=np.float64) - pivot, axis=1).max())


@dataclass(frozen=True)
class Placement:
    """One part positioned in a bin: `orientation` quarter turns
    counterclockwise about its local origin, translated by `position` into
    bin-local millimeters (origin at the interior's minimum corner), then
    turned a further `angle` radians about `Pivot`.

    With `angle` at zero - every placement in the 90-degree mode - the last
    step is skipped entirely and this is exactly the transform it always
    was. With a non-zero angle, `position` stops being the placed part's
    bounding-box corner and becomes only the anchor the pivot is measured
    from; `Bounds` is then the only honest source for the box.
    """

    part_id: int
    position: np.ndarray  # (2,) bin-local mm
    orientation: int = 0
    angle: float = 0.0  # radians, on top of the quarter turns

    @property
    def pose(self) -> Pose:
        return Pose(self.orientation, self.angle)

    def Pivot(self, part: Part) -> np.ndarray:
        """The bin-coordinate point the free angle turns about: the center
        of the quarter-turned bounding box.

        Independent of `angle` by construction, which is the property the
        torque in `energy.ComputeEnergy` is differentiated against. The
        center rather than the corner because a part should spin in place
        rather than swing around one of its ends - a corner pivot would
        turn every angular step into a large translation as well, and the
        step cap would have to be tightened until the angle barely moved.
        """
        return self.position + RotatedSize(part.size, self.orientation) / 2.0

    def ToWorld(self, part: Part) -> np.ndarray:
        """The part's contour placed into bin coordinates."""
        turned = RotatePoints(part.contour, self.orientation, part.size) + self.position
        return SpinPoints(turned, self.angle, self.Pivot(part))

    def SamplesToWorld(self, part: Part) -> np.ndarray:
        """The part's boundary sample points placed into bin coordinates."""
        turned = RotatePoints(part.samples, self.orientation, part.size) + self.position
        return SpinPoints(turned, self.angle, self.Pivot(part))

    def ToLocal(self, part: Part, points: np.ndarray) -> np.ndarray:
        """Bin coordinates back into this part's own local frame, ready to
        query against its distance field. The inverse of ToWorld.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        unspun = SpinPoints(points, -self.angle, self.Pivot(part))
        centered = unspun - self.position
        return RotatePoints(centered, -self.orientation, RotatedSize(part.size, self.orientation))

    def ToBin(self, part: Part, vectors: np.ndarray) -> np.ndarray:
        """Direction vectors from this part's local frame into the bin's.

        The rotation half of `ToWorld`, for gradients rather than
        positions. A field's derivative comes back in the frame of the part
        that owns the field, and has to make the same trip its points did.
        """
        return SpinVectors(RotateVectors(vectors, self.orientation), self.angle)

    def Bounds(self, part: Part) -> tuple[np.ndarray, np.ndarray]:
        """The placed part's axis-aligned bounding box, as (minimum,
        maximum) corners.

        Exact in both branches, and the branch matters. An upright part's
        box is its rotated size hung off `position` - no points touched. A
        turned one has to be measured off the contour, because a rotated
        box is not the box of the rotated shape and using it would report a
        part as overlapping neighbours it clears.
        """
        if self.angle == 0.0:
            return self.position, self.position + RotatedSize(part.size, self.orientation)
        low, high = PoseBounds(part, self.pose)
        return self.position + low, self.position + high


@dataclass(frozen=True)
class Layout:
    """A solved arrangement: the grid size chosen and where every part
    landed inside it.
    """

    grid: tuple[int, int]
    placements: dict[int, Placement]
    inset: float = DEFAULT_INTERIOR_INSET_MM

    @property
    def cells(self) -> int:
        return self.grid[0] * self.grid[1]

    def Interior(self) -> Container:
        """The bin interior this layout was packed into."""
        return BuildContainer(self.grid[0], self.grid[1], self.inset)

    def Envelope(self) -> np.ndarray:
        """That interior's boundary as a polygon."""
        return self.Interior().Polygon()
