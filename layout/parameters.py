"""Everything tunable about a layout, in one place.

Held apart from the energy that consumes it because almost nothing that
needs these is computing an energy: the loader sizes distance fields from
them, the packer reads the grid limit, the solid generator takes the
pocket offset, the GUI edits them. Living in `energy.py` meant six of nine
modules importing a configuration object from a module they otherwise had
no business touching.

The clearances are the interesting part - see D5. They used to be
*derived* from `pocket_offset`, because a part was an object and the
space a pocket would need around it had to be reserved by hand. Since
pockets became geometry the offset is already in the shape being packed,
and the clearances are what is left over: the printable divider and the
printable wall, and nothing else.
"""

from dataclasses import dataclass

from layout.container import (
    DEFAULT_INTERIOR_INSET_MM,
    DIVIDER_WIDTH_MM,
    MIN_WALL_MM,
)
from layout.part import DEFAULT_RESOLUTION_MM
from layout.pocket import (
    DEFAULT_OFFSET_MM,
    DEFAULT_RESOLUTION_MM as POCKET_RESOLUTION_MM,
    DEFAULT_SIMPLIFY_MM as POCKET_SIMPLIFY_MM,
)

# How much freedom a part has to turn inside its bin. The values are what a
# user types at `--rotation`, so they read as angles rather than as jargon.
#
# QUARTER is D1 as written and the default: four exact orientations, no
# torque anywhere in the model. EIGHTH adds the four diagonals - still a
# discrete set, so it costs the search nothing but a longer candidate list
# and still never interpolates a field. FREE hands the angle to the
# relaxation as a real degree of freedom, which is the only one of the
# three that needs a torque, an angular velocity, and a bound in
# `packer.ProvablyTooSmall` that does not assume a finite set of angles.
QUARTER_TURNS = "90"
EIGHTH_TURNS = "45"
FREE_ROTATION = "free"
ROTATIONS = (QUARTER_TURNS, EIGHTH_TURNS, FREE_ROTATION)


@dataclass
class LayoutParameters:
    """Everything tunable about a layout, read almost everywhere in this
    package and by every front end.

    **The clearances no longer carry the pocket offset, because the parts
    do.** What gets packed is a `Part`, and a Part is its *pocket* - the
    object already grown by `pocket_offset` before the solver ever sees
    it. So the space between two of them is divider, all of it, and
    `c_pair` is `DIVIDER_WIDTH_MM` flat. It used to be
    `2*pocket_offset + DIVIDER_WIDTH_MM`, which was the same number
    reached from the other end: back when a part was an *object*, the
    room its pocket would eventually need had to be reserved in the
    clearance instead. Keeping that form now would count the offset
    twice - object to object would come out at `4*pocket_offset +
    DIVIDER_WIDTH_MM` - and every bin would be packed far looser than
    asked for. Explicit values still override.

    **Why this is one object and not three.** The fields fall into three
    groups that different people touch - what the answer must satisfy
    (`pocket_offset`, the clearances, `inset`, `max_grid`,
    `admissible_grids`), how hard to look for it (`seed`, `iterations`,
    `restarts`, `patience`, `placement_tries`, `spacing_iterations`), and
    how the searcher moves (`resolution`, `step_scale`, `damping`,
    `jitter`, `max_step`, `spacing_margin`). No module reads more than half
    of them, which looks like an argument for splitting.

    It is not, because a single derivation chain runs straight through all
    three: `c_pair` plus `resolution` gives `c_pair_enforced`, plus
    `spacing_margin` gives `spacing_pair`, which gives `pad` - and `pad`
    is what the *rasterizer* sizes distance fields from. Split the groups
    and that chain is scattered across three types that have to reach into
    each other. One object with grouped fields keeps it derivable in one
    place.
    """

    # How much larger than its object each pocket is cut. Read by
    # `loading.BuildParts` on the way in and by `solid` on the way out,
    # and - since pockets became geometry - by nothing in between: the
    # solver packs shapes that already have it.
    pocket_offset: float = DEFAULT_OFFSET_MM

    # The raster the dilation is traced on, and how hard the traced
    # outline is simplified afterwards. Separate from `resolution`, which
    # sizes the *solver's* fields, because the two buy different things -
    # see the `pocket` module docstring. Finer costs time quadratically
    # and buys a tighter fit; the test suite turns it down.
    pocket_resolution: float = POCKET_RESOLUTION_MM
    pocket_simplify: float = POCKET_SIMPLIFY_MM
    pair_clearance: float | None = None
    wall_clearance: float | None = None
    resolution: float = DEFAULT_RESOLUTION_MM
    inset: float = DEFAULT_INTERIOR_INSET_MM

    # Largest bin dimension the search will try, in cells. Seven rather
    # than six because six does not hold a real household: of the eighteen
    # objects in `test_data/`, four need a seven-cell bin - huge_server at
    # 260mm, serving_spoon at 274mm, server at 251mm and the knife at
    # 243mm. The knife is the instructive one, since a six-cell interior is
    # 246.3mm and it needs 247.0mm with its wall clearance - it misses by
    # 0.7mm and is reported unpackable at every size.
    #
    # The argument against is that a seven-cell bin is 294mm long and wants
    # a drawer to match. That argument is `admissible_grids`' to make
    # rather than this cap's: once drawers are known,
    # `drawer.AdmissibleFootprints` filters this down to the footprints
    # that actually fit one, so the cap only binds when nobody has said
    # what they own. There the two errors are not symmetric - too high
    # costs a few extra candidate grids, which the packer's bounds reject
    # instantly, and too low costs an object that cannot be packed at all.
    max_grid: int = 7
    seed: int = 0

    # One of ROTATIONS - see the constants above for what each buys. This
    # sits in the "what the answer must satisfy" group rather than the
    # "how hard to look" one on purpose: it changes which arrangements are
    # legal, not merely which ones get found, so two results computed at
    # different settings are not comparable and a session records it.
    rotation: str = QUARTER_TURNS

    # Bin footprints anything downstream will accept, or None for "any up
    # to max_grid" - in practice from `drawer.AdmissibleFootprints`, which
    # says why. A frozenset rather than a predicate so that a parameter set
    # stays comparable and printable.
    admissible_grids: frozenset[tuple[int, int]] | None = None

    # Solver budget. `patience` abandons an attempt whose best energy has
    # not improved for that many iterations: a wedged arrangement does not
    # come unwedged by being pushed harder, and without it a hopeless bin
    # costs the full iteration budget on every one of its restarts.
    iterations: int = 400
    restarts: int = 24
    patience: int = 25
    # Random positions tried after the contact sweep comes up empty. Cheap
    # in practice: an easy bin is solved by a contact and never reaches
    # these, and only the concave nestings that bounding-box contacts
    # cannot express spend the budget.
    placement_tries: int = 150

    # Descent shape. `step_scale` is applied to a force already divided by
    # the part's sample count, so it means roughly "millimeters per
    # millimeter of average violation" and does not need retuning when the
    # raster resolution changes.
    step_scale: float = 0.6
    damping: float = 0.6
    jitter: float = 0.35

    # The angular counterpart of `step_scale`, used only when `rotation` is
    # FREE. It is the one number free rotation adds, and it starts equal to
    # `step_scale` because `placement.PoseInertia` normalizes torque so
    # that the two are directly comparable: at 0.6 apiece, one step turns a
    # part far enough that its outermost point travels about as far as a
    # translation step would move the whole part.
    #
    # Separate rather than shared so rotation can be slowed without
    # slowing translation. A relaxation that spins faster than it slides
    # tends to windmill a long part through its neighbours, and the two
    # failures want different fixes.
    #
    # There is deliberately no angular `damping`, `jitter` or `max_step`.
    # Damping is dimensionless, so it transfers unchanged; the other two
    # are millimetres and become radians by dividing by
    # `placement.PoseRadius` - what the part's furthest point may move,
    # which is the reason those limits exist in the first place.
    angular_step_scale: float = 0.6

    # How far past each clearance the spacing springs reach, and how long
    # the balancing pass gets.
    #
    # Longer is not better, and the curve has a clear optimum. Measured on
    # the three spoons from one fixed starting layout, the spread of the
    # gaps came out 3.45 / 2.74 / 1.98 / 2.04 / 2.92 / 3.81mm at margins of
    # 1.0 / 1.5 / 2.0 / 2.5 / 3.0 / 4.0. A rest length far beyond the slack
    # the bin actually has compresses every spring against the hard limits
    # instead of letting them balance, and contacts that 2.0mm leaves at
    # 1.6mm of slack get squeezed to 0.5mm at 4.0mm. It also costs raster
    # memory, since `pad` is sized from it.
    spacing_margin: float = 2.0
    spacing_iterations: int = 150

    # Hard cap on how far a part may move in one iteration. This is not
    # only for stability: a part that jumps several millimeters can land
    # more than halfway through another, which is exactly the regime where
    # the forces reverse (see ComputeEnergy). Capping the step keeps the
    # solver inside the range where its gradient is trustworthy.
    max_step: float = 0.6

    def __post_init__(self) -> None:
        # Checked here rather than where it is read, because every reader
        # tests for one mode and takes the other branch otherwise. A typo
        # would land in the eighth-turn branch of `solver.CandidatePoses`
        # and be reported as "at any eighth turn" by `packer` - the
        # quietest possible failure, a perfectly good layout that simply
        # ignored what was asked for.
        if self.rotation not in ROTATIONS:
            raise ValueError(f"rotation must be one of {', '.join(ROTATIONS)}, got {self.rotation!r}")

    @property
    def free_rotation(self) -> bool:
        """Whether the angle is a degree of freedom the relaxation moves,
        rather than one the restart loop picks from a list.

        The distinction almost everything downstream actually cares about -
        90 and 45 differ only in how long the candidate list is, while FREE
        is the one that needs a torque and invalidates any bound that
        assumes a finite set of angles.
        """
        return self.rotation == FREE_ROTATION

    @property
    def c_pair(self) -> float:
        """Minimum pocket-to-pocket spacing: a printable divider, since
        that is all the gap between two pockets has to be.
        """
        if self.pair_clearance is not None:
            return self.pair_clearance
        return DIVIDER_WIDTH_MM

    @property
    def c_wall(self) -> float:
        """Minimum pocket-to-wall spacing: a printable wall."""
        if self.wall_clearance is not None:
            return self.wall_clearance
        return MIN_WALL_MM

    @property
    def raster_margin(self) -> float:
        """Extra separation the solver drives to, covering the distance
        field's own measurement error.

        Part-to-part distance is read off a rasterized field and comes back
        short by up to the discretization error - measured at 0.04mm on a
        layout the solver considered finished, well inside one raster cell.
        Without this the solver stops at what it *measures* as `c_pair` and
        the true gap is a hair under, quietly falsifying the guarantee that
        zero energy means every clearance is met.

        One raster cell is comfortably above the observed error and costs
        well under a tenth of the clearance it protects. Part-to-*wall*
        distance needs no such thing: the container is analytic.
        """
        return self.resolution

    @property
    def c_pair_enforced(self) -> float:
        """What the energy actually drives part separation to, so that the
        separation exact geometry reports is at least `c_pair`.
        """
        return self.c_pair + self.raster_margin

    @property
    def spacing_pair(self) -> float:
        """The spacing springs' rest length between parts.

        Every active neighbour shares this one target, which is what makes
        the gaps come out even: where parts are mutually blocked, equal
        springs balance at equal compression. Nothing imposes uniformity -
        it falls out of them all pulling toward the same length.

        It only ever pushes apart. Equalizing the *variance* of the gaps
        would have been the other reading of "make them the same", and it
        is the wrong one: it would happily drag a roomy gap down toward a
        cramped one, which is worse in every way that matters to a print.
        """
        return self.c_pair_enforced + self.spacing_margin

    @property
    def spacing_wall(self) -> float:
        """The same rest length against the bin wall.

        Measured from `c_wall` rather than `c_pair`, so what the springs
        equalize is the *slack* over each contact's own clearance. The two
        clearances differ (0.95 vs 1.2 by D5), and treating the raw
        distances as comparable would systematically favour one.
        """
        return self.c_wall + self.spacing_margin

    @property
    def pad(self) -> float:
        """How far each part's distance field must reach beyond itself.

        A part only feels another once a sample lands inside the other's
        raster, so a field that stops short of the enforced clearance would
        let parts pass straight through each other at exactly the distance
        it is meant to protect.

        It also has to outreach the spacing springs, which is now the
        binding requirement: a spring whose rest length sat outside the
        raster would read every neighbour as infinitely far and pull on
        nothing. That is the real limit on how far parts can be spread -
        not the optimizer, but how far the field can see.
        """
        return self.spacing_pair + 1.0


def FromOverrides(**overrides) -> "LayoutParameters":
    """The tuned defaults, with only the given fields overridden.

    For a caller translating parsed flags into parameters: a flag the user
    did not pass should keep its tuned default rather than the caller
    re-specifying that default itself, which is how the two would
    eventually drift. `layout_cli` and `layout_demo` each expose a
    different subset of fields as flags, so each still builds its own
    overrides dict from its own `argparse.Namespace` - this is only the
    "apply what was actually given" rule they both followed separately.
    """
    return LayoutParameters(**{name: value for name, value in overrides.items() if value is not None})
