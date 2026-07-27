"""Everything tunable about a layout, in one place.

Held apart from the energy that consumes it because almost nothing that
needs these is computing an energy: the loader sizes distance fields from
them, the packer reads the grid limit, the solid generator takes the
pocket offset, the GUI edits them. Living in `energy.py` meant six of nine
modules importing a configuration object from a module they otherwise had
no business touching.

The clearances are the interesting part - see D5. They are *derived* from
`pocket_offset` rather than set independently, because they are not
independent: a pocket is cut that much larger than its object, so two
pockets `c_pair` apart leave a divider of `c_pair - 2*pocket_offset`,
which has to stay printable.
"""

from dataclasses import dataclass

from pipeline.layout.container import (
    DEFAULT_INTERIOR_INSET_MM,
    DIVIDER_WIDTH_MM,
    MIN_WALL_MM,
)
from pipeline.layout.part import DEFAULT_RESOLUTION_MM


@dataclass
class LayoutParameters:
    """Tunables for the packer.

    The three clearances are derived from `pocket_offset` rather than set
    independently, because they are not independent: a pocket is cut
    `pocket_offset` larger than its object so the object drops in, so two
    pockets `pair_clearance` apart leave a divider of
    `pair_clearance - 2*pocket_offset`, which has to stay printable.
    Setting them separately invites a layout whose dividers are too thin to
    print. Explicit values still override.
    """

    pocket_offset: float = 1.0
    pair_clearance: float | None = None
    wall_clearance: float | None = None
    resolution: float = DEFAULT_RESOLUTION_MM
    pair_weight: float = 1.0
    wall_weight: float = 1.0
    inset: float = DEFAULT_INTERIOR_INSET_MM
    max_grid: int = 6
    seed: int = 0

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

    @property
    def c_pair(self) -> float:
        """Minimum part-to-part spacing: enough for both pockets' offsets
        plus a printable divider between them.
        """
        if self.pair_clearance is not None:
            return self.pair_clearance
        return 2.0 * self.pocket_offset + DIVIDER_WIDTH_MM

    @property
    def c_wall(self) -> float:
        """Minimum part-to-wall spacing: the pocket's offset plus a
        printable wall.
        """
        if self.wall_clearance is not None:
            return self.wall_clearance
        return self.pocket_offset + MIN_WALL_MM

    @property
    def raster_margin(self) -> float:
        """Extra separation the solver drives to, covering the distance
        field's own measurement error.

        Part-to-part distance is read off a rasterized field and comes back
        short by up to the discretization error - measured at 0.04mm on a
        layout the solver considered finished, against a 3.2mm clearance.
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
        clearances differ (1.95 vs 3.2 by D5), and treating the raw
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
