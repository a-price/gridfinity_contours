# Layout Optimization Design

Built through M9. Companion implementation plan:
[layout_roadmap.md](layout_roadmap.md); the view from above is
[architecture.md](architecture.md).

## Problem

The capture pipeline turns a photo into one real-world-scale contour per
object. `solid.py` then wraps a *single* contour in the smallest
Gridfinity bin that holds it. That is wasteful: a spoon needs a 1x4 bin
almost all of which is empty, and a drawer full of one-tool-per-bin
shadow boxes squanders most of its grid.

We want the opposite: given a set of contours, drop them into one bin and
shuffle them around until they all fit inside the smallest practical
number of grid cells, without overlapping each other or the bin walls.

This is 2D irregular strip/bin packing (a.k.a. nesting), which is
NP-hard, so we are after a good arrangement found quickly, not a proven
optimum.

## Scope

Two phases, in this order:

1. **Arrangement** (this document's focus). Given an explicit set of
   contours that the user has decided should share a bin, find the
   smallest `N x M` grid and a collision-free placement of every contour
   inside it.
2. **Grouping**. Given all the contours, decide the partition into bins
   that minimizes total cells. Phase 1 is the feasibility oracle this
   search calls; see [Grouping](#grouping).

## Inputs and outputs

Input is the pipeline's existing currency: real-world millimeter polygons,
`dict[int, np.ndarray]` of `(K, 2)` float arrays, as produced by
`Rectify.Run` ([rectify.py:27](../pipeline/rectify.py#L27)). Contours are
already Douglas-Peucker-simplified and closed implicitly (last point
joins first). They are *not* required to be convex — concavity is the
whole point, since a wrench's fork is free space another part can nest
into.

Output is a `Layout`: the chosen grid size plus, per contour, a rigid
placement (translation + one of four orientations) into bin-local mm
coordinates whose origin is the bin's interior corner.

Note that the existing writers could not consume this as-is. Both
`WriteSvg` and `WritePdf` routed through `AlignContoursToPca`, which
re-aligns *each contour into its own local frame* — correct for exporting
one shape, fatally wrong for a layout, since it would move every part
back onto its own origin and discard the arrangement. M5 factored both
into an align-then-write wrapper over a write-these-coordinates core
(`WriteShapesSvg`/`WriteShapesPdf`, taking a `Shape`
([svg_writer.py](../pipeline/svg_writer.py))); the preview uses the core.

The `.scad` is generated from a `Layout` by
[pipeline/layout/solid.py](../pipeline/layout/solid.py) — a separate
module, so layout stays pure geometry. It supersedes the root
`solid.py`, whose single-contour path predates `Layout`.

That older path subtracts its cutout from *outside* `bin_render` and then
unions `bin_render_base` back on top. It is sound — the outer union
restores the base the difference removed, verified by intersecting the
result with a slab in the base under the cutout and finding it solid —
but the pocket depth is implicit: a bare `linear_extrude()` cuts 100mm
upward from z=0, so the floor lands wherever the base happens to start.
The new generator passes pockets as children of `bin_render`, the
mechanism the library provides, which places them at the top of the
infill extending downward by a stated depth and never cuts the base at
all.

## Container geometry

From the vendored spec constants
([standard.scad](../gridfinity-rebuilt-openscad/src/core/standard.scad)):

| Quantity | Value | Source |
| --- | --- | --- |
| Grid pitch | 42 mm | `GRID_DIMENSIONS_MM` |
| Bin outer footprint | `42*N - 0.5` mm | `BASE_TOP_DIMENSIONS`, 0.5 mm inter-bin gap |
| Outer corner radius | 3.75 mm | `BASE_TOP_RADIUS` |
| Stacking lip intrusion | 2.6 mm | `STACKING_LIP_SIZE.x` (includes wall) |
| Min wall | 0.95 mm | `d_wall` |
| Divider width | 1.2 mm | `d_div` |

So the usable interior of an `N x M` bin is a rounded rectangle of
`42*N - 5.7` by `42*M - 5.7` mm (36.3 mm for a 1x1, matching the
commonly-quoted ~36 mm usable width) with corner radius
`max(0, 3.75 - 2.6) = 1.15` mm.

These become named constants in the layout module rather than being
re-derived at each call site, and the interior inset is a parameter — a
lipless bin gives back 2 x 1.65 mm, which matters at 1x1.

## Decisions

### D1: Discrete orientations, continuous translation

Each part has 2 continuous DOF (x, y) and one discrete DOF: orientation
in {0°, 90°, 180°, 270°} relative to its PCA-aligned frame (`PCABox`,
[contour_extraction.py:7](../pipeline/contour_extraction.py#L7)), which
already levels each shape along its principal axis.

The consequence is worth stating plainly: **there is no torque in the
force model.** Orientation is not something the physics relaxes; it is a
discrete variable the outer search chooses and the restart loop
perturbs. This removes rotational inertia, angular damping, and the
need to re-derive a part's field at arbitrary angles — a 90° rotation of
a raster field is an axis swap and flip, exact and free, whereas an
arbitrary angle needs interpolation and reintroduces sampling error on
every step.

Rejected: continuous rotation. It buys a few percent density on
diagonal-friendly shapes and costs a materially more complex solver.
Rejected for now: mirroring. Flipping a part means the tool sits upside
down in its pocket, which is fine for a wrench and wrong for a
screwdriver with a printed logo; it becomes a per-part opt-in flag
(`allow_flip`, default off) if a real layout needs it.

### D2: Signed distance fields for collision, not polygon booleans

Each part gets a signed distance field (SDF) rasterized once in its own
local frame, negative inside:

```
sdf = cv2.distanceTransform(~mask) - cv2.distanceTransform(mask)
```

at a configurable resolution (default 0.25 mm/px — a 200 mm part is then
an 800 px field, well under a megabyte at float32). The gradient is
precomputed alongside it.

Overlap between parts `i` and `j` is measured by sampling points along
`∂i`, transforming them into `j`'s frame, and reading `sdf_j`. Any sample
with `sdf_j < clearance` is a violation, with penetration depth
`clearance - sdf_j` and a push direction of `+∇sdf_j` (up the distance
gradient, i.e. out of `j`).

Why this and not the alternatives:

- **vs. no-fit polygons (NFP).** NFP is the textbook nesting primitive
  and gives exact, robust contact geometry, but computing NFPs for
  non-convex polygon pairs is notoriously fiddly (degenerate touching
  cases, holes in the NFP) and must be redone per orientation pair. With
  4 orientations and `n` parts that is `16 * n^2 / 2` NFPs.
- **vs. Shapely intersection area.** Exact and easy, but the *area* of
  overlap is a poor descent signal — its gradient vanishes for a
  vertex-on-edge touch and it says nothing about which way to move to
  separate deeply-overlapping parts. Penetration depth from an SDF is a
  direct, well-conditioned direction.
- **vs. convex decomposition + GJK/EPA.** Fast and exact, but the
  decomposition of a 40-vertex simplified contour is another dependency
  and another source of edge cases.

The SDF is approximate — it discretizes at the raster resolution — which
is why the clearance parameter is tuned to swallow it (see D5). It also
handles concavity for free, which is the property that makes nesting
efficient at all.

Sample points are the polygon's vertices *plus* points resampled along
each edge at ~1 raster cell spacing, so a long straight edge cannot slide
through a thin feature unnoticed.

Pairwise checks are evaluated in both directions (`∂i` against `sdf_j`
and `∂j` against `sdf_i`) and the resulting force applied equal and
opposite. Besides being physically right, this catches the degenerate
case where one part fully contains another with no boundary samples in
range.

### D3: Energy formulation — repulsion is the whole objective

```
E = Σ_{i<j} w_pair * Σ_{p ∈ ∂i,∂j} max(0, c_pair - d)^2
  + Σ_i     w_wall * Σ_{p ∈ ∂i}     max(0, c_wall - d_container(p))^2
```

Parts repel each other; the container's interior boundary repels
everything outward-crossing (equivalently: the container SDF is positive
inside, and any sample below `c_wall` gets pushed back in). No other
terms.

Two properties fall out of this:

- `E = 0` **is exactly the feasibility predicate.** A layout is
  acceptable iff the energy reaches zero, so the solver's stopping
  condition and the packer's success condition are the same number. No
  separate collision re-check pass, no threshold to tune.
- **No compaction term is needed.** Within a fixed `N x M` bin every
  feasible layout is *acceptable* — the bin is already the size it is.
  Compaction happens in the outer loop by *shrinking the container*
  (trying a smaller grid) rather than by adding an attractive force.

This is the piece most likely to get "improved" into an attraction-to-
center term. Resist it: it fights the wall force, adds a weight to tune,
and buys nothing the grid-size search doesn't already buy.

**Corrected.** This section originally went on to claim that "repulsion
alone spreads parts evenly, which is what we want anyway". It does not,
and the reason is the first property above rather than any defect in the
forces: the energy is zero the instant nothing is too close, so the
descent stops at the *first* feasible arrangement and has no reason to
prefer one over another. Measured on the three spoons, the slack over the
clearances ran 0.00, 0.02, 0.44, 2.30, 4.79mm — one contact exactly at
the limit while another had nearly 5mm spare. Feasible layouts are
interchangeable to the solver but plainly not to a printed bin, which is
what D8 addresses.

### D8: Even spacing — a spring network at an inflated rest length

Once a layout is feasible, a second pass evens out the gaps
([spacing.py](../pipeline/layout/spacing.py)). Every pair of parts near
enough to sense each other, and every part near a wall, gets a quadratic
spring pulling toward one shared rest length, `clearance + spacing_margin`.
Where they cannot all reach it — which in a bin worth packing is
everywhere — equal springs balance at equal compression, so the gaps come
out even without anything measuring evenness.

It is the existing energy at wider clearances, not a second force model:
`LayoutParameters` already accepts explicit clearance overrides, so the
pass builds an inflated copy and reuses `ComputeEnergy` unchanged. That
matters for trust — there is no separate gradient to keep consistent with
the first, and the finite-difference tests already cover it.

Three things this deliberately is not:

- **Not variance minimization.** "Make the gaps the same" also reads as
  "minimize their spread", and that version is wrong: it would drag a
  roomy gap *down* toward a cramped one. A shared rest length only ever
  pushes apart. (Individual gaps can still close as a part is pushed off
  something else — the guarantee is about the springs, not every gap.)
- **Not equal spacing between everything.** Only parts within the
  distance field's reach interact. That is not a threshold anyone picked:
  past `pad` a part cannot sense another at all, so the field's range
  *is* the definition of "neighbour". Spoons at opposite ends of a bin
  are not neighbours and must not be equalized against each other.
- **Not a risk to feasibility.** The springs run at clearances that
  strictly contain the true ones, and every candidate is re-checked
  against the true clearances before being kept, falling back to the
  arrangement passed in. The pass can only improve a layout or leave it
  alone.

`spacing_margin` has a real optimum rather than "as far as possible".
From one fixed starting layout on the spoons, the spread of the gaps came
out 3.45 / 2.74 / **1.98** / 2.04 / 2.92 / 3.81mm at margins of 1.0 / 1.5
/ **2.0** / 2.5 / 3.0 / 4.0. A rest length well beyond the slack the bin
actually has just compresses everything against the hard limits instead
of letting the springs balance — at 4.0 two contacts get squeezed to
0.5mm that 2.0 leaves at 1.6mm. It also sets `pad`, so it costs raster
memory on every part. Note this is tuned on one fixture; the shape of the
curve is the durable finding, not the exact figure.

End to end on the spoons, the slack over the clearances went from
0.00/0.02/0.44/2.30/4.79 to 0.03/0.88/0.99/1.87/2.95 — the spread nearly
halved, at no measurable cost to the pack (3.02s against 3.14s without),
because the pass stops on the same patience rule the main relaxation uses.
The one contact still at 0.03mm is the big spoon's length against the bin:
200.26mm of spoon in a 204.3mm interior leaves about 0.05mm total, so it
is geometry, not the solver.

#### The under-filled bin, and why the springs cannot reach it

The springs go slack the moment nothing is within reach of anything else,
so a bin with room to spare is not spaced at all: every pair reads
`DISTANT_MM` and the parts stay where bottom-left fill dropped them.

**Lengthening the springs does not fix this, and the reason is not `pad`.**
`pad` is *derived* — `spacing_pair + 1` — so it is not a lever in its own
right; and the pair force is `max(0, rest − d)²`, which is zero past the
rest length however large the raster is. Growing the bounding box alone
changes nothing at all.

Raising `spacing_margin`, which does grow both, changes nothing either.
`SpringParameters` inflates the *wall* clearance by the same amount, and
at bin scale every part violates all four walls at once; the descent
stalls and the placements come back identical at margins of 21, 42, 80 and
160mm. Inflating only the pair springs does distribute the parts — and
costs the tight bins the entire point of the pass, taking the spoons in
5x2 from a gap spread of 0.04mm to 24.57mm. There is no setting of this
model that serves both regimes, because a repulsion-only energy has
nothing to say about a part that is far from everything.

So the roomy bin is handled geometrically instead, by `Distribute`:
center the arrangement, scale it out about its center, then center it
again. The third step is the load-bearing one. Inflation scales part
*centers* while centering centers the *bounding box*, so scaling undoes
the centering it started from — whichever part sits furthest out reaches
its wall first and stops the scale, leaving the rest with whatever they
happened to get, measured at 7.0mm against 2.2mm with one part jammed in a
corner. It is deliberately not iterated: re-centering frees room on the
pinned side, and another round would simply pin the other one.

The inflation stops at `spacing_wall`, not at the bare clearance. The bare
clearance is the *legal minimum*, so inflating to it spends every
millimetre of slack on the gaps between parts and leaves none at the rim —
five parts in a 4x3 came out with one 2.2mm off the wall against
inter-part gaps of tens of millimetres. `spacing_wall` is what the springs
already drive a wall contact to when they have the room, so stopping there
makes the two passes agree instead of the later one overriding the
earlier. It only ever reduces how far the inflation pushes, so no bin that
fits today stops fitting, and a tight bin — where there is no room to
inflate — is untouched.

### D4: Solver — damped descent with annealed restarts

Within a fixed grid size, one *attempt* is:

1. Initialize positions (see below) and pick an orientation per part.
2. Iterate: accumulate forces, `v = damping * v + step * F`,
   `x += v`, plus a jitter term whose magnitude decays over the
   iteration budget.
3. Stop early on `E < epsilon` (success), or when the budget is spent
   (failure).

The jitter is what makes this more than gradient descent: the energy
landscape is full of local minima where two parts are wedged apart by a
third, and a purely downhill solver parks in them. Decaying noise is
simulated annealing in all but name, and is cheap.

If an attempt fails, restart with a different seed *and a different
orientation assignment* — orientation is discrete, so the restart loop is
the only thing that can explore it. Cheap heuristic for the assignment:
start from all-0°, and on each restart flip a random subset to 90°.

Initialization: place parts in decreasing area order (big parts first is
the standard nesting heuristic; they are the constrained ones) on a
jittered lattice sized to the bin. Deterministic given a seed, which is
a hard requirement — the same contours must produce the same bin twice,
or the printed part won't match the sheet.

Everything is vectorized over sample points with NumPy; the per-pair loop
stays in Python, which is fine for the ~10 parts a bin realistically
holds.

### D5: Clearances derive from print reality, not from geometry

Three separate numbers, all parameters:

- `pocket_offset` (default 1.0 mm) — how much bigger the pocket is than
  the object, so the object actually drops in. This is `solid.py`'s
  existing `offset(r=1)`.
- `c_pair` (default 3.2 mm) — minimum part-to-part spacing, which must
  be at least `2 * pocket_offset + d_div` (1.2 mm) so the divider between
  two pockets is printable.
- `c_wall` (default 2.0 mm) — minimum part-to-wall spacing, at least
  `pocket_offset + d_wall`.

Deriving `c_pair` and `c_wall` from `pocket_offset` by default (rather
than leaving three independent knobs) keeps them consistent; explicit
overrides stay available. The defaults also leave headroom over the SDF's
0.25 mm discretization error, so raster approximation never causes a real
collision.

Layout operates on the *raw* contours and enforces spacing between them;
the pocket offset is applied downstream at solid-generation time. Keeping
the offset out of the layout geometry means changing print tolerance does
not invalidate a layout.

### D6: Grid size search — smallest area first

Enumerate candidate `(N, M)` in increasing order of `N * M`, breaking
ties toward square-ish, and return the first that yields `E = 0`.

Pruned by two lower bounds, both cheap and both sound:

- **Area:** the parts' clearance bands, counted in, must fit in the
  interior — each part's area dilated by `c_pair / 2`, summed, against the
  interior's area. Two parts `c_pair` apart have exactly touching
  dilations, so the dilations of a feasible layout are disjoint and
  nesting never beats this. Dilated area is measured off each part's own
  distance field (`sdf <= r`) rather than from a perimeter formula, which
  would overcount a concave shape whose dilation folds into itself and
  could then reject a size that actually fits.
- **Extent:** the largest part's oriented bounding box must fit inside
  the interior in at least one orientation.

**No slack term on the extent bound.** An earlier draft required a raster
cell of it, reasoning that a part clearing its run by less than the
discretization error could never really be placed. That is wrong, and the
fixtures disprove it: `big_spoon` has a 0.135 mm window in a 5-cell run
and the solver seats it there reliably, with measured wall margins of
2.064 mm and 1.972 mm against a required 1.95 mm.

The distinction is worth holding onto, because it decides where margins
are needed at all. **Part-to-wall distance is analytic** — the container
is a rounded rectangle with a closed-form distance function (D2) — so it
carries no raster error and needs no slack. **Part-to-part distance is
rasterized**, so it does, which is why the solver's contact positions are
offset by two raster cells. Same geometry, two different error regimes.

Cap `N` and `M` at a configurable max (default 6, past which nothing fits
in a normal drawer) and report failure rather than searching forever.
Failure at a given size is probabilistic — the solver may just have been
unlucky — so a size is only rejected after its full restart budget, and
the report distinguishes "provably too small" (bound violated) from
"could not find an arrangement".

### D7: Package, CLI, and GUI stage

Layout is a package rather than a module — it is a subsystem with several
distinct concerns, and one file holding all of them passed 800 lines
before the solver was even written, against a house style of 100-375.

```
pipeline/layout/container.py   the bin interior, from the Gridfinity spec
pipeline/layout/part.py        a contour and its signed distance field
pipeline/layout/placement.py   a part positioned in a bin
pipeline/layout/parameters.py  everything tunable, in one place
pipeline/layout/energy.py      clearance violation and the forces to fix it
pipeline/layout/descent.py     moving parts along those forces
pipeline/layout/solver.py      arranging parts inside a bin of fixed size
pipeline/layout/packer.py      choosing the bin size, with the bounds that
                               reject one without running the solver
pipeline/layout/grouping.py    partitioning parts across several bins
pipeline/layout/drawer.py      which bins share a drawer, in whole cells
pipeline/layout/floorplan.py   a whole drawer at true scale, to print
pipeline/layout/loading.py     getting parts in, from SVGs or from
                               contours you already have
pipeline/layout/spacing.py     evening out the gaps once a layout fits
pipeline/layout/preview.py     drawing a solved layout at true scale
pipeline/layout/solid.py       the printable bin, as OpenSCAD
pipeline/layout/render.py      the same drawing, rasterized for a screen
pipeline/layout/field.py       a part's distance field, false-colored
pipeline/layout/plan.py        the whole stack: parts to bins to drawers
pipeline/layout/session.py     saving a floorplan and resuming it
pipeline/layout/verify.py      independent checks, no code shared with above
pipeline/layout/*_test.py      one test module per source module
render_demo.py                 one committed still per drawing path
pipeline/layout_stage.py       Stage subclass, group box, Qt
pipeline/field_stage.py        the same, for the field viewer
pipeline/floorplan_stage.py    the same, for the whole-library floorplan
layout_cli.py                  headless entry point
layout_gui.py                  interactive entry point
field_gui.py                   the field viewer's window
floorplan_gui.py               the whole-library window
```

One test module per source module, strictly — a test lives beside the
module it exercises, not beside the one that happens to call it. Tests for
the container, for a part's field derivative, and for vector rotation had
all accumulated in `energy_test.py` simply because the energy code is what
consumes them; the cost is that changing `container.py` gives no hint that
its tests live somewhere else entirely.

Dependencies run one way: `container` and `part` depend on nothing local,
`placement` on `part`, `parameters` on `container` and `part`, `energy` and
`descent` on those, `loading`, `spacing`, `solver`, `packer`, `preview`,
`render`, `field` and `solid` above that, `grouping` on top of `packer`,
`drawer` on top of that, `floorplan` on top of `drawer`, `preview` and
`render`, `plan` above those, `session` on top of `plan`, and `verify`
deliberately to one side.

`plan` is the top, and it exists because the join between grouping and
the drawer level was the one seam
[architecture.md](architecture.md) called built but *unplumbed*: `Group`
returns a list of layouts, `Assign` places their footprints, and until
M10 nothing but `layout_demo` ran both. Two things live there and nowhere
else. The **feedback edge** — grouping is restricted to footprints some
drawer can actually hold, before it starts, because proposing a bin no
drawer could store wastes the entire stochastic search underneath it. And
**progress**, which at this level is a feature rather than a debug hook:
the search runs for minutes, so "what would I get if I stopped now" has
to be answerable at every moment. Both underlying searches already report
— `grouping.Step` and `drawer.Trial`, at thousands of events a second in
two different vocabularies — and `plan` narrows them to one throttled
`Progress` carrying the best complete answer so far.

That word *complete* is load-bearing. First-fit builds its grouping up one
part at a time, so its early steps describe a few parts in a few bins and
score beautifully on cell count while holding almost nothing; reporting
one of those as "best so far" claimed 1 bin and 2 cells for a four-part
library. That is not a worse answer, it is not an answer, and the filter
against it is the part of the progress channel most likely to be
"simplified" back out.

Those fragments are still *drawn*, though, under a phase of their own
(`FILLING`). They are made of real bins holding real parts, and they exist
within a pack or two of pressing Plan where the first complete grouping is
a minute away — so a window that ignored them would have nothing to show
for the minute it can least afford to look hung. The distinction is kept
where it belongs: the report describes them as a fraction of the library
rather than as an arrangement of it, and `BuildPlan` never hands one back
as a plan, so a search stopped mid-fill returns no floorplan rather than
half of one.

Cancellation rides the same channel, because `Group` takes no `cancelled`
predicate and the observer is the only hook into it. The best grouping
seen is kept as the search runs, so stopping costs the time the search
had left rather than the answer it had found.

#### Resuming a floorplan

A library grows one object at a time, and by the time the second one
arrives half the answer is already printed and sitting in a drawer. So
the question is not "where does everything go" but *"where does this go,
and what do I have to reprint"* — which is what
[session.py](../pipeline/layout/session.py) makes askable.

`BuildPlan` takes an optional `start` grouping. Given one, it opens a bin
for any part the grouping does not account for and hands the whole thing
to `Improve` rather than to `Group`. **The stability that buys is a
property of `Improve`, not of anything in `plan`**: the local search only
ever accepts a move that lowers the total cell count, and a bin no
accepted move touched is carried through as the very same `Layout` — same
grid, same placements, to the millimetre. Measured on a four-object
library, adding a fifth left one bin untouched and changed one. Starting
over would have found an equally good answer sharing nothing with the
bins already on the shelf.

Three things follow for the file format. It stores the **contours
themselves, not paths** — part ids come from the order `ReadContours`
encounters files, so a grouping referring to source files would silently
mean something else after a rename. It stores the **geometry parameters**,
because placements satisfied the clearances they were solved against and
reopening at a wider pocket offset yields a floorplan that looks settled
and is not — `Verify` runs `CheckLayout` on the way in and says so. And
it stores neither `admissible_grids`, which `BuildPlan` derives from the
drawers, nor the search budget, since how hard the last run looked says
nothing about whether its answer still holds.

#### Getting the bins out

The drawer map says where bins go. What you *make* is a bin, and until
`FloorplanStage.Export` wrote them, nothing in the project produced a
solid for a bin the **grouping** search had chosen — `LayoutStage` and
`layout_cli` both hold exactly one bin, packed directly.

That left one route from a library plan to something printable: read the
plan's report, load those objects into `layout_gui.py`, and pack them
again. It does not work. `Pack` runs its own grid search and its own
stochastic solve, so the same objects come back in a different
arrangement — and the bin coming off the printer would not match the
floorplan printed to lay it out with. The failure is silent and only
shows up when the tools do not sit in their pockets.

So the export writes them: `NAME.pdf` for the map, then `NAME_bin0.svg`,
`NAME_bin0.pdf` and `NAME_bin0.scad` per bin, cut from the layouts the
plan actually chose. Bin numbers are zero-padded to a fixed width so a
directory of a dozen sorts the way the report lists them.

`WriteScad` can refuse a layout whose dividers are thinner than the
minimum at the current pocket offset. That is a property of one bin, so
it is collected per bin and reported rather than raised — the sheets are
already written and the other bins are finished, and losing an eleven-bin
export over the twelfth would be the wrong trade.

**A plan can arrive here with no drawer assignment.** Stop the search
after the grouping lands but before the drawers are searched and
`BuildPlan` returns bins and no assignment — correctly, since a grouping
still being improved says nothing about drawers. Save from that state and
the session records none either, so reopening it gave a floorplan that
could be looked at but not exported: *"nothing planned to export"* about
five perfectly good bins. The fix is to run the drawer search rather than
refuse, which is obvious once the two halves are priced — grouping is
stochastic and takes minutes, while `Assign` is exact and took 0.1ms on
that same five-bin, six-drawer floorplan. The session saved the expensive
half; the cheap half is not worth a person's time to redo. It is filled in
at `Load`, so a reopened floorplan is a whole one however it was saved.

#### Pinning a bin

`start` is a *hint*. It gives the local search a good place to begin and
the search is free to take it apart, which it usually does — that is the
point of a search. A **pin** is the other thing: `BuildPlan(pinned=[...])`
holds those bins out of the grouping search altogether, so the parts
inside them never reach it and the bins come back in the answer as the
same `Layout` objects that went in. There is no heuristic involved and
nothing to tune; the guarantee is structural.

That covers two cases the cell count cannot see. A bin already printed and
sitting in a drawer, where no cheaper arrangement is worth the reprint.
And a grouping that is right for reasons outside the model — objects used
together, or one bin that has to stay shallow.

It is also how a big library stays workable. Measured on a five-object
library at the test parameters, pinning all but one bin cut the grouping
search from 35 reported steps and 4.4s to 8 steps and 1.7s, reaching the
same 2 bins / 6 cells. The saving scales with what you pin, which is why
adding one tool to a settled library ought to be a search over one object.

Two boundaries are worth naming. Pinned bins are laid down **first**, so
they hold ids `0..k-1` and a pin does not change bin number every time you
re-plan — otherwise the checklist in the window would tick a different bin
each run. And a pin holds a bin *together*, not *still*: `Assign` still
chooses which drawer cell it goes in, because sliding a bin along a shelf
costs nothing and refusing to would turn a pin into an obstacle. Pins are
saved with the session, since a pin is a claim about the physical world
and re-ticking a dozen boxes on every open is how one gets lost.

`Changes` compares two groupings by contents *and* placements. Same parts
in different positions is a different bin as far as a printed pocket goes,
and reporting one of those as unchanged is the single error it must not
make.

`field` is the odd one out in that it consumes nothing and is consumed by
nothing — it exists to be *looked at*. Both phases of the search read a
part's distance field and neither draws it, so when a pack behaves oddly
there was no way to ask whether the field was what it looked like. It
draws two things: the distance itself, contoured, with the clearance
levels that mean something on a part's own field marked (`c_pair_enforced`
and `spacing_pair`, but not `c_wall` — that one is measured against the
container's analytic depth and is not a level of any part's field); and
the length of the field's gradient, which is where the medial axis shows
up. That second view is the one worth having: D3's forces stop being
trustworthy exactly on those creases, and the deviation is two-sided —
opposing walls leave the interpolated gradient near zero, perpendicular
ones near `sqrt(2)`, and shading by raw magnitude would hide half of it.

`parameters` is separate from `energy` because six of the nine modules
that need a `LayoutParameters` compute no energy at all — the loader sizes
distance fields from it, the packer reads the grid limit, the solid
generator takes the pocket offset, the GUI edits it. It had grown to 40%
of `energy.py` while being the configuration of the whole subsystem.

`descent` is separate for the opposite reason: two passes descend an
energy — the solver's relaxation and the spacing pass — and both had their
own copy of the same step. The step is where the tuning lives (per-sample
normalization, damping, and the cap that keeps a move short enough for the
gradient to stay trustworthy), so two copies meant retuning one and
silently leaving the other. The passes still differ in what they descend
and when they stop; only the integration is shared.

Nothing in the package imports Qt, so all of it is unit-testable without
a display.

The stage stays outside the package, mirroring the existing pattern
([contour_extraction.py](../pipeline/contour_extraction.py) vs.
[contour_selection_stage.py](../pipeline/contour_selection_stage.py)): a
thin adapter that owns a `LayoutParameters` and builds its group box via
`CreateGroupBox`, keeping the geometry free of the GUI.

The CLI reads contours from a file and writes a layout preview, so the
packer can be iterated on without launching the GUI or re-running SAM2 —
important, because tuning a stochastic solver through a Qt event loop is
miserable. Its inputs are either an SVG this project wrote or a JSON
contour dump ([contour_io.py](../pipeline/contour_io.py)); the SVG export
stage writes the dump alongside its other outputs, since the SVG itself
is a *picture* of the contours — PCA-aligned per shape and rounded for
drawing — rather than the contours.

The preview is drawn on the bin's **outer footprint**, not its interior,
so the printed sheet can be checked against a real bin rim-to-rim. Only
part outlines are written as `<polygon>`; the rim, interior, and cell
grid are `<polyline>`. Since `LoadSvgContours` reads only `<polygon>`, a
preview reads back as exactly the parts in it, with no annotation to
filter out. Nothing in the drawing path flips a coordinate: the layout
frame is whatever frame the contours arrived in, and a flip would mirror
every part relative to an export already checked against real objects.
A mirrored outline is the one error a printed template cannot survive —
it measures correctly and still will not fit, because a reflected tool
sits upside down in its pocket (D1).

The stage is hosted by its **own window**
([layout_gui.py](../layout_gui.py)), not registered into `SVGGui`. M6
originally planned the latter, and building it showed why it does not
work: `SVGGui` captures one photo — one segmentation, one calibration,
one set of clicks — while packing needs many objects, which in practice
arrive from many sessions. The three spoon fixtures in `test_data/` are
three separate captures. A Pack button inside the capture window could
only ever pack what happened to be in the current frame. A second symptom
pointed the same way: a layout lives in bin millimeters, so it cannot
overlay the photo — it has to replace the image view entirely.

`LayoutGui` therefore starts where capture leaves off. It loads the dumps
and SVGs those sessions wrote, *accumulating* across files, and packs the
collection. The two may eventually meet under a common entry point, but
that workflow is not understood well enough yet to design for.

Either way the stage runs only when explicitly triggered by its Pack
button, never on a parameter edit — packing takes seconds, and the panel's
controls are exactly the kind a person drags.

## Grouping

Phase 2 turns "pack this set" into "partition these parts into bins"
([grouping.py](../pipeline/layout/grouping.py)). The arrangement packer is
the feasibility oracle; the search around it:

1. Sort parts by area, descending.
2. First-fit-decreasing into open bins, where "fits" = the packer
   succeeds at the bin's current size.
3. Improve with local search: move or swap parts between bins, keeping
   changes that reduce total cells.

Two things make this affordable: the area lower bound rejects most
candidate moves without ever running the solver, and results are cached
by (frozen set of part ids, grid size).

**Stage 2 deliberately never grows a bin**, which the earlier draft of
this section left open ("a bin may grow up to a user-set maximum
footprint"). A part that fits only once its bin gets larger is not a fit,
it is a *trade* — this bin costs more, in exchange for one fewer bin
elsewhere — and pricing that trade is exactly what stage 3 does. Letting
first-fit take it greedily, in whatever order the parts happen to arrive,
would commit to it without ever comparing it against the alternative.
Growth still happens; it happens where it can be compared.

**The lower bound must be one-sided, and that is the property worth
testing** — the same lesson M4 recorded about the packer's own bounds, now
load-bearing in a second place. The search discards a candidate move when
even the bound cannot beat what the move would replace. A bound that ever
came out *above* the true cost would silently discard improvements, and
the grouping would come back larger looking perfectly correct.

The local search takes the **first** improvement rather than the best:
candidates are cheap to generate and expensive to price, so taking the
first that helps and re-generating beats pricing them all. It needs no
iteration cap, because every accepted change strictly lowers a
non-negative integer.

Measured on the three `test_data/` spoons: one bin each costs 22 cells
(5x2 + 5x2 + 2x1), and grouping returns a single 5x2 at **10 cells**. The
local search reaches that on its own — started from one bin per spoon,
without first-fit's help, it still collapses to one bin — which is what
distinguishes a search that works from one that never fires.

### Semantic coherence (much later)

Minimizing cells alone produces bins that are geometrically tight and
semantically nonsense: a bin holding a spoon, a hammer, and a camera lens
because they happened to tessellate. A good shadow box groups *like*
things. Given an embedding vector per object, we want a bin of
assorted spoons to score better than that grab bag.

**Measure.** Von Neumann entropy of the bin's normalized similarity Gram
matrix: build `G_ij = cos(x_i, x_j)` over the bin's embeddings, set
`ρ = G / n` (trace 1, since `G_ii = 1`), and take
`S = -Σ λ_k log λ_k` over `ρ`'s eigenvalues. This is `log` of the Vendi
score (Friedman & Dieng, 2023). Verified behavior:

| Bin contents | S (nats) |
| --- | --- |
| 5 identical items | 0.000 |
| 4 assorted spoons | 0.347 |
| spoon + hammer + lens | 1.099 = `log 3` |
| n mutually dissimilar items | `log n` (max) |

which is exactly the requested ordering.

Rejected: **differential entropy of a Gaussian fit**
(`½ log det 2πe Σ`), the obvious first thing to reach for. A bin holds
~10 objects and embeddings are hundreds of dimensions, so the covariance
is always rank-deficient — on 10 samples in 512-d the log-determinant
comes out at −17045 and heads to −∞ with more dimensions. It needs
shrinkage (`Σ + λI`) and the answer then depends mostly on `λ`. The von
Neumann form needs no density model and is well-defined for any `n < d`.
Also rejected: mean pairwise distance (not an entropy, no diminishing
return for near-duplicates) and cluster-assignment Shannon entropy
(genuine entropy, but requires a global clustering step and inherits its
`k`).

**Three consequences that constrain the search:**

1. **Entropy is a regularizer, not an objective.** Splitting every object
   into its own bin drives total entropy to exactly 0 — a singleton bin
   is always perfectly coherent. Entropy on its own therefore prefers
   maximal fragmentation, the precise opposite of the goal. It only ever
   works as a counterweight to cell-count pressure: packing wants to cram
   things together, entropy resists cramming *unlike* things together,
   and the two are in tension by construction.
2. **The exchange rate is the real knob.** Scalarize as
   `Σ_b (cells_b + λ · S_b)`, where `λ` has units of cells-per-nat and
   answers "how many extra grid cells will I spend to keep the hammer out
   of the spoon bin?". Start with `λ → 0+`, i.e. entropy purely as a
   tiebreak among equal-cell-count partitions — there are usually many,
   and this gets most of the benefit with no density regression and no
   parameter to defend. Promote to a weighted term only if tiebreaking
   proves too weak.
3. **Entropy is not monotone under insertion, so it cannot prune.** The
   area bound works because adding a part can only increase required
   area, letting the search reject moves without running the solver.
   Entropy has no such property: two dissimilar items score 0.693, and
   adding a *third* item that duplicates the first *drops* it to 0.637.
   Any move that adds a part to a bin can raise or lower the entropy
   term, so the branch-and-bound structure of the M8 search does not
   extend to it. The area bound still prunes on the geometry half; the
   entropy half must be evaluated, not bounded.

**Embeddings stay out of scope.** The layout module accepts
`embeddings: dict[int, np.ndarray]` and never learns where they came
from. Candidate providers, in rough order of preference: CLIP image
embeddings of the masked crop (genuinely semantic; `transformers` is
already a dependency, so it costs a weights download, not a new package);
SAM2 encoder features pooled over the existing mask (free — the model is
already loaded and the mask already exists — but trained for
segmentation, so more geometric than semantic); plain shape descriptors
off the contour we already have (zero new dependencies, and arguably
"different size spoons" is partly a shape claim anyway). Deciding among
them is an experiment to run once the harness exists, not a decision to
make now.

Two smaller notes. The kernel has a free bandwidth — raw cosine, or
`exp(-d/σ)` — that sets how fast things count as different; it interacts
with `λ` and both should be tuned together or not at all. And semantic
grouping is exactly what a user has strong opinions about, so
explicitly pinned groups ("all my screwdrivers together") must override
the term rather than be scored against it.

## Testing

The solver is stochastic, so tests assert invariants and seeded
determinism rather than exact coordinates:

- **Geometry units.** SDF sign and magnitude at known points of a known
  polygon; 90° rotation of a field equals the field of the rotated
  polygon.
- **Feasibility.** For a synthetic set (rectangles with known optimal
  packing), the packer finds the known-optimal grid size.
- **No overlap.** For randomized part sets, every successful layout is
  re-checked with an independent, exact polygon test — not the SDF the
  solver itself used. This is the important one: it catches the raster
  approximation lying. `matplotlib.path.Path.intersects_path(..., filled=True)` serves, and matplotlib is already a dependency, so this
  costs nothing; verified to handle both full containment and the
  non-convex case that matters (a bar nested in a U's notch reads as
  *not* overlapping). It reports exact edge contact as overlapping, which
  is harmless given `c_pair` is millimeters.
- **Determinism.** Same input + same seed = byte-identical layout.
- **Bounds.** Every placed part is inside the interior envelope with
  `c_wall` to spare.
- **Regression.** The `test_data/` fixtures, packed, with the resulting
  cell count asserted — flags a density regression from a solver change.

Slow randomized sweeps get the existing `slow` pytest marker.

### Fixtures

`test_data/` holds three real exported spoon contours. Measured:

| Fixture | Bbox (mm) | Verts | Area (mm²) | Solo bin | Long-axis slack |
| --- | --- | --- | --- | --- | --- |
| `big_spoon.svg` | 200.26 x 41.67 | 40 | 3414 | 5x2 | **0.14 mm** |
| `medium_spoon.svg` | 162.76 x 34.89 | 39 | 2356 | 5x2 | 37.67 mm |
| `small_spoon.svg` | 73.93 x 14.20 | 42 | 437 | 2x1 | **0.49 mm** |

(Slack against `c_wall` = 1.95 mm, the value derived from a 1 mm pocket
offset. An earlier draft of this table assumed 2.0 mm.)

They are a better test set than they look, for three reasons.

**They sit on cell boundaries.** `big_spoon` at 200.26 mm needs 204.16 mm
with `c_wall` on both ends, against a 5-cell interior of 204.30 mm — a
0.14 mm window, well under the raster resolution. The solver seats it
there reliably, which is the evidence behind D6 carrying *no* slack term:
the wall constraint is analytic, so a sub-resolution window is real space,
not measurement noise.

To be clear about what does *not* rescue it: rotation cannot. The x-extent
of a rotated `L x W` box is `L cos θ + W sin θ`, whose derivative at
θ = 0 is `W > 0` — so any rotation off-axis makes the long-axis fit
*worse*, not better. This is a point in favor of D1's rejection of
continuous rotation rather than against it.

**Grouping has real headroom here.** One-per-bin costs 22 cells
(10 + 10 + 2 — note `big_spoon` is 41.67 mm wide, over a 1-cell
interior's 36.3 mm, so it needs two rows). All three share a 5x2 at 39%
fill, and a 5x1 is not obviously out of reach at 84% fill. So M8 should
turn 22 cells into 10, or 5 if nesting goes well — a concrete target
rather than a vague "fewer".

**They are the entropy example.** Three different-size spoons is
literally the case
[Semantic coherence](#semantic-coherence-much-later) is specified
against, so M10's measure can be validated on real embeddings of real
objects from day one.

One trap. These files are 1:1 mm (`viewBox` units == mm), while the
current writer pre-scales by 96/25.4 = 3.7795
([svg_writer.py:42](../pipeline/svg_writer.py#L42)) — they predate
`d3c08a3`. **The loader must derive the scale as
`viewBox_width / width_mm`, not hardcode either constant**, or one of the
two formats silently comes in 3.78x wrong.

## Open questions

- **Finger access.** A pocket that exactly fits a flat tool is hard to
  get the tool *out* of. Real shadow boxes add a scoop or a thumb relief.
  Is that a layout concern (reserve space next to each part, which
  changes packing) or a solid-generation concern (subtract a scoop from
  the pocket afterward)? Leaning toward the latter, but it affects
  `c_pair`.
- **Orientation preference.** Should a user be able to pin a part's
  orientation (e.g. "all screwdrivers point the same way") for a layout
  that reads well, at a density cost? Cheap to add as a constraint on the
  restart loop.
- **Part variants.** Multiple copies of the same tool are common. Worth a
  multiplicity count rather than N duplicate contours?
- **Label space.** Gridfinity bins often carry a label tab; if the layout
  must reserve room for one, the usable interior shrinks on one edge.
