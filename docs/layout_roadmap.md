# Layout Optimization Roadmap

Implementation plan for [layout.md](layout.md). Ordered so that each
milestone is independently testable and leaves the tree green
(`make check`).

The ordering principle: build the *verifier* before the solver, and the
headless path before the GUI. A stochastic optimizer whose correctness
check shares code with the optimizer itself is untrustworthy, and one
that can only be run through a Qt event loop is untunable.

## Working agreement

Work stops at the end of each milestone, once it is functional and its
tests pass — not partway through, and not rolling on into the next one.
Each milestone should leave `make check` green and stand as one
reviewable commit, made by hand after inspection. The "Done when" clause
on each milestone is the completion criterion; meeting it is the signal
to stop.

No new dependencies were needed through M8: SDFs use `opencv-python` and
`numpy`, the test oracle uses `matplotlib`, and the SVG loader uses the
standard library — all already in `requirements.in`. Only M9's embedding
provider might add one, and only if CLIP beats the alternatives. If a
milestone seems to need a new package, that is a signal to re-read the
design rather than to add it.

## M1 — Geometry primitives

`pipeline/layout/`, no solver yet.

- [x] Gridfinity constants: pitch, interior inset, corner radius, wall
  and divider minimums, sourced from
  [standard.scad](../gridfinity-rebuilt-openscad/src/core/standard.scad)
  and cited in comments.
- [x] `BuildContainer(n, m, inset).Polygon()` → rounded-rect polygon for
  an `N x M` bin's usable interior. (Originally a free `InteriorEnvelope`
  function; removed in M5's cleanup once nothing but its own tests called
  it, since `Container` is what production actually goes through.)
- [x] `Part`: a contour plus its precomputed raster fields — occupancy
  mask, SDF (negative inside), SDF gradient, boundary sample points
  (vertices plus edge resampling at ~1 raster cell), area, local frame
  origin.
- [x] `Placement`: part id, translation, orientation index (0-3), and
  `ToWorld()` returning the placed polygon in bin coordinates.
- [x] Exact, independent overlap predicate for tests, via
  `matplotlib.path.Path.intersects_path(..., filled=True)` plus a
  containment check (matplotlib is already a dependency; verified to
  handle non-convex nesting and full containment). This is the oracle M4
  is validated against, so it must not share code with the solver.
- [x] SVG contour loader for `test_data/`, deriving scale as
  `viewBox_width / width_mm`. **Do not hardcode 96/25.4** — the existing
  fixtures are 1:1 mm and predate that change, so a hardcoded factor
  breaks one format or the other by 3.78x.

**Done when:** SDF sign/magnitude match hand-computed values for a
rectangle and an L-shape; rotating a part 90° gives a field equal to the
field of the rotated polygon; the independent overlap predicate agrees
with hand-checked cases; the three `test_data/` spoons load at their
measured sizes (200.26, 162.76, 73.93 mm long). **Met** — 46 tests across
`pipeline/layout/{container,part,svg,verify}_test.py`.

Two things M1 turned up that later milestones inherit:

- **`cv2.fillPoly` cannot rasterize these masks.** It rounds polygon
  coordinates to whole pixels and fills inclusively, so an edge landing on
  a half-pixel — which a bounding-box corner always does — gains a row or
  column on the high side but not the low side. That asymmetric half pixel
  is invisible in the mask and surfaced as a distance field disagreeing
  with exact geometry by 0.42 mm at one corner and 0.05 mm at the
  opposite one. `_RasterizePolygon` does a crossing-number test at pixel
  centers instead; error is now 0.05 mm everywhere, uniformly.
- **PCA alignment is canonical only up to a 180° rotation.** Handedness
  correction fixes reflection (D1 forbids mirroring) but cannot see a
  double sign flip, so the same object photographed at two angles produced
  two Parts differing by half a turn. Resolved with a third-moment skew
  tiebreak, which is stable for asymmetric objects and irrelevant for
  symmetric ones. Worth knowing at M8, where caching keyed on a part's
  identity assumes a part is a function of the object and not the photo.

## M2 — Energy and forces

- [x] `LayoutParameters` dataclass: `pocket_offset`, `c_pair`, `c_wall`
  (both derived from `pocket_offset` by default), `raster_resolution`,
  iteration/restart budgets, `seed`, `max_grid`.
- [x] Pair term: sample `∂i` against `sdf_j` and `∂j` against `sdf_i`,
  quadratic in penetration depth, forces applied equal and opposite.
- [x] Wall term: sample every part against the container field.
- [x] `Energy(placements)` returning total energy and per-part force,
  vectorized over sample points.

**Done when:** two identical squares at increasing separation give
energy that is positive-and-decreasing, then exactly zero past `c_pair`;
force directions point along the separating axis; a part straddling the
wall is pushed inward. Finite-difference check: numerical gradient of
`Energy` matches the returned forces. **Met** — 39 tests in
`pipeline/layout/energy_test.py`.

Notes for later milestones:

- **The container is analytic, not rasterized.** A rounded rectangle has
  a closed-form distance function, so it costs no memory, has no
  resolution to tune, and stays meaningful arbitrarily far outside the
  bin — which matters because a part that escapes during relaxation has
  to be pulled back from wherever it went, and a raster would simply end.
- **Forces come from the derivative of the interpolant**, not from a
  separately smoothed gradient raster. M1 stored a normalized `gradient`
  field; M2 dropped it, because a force that is not exactly the gradient
  of the energy it claims to minimize can push uphill near a crease and
  stall the solver. Both now come from the same bilinear interpolant, the
  finite-difference tests hold them together, and each part is ~1.3MB
  lighter.
- **Deep overlap reverses the push — this constrains M3.** See below.

## M3 — Solver

**Read this first: the forces are only trustworthy for shallow overlap.**
A distance field points toward the nearest way out, so once a sample
penetrates past the other part's medial axis, the nearest exit is out the
*far* side and the force flips from separating to attracting. Measured on
two 10mm squares: correct to ~50% overlap, reversed beyond it, and total
energy *falls* toward full overlap — so coincident parts are a spurious
minimum that a descent solver can settle into and report as converged.
This is inherent to penalty methods on distance fields, not a bug to fix
in M2. Clamping the trusted depth band was measured and rejected: it
repairs the mid-range but not near-coincidence, where the failure is
symmetry rather than depth, and it costs a parameter plus an energy that
no longer matches D3. The mitigation is the solver's, hence the first two
boxes below.

- [x] **Constructive initialization, not a jittered lattice.** Place parts
  largest-first, each at a position where it does not overlap anything
  already placed. A lattice can drop two parts on top of each other, which
  starts the descent inside the reversed regime — it has to climb *out* of
  a trap before it can begin working. Starting non-overlapping leaves
  relaxation with only the job it is actually good at: resolving shallow
  clearance violations.
- [x] Abort on `EnergyResult.engulfed` (`containment >= 1.0`), restarting
  rather than descending. Containment is chosen over penetration depth
  deliberately: it is a *fraction*, so 1.0 means swallowed at any scale,
  with no size-dependent threshold to tune. Crossing gets no such
  treatment — overlapping parts are the ordinary midway state of a
  relaxation and resolve themselves.
- [x] Damped descent loop with decaying jitter; early exit on `E = 0`.
- [x] Restart loop, re-seeding positions *and* the orientation
  assignment (orientation is discrete — only restarts explore it).
- [x] `SolveFixedGrid(parts, n, m, params)` → `Layout | None`.

**Done when:** three rectangles with a known-tight packing into a 1x3 are
placed without overlap, repeatably; the same seed reproduces a
byte-identical layout; a deliberately over-full bin fails cleanly rather
than returning an overlapping layout; and a run started from deliberately
stacked parts either separates them or reports failure — it must never
return a layout with parts on top of each other. **Met** — 29 tests in
`pipeline/layout/solver_test.py`.

Three things M3 turned up, all of which M4 inherits:

- **Random placement cannot initialize a dense bin — it must be
  bottom-left fill.** "Bounded retries at random positions" was the
  original plan and it does not work: on four 50x30 parts in a 3x2, which
  packs by hand as an obvious 2x2, random init succeeded **0 times in 30
  attempts**, and raising the candidate budget sixteen-fold changed
  nothing. Once parts fill most of the interior the feasible region is a
  vanishing fraction of it. Sweeping *contact positions* — flush against a
  wall or against an already-placed part, `O(n^2)` of them — took that
  case from 3/6 seeds in 4.96s to **6/6 in 0.02s**. A random tail is kept
  after the contact sweep, because contacts come from bounding boxes and
  cannot express tucking one spoon's bowl into another's handle.
- **Contacts must clear the raster's own error.** A part placed at
  *exactly* `c_pair` measures as violating it: the field is rasterized, so
  separation reads short by the ~0.05mm discretization error, and a
  hand-built perfect packing prices at positive energy rather than zero.
  Contacts are offset by two raster cells. Without that, every contact
  looked infeasible and the sweep silently degraded into the random search
  it was meant to replace — the bug hid as "BLF didn't help".
- **An easy bin is now seed-independent.** Bottom-left fill is
  deterministic and the first attempt uses it unchanged, so the seed only
  matters once that attempt fails. Reproducibility for everyday layouts
  comes for free; the stochastic restarts only engage on hard ones.

Tuned against measurement rather than guessed: `patience` 25 (matches 50's
solve rate everywhere, ~25% faster — restarting beats grinding), and
orientations that cannot fit the bin are filtered before an attempt starts
(the spoons in a 5x1 now fail in 0.00s instead of 13s, correctly, since
the big spoon is 41.67mm across a 36.3mm interior).

## M4 — Grid size search

**Correction carried in from M3: the extent bound needs no slack term.**
This roadmap previously demanded a raster cell of it, on the reasoning
that a part clearing its run by less than the discretization error could
never actually be placed. M3 disproves that — `big_spoon` has a 0.135 mm
window in a 5-cell run and the solver seats it there reliably, measured
wall margins 2.064 mm and 1.972 mm against a required 1.95 mm. Part-to-
*wall* distance is analytic (the container has a closed-form distance
function), so it carries no raster error; only part-to-*part* distance
does, which is why contact positions are offset and the extent bound is
not. Requiring slack here would reject bins that genuinely fit.

- [x] Area lower bound: each part's area dilated by `c_pair / 2`, summed,
  against the interior's area. Measure the dilated area off the part's own
  distance field (`sdf <= r`) — a perimeter formula overcounts a concave
  shape whose dilation folds into itself, which would make the bound
  unsound and reject sizes that fit.
- [x] Extent bound: largest part's oriented bbox must fit in at least one
  orientation. `FittingOrientations` already computes exactly this and
  already short-circuits `SolveFixedGrid`; M4 reuses it.
- [x] Candidate enumeration by increasing `N * M`, square-ish tiebreak,
  capped at `max_grid`. Only `n >= m` is generated: a 2x5 is a 5x2 turned
  a quarter turn and every part can turn too, so enumerating both would
  double the search to rediscover each answer sideways.
- [x] `Pack(parts, params)` → layout plus a report distinguishing
  "provably too small" from "no arrangement found". Resolved by testing
  the bounds in `Pack` *before* dispatching to the solver, so
  `SolveFixedGrid` keeps its `Layout | None` signature and the two
  outcomes are separated by which code path produced them.
- [x] **On failure at a feasible-looking size, step up and return the
  larger bin that did work**, naming the abandoned sizes in
  `PackResult.skipped`.
- [x] Randomized validation sweep, marked `slow`: every successful layout
  re-verified against exact polygon geometry.

**Done when:** synthetic sets with known optimal cell counts pack to that
count; the three `test_data/` spoons pack into a 5x2 (39% fill) and the
report cleanly explains any smaller size it rejected; the sweep finds
zero overlaps across a few hundred random cases.
**This is the gate for everything downstream** — if the sweep is not
clean, the raster resolution or clearance defaults are wrong, and no
amount of GUI work will fix that. **Met** — 23 tests in
`pipeline/layout/packer_test.py`, sweep clean over 120 randomized sets.

**The gate earned its place on its first run.** It failed, on exactly the
class of defect it was written to catch: two parts came out 3.157mm apart
under a 3.200mm clearance. Zero energy was true and the clearance was
still violated, because the solver reads separation off a rasterized field
that comes back short by the discretization error — so "E = 0 means every
clearance is met", relied on since M2, was quietly false by 0.04mm.

The fix is `c_pair_enforced = c_pair + one raster cell`: the energy drives
to the larger value so that what *exact* geometry reports clears the real
one. Costs under a tenth of the clearance it protects. Note the asymmetry
with the wall term, which needs nothing of the sort — the container is
analytic, so only part-to-part distance carries raster error. That is the
same split that governs the extent bound above, and it is now pinned by a
test over 400 randomized arrangements rather than left as reasoning.

Two smaller notes:

- **The bounds must be one-sided, and that is the property worth testing.**
  A bound that wrongly rejects inflates every result it touches and nothing
  downstream notices — the packer just returns a bigger bin and looks
  correct. Hence measuring dilated area off the distance field rather than
  from `area + perimeter*r + pi*r^2`, which overcounts a concave shape
  whose dilation folds into itself.
- **Hand-computed "known optimal" cell counts are easy to get wrong, and
  they fail flatteringly.** The four-part synthetic case was written
  expecting 4 cells; the packer found 3, which is genuinely better. The
  arithmetic is now spelled out per case in the test.

## M5 — Headless CLI

- [x] `layout_cli.py`: read contours (JSON dumps or any SVG this project
  wrote), pack, write a preview. Flags override `LayoutParameters` only
  where actually passed, so the tuned defaults are not restated where
  they would drift.
- [x] Contour serialization helpers shared with the GUI
  ([contour_io.py](../pipeline/contour_io.py)), so a session's contours
  can be dumped once and iterated on offline. `SvgExportStage` writes
  `<filename>.json` alongside the SVG and PDF — without a producer the
  format has no source, and the SVG cannot serve as one (it is
  per-shape PCA-aligned and rounded to four decimals for drawing).
- [x] **Refactor the writers first.** Split each into an align-then-write
  wrapper over a write-these-coordinates core (`WriteShapesSvg`,
  `WriteShapesPdf`, taking a `Shape` carrying geometry plus stroke).
  Existing tests unchanged and passing.
- [x] Preview ([preview.py](../pipeline/layout/preview.py)) on top of that
  core, as both SVG and PDF: one polygon per placed part, plus the rim,
  interior outline, and cell grid as polylines. Drawn on the bin's outer
  footprint rather than its interior, so the sheet can be laid under a
  real bin and checked rim-to-rim.

**Added in M6:** the same progress and cancellation hooks the window uses
also drive the CLI. A live one-line display when stdout is a terminal
(suppressed in a pipe, where a self-rewriting line is just noise, and by
`--quiet`), and the first Ctrl-C stops the search and prints what it had
already ruled out instead of a traceback — exiting 130, so a wrapping
script can tell "you stopped it" from "it did not fit".

**Corrected here:** the PDF writer went through `pyplot`, which selects a
global *interactive* backend on import — every test module had been
compensating with its own `matplotlib.use("Agg")`, which is exactly why
the suite passed while a real headless CLI run aborted trying to open a
Qt display. Now built on `Figure` + an explicit PDF canvas, which needs
no backend and no global state.

**Done when:** a real captured contour set packs from the command line
and the printed preview measures correctly against a physical bin.
The three spoons pack to 5x2 in ~4s from the command line; measuring the
print against a physical bin is Andrew's check.

## M6 — GUI stage

- [x] `pipeline/layout_stage.py`: `LayoutStage(Stage)` with a "Layout"
  group box — pocket offset, max grid size, seed, and a "Pack" button.
  Offset rather than two independent clearance boxes: they are derived
  from it (D5), and typing them separately invites a divider too thin to
  print. The derived values are shown read-only beside it.
- [x] Explicit trigger only, like
  [SvgExportStage](../pipeline/svg_export_stage.py) — packing takes
  seconds and must not run on slider drags. Pack is also disabled while a
  pack runs, since the status label pumps the event loop.
- [x] `pipeline/layout/render.py`: rasterize `preview.LayoutShapes` for
  the image view, so the screen and the printed sheet cannot drift apart.
- [x] Report grid size, any failure reason, and whether a smaller size was
  skipped, in the panel.
- [x] Run the pack on a worker thread, reporting progress per restart and
  offering Cancel. See below.

**Changed here — the stage does not go in `SVGGui`.** The plan was to
register it downstream of rectification; building it showed that
`SVGGui` captures *one* photo (one segmentation, one calibration, one set
of clicks) while packing needs many objects, which arrive from many
sessions — the three spoon fixtures are three separate captures. A Pack
button there could only pack the current frame. It lives in its own
window, [layout_gui.py](../layout_gui.py), which loads dumps and SVGs and
accumulates them across files. A common entry point over both may come
later, once that workflow is understood.

**Threading, not event-loop pumping.** The first version ran the pack
inline and called `QApplication.processEvents()` between restarts to keep
the progress label painting. It worked, but it needed two re-entrancy
guards (disable Pack, then disable the whole panel) precisely because
pumping makes every widget live again mid-computation, and the window
still could not repaint or resize. The packer touches no Qt, so it moved
to a `QThread` (`PackWorker`) that reports progress by signal; `pack()`
now returns in well under a millisecond. That in turn made Cancel
possible, which matters when the spoons take ~8 seconds. A cancelled
search is recorded as `CANCELLED`, never `NOT_FOUND` — "you stopped me"
is not evidence about the bin, and must not land in `skipped` claiming a
tighter packing might exist at a size never actually searched.

`LayoutStage.Run` touches no widgets as a result: progress arrives through
a callback the window marshals onto the UI thread, because a label written
from a worker thread is undefined behavior in Qt rather than merely poor
style.

**Done when:** a full contours-to-layout run works in the GUI and the
stage never blocks on an upstream parameter change. The three spoon
captures load and pack to 5x2 in the window.

## M7 — Solid generation

- [x] [pipeline/layout/solid.py](../pipeline/layout/solid.py) takes a
  `Layout` — one pocket per part, bin sized from the layout's grid.
  A new module rather than an extension of the root `solid.py`, which
  keeps layout in one package and leaves the old single-contour path
  alone; the root script's import-time side effect (it wrote `test.scad`
  on import) is now behind a `__main__` guard.
- [x] `pocket_offset` is applied here, not in layout, so changing print
  tolerance re-cuts the solid instead of invalidating the arrangement.
  `layout_cli.py --solid-offset` exercises that; `ThinnestWalls` keeps it
  honest by refusing a tolerance the layout never budgeted for.
- [x] Wired into both front-ends: the CLI writes `<out>.scad` alongside
  the preview, and the GUI's Export writes all three.
- [ ] Verify a printed multi-pocket bin against the real objects.

**On the cutout mechanism.** Pockets are passed as *children* of
`bin_render`, which places them at the top of the infill extending
downward — that is where the pocket depth and its limit come from. The
older top-level `difference()` against `bin_render(...)` is *not* broken,
contrary to a claim made here in an earlier draft: it cuts the base and
then unions `bin_render_base` back on top, and the base survives.
(Checked by intersecting the result with a slab in the base under the
cutout: solid with the outer union, empty without it.) What the older
form does leave implicit is the depth — a bare `linear_extrude()` cuts
100mm upward from z=0, so the floor lands wherever the base happens to
start rather than somewhere chosen.

**One thing that would have survived review and failed on the bed:**

- **OpenSCAD is y-up; the layout frame is the printed page's, y-down.**
  Emitting the coordinates unchanged mirrors every pocket — it measures
  correctly on every axis and still will not hold the tool (D1). Pinned
  by a test that checks the emitted outline's winding is reversed, since
  a 180° rotation would pass a bounding-box check just as well.

**Done when:** a generated `.scad` renders and a test print holds every
object with the divider walls intact. The three spoons render to a
manifold solid (9639 facets, `Simple: yes`) with a 3.23mm thinnest
divider against the 1.2mm minimum; the physical print is Andrew's check.

## M8 — Grouping

See [Grouping](layout.md#grouping).

- [x] First-fit-decreasing over open bins, with the packer as the
  feasibility oracle.
- [x] Cache keyed by (frozenset of part ids, grid size); prune candidate
  moves with the area bound before ever calling the solver.
- [x] Local search: move/swap parts between bins, keep improvements.
- [ ] Wire into a front end — see below.

**Done when:** the three `test_data/` spoons group from 22 cells
(one-per-bin: 10 + 10 + 2) down to 10 or fewer, with every resulting bin
passing the independent overlap check. **Met** — 22 tests in
`pipeline/layout/grouping_test.py`; the spoons group to a single 5x2 at
**10 cells**, every bin clean under `CheckLayout`.

**The gate was only half met when this was built.** M4's sweep is clean,
but M7's physical print is still outstanding, so the clearances this
optimizes against remain derived rather than measured. The algorithm does
not depend on their values — revising `c_pair` changes the cell counts
grouping reports, not the search that produces them — but the 22-to-10
figure above is stated at the current defaults and will need re-measuring
if the print says they are wrong.

**Two things settled here that the design left open:**

- **First-fit does not grow a bin.** The design said a bin "may grow up to
  a user-set maximum footprint" during first-fit. Building it showed that
  growth is not a fit but a *trade* — this bin costs more in exchange for
  one fewer bin elsewhere — and stage 3 exists precisely to price trades.
  Taken greedily in part-arrival order, it would be committed to without
  ever being compared. Bins still grow; they grow where the alternative is
  visible.
- **The cache key in the design is the right one, and it is why the oracle
  answers two questions rather than one.** "Does this set fit this size"
  is what first-fit asks; "what is this set's smallest size" is what the
  local search asks. Building the second on the first makes them share one
  cache, so answering either partly answers the other. Going through
  `packer.Pack` instead would have kept the two apart.

**The bound is load-bearing in a way the packer's is not.** M4's bounds
save the solver from hopeless bin *sizes*; here the same bound decides
which candidate *moves* are worth pricing at all, and it runs on every one
of them. The one-sidedness requirement carries over unchanged and is
tested directly — measured on two 30x30 squares, which cannot gain by
sharing, the entire local search completes with **zero** solver calls.

**Not wired into a front end.** The CLI and GUI still pack one explicit
set into one bin, which is M5 and M6's contract; grouping changes the
shape of the output from a layout to a list of them, and the preview,
export, and solid paths all assume the former. That is a milestone's worth
of work on its own rather than a loose end of this one.

## M9 — Semantic coherence

Much later, and only on top of a working M8 — see
[Semantic coherence](layout.md#semantic-coherence-much-later). The
partition value function gains an entropy term so that a bin of assorted
spoons beats a bin holding a spoon, a hammer, and a camera lens.

- [ ] `BinEntropy(embeddings)` — von Neumann entropy of the normalized
  cosine Gram matrix. Pure function, no packing dependency, testable on
  synthetic vectors before any real embedding provider exists.
- [ ] Validate on `test_data/`: three different-size spoons is exactly
  the motivating case, so their embeddings must score well below a
  spoon-plus-unlike-object bin on real data, not just synthetic vectors.
- [ ] Embedding provider behind a narrow interface; layout takes
  `dict[int, np.ndarray]` and stays agnostic. Bake-off between CLIP on
  the masked crop, pooled SAM2 encoder features, and contour shape
  descriptors — an experiment, not a design decision.
- [ ] Entropy as a **tiebreak only** among equal-cell-count partitions
  (`λ → 0+`). Ship this before any weighted form.
- [ ] Only if tiebreaking proves too weak: the weighted objective
  `Σ_b (cells_b + λ · S_b)`, with `λ` surfaced in cells-per-nat.
- [ ] User-pinned groups that override the term outright.

**Done when:** on a mixed set, the tiebreak visibly reorders bins toward
coherent groupings *without* increasing total cell count — that last
clause is the whole test.

**Watch for:** the local search in M8 prunes candidate moves with the area
bound. Entropy cannot be pruned that way (it is not monotone under
insertion — see the design doc), so the entropy half of every surviving
candidate must be evaluated. If M8's search is already near its time
budget, this is where it tips over; cache per-bin entropy keyed by the
bin's frozenset of part ids, exactly as M8 caches packing results.

## Risks

- **Local minima.** The construction heuristic that was this risk's
  fallback is now M3's primary initialization, for the deep-overlap reason
  above. If annealed restarts still fail routinely on feasible-looking
  sets, the next lever is the jitter schedule, not another initializer.
- **Raster resolution vs. speed.** 0.25 mm/px is a guess. If pair
  evaluation dominates runtime, drop to 0.5 mm/px and widen clearances to
  compensate — but only with M4's sweep confirming zero overlaps at the
  new setting.
- **Clearance defaults.** `c_pair = 3.2 mm` is derived, not measured. The
  first physical print at M7 is the real test; expect to revise it.
- **Scope creep into grouping.** M8 is genuinely valuable and genuinely
  tempting to start early. It is worthless on an unreliable oracle.
- **Entropy as a standalone objective.** M9's term is a regularizer on a
  cell-count objective that opposes it. Optimized on its own it prefers
  one object per bin, since a singleton bin has entropy 0. If a future
  change ever makes entropy the primary term, that degeneracy is what
  will be observed.
