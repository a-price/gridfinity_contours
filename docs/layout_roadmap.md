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

No new dependencies are needed through M8: SDFs use `opencv-python` and
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
- [x] `InteriorEnvelope(n, m, inset)` → rounded-rect polygon for an
  `N x M` bin's usable interior.
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

- [ ] Area lower bound (part areas dilated by `c_pair / 2`) and extent
  lower bound (largest part's oriented bbox must fit **with at least one
  raster cell of slack** — `big_spoon` clears a 5-cell run by 0.04 mm,
  well under the 0.25 mm resolution, and a bound without the slack term
  will call that feasible and waste a full restart budget on it).
- [ ] Candidate enumeration by increasing `N * M`, square-ish tiebreak,
  capped at `max_grid`.
- [ ] `Pack(contours, params)` → layout plus a report distinguishing
  "provably too small" from "no arrangement found".
- [ ] Randomized validation sweep, marked `slow`: every successful layout
  re-verified with M1's independent overlap predicate.

**Done when:** synthetic sets with known optimal cell counts pack to that
count; the three `test_data/` spoons pack into a 5x2 (39% fill) and the
report cleanly explains any smaller size it rejected; the sweep finds
zero overlaps across a few hundred random cases.
**This is the gate for everything downstream** — if the sweep is not
clean, the raster resolution or clearance defaults are wrong, and no
amount of GUI work will fix that.

## M5 — Headless CLI

- [ ] `layout_cli.py`: read contours (JSON of the pipeline's
  `dict[int, ndarray]`), pack, write a preview SVG.
- [ ] Contour serialization helpers shared with the GUI, so a session's
  contours can be dumped once and iterated on offline.
- [ ] **Refactor the writers first.** `WriteSvg` and `WritePdf` both call
  `AlignContoursToPca`, which re-aligns each contour into its own frame
  and would destroy the arrangement. Split each into an
  align-then-write wrapper (existing behavior, existing tests unchanged)
  over a write-these-coordinates core.
- [ ] Preview SVG on top of that core: one polygon per placed part plus
  the bin outline and cell grid, at true mm scale, keeping
  [svg_writer.py](../pipeline/svg_writer.py)'s unit conventions (the
  96/25.4 pre-scaling in particular).

**Done when:** a real captured contour set packs from the command line
and the printed preview measures correctly against a physical bin.

## M6 — GUI stage

- [ ] `pipeline/layout_stage.py`: `LayoutStage(Stage)` with a "Layout"
  group box — clearance spin boxes, max grid size, seed, and a "Pack"
  button.
- [ ] Explicit trigger only, like
  [SvgExportStage](../pipeline/svg_export_stage.py) — packing takes
  seconds and must not run on slider drags.
- [ ] Register in `SVGGui` downstream of rectification; render the
  resulting layout in the image view.
- [ ] Report grid size and any failure reason in the panel.

**Done when:** a full photo-to-layout run works in the GUI and the stage
never blocks on an upstream parameter change.

## M7 — Solid generation

- [ ] Extend `solid.py` to take a `Layout` — many pockets, bin sized from
  the layout's grid rather than from one contour's bbox.
- [ ] Apply `pocket_offset` here, not in layout (so changing print
  tolerance does not invalidate a layout).
- [ ] Verify a printed multi-pocket bin against the real objects.

**Done when:** a generated `.scad` renders and a test print holds every
object with the divider walls intact.

## M8 — Grouping

Only after M4's validation sweep is clean and M7 has produced a real
print — see [Grouping](layout.md#grouping-future-work).

- [ ] First-fit-decreasing over open bins, with `Pack` as the
  feasibility oracle.
- [ ] Cache keyed by (frozenset of part ids, grid size); prune candidate
  moves with the area bound before ever calling the solver.
- [ ] Local search: move/swap parts between bins, keep improvements.

**Done when:** the three `test_data/` spoons group from 22 cells
(one-per-bin: 10 + 10 + 2) down to 10 or fewer, with every resulting bin
passing the independent overlap check.

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
