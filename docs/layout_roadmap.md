# Layout Optimization Roadmap

Implementation plan for [layout.md](layout.md). Ordered so that each
milestone is independently testable and leaves the tree green
(`make check`).

The ordering principle: build the *verifier* before the solver, and the
headless path before the GUI. A stochastic optimizer whose correctness
check shares code with the optimizer itself is untrustworthy, and one
that can only be run through a Qt event loop is untunable.

## M1 — Geometry primitives

`pipeline/layout.py`, no solver yet.

- [ ] Gridfinity constants: pitch, interior inset, corner radius, wall
  and divider minimums, sourced from
  [standard.scad](../gridfinity-rebuilt-openscad/src/core/standard.scad)
  and cited in comments.
- [ ] `InteriorEnvelope(n, m, inset)` → rounded-rect polygon for an
  `N x M` bin's usable interior.
- [ ] `Part`: a contour plus its precomputed raster fields — occupancy
  mask, SDF (negative inside), SDF gradient, boundary sample points
  (vertices plus edge resampling at ~1 raster cell), area, local frame
  origin.
- [ ] `Placement`: part id, translation, orientation index (0-3), and
  `ToWorld()` returning the placed polygon in bin coordinates.
- [ ] Exact, independent overlap predicate for tests — polygon-based, not
  SDF-based. This is the oracle M4 is validated against, so it must not
  share code with the solver.
- [ ] SVG contour loader for `test_data/`, deriving scale as
  `viewBox_width / width_mm`. **Do not hardcode 96/25.4** — the existing
  fixtures are 1:1 mm and predate that change, so a hardcoded factor
  breaks one format or the other by 3.78x.

**Done when:** SDF sign/magnitude match hand-computed values for a
rectangle and an L-shape; rotating a part 90° gives a field equal to the
field of the rotated polygon; the independent overlap predicate agrees
with hand-checked cases; the three `test_data/` spoons load at their
measured sizes (200.26, 162.76, 73.93 mm long).

## M2 — Energy and forces

- [ ] `LayoutParameters` dataclass: `pocket_offset`, `c_pair`, `c_wall`
  (both derived from `pocket_offset` by default), `raster_resolution`,
  iteration/restart budgets, `seed`, `max_grid`.
- [ ] Pair term: sample `∂i` against `sdf_j` and `∂j` against `sdf_i`,
  quadratic in penetration depth, forces applied equal and opposite.
- [ ] Wall term: sample every part against the container field.
- [ ] `Energy(placements)` returning total energy and per-part force,
  vectorized over sample points.

**Done when:** two identical squares at increasing separation give
energy that is positive-and-decreasing, then exactly zero past `c_pair`;
force directions point along the separating axis; a part straddling the
wall is pushed inward. Finite-difference check: numerical gradient of
`Energy` matches the returned forces.

## M3 — Solver

- [ ] Deterministic seeded initialization: parts placed largest-first on
  a jittered lattice.
- [ ] Damped descent loop with decaying jitter; early exit on `E = 0`.
- [ ] Restart loop, re-seeding positions *and* the orientation
  assignment (orientation is discrete — only restarts explore it).
- [ ] `SolveFixedGrid(parts, n, m, params)` → `Layout | None`.

**Done when:** three rectangles with a known-tight packing into a 1x3 are
placed without overlap, repeatably; the same seed reproduces a
byte-identical layout; a deliberately over-full bin fails cleanly rather
than returning an overlapping layout.

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
- [ ] Preview SVG: one polygon per placed part plus the bin outline and
  cell grid, at true mm scale, reusing
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

- **Local minima.** If the annealed restarts routinely fail on
  feasible-looking sets, the fallback is a bottom-left-fill construction
  heuristic to seed the relaxation instead of a jittered lattice. Decide
  at M3; do not build both up front.
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
