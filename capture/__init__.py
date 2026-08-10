"""One photograph to one contour per object, in real millimetres.

The algorithms, and the configurable widget over each:

    segmenter          + segmenter_stage           SAM2, from clicks
    morphology         + morphology_stage          mask cleanup, symmetry
    contour_extraction + contour_selection_stage   outlines, and which to keep
    calibration        + calibration_stage         ArUco markers to a homography
    rectify                                        that homography, applied
                       + svg_export_stage          the three exported files

**Why the pairs live together.** A `*_stage` is the configurable UI over
one step of the capture graph: it holds that step's parameters, builds the
group box that edits them, and re-runs everything downstream when one
changes. That is a real contract - `core.Pipeline` sequences these, and
`silhouette_gui` is the window it drives. Algorithm and stage are one
concept in two layers, so separating them by Qt-ness would file half of
each pair away from the other for a distinction the `_stage` suffix
already makes.

The algorithms are Qt-free and the stages are not. That split is worth
keeping *within* the package: `morphology.py` can be tested without a
display, and only `morphology_stage.py` knows a slider exists.

Note that `pipeline.field_stage`, `pipeline.layout_stage` and
`pipeline.floorplan_stage` are named for this pattern but do not follow
it - no `Pipeline` sequences them, and the latter two run on a worker
thread with progress and cancellation. See the note in `layout_gui`.

Design: docs/capture.md.
"""
