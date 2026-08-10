"""One photograph to one contour per object, in real millimetres.

The algorithms, and the configurable widget over each:

    segmenter          + segmenter_stage           SAM2, from clicks
    morphology         + morphology_stage          mask cleanup, symmetry
    contour_extraction + contour_selection_stage   outlines, and which to keep
    calibration        + calibration_stage         ArUco markers to a homography
    rectify                                        that homography, applied
                       + svg_export_stage          the three exported files

    pipeline                                       what sequences the above

**Why the pairs live together.** A `*_stage` is the configurable UI over
one step of the capture graph: it holds that step's parameters, builds the
group box that edits them, and re-runs everything downstream when one
changes. That is a real contract - `pipeline.Pipeline` sequences these,
and `silhouette_gui` is the window it drives. Algorithm and stage are one
concept in two layers, so separating them by Qt-ness would file half of
each pair away from the other for a distinction the `_stage` suffix
already makes.

`pipeline` is here rather than somewhere shared because this is the only
place it is used. It was `pipeline/core.py` back when the package holding
it also held the layout solver and the Qt widget helpers, neither of which
ever wanted it.

The algorithms are Qt-free and the stages are not. That split is worth
keeping *within* the package: `morphology.py` can be tested without a
display, and only `morphology_stage.py` knows a slider exists.

The planner windows own a similar-looking object each. Those are not
stages and no longer claim to be - see `panels`.

Design: docs/capture.md.
"""
