"""The control panel behind each planner window.

    layout_panel      pack a set of contours into one bin
    floorplan_panel   group a whole library into bins, and bins into drawers
    field_panel       show one contour's distance field

**Not stages, which is why they are not called that.** A capture stage is
one node of `capture.pipeline.Pipeline`: it recomputes from its parameters
and everything downstream re-runs synchronously the moment it returns. Two
of these run a search on a worker thread instead - seconds for one bin,
minutes for a whole library - so "re-run downstream as soon as it returns"
is exactly wrong: the display would fire while the search was still going.
They take `progress` and `cancelled` callbacks and are sequenced by a
finished signal instead. See the note in `layout_gui`.

`field_panel` is the honest oddity: it has nothing downstream at all, so
no sequencing question arises. It is here because it is the same *kind* of
thing - a panel of controls owning the state one window works on.

**Each belongs to exactly one window**, and is split out from it so the
half that is not widgets can be tested without one. That split is what the
three test modules here buy; the windows keep their own tests for the
wiring. The windows themselves are still at the repository root.
"""
