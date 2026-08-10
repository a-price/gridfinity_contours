"""Shape math, with no I/O and no project concepts in it.

    pca_box   an oriented bounding box from a point set's principal axes

Small on purpose rather than by accident. `PCABox` is the most widely
shared thing in this repository - `export.svg_writer` aligns contours with
it, `capture.morphology` measures masks with it, `layout.part`
builds a part's local frame from it, and `silhouette_gui` transforms
clicked points with it. Four consumers spanning every layer, which is
exactly why it cannot live in any one of them.

It used to sit in `capture.contour_extraction`, so a module named for one
step of the capture pipeline was imported by the layout solver purely to
reach a bounding box. Moving it here is what lets the layout package stop
depending on the capture package at all.
"""
