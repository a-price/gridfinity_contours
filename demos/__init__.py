"""Scripts that write every picture in the README and the design docs.

    capture_demo      the capture window animated: photo to contour
    layout_demo       the three layout searches: pack, group, drawer
    screenshot_demo   one still of each window, panel and all
    render_demo       one still per drawing path, for review diffs
    solid_demo        the printable bin itself, rendered by OpenSCAD

**Run as modules, not as scripts.** `python demos/render_demo.py` puts
`demos/` on `sys.path` rather than the repository root, so every
`from layout...` and `from silhouette_gui...` in here fails. This
package exists to make the module form work:

    .venv/bin/python3 -m demos.render_demo --out docs/media

For the same reason the fixture paths in these scripts resolve against the
repository root - one level up - rather than the working directory, so a
picture comes out identical wherever it was generated from.

Nothing imports these; they sit at the top of the dependency graph and
reach down into `pipeline` and the GUIs. `make media` runs the lot.

What each one costs, what it needs installed, and why these are committed
artifacts rather than test assertions: docs/media.md.
"""
