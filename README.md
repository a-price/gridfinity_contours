# Gridfinity Contours

Turns a photo of an oddly-shaped object into a custom-fit
[Gridfinity](https://gridfinity.xyz/) bin: photograph the object next to a
printed calibration sheet, interactively segment it, extract a simplified
real-world-scale contour, then feed that contour into a bin generator to
produce a 3D-printable model with a cutout matching the object's outline.

## Tools

### `silhouette.py` — main app

An interactive PyQt5 GUI that runs the whole capture pipeline:

1. **Load** a photo of the object.
2. **Segment** it by clicking points on the image - left-click adds an
   interior (positive) point, right-click an exterior (negative) point -
   which are fed to a SAM2 model to produce a mask. Only the connected
   component(s) touched by a positive click are kept, so a stray disjoint
   blob elsewhere in the frame doesn't leak into the result.
3. **Clean up** the mask: morphological closing to smooth jitter, a
   minimum-area filter to drop small noise blobs, and optional
   lateral/longitudinal symmetry (many manmade objects are symmetric) -
   combined with the original mask via AND (carve out one-sided errors) or
   OR (fill in an occluded side from its mirror), pivoted on the mask's
   PCA-aligned bounding box.
4. **Extract and simplify** the object's contour, with a PCA-aligned
   bounding box.
5. **Calibrate**: detects ArUco markers on a printed calibration sheet
   (see `generate_aruco_sheet.py` below) and solves for the camera pose to
   recover real-world (mm) coordinates. If no sheet is in frame, contours
   fall back to pixel space instead of failing.
6. **Select** a contour to see it rectified to real-world units
   automatically - a text preview updates live, no button needed.
7. **Export** writes the selected, rectified contour to an SVG file (1
   unit = 1mm, PCA-aligned so it comes out level), ready to import into a
   CAD tool like Fusion 360. A same-scale PDF (`<filename>.svg.pdf`) is
   written alongside it - print that one instead of the SVG directly if a
   printed SVG comes out the wrong size, since not every SVG viewer/print
   path honors its embedded physical units (a PDF's page size is
   unambiguous).

Each stage's tunable parameters live in its own group box in the control
panel (Segmentation, Calibration, Mask Cleanup, Contour Selection, SVG
Export).

Clicking the image view does one of three things, chosen from the "Click
Mode" dropdown:

- **Add Segmentation Points** (default) - left/right click adds an
  interior/exterior point for SAM2.
- **Select a Contour** - click a detected object to select/deselect it,
  rectifying and previewing it automatically.
- **Select a Fiducial** - click to toggle a calibration fiducial, for
  calibration strategies that support manual selection (not the default
  ArUco one, which auto-matches by marker ID).

Run it with:

```
.venv/bin/python3 silhouette.py [path/to/photo.jpg]
```

The image path is optional; if omitted, use the "Load Image" button.

### `generate_aruco_sheet.py`

Generates a letter-size PDF calibration sheet with 4 ArUco markers at known
real-world (mm) positions - print it at 100% scale (no "fit to page") and
keep it in frame when photographing an object, so `silhouette.py` can
recover real-world units automatically. Its defaults already match
`ArucoCalibration`'s expected marker layout, so no configuration is needed
if you don't touch the constants in either file.

```
.venv/bin/python3 generate_aruco_sheet.py [output.pdf]
```

### `solid.py`

Takes a set of contour points (paste `silhouette.py`'s export output into
the `points` array at the bottom of the file) and generates a Gridfinity
bin `.scad` file - sized to the standard 42mm grid - with a cutout matching
the object's outline. Requires the `gridfinity-rebuilt-openscad` submodule
(see Installation) and OpenSCAD to render the result into an STL.

### `postprocess_gcode_for_prusa_i3.py`

A print-time utility: inserts an `M600` color-change pause into
already-sliced gcode at a given layer height, for Prusa i3-family printers.
Useful for making "shadow" prints where the bottom of the cutout is a
different color from the top, for printers without AMS.

```
python3 postprocess_gcode_for_prusa_i3.py path/to/file.gcode [height_mm]
```

## Installation

1. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies: `pip install -r requirements.txt`
3. If you'll use `solid.py`, fetch the OpenSCAD submodule it depends on:
   `git submodule update --init`
4. The SAM2 model (`facebook/sam2-hiera-tiny`) is loaded offline
   (`local_files_only=True`) by default, so it needs to already be cached
   locally (e.g. via `huggingface-cli download facebook/sam2-hiera-tiny`)
   before first use. Alternatively, pass `--download-model` to
   `silhouette.py` to let it fetch the model from the Hugging Face Hub on
   first run if it isn't cached yet.

## Development

Config for `black`/`pytest`/`pyright` lives in `pyproject.toml`; `mdformat`
reads only `.mdformat.toml`, not `pyproject.toml`. A `Makefile` wraps all of
them:

- `make format` - apply `black` and `mdformat` formatting
- `make format-check` - check formatting without changing files
- `make lint` - `pyflakes`
- `make typecheck` - `pyright`
- `make test` - `pytest`
- `make check` - all of the above, stopping at the first failure
- `make requirements` - regenerate `requirements.txt` from `requirements.in`
  (see below)

`requirements.in` lists only direct dependencies (runtime + dev tooling);
`requirements.txt` is the fully-pinned, `pip-compile`-generated lockfile
(every transitive dependency, annotated with `# via <package>` showing why
it's there) - install from `requirements.txt`, but edit `requirements.in`
and run `make requirements` to change a direct dependency or its version.
`make requirements` needs `pip-tools` (`pip install pip-tools`); it's a
meta-tool for maintaining the lockfile, not a project dependency itself,
so it isn't in `requirements.in`.

Tests are plain `pytest`; the `slow` marker flags tests that exercise the
real SAM2 model end-to-end (slower, needs the cached weights):

```
.venv/bin/python3 -m pytest              # everything
.venv/bin/python3 -m pytest -m "not slow"  # skip the SAM2-dependent tests
```
