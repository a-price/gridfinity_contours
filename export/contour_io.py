"""Dumping a session's contours to a file and reading them back.

The point is to break the pipeline's dependence on a live GUI session.
Extracting contours from a photo is interactive and slow; packing them is
neither, and iterating on the packer by re-clicking the same spoons every
time would be miserable. One dump, then as many headless runs as the
search needs.

JSON rather than .npz because these files are small, are meant to be
diffable, and are the one place a human might reasonably hand-edit a
coordinate.
"""

import json
from typing import Any

import numpy as np

from export.json_writer import SchemaPointer, WriteJson

# Bumped only when a reader could otherwise misinterpret an older file.
# Recorded so that a future format change fails loudly on sight instead of
# quietly reading millimeters as something else.
CONTOUR_FORMAT_VERSION = 1

UNITS = "mm"

# A contour has no dataclass of its own - it is a bare `np.ndarray`
# everywhere in this project - so its wire shape lives here instead, beside
# the one module that already treats it as an authority (architecture.md:
# "this is where [the millimetre frame] is written down"). `minItems: 3`
# matches `_ParseContour`'s "a polygon needs at least 3" check.
POINT_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 2,
    "maxItems": 2,
}
CONTOUR_SCHEMA = {
    "type": "array",
    "items": POINT_SCHEMA,
    "minItems": 3,
}

# The whole file `SaveContours`/`LoadContours` read and write.
CONTOUR_DUMP_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Gridfinity Contours contour dump",
    "type": "object",
    "properties": {
        # Not required: only files this build wrote have it, and an older
        # dump without one is still a valid dump.
        "$schema": {"type": "string"},
        "version": {"const": CONTOUR_FORMAT_VERSION},
        "units": {"const": UNITS},
        "contours": {
            "type": "object",
            "propertyNames": {"pattern": "^[0-9]+$"},
            "additionalProperties": CONTOUR_SCHEMA,
        },
    },
    "required": ["version", "units", "contours"],
    "additionalProperties": False,
}


def SaveContours(path: str, contours: dict[int, np.ndarray]) -> None:
    """Write real-world millimeter contours (e.g. Rectify.contours) to
    `path`.

    Coordinates are written at full precision rather than rounded: this is
    an intermediate the packer reads back, not a drawing, and the
    clearances it decides are tenths of a millimeter.
    """
    if not contours:
        raise ValueError("no contours to save")

    payload = {
        "$schema": SchemaPointer(path, "contour_dump"),
        "version": CONTOUR_FORMAT_VERSION,
        "units": UNITS,
        "contours": {
            str(contour_id): np.asarray(points, dtype=np.float64).reshape(-1, 2).tolist()
            for contour_id, points in sorted(contours.items())
        },
    }
    WriteJson(path, payload)


def _ParseContour(contour_id: str, points: Any) -> tuple[int, np.ndarray]:
    """One id/points entry, validated into the shape the packer expects.

    Everything downstream assumes an (N, 2) array of finite millimeters and
    would fail much further from the cause - a distance field with a NaN in
    it produces a layout that is merely wrong, not one that errors.
    """
    try:
        key = int(contour_id)
    except ValueError:
        raise ValueError(f"contour key '{contour_id}' is not an integer id")

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"contour {key} has shape {array.shape}, expected (N, 2)")
    if len(array) < 3:
        raise ValueError(f"contour {key} has {len(array)} points; a polygon needs at least 3")
    if not np.isfinite(array).all():
        raise ValueError(f"contour {key} contains a non-finite coordinate")
    return key, array


def LoadContours(path: str) -> dict[int, np.ndarray]:
    """Read a contour dump back as the `dict[int, ndarray]` the rest of the
    pipeline passes contours around in.
    """
    with open(path) as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a contour file")

    version = payload.get("version")
    if version != CONTOUR_FORMAT_VERSION:
        raise ValueError(f"{path} is format version {version}, this build reads {CONTOUR_FORMAT_VERSION}")

    units = payload.get("units")
    if units != UNITS:
        raise ValueError(f"{path} is in '{units}', expected '{UNITS}'")

    raw = payload.get("contours")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} contains no contours")

    return dict(_ParseContour(contour_id, points) for contour_id, points in raw.items())
