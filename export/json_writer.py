"""Compact JSON: every file this project reads back later -
`contour_io`'s dumps, `layout.session`'s floorplans, `layout.plan`'s
drawer lists - writes it, so all three should look alike.

Python's `json.dump(..., indent=N)` indents every nesting level uniformly,
which is fine for the dicts these files are mostly made of but disastrous
for the coordinate arrays inside them: a single `[x, y]` point costs four
lines instead of one, and a captured contour with forty points costs a
page. A real six-object floorplan ran to over a thousand lines this way.

`Dumps` keeps that indentation for dicts and for any list holding
something other than plain numbers, strings, booleans or nulls - or other
such lists, nested arbitrarily deep - but writes a list of only those on
one line: a grid size, a placement's position, or a whole contour's worth
of points at once. That is the one rule: an array of nothing but scalars,
however deeply nested, is one line; everything else nests normally. It
changes no value's own text - a float renders exactly as `json.dumps`
would render it, just without the whitespace between it and its neighbours
- so it costs nothing in round-trip fidelity, and it turns a forty-point
contour from forty lines into one. `contour_io`'s own docstring is the
reason that matters here: these files are meant to be diffable, and are
the one place a human might reasonably hand-edit a coordinate.
"""

import json
import os
from typing import Any

_SCALAR_TYPES = (type(None), bool, int, float, str)


def _IsScalarArray(value: Any) -> bool:
    """Whether `value` is a list holding nothing but scalars and/or other
    such lists - so `[[1.0, 2.0], [3.0, 4.0]]` qualifies exactly as
    `[1.0, 2.0]` does, however many levels deep the nesting goes.
    """
    return isinstance(value, list) and all(isinstance(item, _SCALAR_TYPES) or _IsScalarArray(item) for item in value)


def _Format(value: Any, indent: int, depth: int) -> str:
    pad = " " * (indent * depth)
    inner_pad = " " * (indent * (depth + 1))

    if isinstance(value, dict):
        if not value:
            return "{}"
        items = ",\n".join(
            f"{inner_pad}{json.dumps(str(key))}: {_Format(item, indent, depth + 1)}" for key, item in value.items()
        )
        return f"{{\n{items}\n{pad}}}"

    if isinstance(value, list):
        if not value:
            return "[]"
        if _IsScalarArray(value):
            return json.dumps(value)
        items = ",\n".join(f"{inner_pad}{_Format(item, indent, depth + 1)}" for item in value)
        return f"[\n{items}\n{pad}]"

    return json.dumps(value)


def Dumps(payload: Any, indent: int = 1) -> str:
    """`payload` as JSON text, indented like `json.dumps(payload,
    indent=indent)` except that an array holding only scalars - or nested
    arrays of them, a contour's `[[x, y], ...]` included - is written on
    one line rather than one element per line. No trailing newline - see
    `WriteJson` for the file convention every writer here follows.
    """
    return _Format(payload, indent, 0)


def WriteJson(path: str, payload: Any, indent: int = 1) -> None:
    """Write `payload` to `path` as compact JSON, with the trailing
    newline every JSON file in this project ends with.
    """
    with open(path, "w") as handle:
        handle.write(Dumps(payload, indent))
        handle.write("\n")


def SchemaPointer(output_path: str, schema_name: str) -> str:
    """The `$schema` value for a file about to be written to
    `output_path`, pointing at `export/schema/<schema_name>.schema.json`.

    Relative rather than absolute, and computed fresh for every call rather
    than fixed once: these files get committed - this repo's own
    floorplan.json among them - and an absolute path would bake one
    machine's home directory into that history. A relative URI is what a
    validating editor resolves the way a browser resolves a relative href -
    against the document's own location - so this keeps working no matter
    where `output_path` ends up.
    """
    schema_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema")
    schema_path = os.path.join(schema_dir, f"{schema_name}.schema.json")
    pointer = os.path.relpath(schema_path, os.path.dirname(os.path.abspath(output_path)))
    return pointer.replace(os.sep, "/")
