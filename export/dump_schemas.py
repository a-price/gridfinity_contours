"""Dumping the schemas beside each format's dataclasses to real
`export/schema/*.schema.json` files.

The Python dicts - `export.contour_io.CONTOUR_DUMP_SCHEMA`, `layout.plan.
DRAWERS_FILE_SCHEMA`, `layout.session.SESSION_SCHEMA`, and the fragments
each of them is built from - are the source of truth, kept beside the
dataclass each one describes so a change to a dataclass and a change to
its schema show up in the same diff. These files are the mechanical,
un-hand-editable *output* of that source of truth: something an external
editor can point a `$schema` at when hand-editing a contour dump, the one
file this project expects a human to touch by hand (see contour_io.py).

    make schema          # regenerate
    make schema-check    # fail if what's committed is stale

`--check` compares against what generation would produce without writing,
for the Makefile target - safe to run on every `make check`, unlike
`docs-check`, because this generation is pure deterministic Python with no
external tool whose version could shift the output.
"""

import argparse
import os
import sys
from typing import Sequence

from export.contour_io import CONTOUR_DUMP_SCHEMA
from export.json_writer import Dumps
from layout.plan import DRAWERS_FILE_SCHEMA
from layout.session import SESSION_SCHEMA

_OUT = os.path.join(os.path.dirname(__file__), "schema")

# One file per format this project reads and writes. The name is the file
# stem an editor's `$schema` would point at, not the dict's own Python name.
SCHEMAS = {
    "contour_dump": CONTOUR_DUMP_SCHEMA,
    "drawers": DRAWERS_FILE_SCHEMA,
    "session": SESSION_SCHEMA,
}


def Main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check", action="store_true", help="fail if export/schema/*.schema.json are stale, without writing"
    )
    args = parser.parse_args(argv)

    stale = []
    for name, schema in sorted(SCHEMAS.items()):
        path = os.path.join(_OUT, f"{name}.schema.json")
        text = Dumps(schema) + "\n"
        if args.check:
            current = open(path).read() if os.path.exists(path) else None
            if current != text:
                stale.append(path)
            continue
        os.makedirs(_OUT, exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    if stale:
        print("stale: " + ", ".join(stale) + " - run 'make schema' and commit the result", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(Main())
