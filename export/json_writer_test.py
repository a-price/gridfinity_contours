import json
import os

import pytest

from export.json_writer import Dumps, SchemaPointer, WriteJson


def test_a_scalar_array_is_one_line():
    assert Dumps([1.0, 2.0]) == "[1.0, 2.0]"


def test_a_list_of_scalar_arrays_is_one_line():
    """The case that matters most: a contour is a list of points, and the
    whole thing renders on one line - so a forty-point contour costs one
    line, not forty.
    """
    assert Dumps([[1.0, 2.0], [3.0, 4.0]]) == "[[1.0, 2.0], [3.0, 4.0]]"


def test_scalar_arrays_nest_one_line_regardless_of_depth():
    assert Dumps([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]) == "[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]"


def test_a_list_of_dicts_stays_one_per_line():
    payload = [{"a": 1}, {"a": 2}]
    assert Dumps(payload) == '[\n {\n  "a": 1\n },\n {\n  "a": 2\n }\n]'


def test_a_dict_indents_normally():
    payload = {"version": 1, "grid": [3, 2]}
    assert Dumps(payload) == '{\n "version": 1,\n "grid": [3, 2]\n}'


def test_empty_containers_render_without_a_newline():
    assert Dumps({}) == "{}"
    assert Dumps([]) == "[]"
    assert Dumps({"contours": []}) == '{\n "contours": []\n}'


def test_indent_width_is_respected():
    assert Dumps({"a": [{"b": 1}]}, indent=2) == '{\n  "a": [\n    {\n      "b": 1\n    }\n  ]\n}'


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 1, "units": "mm", "contours": {"0": [[1.5, -2.25], [3.0, 4.0], [0.0, 0.0]]}},
        {"nested": {"a": [1, 2, [3, 4]], "b": None, "c": True, "d": 'text with "quotes" and \\backslash'}},
        [1, 2.5, None, True, False, "x"],
        {"empty_list": [], "empty_dict": {}},
        3.14159265358979,
        "a bare string",
        None,
    ],
)
def test_round_trips_to_the_same_value_as_stdlib_json(payload):
    """Whitespace differs from `json.dumps`; the value it parses back to
    must not. This is the property that matters for every real caller -
    `contour_io`, `layout.session`, `layout.plan` - since they all read
    these files back with plain `json.load`.
    """
    assert json.loads(Dumps(payload)) == json.loads(json.dumps(payload))


def test_write_json_ends_with_a_trailing_newline(tmp_path):
    path = tmp_path / "out.json"

    WriteJson(str(path), {"a": [1, 2]})

    text = path.read_text()
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert json.loads(text) == {"a": [1, 2]}


def test_schema_pointer_resolves_relative_to_the_output_file(tmp_path):
    """A `$schema` value has to be a relative URI - resolved against the
    document's own location, the way a browser resolves a relative href -
    so this checks the math the same way an editor would: joined back onto
    the output file's own directory, it must land on the real generated
    schema, wherever the output file happens to be.
    """
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    output_path = nested / "session.json"

    pointer = SchemaPointer(str(output_path), "session")

    assert not pointer.startswith("/")
    resolved = os.path.normpath(os.path.join(nested, pointer))
    assert resolved == os.path.join(os.path.dirname(__file__), "schema", "session.schema.json")
