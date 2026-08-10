import json

import numpy as np
import pytest

from export.contour_io import CONTOUR_FORMAT_VERSION, LoadContours, SaveContours

_RECT = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]])
_TRIANGLE = np.array([[1.0, 1.0], [5.0, 1.0], [3.0, 4.0]])


def _write(tmp_path, payload) -> str:
    path = tmp_path / "contours.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_contours_survive_a_round_trip(tmp_path):
    path = str(tmp_path / "contours.json")
    contours = {0: _RECT, 7: _TRIANGLE}

    SaveContours(path, contours)
    loaded = LoadContours(path)

    assert sorted(loaded) == [0, 7]
    for contour_id, points in contours.items():
        np.testing.assert_allclose(loaded[contour_id], points)


def test_a_round_trip_does_not_round_coordinates(tmp_path):
    # The clearances this feeds are tenths of a millimeter, so a dump that
    # quietly rounded would change which grid size the packer chose.
    path = str(tmp_path / "contours.json")
    precise = np.array([[0.123456789, 1.987654321], [30.000000001, 0.5], [15.5, 22.333333333]])

    SaveContours(path, {0: precise})

    np.testing.assert_array_equal(LoadContours(path)[0], precise)


def test_integer_ids_come_back_as_integers(tmp_path):
    # JSON object keys are strings; ids that came back as "0" would key
    # nothing the rest of the pipeline looks up.
    path = str(tmp_path / "contours.json")

    SaveContours(path, {3: _RECT})

    assert list(LoadContours(path)) == [3]


def test_saving_nothing_raises(tmp_path):
    with pytest.raises(ValueError):
        SaveContours(str(tmp_path / "empty.json"), {})


def test_a_float32_contour_saves(tmp_path):
    # The GUI's contours arrive as float32, which json.dump cannot
    # serialize without the explicit conversion.
    path = str(tmp_path / "contours.json")

    SaveContours(path, {0: _RECT.astype(np.float32)})

    np.testing.assert_allclose(LoadContours(path)[0], _RECT)


def test_a_future_format_version_is_refused(tmp_path):
    path = _write(tmp_path, {"version": CONTOUR_FORMAT_VERSION + 1, "units": "mm", "contours": {"0": _RECT.tolist()}})

    with pytest.raises(ValueError, match="version"):
        LoadContours(path)


def test_other_units_are_refused(tmp_path):
    # Reading inches as millimeters would pack a layout 25x too small and
    # report success, so this has to fail rather than guess.
    path = _write(tmp_path, {"version": CONTOUR_FORMAT_VERSION, "units": "in", "contours": {"0": _RECT.tolist()}})

    with pytest.raises(ValueError, match="in"):
        LoadContours(path)


def test_an_empty_contour_set_is_refused(tmp_path):
    path = _write(tmp_path, {"version": CONTOUR_FORMAT_VERSION, "units": "mm", "contours": {}})

    with pytest.raises(ValueError, match="no contours"):
        LoadContours(path)


def test_a_non_integer_id_is_refused(tmp_path):
    path = _write(tmp_path, {"version": CONTOUR_FORMAT_VERSION, "units": "mm", "contours": {"spoon": _RECT.tolist()}})

    with pytest.raises(ValueError, match="integer id"):
        LoadContours(path)


def test_a_contour_with_the_wrong_shape_is_refused(tmp_path):
    path = _write(tmp_path, {"version": CONTOUR_FORMAT_VERSION, "units": "mm", "contours": {"0": [[1.0, 2.0, 3.0]]}})

    with pytest.raises(ValueError, match=r"expected \(N, 2\)"):
        LoadContours(path)


def test_a_contour_with_too_few_points_is_refused(tmp_path):
    path = _write(tmp_path, {"version": CONTOUR_FORMAT_VERSION, "units": "mm", "contours": {"0": [[0.0, 0.0]]}})

    with pytest.raises(ValueError, match="at least 3"):
        LoadContours(path)


def test_a_non_finite_coordinate_is_refused(tmp_path):
    # A NaN survives rasterization into a distance field and produces a
    # layout that is merely wrong rather than one that errors, so it has to
    # be caught at the door.
    path = _write(tmp_path, {"version": CONTOUR_FORMAT_VERSION, "units": "mm", "contours": {"0": [[0, 0], [1, 0]]}})
    payload = json.loads(open(path).read())
    payload["contours"]["0"] = [[0.0, 0.0], [1.0, 0.0], [float("nan"), 1.0]]
    with open(path, "w") as f:
        json.dump(payload, f)

    with pytest.raises(ValueError, match="non-finite"):
        LoadContours(path)


def test_a_file_that_is_not_a_contour_dump_is_refused(tmp_path):
    path = _write(tmp_path, [1, 2, 3])

    with pytest.raises(ValueError, match="not a contour file"):
        LoadContours(path)
