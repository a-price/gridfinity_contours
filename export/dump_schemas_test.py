import json

import jsonschema

import export.dump_schemas as dump_schemas
from export.dump_schemas import SCHEMAS, Main


def _redirect(monkeypatch, tmp_path):
    monkeypatch.setattr(dump_schemas, "_OUT", str(tmp_path))


def test_it_writes_one_valid_schema_document_per_format(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)

    assert Main([]) == 0

    for name in SCHEMAS:
        payload = json.loads((tmp_path / f"{name}.schema.json").read_text())
        # Not just "a file exists" - each one has to be a well-formed
        # schema document by JSON Schema's own rules, not merely a dict
        # this project's writer happened to produce.
        jsonschema.Draft202012Validator.check_schema(payload)


def test_check_passes_once_up_to_date(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    Main([])

    assert Main(["--check"]) == 0


def test_check_reports_a_stale_file_without_overwriting_it(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    Main([])
    stale = tmp_path / "session.schema.json"
    stale.write_text("{}\n")

    assert Main(["--check"]) == 1
    assert stale.read_text() == "{}\n"


def test_check_reports_a_missing_file_without_writing_it(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)

    assert Main(["--check"]) == 1
    assert not (tmp_path / "session.schema.json").exists()


def test_regenerating_is_idempotent(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    Main([])
    first = {name: (tmp_path / f"{name}.schema.json").read_text() for name in SCHEMAS}

    Main([])

    for name in SCHEMAS:
        assert (tmp_path / f"{name}.schema.json").read_text() == first[name]
