"""fit.meta.json versions: where the live model and config files are."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from camdl_scope import fitmeta
from camdl_scope.verdict import UnsupportedSchema
from tests.fixtures.make_golden_store import RUN_DIR, build


def _store(tmp_path: Path) -> tuple[Path, Path]:
    store = tmp_path / "proj" / "results" / "fits"
    store.mkdir(parents=True)
    build(store)
    run = store / RUN_DIR
    (tmp_path / "proj" / "models").mkdir()
    (tmp_path / "proj" / "models" / "m.camdl").write_text("model live {}\n")
    return store, run


def _set_meta(run: Path, **kw) -> None:
    meta = json.loads((run / "fit.meta.json").read_text())
    for k in ("model_path", "fit_toml_path", "schema_version", "source_model", "source_config"):
        meta.pop(k, None)
    meta.update(kw)
    (run / "fit.meta.json").write_text(json.dumps(meta))


def _source(store: Path, monkeypatch) -> dict:
    monkeypatch.setenv("CAMDL_SCOPE_STORE", str(store))
    from camdl_scope.api.app import app
    return TestClient(app).get(f"/api/runs/{RUN_DIR}/source").json()["model"]


def test_v1_model_path_is_relative_to_the_project_root(tmp_path, monkeypatch):
    store, run = _store(tmp_path)
    (run / "model.camdl.original").unlink(missing_ok=True)
    _set_meta(run, model_path="models/m.camdl", fit_toml_path="fits/m.toml")
    m = _source(store, monkeypatch)
    assert m["present"] and m["origin"] == "live" and "model live" in m["text"]


def test_v2_source_model_is_relative_to_the_fit_folder(tmp_path, monkeypatch):
    store, run = _store(tmp_path)
    (run / "model.camdl.original").unlink(missing_ok=True)
    # fit folder is <proj>/results/fits/<run>; the model is <proj>/models/m.camdl
    _set_meta(run, schema_version=2, source_model="../../../models/m.camdl",
              source_config="../../../fits/m.toml")
    m = _source(store, monkeypatch)
    assert m["present"] and m["origin"] == "live" and "model live" in m["text"]
    assert m["path"] == "../../../models/m.camdl"


def test_v2_path_is_not_read_against_the_project_root(tmp_path, monkeypatch):
    store, run = _store(tmp_path)
    (run / "model.camdl.original").unlink(missing_ok=True)
    _set_meta(run, schema_version=2, source_model="models/m.camdl")  # wrong base for v2
    assert not _source(store, monkeypatch)["present"]


def test_the_archived_copy_is_preferred_to_the_live_file(tmp_path, monkeypatch):
    store, run = _store(tmp_path)
    (run / "model.camdl.original").write_text("model archived {}\n")
    _set_meta(run, schema_version=2, source_model="../../../models/m.camdl")
    m = _source(store, monkeypatch)
    assert m["origin"] == "leaf" and "archived" in m["text"]


def test_unknown_schema_version_is_refused_by_name(tmp_path, monkeypatch):
    store, run = _store(tmp_path)
    (run / "model.camdl.original").unlink(missing_ok=True)
    _set_meta(run, schema_version=3, source_model="x.camdl")
    m = _source(store, monkeypatch)
    assert not m["present"] and "schema_version 3" in m["path"]
    with pytest.raises(UnsupportedSchema):
        fitmeta.schema_version({"schema_version": 3})


def test_names_follow_the_version():
    assert fitmeta.config_ref({"schema_version": 2, "source_config": "../c/fit.toml",
                               "fit_toml_path": "old.toml"}) == "../c/fit.toml"
    assert fitmeta.config_ref({"fit_toml_path": "old.toml"}) == "old.toml"
    assert fitmeta.model_ref({"schema_version": 2, "model_path": "old.camdl"}) == ""
