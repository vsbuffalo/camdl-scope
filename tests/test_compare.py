"""Model-comparison reader tests. The watcher is a *pure consumer* of ``camdl
compare`` — the elpd/Δelpd math and the evidence scale are camdl's and tested
there. These cover what the watcher itself owns: prequential discovery, the
stable-name compare.toml generation, and the shell-out orchestration (gated on
the ``camdl`` binary, skipped when it's absent).
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from camdl_scope import compare


def _write_prequential(path: Path, level: float, n: int = 60) -> None:
    """A minimal valid prequential.json (the fields ``camdl compare`` reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [
        {
            "t": float(1 + i), "y_obs": 10.0, "y_pred_samples": [],
            "log_score": level + math.sin(i * 0.3) * 0.4,
            "crps": 2.0, "pit": 0.5, "ess": 800.0,
        }
        for i in range(n)
    ]
    path.write_text(json.dumps(
        {"schema_version": 1, "t0": 1, "provenance": "plug_in",
         "steps": steps, "warnings": []}
    ))


def test_find_prequential_at_bounded_depth(tmp_path):
    # camdl writes it under a posterior stage's seed dir (depth 2 below run_dir).
    target = tmp_path / "01-posterior-aaaa" / "seed_0-bbbb" / "prequential.json"
    _write_prequential(target, level=-2.0)
    found = compare.find_prequential(tmp_path)
    assert found == target


def test_find_prequential_absent_is_none(tmp_path):
    assert compare.find_prequential(tmp_path) is None


def test_write_compare_toml_uses_stable_names_and_quotes_paths():
    specs = [
        compare.CompareSpec(name="run a", path=Path("/tmp/a dir/prequential.json")),
        compare.CompareSpec(name="run_b", path=Path("/tmp/b/prequential.json")),
    ]
    toml = compare._write_compare_toml(specs, baseline="run_b", metrics=None)
    assert 'format = "json"' in toml
    assert 'baseline = "run_b"' in toml
    # Each model is named explicitly (bare paths would collide every row to
    # "prequential.json"), and a path with a space is a quoted basic string
    # (paths are resolved to absolute, so match the quoted tail, not the prefix).
    assert 'name = "run a"' in toml
    assert '/a dir/prequential.json"' in toml


@pytest.mark.skipif(
    shutil.which("camdl") is None, reason="camdl binary not on PATH"
)
def test_run_compare_shells_out_and_parses(tmp_path):
    a = tmp_path / "a" / "prequential.json"
    b = tmp_path / "b" / "prequential.json"
    _write_prequential(a, level=-2.0)  # better elpd
    _write_prequential(b, level=-2.3)  # worse, same horizon → commensurable
    specs = [
        compare.CompareSpec(name="model_a", path=a),
        compare.CompareSpec(name="model_b", path=b),
    ]
    data, commensurable, _notes = compare.run_compare(specs, baseline="model_a")
    assert commensurable is True
    assert data["baseline"] == "model_a"
    names = {r["name"] for r in data["rows"]}
    assert names == {"model_a", "model_b"}
    # The baseline has no Δ; the other does, and is worse (negative).
    by = {r["name"]: r for r in data["rows"]}
    assert by["model_a"]["delta_elpd"] is None
    assert by["model_b"]["delta_elpd"] < 0


def test_standalone_pfilter_runs_are_compare_candidates(tmp_path, monkeypatch):
    """camdl writes one [method] per config, so a particle-filter scoring run is
    a run folder of its own with no draws or estimate. The run list omits it;
    the Compare picker must still offer it, and /api/compare must resolve it."""
    store = tmp_path / "results" / "fits"
    for label, level in (("pf_pois-2dc18961", -2.0), ("pf_nb-c0ede1ae", -2.5)):
        _write_prequential(
            store / label / "pfilter-d7e5f501" / "seed_1-39ae1d69" / "prequential.json", level
        )
    found = {r.run_id: r.label for r in compare.discover_scored(store)}
    assert found == {"pf_pois-2dc18961": "pf_pois", "pf_nb-c0ede1ae": "pf_nb"}

    monkeypatch.setenv("CAMDL_SCOPE_STORE", str(store))
    from fastapi.testclient import TestClient
    from camdl_scope.api.app import app

    c = TestClient(app)
    got = c.get("/api/compare/candidates").json()
    assert {x["run_id"] for x in got["candidates"]} == set(found)
    assert got["n_without_prequential"] == 0
    assert c.get("/api/runs").json() == []  # the run list still omits them
    if shutil.which(compare.CAMDL_BIN) is None:
        pytest.skip("camdl not on PATH")
    r = c.get("/api/compare", params=[("runs", "pf_pois-2dc18961"), ("runs", "pf_nb-c0ede1ae")])
    assert r.status_code == 200, r.text
    assert {row["label"] for row in r.json()["rows"]} == {"pf_pois", "pf_nb"}


def test_json_caveats_and_row_warnings_become_notes():
    """camdl puts its advisories in named JSON fields, not stderr."""
    data = {
        "optimism_caveat": "scores are plug-in and in-sample",
        "data_mismatch": None,
        "se_caveat": "se is a normal approximation",
        "rows": [
            {"name": "a", "lag1_autocorr_delta_elpd": None, "warnings": []},
            {"name": "b", "lag1_autocorr_delta_elpd": 0.8425,
             "warnings": [{"kind": "starts_at_prior"}]},
        ],
    }
    assert compare.json_notes(data) == [
        "scores are plug-in and in-sample",
        "se is a normal approximation",
        "lag-1 autocorrelation of the per-step Δelpd vs baseline: b +0.84",
        "b: starts at prior",
    ]
