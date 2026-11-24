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
from tests.storekit import StoreKit


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


def test_find_prequential_reads_the_runs_own_folder_only(tmp_path):
    kit = StoreKit.create(tmp_path)
    leaf = kit.fit_run(kit.problem(), algorithm="pfilter")
    target = leaf / "prequential.json"
    _write_prequential(target, level=-2.0)
    assert compare.find_prequential(leaf) == target
    # A score inside one of the run's results (a prediction) is not the run's.
    other = kit.fit_run(kit.problem("seir"), algorithm="pfilter")
    _write_prequential(other / "predict" / "x" / "prequential.json", level=-2.0)
    assert compare.find_prequential(other) is None


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
    store = tmp_path / "camdl-store"
    kit = StoreKit.create(store)
    scoring = {}
    for model, level in (("pf_pois", -2.0), ("pf_nb", -2.5)):
        problem = kit.problem(model)
        leaf = kit.fit_run(problem, algorithm="pfilter", config=f"{model}.toml")
        _write_prequential(leaf / "prequential.json", level)
        scoring[f"{problem.name}~{leaf.name}"] = model
    # A listed posterior run with a score of its own, and one without.
    scored_mh = kit.fit_run(kit.problem("sir_mh"), config="sir_mh.toml")
    _write_prequential(scored_mh / "prequential.json", -2.2)
    kit.fit_run(kit.problem("sir_plain"), config="sir_plain.toml")

    found = {r.run_id: r.label for r in compare.discover_scored(store)}
    assert found == scoring

    monkeypatch.setenv("CAMDL_STORE", str(store))
    from fastapi.testclient import TestClient
    from camdl_scope.api.app import app

    c = TestClient(app)
    listed = {r["run_id"] for r in c.get("/api/runs").json()}
    # The run list omits the scoring runs and shows the two posterior runs.
    assert len(listed) == 2 and not (listed & set(scoring))
    mh_id = scored_mh.parents[2].name  # its problem folder, which holds one run
    got = c.get("/api/compare/candidates").json()
    assert {x["run_id"] for x in got["candidates"]} == {*scoring, mh_id}
    assert got["n_without_prequential"] == 1
    if shutil.which(compare.CAMDL_BIN) is None:
        pytest.skip("camdl not on PATH")
    r = c.get("/api/compare", params=[("runs", rid) for rid in scoring])
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


def test_notes_name_runs_by_label_unless_two_share_one():
    from camdl_scope.api.routes import _name_runs

    labels = {"sir-aa~seed_1-bb": "pf_pois", "sir_nb-cc~seed_1-dd": "pf_nb"}
    assert _name_runs("sir_nb-cc~seed_1-dd: starts at prior", labels) == "pf_nb: starts at prior"
    same = {"sir-aa~seed_1-bb": "pf", "sir-aa~seed_2-ee": "pf"}
    # Two runs called "pf" would make the note ambiguous; the ids stay.
    assert _name_runs("sir-aa~seed_2-ee: starts at prior", same) == "sir-aa~seed_2-ee: starts at prior"
