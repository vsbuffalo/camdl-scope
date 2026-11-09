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
