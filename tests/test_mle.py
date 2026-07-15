"""MLE ('scout') fit discovery + reader + endpoint — over a crafted store."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from camdl_watch import ingest
from camdl_watch.mle import read_mle


def _scout_run(store: Path, run: str = "opt-aa") -> Path:
    """A pure MLE run: fit.meta.json + fit.toml + one scout seed leaf with an
    mle_params.toml (θ̂) and chain_results.tsv (per-restart optima)."""
    run_dir = store / run
    seed = run_dir / "01-scout-hh" / "seed_1-hh"
    seed.mkdir(parents=True)
    (run_dir / "fit.meta.json").write_text(json.dumps({
        "model_path": "models/opt.camdl",
        "estimated": ["a", "b"],
    }))
    (run_dir / "fit.toml.original").write_text(
        '[stages.scout]\nalgorithm = "nl-sbplx"\nbackend = "ode"\n'
    )
    (seed / "mle_params.toml").write_text(
        "a = 1.5\nb = 2.5\n\n[provenance]\nlog_likelihood = -10.0\nstage = \"scout\"\n"
    )
    # 3 restarts: two converged (best a=1.5), one failed (sentinel).
    (seed / "chain_results.tsv").write_text(
        "chain\tloglik\tstatus\tn_evals\ta\tb\n"
        "1\t-10.0\txtol_reached\t500\t1.5\t2.5\n"
        "2\t-12.0\txtol_reached\t480\t1.2\t2.9\n"
        "3\t-1e100\txtol_reached\t100\t9.9\t9.9\n"
    )
    return run_dir


def test_read_mle_estimate_and_restarts(tmp_path):
    run_dir = _scout_run(tmp_path / "fits")
    seed = run_dir / "01-scout-hh" / "seed_1-hh"
    fit = read_mle(seed, ["a", "b"])
    assert fit is not None
    assert fit.loglik == -10.0
    assert fit.n_restarts == 3 and fit.n_converged == 2
    # Restarts are best-first; the sentinel one is last.
    assert [r.chain for r in fit.restarts] == [1, 2, 3]
    assert fit.restarts[-1].loglik <= -1e99
    # θ̂ + converged-restart spread (the failed restart's 9.9 is excluded).
    a = next(p for p in fit.params if p.name == "a")
    assert a.value == 1.5 and (a.restart_lo, a.restart_hi) == (1.2, 1.5)


def test_mle_run_is_discovered_and_kind_tagged(tmp_path):
    _scout_run(tmp_path / "fits")
    runs = ingest.discover_runs(tmp_path / "fits")
    assert len(runs) == 1
    meta = runs[0]
    assert meta.fit_kind == "mle"
    assert meta.algorithm == "nl-sbplx" and meta.backend.value == "ode"
    assert meta.chain_paths == {}  # no draws


def test_mle_endpoints(tmp_path, monkeypatch):
    store = tmp_path / "results" / "fits"
    store.mkdir(parents=True)
    _scout_run(store, "opt-bb")
    monkeypatch.setenv("CAMDL_WATCH_STORE", str(store))
    from camdl_watch.api.app import app

    c = TestClient(app)
    run = c.get("/api/runs").json()[0]
    assert run["fit_kind"] == "mle" and run["status"] == "done"

    body = c.get(f"/api/runs/{run['run_id']}/mle").json()
    assert body["loglik"] == -10.0
    assert body["n_restarts"] == 3 and body["n_converged"] == 2
    assert [p["name"] for p in body["params"]] == ["a", "b"]
    assert body["restarts"][0]["chain"] == 1  # best first

    # A posterior-only run 404s on /mle.
    assert c.get("/api/runs/nope/mle").status_code == 404
