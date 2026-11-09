"""MLE ('scout') fit discovery + reader + endpoint — over a crafted store."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from camdl_scope import ingest
from camdl_scope.mle import read_mle


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
    monkeypatch.setenv("CAMDL_SCOPE_STORE", str(store))
    from camdl_scope.api.app import app

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


def test_restart_table_reads_the_current_filename_and_columns(tmp_path: Path):
    """camdl renamed ``chain_results.tsv`` to ``chain_evaluations.tsv``, gave it
    a leading ``# <version>`` stamp, and replaced NLopt's ``status``/``n_evals``
    with IF2's ``se``/``ess_mean``/``ess_min``. Reading only the old name left a
    current MLE fit reporting zero restarts and no spread — a point estimate
    with nothing to say about whether the optimizer found the mode."""
    seed = tmp_path / "if2-802e7013" / "seed_1-06cbd6b3"
    seed.mkdir(parents=True)
    (seed / "mle_params.toml").write_text(
        "beta = 0.6148\ngamma = 0.2388\n\n[provenance]\nlog_likelihood = -543.4\n"
    )
    (seed / "chain_evaluations.tsv").write_text(
        "# 0.1.0+57c607c1 (2026-09-10)\n"
        "chain\tloglik\tse\tess_mean\tess_min\tbeta\tgamma\n"
        "1\t-543.4\t0.99\t2986.9\t12.1\t0.6148\t0.2388\n"
        "2\t-547.2\t0.98\t3359.7\t169.4\t0.7508\t0.2762\n"
        "3\t-1e100\t0.00\t0.0\t0.0\t9.9\t9.9\n"
    )

    fit = read_mle(seed, ["beta", "gamma"])
    assert fit is not None
    assert fit.n_restarts == 3 and fit.n_converged == 2
    # The columns IF2 does not write are simply absent, not a parse failure.
    assert fit.restarts[0].status == "" and fit.restarts[0].n_evals == 0
    beta = next(p for p in fit.params if p.name == "beta")
    assert beta.value == 0.6148
    assert (beta.restart_lo, beta.restart_hi) == (0.6148, 0.7508)
