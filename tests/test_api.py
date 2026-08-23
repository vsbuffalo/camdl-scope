"""API tests — the typed JSON projection of the run store.

Each test points ``CAMDL_WATCH_STORE`` at a freshly-built golden store *before*
importing the FastAPI app (so ``current_store()`` reads it) and drives the app
through Starlette's ``TestClient``. The golden store carries the full
``docs``/``schema`` sidecar, so these assert the doc-labelled posterior contract
the first frontend screen (a forest plot) depends on.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.fixtures.make_golden_store import (
    PARAM_NAMES,
    POSTERIOR_DIR,
    RUN_DIR,
    SEED_DIR,
    build,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient over the app, pointed at a golden store in ``tmp_path``."""
    build(tmp_path)
    monkeypatch.setenv("CAMDL_WATCH_STORE", str(tmp_path))
    # current_store() reads the env fresh per request, so importing once is fine.
    from camdl_watch.api.app import app

    return TestClient(app)


def test_health_sees_the_store(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["runs"] == 1


def test_list_runs_one_documented_run(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    runs = r.json()
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == RUN_DIR
    # target_sweeps rides on the summary (from fit.toml) so the run bar can draw a
    # completion fraction against max_iter even without a progress.json heartbeat.
    assert run["target_sweeps"] == 600  # FIT_TOML: sweeps = 600
    assert run["max_iter"] is not None
    assert run["has_docs"] is True
    assert run["n_params"] == 6
    assert run["n_chains"] == 2
    # The real chain ids, ascending — the chain selector keys off these rather
    # than synthesizing 0..n-1 (a store may number chains from 1).
    assert run["chain_ids"] == [0, 1]
    assert run["status"] == "done"
    assert run["algorithm"] == "pgas"
    assert run["backend"] == "chain_binomial"


def test_run_detail_schema_streams_and_dimensions(client):
    runs = client.get("/api/runs").json()
    run_id = runs[0]["run_id"]
    r = client.get(f"/api/runs/{run_id}")
    assert r.status_code == 200
    detail = r.json()

    streams = {s["name"]: s for s in detail["streams"]}
    assert "cases" in streams
    assert streams["cases"]["index_dims"] == ["patch"]
    assert streams["cases"]["likelihood"] == "neg_binomial"

    dims = {d["name"]: d for d in detail["dimensions"]}
    assert "patch" in dims
    assert dims["patch"]["levels"] == ["Bo", "Bombali"]

    assert detail["available_streams"] == ["cases"]
    assert detail["target_sweeps"] == 600
    assert detail["estimated"] == [
        "beta", "sigma", "gamma", "rho", "k_raw_Bo", "k_raw_Bombali",
    ]


def test_run_detail_404(client):
    r = client.get("/api/runs/does-not-exist")
    assert r.status_code == 404


def test_posterior_doc_labelled_params(client):
    runs = client.get("/api/runs").json()
    run_id = runs[0]["run_id"]
    r = client.get(f"/api/runs/{run_id}/posterior", params={"warmup_pct": 50})
    assert r.status_code == 200
    body = r.json()

    assert body["warmup_pct"] == 50
    assert body["n_tail"] > 0
    params = {p["name"]: p for p in body["params"]}
    estimated = [p["name"] for p in body["params"] if not p["is_objective"]]
    objectives = [p["name"] for p in body["params"] if p["is_objective"]]
    assert estimated == [
        "beta", "sigma", "gamma", "rho", "k_raw_Bo", "k_raw_Bombali",
    ]
    # The pooled objectives close the forest (Stan's lp__) — flagged, prior-free,
    # and sourced as "derived", never mixed in with the estimands.
    assert objectives == ["log_posterior", "log_likelihood"]
    for o in objectives:
        assert params[o]["prior"] is None
        assert params[o]["symbol"] is None
        assert params[o]["source"] == "derived"

    beta = params["beta"]
    assert beta["symbol"] == "β"
    assert beta["reference"] == "Anderson & May 1991"
    assert "transmission" in beta["description"]
    assert beta["prior"] == "LogNormal(μ=-0.6, σ=0.4)"
    assert beta["source"] == "fit_toml"

    # Expanded coordinate resolves to its base (k_raw) doc block.
    k_bo = params["k_raw_Bo"]
    assert k_bo["symbol"] == "k"
    assert k_bo["prior"] == "Normal(μ=0, σ=1)"
    assert k_bo["bounds"] == [-5.0, 5.0]

    # rho is documented but has no @ref.
    assert params["rho"]["symbol"] == "ρ"
    assert params["rho"]["reference"] is None
    assert params["rho"]["prior"] == "Beta(α=3, β=6)"

    # Every row (params and objectives) ships finite, ordered quantiles.
    for p in body["params"]:
        assert p["q05"] <= p["q25"] <= p["q50"] <= p["q75"] <= p["q95"]
        for key in ("mean", "sd", "q05", "q50", "q95"):
            assert isinstance(p[key], float)
    # Estimated params carry R̂/ESS from camdl's authoritative summary.
    for p in body["params"]:
        if not p["is_objective"]:
            assert p["rhat"] is not None
            assert p["ess"] is not None


def test_posterior_404(client):
    r = client.get("/api/runs/nope/posterior")
    assert r.status_code == 404


# --- Source tab: leaf-archived model source (gh#353) -------------------------


def test_source_prefers_the_leaf_archived_model(client, tmp_path):
    """When the fit leaf carries ``model.camdl.original`` (gh#353), the Source
    tab serves that self-contained copy — not a checkout-relative live read —
    and marks its origin ``leaf``. The golden fit's recorded ``model_path`` is a
    non-existent ``.ir.json``, so a live read would miss; the leaf copy wins."""
    model_src = "compartments { S, I, R }\nparameters { beta ~ log_normal() }\n"
    (tmp_path / RUN_DIR / "model.camdl.original").write_text(model_src)

    r = client.get(f"/api/runs/{RUN_DIR}/source")
    assert r.status_code == 200
    body = r.json()

    model = body["model"]
    assert model["present"] is True
    assert model["origin"] == "leaf"
    assert model["text"] == model_src
    assert model["html"]  # highlighted, non-empty

    # fit.toml is always leaf-archived.
    assert body["fit_toml"]["present"] is True
    assert body["fit_toml"]["origin"] == "leaf"


def test_source_falls_back_to_live_when_no_leaf_copy(client):
    """Older fits predate source archiving: with no ``model.camdl.original`` in
    the leaf, the model falls back to a live read of the recorded path. The
    golden fit points at a missing ``.ir.json``, so it reads ``present: false``
    with origin ``live`` — never a leaf hit."""
    r = client.get(f"/api/runs/{RUN_DIR}/source")
    assert r.status_code == 200
    model = r.json()["model"]
    assert model["origin"] == "live"
    assert model["present"] is False


def test_source_404(client):
    r = client.get("/api/runs/nope/source")
    assert r.status_code == 404


# --- Chain selector: drop a stuck chain from draws/traces/diagnostics --------


def test_draws_chain_filter_pools_only_selected(client):
    """`?chains=0` pools draws from chain 0 only — every row carries chain 0."""
    full = client.get(f"/api/runs/{RUN_DIR}/draws?warmup_pct=0").json()
    assert set(full["chain"]) == {0, 1}  # golden store has 2 chains

    one = client.get(f"/api/runs/{RUN_DIR}/draws?warmup_pct=0&chains=0").json()
    assert set(one["chain"]) == {0}
    assert one["n_draws"] < full["n_draws"]


def test_traces_chain_filter_drops_series(client):
    """`?chains=1` leaves each parameter with a single chain-1 series."""
    r = client.get(f"/api/runs/{RUN_DIR}/traces?chains=1")
    assert r.status_code == 200
    for pt in r.json()["traces"]:
        assert [s["chain"] for s in pt["series"]] == [1]


def test_diagnostics_chain_filter_recomputes_live(client):
    """Dropping to one chain forces the live arviz path (camdl's all-chain
    summary can't be recomputed for a subset), so `source` is `live`, `n_chains`
    is 1, and R̂ — undefined for a single chain — comes back null."""
    r = client.get(f"/api/runs/{RUN_DIR}/diagnostics?chains=0")
    assert r.status_code == 200
    body = r.json()
    assert body["n_chains"] == 1
    assert body["source"] == "live"
    assert all(p["rhat"] is None for p in body["params"])


def test_chain_filter_unknown_ids_fall_back_to_all(client):
    """A selection that would keep nothing (all ids unknown) is treated as
    "all" — a request can never blank the run."""
    full = client.get(f"/api/runs/{RUN_DIR}/draws?warmup_pct=0").json()
    bogus = client.get(f"/api/runs/{RUN_DIR}/draws?warmup_pct=0&chains=99").json()
    assert bogus["n_draws"] == full["n_draws"]
    assert set(bogus["chain"]) == {0, 1}


# --- A warming chain must not suppress diagnostics for the ready ones --------


def _chain_buffer(cid: int, n: int):
    """A minimal ChainBuffer carrying `n` sweeps (0..n-1) for _drop_warming."""
    from camdl_watch.state import ChainBuffer

    buf = ChainBuffer(cid=cid, path=Path("x"))
    buf.iters = np.arange(n, dtype=np.int64)
    return buf


def test_drop_warming_chains_prunes_empty_and_short():
    """Chains with fewer than the arviz floor of post-cutoff draws are dropped;
    the ready ones remain, so a rectangular array can still be built."""
    from camdl_watch.api.routes import _drop_warming_chains

    rs = SimpleNamespace(
        chains={0: _chain_buffer(0, 500), 1: _chain_buffer(1, 500),
                2: _chain_buffer(2, 500), 3: _chain_buffer(3, 0)},
    )
    dropped = _drop_warming_chains(rs, cutoff=250)
    assert dropped == 1
    assert set(rs.chains) == {0, 1, 2}


def test_drop_warming_chains_noop_when_none_ready():
    """If no chain clears the floor, leave the run untouched so the usual
    "no draws yet" path applies rather than blanking it."""
    from camdl_watch.api.routes import _drop_warming_chains

    rs = SimpleNamespace(chains={0: _chain_buffer(0, 2), 1: _chain_buffer(1, 2)})
    assert _drop_warming_chains(rs, cutoff=0) == 0  # 2 draws each < floor of 4
    assert set(rs.chains) == {0, 1}


def test_diagnostics_survive_a_warming_chain(client, tmp_path):
    """A run with 2 full chains and a 3rd that has produced no draws (header
    only) must still return R̂/ESS — computed over the 2 ready chains — instead
    of the empty ``n_tail=0`` the whole-run short-circuit used to yield. The
    warming chain is reported via ``n_chains_warming``."""
    _add_warming_chain(tmp_path)

    body = client.get(f"/api/runs/{RUN_DIR}/diagnostics?warmup_pct=50").json()
    assert body["n_chains"] == 2  # the two chains with draws
    assert body["source"] == "live"  # a dropped chain forces the live path
    assert body["n_tail"] > 0
    assert len(body["params"]) > 0
    assert any(p["rhat"] is not None for p in body["params"])
    # The golden store's run has stopped, so a chain that never wrote a draw is
    # DEAD, not warming — reporting it as warming tells the reader to wait for
    # draws that will never arrive (see _dead_chain_ids).
    assert body["n_chains_dead"] == 1
    assert body["dead_chain_ids"] == [2]
    assert body["n_chains_warming"] == 0


def test_zero_draw_chain_is_warming_not_dead_while_the_run_is_live(
    client, tmp_path, monkeypatch
):
    """The same header-only chain in a RUNNING fit is still starting up. Only
    the run's status separates the two readings."""
    from camdl_watch.api import routes as routes_mod
    from camdl_watch.state import Status

    _add_warming_chain(tmp_path)
    monkeypatch.setattr(
        routes_mod, "_dead_chain_ids",
        lambda rs: [] if rs.status in (Status.RUNNING, Status.WARMING)
        else sorted(c for c, b in rs.chains.items() if b.n == 0),
    )
    body = client.get(f"/api/runs/{RUN_DIR}/diagnostics?warmup_pct=50").json()
    assert body["n_chains_dead"] + body["n_chains_warming"] == 1


def _add_warming_chain(tmp_path):
    """Add a header-only (zero-draw) 3rd chain to the golden run — the reader
    leaves its ``aux`` empty, reproducing a chain still in burn-in."""
    header = "\t".join(["sweep", *PARAM_NAMES, "log_likelihood", "log_posterior"])
    warming = tmp_path / RUN_DIR / POSTERIOR_DIR / SEED_DIR / "chain_2" / "trace.tsv"
    warming.parent.mkdir(parents=True, exist_ok=True)
    warming.write_text(header + "\n")


def test_draws_objectives_survive_a_warming_chain(client, tmp_path):
    """A warming chain carries no aux columns, so gating objectives on *every*
    chain used to drop log_posterior/log_likelihood from the pair plot entirely.
    They must stay available — gated only on the chains that produced draws."""
    _add_warming_chain(tmp_path)
    body = client.get(f"/api/runs/{RUN_DIR}/draws?warmup_pct=50").json()
    assert body["objectives"] == ["log_posterior", "log_likelihood"]
    # And the pooled objective columns are populated (from the 2 real chains).
    assert len(body["draws"]["log_posterior"]) == body["n_draws"] > 0


# --- Upstream ESS/iteration + ESS/second (thinning-invariant) ----------------


def test_efficiency_metrics_formula_and_thinning_invariance():
    """`_efficiency_metrics` mirrors camdl's `fit summary`: min-param ESS over
    (n_samples × thin) and over wall-clock, keyed off the slowest parameter — and
    invariant to thinning ((500 draws, thin 1) == (50 draws, thin 10))."""
    from camdl_watch.api.routes import _efficiency_metrics
    from camdl_watch.state import ChainSummary

    def summ(thin, wall):
        return ChainSummary(
            stage="posterior", n_chains=2, rhat={},
            ess={"beta": 145.0, "sigma": 300.0},  # min ESS is 145
            ess_per_chain={}, thin=thin, wall_time_secs=wall,
        )

    epi, eps = _efficiency_metrics(summ(1, 11.8), n_samples=500)
    assert epi == pytest.approx(145.0 / 500)  # 0.290 per raw iteration
    assert eps == pytest.approx(145.0 / 11.8)  # 12.29 per second

    # 50 kept draws × thin 10 == the SAME 500 raw iterations → identical ESS/iter.
    epi_thinned, _ = _efficiency_metrics(summ(10, 5.0), n_samples=50)
    assert epi_thinned == pytest.approx(epi)

    # No summary, no usable ESS, or no wall-clock → graceful None.
    assert _efficiency_metrics(None, 500) == (None, None)
    empty = ChainSummary(stage="", n_chains=0, rhat={}, ess={}, ess_per_chain={})
    assert _efficiency_metrics(empty, 500) == (None, None)
    _, eps_nowall = _efficiency_metrics(summ(1, None), n_samples=500)
    assert eps_nowall is None


def test_diagnostics_report_ess_per_iter_from_summary(client):
    """The golden run's authoritative summary yields a run-level ESS/iteration
    (no run.json → ESS/second stays None)."""
    body = client.get(f"/api/runs/{RUN_DIR}/diagnostics?warmup_pct=50").json()
    assert body["ess_per_iter"] is not None and body["ess_per_iter"] > 0
    assert body["ess_per_sec"] is None  # golden fixture writes no run.json


def test_diagnostics_ess_per_iter_honors_thin_and_walltime(client, tmp_path):
    """Reading `thin` from the summary and `wall_time_seconds` from run.json:
    thinning ×10 makes ESS/iteration 10× smaller (same raw iterations per kept
    draw), and ESS/second appears once wall-clock is recorded."""
    import json as _json

    seed = tmp_path / RUN_DIR / POSTERIOR_DIR / SEED_DIR
    summ_path = seed / "pgas_summary.json"
    summary = _json.loads(summ_path.read_text())

    summary["thin"] = 1
    summ_path.write_text(_json.dumps(summary))
    base = client.get(f"/api/runs/{RUN_DIR}/diagnostics").json()["ess_per_iter"]

    summary["thin"] = 10
    summ_path.write_text(_json.dumps(summary))
    (seed / "run.json").write_text(
        _json.dumps({"kind": "fit_stage", "inputs": {"wall_time_seconds": 8.0}})
    )
    thinned = client.get(f"/api/runs/{RUN_DIR}/diagnostics").json()

    assert thinned["ess_per_iter"] == pytest.approx(base / 10)
    assert thinned["ess_per_sec"] is not None and thinned["ess_per_sec"] > 0


def test_diagnostics_live_ess_per_iter_while_running(client, tmp_path):
    """A still-sampling run (no authoritative summary) still reports an
    ESS/iteration — a live arviz estimate over the post-warm-up tail — so the
    efficiency strip isn't blank during monitoring. ESS/second stays absent (no
    final wall-clock)."""
    seed = tmp_path / RUN_DIR / POSTERIOR_DIR / SEED_DIR
    (seed / "pgas_summary.json").unlink()  # drop the summary → looks live
    (seed / "diagnostics.json").unlink(missing_ok=True)

    body = client.get(f"/api/runs/{RUN_DIR}/diagnostics?warmup_pct=50").json()
    assert body["source"] == "live"
    assert body["ess_per_iter"] is not None and body["ess_per_iter"] > 0
    assert body["ess_per_sec"] is None


def test_diagnostics_read_nuts_summary(client, tmp_path):
    """The path is method-agnostic: nuts writes `nuts_summary.json` with the same
    rhat/ess/thin keys, and the watcher must read it (it previously only looked
    for pgas/pmmh). Renaming the golden summary is enough to exercise it."""
    seed = tmp_path / RUN_DIR / POSTERIOR_DIR / SEED_DIR
    (seed / "pgas_summary.json").rename(seed / "nuts_summary.json")

    body = client.get(f"/api/runs/{RUN_DIR}/diagnostics?warmup_pct=50").json()
    assert body["source"] == "camdl"  # summary found → authoritative
    assert body["ess_per_iter"] is not None and body["ess_per_iter"] > 0


def test_efficiency_is_withheld_when_an_assessed_param_has_no_ess():
    """camdl-scope#4: filtering out a null ESS took the minimum over the
    CONVERGED parameters only, so the metric IMPROVED as the fit got worse. A
    parameter with a finite R̂ and no ESS means "chains disagree" — the blank is
    the diagnosis, and the minimum is unreportable."""
    from camdl_watch.api.routes import _efficiency_metrics, _min_ess
    from camdl_watch.state import ChainSummary

    # tau was assessed (R̂ 2.6) and has no ESS -> unreportable, tau named.
    bad = ChainSummary(
        stage="pgas", n_chains=4,
        rhat={"beta": 1.01, "tau": 2.6},
        ess={"beta": 559.0, "tau": None},
        ess_per_chain={}, thin=1, wall_time_secs=10.0,
    )
    assert _min_ess(bad) == (None, ["tau"])
    assert _efficiency_metrics(bad, 500) == (None, None)

    # The same fit improved: tau now reports, so the minimum is stateable — and
    # it is WORSE than the old filtered number, which is the whole point.
    good = ChainSummary(
        stage="pgas", n_chains=4,
        rhat={"beta": 1.01, "tau": 1.02},
        ess={"beta": 559.0, "tau": 73.0},
        ess_per_chain={}, thin=1, wall_time_secs=10.0,
    )
    min_ess, missing = _min_ess(good)
    assert missing == [] and min_ess == pytest.approx(73.0)


def test_unassessed_param_does_not_withhold_but_still_bounds_the_minimum():
    """A parameter with no finite R̂ was never assessable across chains (a
    constant column, or an excluded-chains view), so it must not trigger the
    withholding — otherwise every filtered view loses its efficiency line."""
    from camdl_watch.api.routes import _min_ess
    from camdl_watch.state import ChainSummary

    s = ChainSummary(
        stage="pgas", n_chains=1,
        rhat={"beta": 1.01, "const": None},
        ess={"beta": 559.0, "const": 12.0},
        ess_per_chain={}, thin=1,
    )
    min_ess, missing = _min_ess(s)
    assert missing == []           # `const` has no R̂ -> not evidence of disagreement
    assert min_ess == pytest.approx(12.0)  # but its ESS still bounds the run
