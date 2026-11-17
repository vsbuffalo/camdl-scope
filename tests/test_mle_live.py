"""An optimizer run is visible from its first heartbeat, and each start's result
shows as it finishes (camdl gh#1197: ``chain_results.tsv`` is created with its
header first and each start's row appended when it finishes, in completion
order; ``mle_params.toml`` is written only at the end)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from camdl_scope import ingest
from camdl_scope.assembly import build_run_overview
from camdl_scope.mle import read_mle
from camdl_scope.state import Status

HEADER = "chain\tloglik\tstatus\tn_evals\ta\tb\n"
# Rows in completion order, not chain order.
ROW_2 = "2\t-12.0\tmaxeval_reached\t400\t1.2\t2.9\n"
ROW_0 = "0\t-10.0\tmaxeval_reached\t400\t1.5\t2.5\n"
ROW_3 = "3\t-11.0\tftol_reached\t250\t1.4\tinf\n"
ROW_1 = "1\t-13.0\txtol_reached\t300\t1.1\t3.0\n"


def _folder(store: Path, name: str = "opt-aa") -> Path:
    run_dir = store / name
    run_dir.mkdir(parents=True)
    (run_dir / "fit.meta.json").write_text(json.dumps({
        "model_path": "models/opt.camdl", "estimated": ["a", "b"],
    }))
    (run_dir / "fit.toml.original").write_text(
        '[method]\nalgorithm = "nl-sbplx"\nbackend = "ode"\nchains = 4\n'
    )
    return run_dir


def _leaf(run_dir: Path, method: str = "nl-sbplx-97108e67", seed: str = "seed_1-aa",
          *, n_chains: int = 4, algorithm: str = "nl-sbplx") -> Path:
    leaf = run_dir / method / seed
    leaf.mkdir(parents=True)
    (leaf / "run.json").write_text(json.dumps({
        "kind": "fit_method", "status": "running", "run_id": f"rid-{seed}",
        "inputs": {
            "method": algorithm, "n_chains": n_chains,
            "algorithm": {"algorithm": algorithm, "backend": "ode", "chains": n_chains},
        },
    }))
    return leaf


def _beat(leaf: Path, *, age: float = 0.0, state: str = "running",
          step: int = 11871, total: int = 40000, phase: str = "optimizing") -> None:
    body: object = (
        {"running": {"phase": phase, "step": step, "total": total}}
        if state == "running" else state
    )
    (leaf / "progress.json").write_text(json.dumps({
        "updated_at": int(time.time() - age), "pid": 4242, "state": body,
    }))


def _finish(leaf: Path) -> None:
    """What camdl leaves at the end: the table rewritten in chain order, the
    point estimate written, the heartbeat marked done."""
    (leaf / "chain_results.tsv").write_text(HEADER + ROW_0 + ROW_1 + ROW_2 + ROW_3)
    (leaf / "mle_params.toml").write_text(
        "a = 1.5\nb = 2.5\n\n[provenance]\nlog_likelihood = -10.0\n"
    )
    _beat(leaf, state="done")


def _status(meta) -> Status:
    return build_run_overview(meta).status


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------


def test_heartbeat_only_optimizer_is_listed_as_running_mle(tmp_path):
    leaf = _leaf(_folder(tmp_path))
    _beat(leaf)
    runs = ingest.discover_runs(tmp_path, include_warming=True)
    assert len(runs) == 1
    meta = runs[0]
    assert meta.fit_kind == "mle" and meta.algorithm == "nl-sbplx"
    assert meta.run_id == "opt-aa"
    assert _status(meta) is Status.RUNNING

    fit = read_mle(leaf, ["a", "b"], n_expected=4)
    assert fit is not None and fit.provisional
    assert fit.n_restarts == 0 and fit.loglik is None
    assert all(p.value is None for p in fit.params)


def test_stale_heartbeat_with_no_rows_stays_hidden(tmp_path):
    leaf = _leaf(_folder(tmp_path))
    _beat(leaf, age=3600)
    (leaf / "chain_results.tsv").write_text(HEADER)
    assert ingest.discover_runs(tmp_path, include_warming=True) == []


def test_header_only_table_reads_as_no_starts_finished(tmp_path):
    leaf = _leaf(_folder(tmp_path))
    _beat(leaf)
    (leaf / "chain_results.tsv").write_text(HEADER)
    [meta] = ingest.discover_runs(tmp_path, include_warming=True)
    assert meta.fit_kind == "mle" and _status(meta) is Status.RUNNING
    fit = read_mle(leaf, ["a", "b"], n_expected=4)
    assert fit is not None and fit.provisional and fit.n_restarts == 0


def test_partial_rows_give_a_provisional_best_so_far(tmp_path):
    leaf = _leaf(_folder(tmp_path))
    _beat(leaf)
    # Completion order 2, 0, 3, plus a torn append that has not ended yet.
    (leaf / "chain_results.tsv").write_text(HEADER + ROW_2 + ROW_0 + ROW_3 + "1\t-13.0\txt")
    fit = read_mle(leaf, ["a", "b"], n_expected=4)
    assert fit is not None and fit.provisional and fit.n_expected == 4
    # The torn line is not a start; rows come back best first.
    assert [r.chain for r in fit.restarts] == [0, 3, 2]
    # The best finished start stands in for the estimate.
    assert fit.loglik == -10.0
    assert {p.name: p.value for p in fit.params} == {"a": 1.5, "b": 2.5}
    # A non-finite value is absent, not a number, and drops out of the spread.
    assert fit.restarts[1].values["b"] is None
    b = next(p for p in fit.params if p.name == "b")
    assert (b.restart_lo, b.restart_hi) == (2.5, 2.9)
    assert fit.n_maxeval == 2 and not fit.all_maxeval


def test_all_finished_starts_at_maxeval_is_flagged(tmp_path):
    leaf = _leaf(_folder(tmp_path))
    _beat(leaf)
    (leaf / "chain_results.tsv").write_text(HEADER + ROW_2 + ROW_0)
    fit = read_mle(leaf, ["a", "b"])
    assert fit is not None and fit.all_maxeval and fit.n_maxeval == 2


def test_stopped_run_keeps_its_finished_starts(tmp_path):
    """Killed partway: a stale heartbeat, some rows, no point estimate."""
    leaf = _leaf(_folder(tmp_path))
    _beat(leaf, age=3600)
    (leaf / "chain_results.tsv").write_text(HEADER + ROW_2 + ROW_0)
    [meta] = ingest.discover_runs(tmp_path, include_warming=True)
    assert meta.fit_kind == "mle"
    assert _status(meta) is Status.STALLED
    fit = read_mle(leaf, ["a", "b"])
    assert fit is not None and fit.provisional and fit.n_restarts == 2


def test_finished_leaf_is_done_and_matches_the_live_table(tmp_path):
    leaf = _leaf(_folder(tmp_path))
    _beat(leaf)
    (leaf / "chain_results.tsv").write_text(HEADER + ROW_2 + ROW_0 + ROW_3 + ROW_1)
    live = read_mle(leaf, ["a", "b"])
    _finish(leaf)
    [meta] = ingest.discover_runs(tmp_path, include_warming=True)
    assert meta.run_id == "opt-aa"  # same entry, no duplicate
    assert _status(meta) is Status.DONE
    done = read_mle(leaf, ["a", "b"])
    assert done is not None and live is not None
    assert not done.provisional and live.provisional
    # The chain-order rewrite gives the same per-start table.
    assert done.restarts == live.restarts
    assert done.loglik == -10.0


def test_live_rerun_beside_a_finished_optimizer_is_its_own_entry(tmp_path):
    run_dir = _folder(tmp_path)
    old = _leaf(run_dir, "nl-sbplx-11111111", "seed_1-aa")
    _finish(old)
    new = _leaf(run_dir, "nl-sbplx-22222222", "seed_2-bb")
    _beat(new)
    runs = ingest.discover_runs(tmp_path, include_warming=True)
    assert [m.run_id for m in runs] == ["opt-aa~seed_1-aa", "opt-aa~seed_2-bb"]
    assert [_status(m) for m in runs] == [Status.DONE, Status.RUNNING]
    assert [m.folder_default for m in runs] == [True, False]
    # The bare folder name still opens the folder's default.
    assert ingest.resolve_run(runs, "opt-aa") is runs[0]


def test_if2_heartbeat_only_leaf_is_an_optimizer(tmp_path):
    leaf = _leaf(_folder(tmp_path), "if2-34d72a87", algorithm="if2")
    _beat(leaf, phase="optimizing")
    [meta] = ingest.discover_runs(tmp_path, include_warming=True)
    assert meta.fit_kind == "mle" and meta.algorithm == "if2"


# --------------------------------------------------------------------------
# Samplers are unaffected, except that a live one is visible sooner
# --------------------------------------------------------------------------


def test_heartbeat_only_sampler_is_listed_with_no_chains(tmp_path):
    leaf = _leaf(_folder(tmp_path), "pgas-79d6825a", algorithm="pgas")
    _beat(leaf, phase="burn_in", step=3, total=500)
    [meta] = ingest.discover_runs(tmp_path, include_warming=True)
    assert meta.fit_kind == "posterior" and meta.chain_paths == {}
    assert _status(meta) is Status.WARMING
    # Without include_warming it is not listed.
    assert ingest.discover_runs(tmp_path) == []


def test_sampler_with_traces_classifies_as_before(tmp_path):
    leaf = _leaf(_folder(tmp_path), "pgas-79d6825a", algorithm="pgas")
    _beat(leaf, phase="sampling", step=20, total=500)
    (leaf / "chain_0").mkdir()
    (leaf / "chain_0" / "trace.tsv").write_text("iter\ta\tb\n1\t0.1\t0.2\n")
    [meta] = ingest.discover_runs(tmp_path, include_warming=True)
    assert meta.fit_kind == "posterior" and list(meta.chain_paths) == [0]
    assert _status(meta) is Status.RUNNING


def test_finished_posterior_is_not_reclassified_by_a_running_heartbeat(tmp_path):
    """A leaf with draws.tsv keeps today's classification whatever its heartbeat."""
    leaf = _leaf(_folder(tmp_path), "pgas-79d6825a", algorithm="pgas")
    (leaf / "chain_0").mkdir()
    (leaf / "chain_0" / "trace.tsv").write_text("iter\ta\tb\n1\t0.1\t0.2\n")
    (leaf / "draws.tsv").write_text("a\tb\n0.1\t0.2\n")
    _beat(leaf, phase="sampling")
    [meta] = ingest.discover_runs(tmp_path, include_warming=True)
    assert meta.fit_kind == "posterior" and list(meta.chain_paths) == [0]


# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = tmp_path / "results" / "fits"
    store.mkdir(parents=True)
    monkeypatch.setenv("CAMDL_SCOPE_STORE", str(store))
    from camdl_scope.api.app import app

    return store, TestClient(app)


def test_running_optimizer_endpoints(client):
    store, c = client
    leaf = _leaf(_folder(store))
    _beat(leaf)
    (leaf / "chain_results.tsv").write_text(HEADER + ROW_2 + ROW_0)

    [run] = c.get("/api/runs").json()
    assert run["fit_kind"] == "mle" and run["status"] == "running"
    assert run["progress"]["phase"] == "optimizing"
    assert (run["progress"]["step"], run["progress"]["total"]) == (11871, 40000)

    body = c.get(f"/api/runs/{run['run_id']}/mle").json()
    assert body["provisional"] is True and body["n_expected"] == 4
    assert body["n_restarts"] == 2 and body["loglik"] == -10.0
    assert body["n_maxeval"] == 2 and body["all_maxeval"] is True
    assert body["restarts"][0]["values"] == {"a": 1.5, "b": 2.5}
    assert body["restarts"][0]["hit_maxeval"] is True

    _finish(leaf)
    [run] = c.get("/api/runs").json()
    assert run["status"] == "done"
    body = c.get(f"/api/runs/{run['run_id']}/mle").json()
    assert body["provisional"] is False and body["n_restarts"] == 4
