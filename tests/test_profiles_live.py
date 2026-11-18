"""A running profile: pending grid values, partial values, progress and liveness,
read from camdl's per-invocation record (``invocations/<id>/invocation.json``)
and heartbeat (``progress.json``)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from camdl_scope.profiles import discover_profiles, load_profile

BASE = "sir-aa"
GRID = [0.018, 0.05, 0.10, 0.15]


def _dir(v: float) -> str:
    return f"leave_s_{v:.4f}-{int(v * 1e4):04d}"


def _store(tmp_path: Path) -> Path:
    store = tmp_path / "results" / "fits"
    store.mkdir(parents=True)
    return store


def _base(store: Path) -> Path:
    b = store.parent / "profiles" / BASE
    b.mkdir(parents=True, exist_ok=True)
    return b


def _start(store: Path, v: float, k: int, loglik: float) -> None:
    leaf = _base(store) / _dir(v) / "nlopt_sbplx_-mm" / "seed_1-ss" / f"start_{k}-kk"
    leaf.mkdir(parents=True)
    (leaf / "run.json").write_text(json.dumps({
        "kind": "profile_point",
        "levels": [{"name": "profile", "label": "sir"},
                   {"name": "point", "label": f"leave_s={v:.4f}"}],
        "inputs": {"best_loglik": loglik, "method": "nl-sbplx", "loglik_type": "ode_marginal"},
    }))
    (leaf / "mle.toml").write_text(f"[focal]\nleave_s = {v}\n\n[mle]\nbeta = 0.9\n")


def _invocation(store: Path, *, state: object = None, age: float = 0.0,
                n_starts: int = 10, inv_id: str = "1700000000-42") -> None:
    d = _base(store) / "invocations" / inv_id
    d.mkdir(parents=True)
    (d / "invocation.json").write_text(json.dumps({
        "schema": "camdl.profile-invocation/v1", "invocation_id": inv_id,
        "started_at": time.time() - 600, "label": "m4l_profile", "algorithm": "nl-sbplx",
        "focal": [{"name": "leave_s", "values": GRID}],
        "n_starts": n_starts, "seeds": [1],
        "points": [{"coords": {"leave_s": v}, "dir": _dir(v)} for v in GRID],
        "total_jobs": n_starts * len(GRID),
    }))
    if state is not None:
        (d / "progress.json").write_text(json.dumps({
            "updated_at": int(time.time() - age), "pid": 7, "state": state}))


def _mid_run(store: Path) -> None:
    """0.05 has no finished start; 0.018 has 1 of 10 (and the highest loglik so
    far); 0.10 and 0.15 are complete."""
    _start(store, 0.018, 0, -4620.0)
    for k in range(10):
        _start(store, 0.10, k, -4631.8 - 0.1 * k)
        _start(store, 0.15, k, -4628.2 - 0.01 * k)


RUNNING = {"running": {"phase": "profiling", "step": 21, "total": 40}}


def test_unstarted_value_is_pending_and_partial_value_is_marked(tmp_path):
    store = _store(tmp_path)
    _mid_run(store)
    _invocation(store, state=RUNNING)
    c = load_profile(store, BASE)
    assert c is not None
    assert [p.coords for p in c.pending] == [(0.05,)]
    assert c.pending[0].n_expected == 10
    by = {p.coords[0]: p for p in c.points}
    assert (by[0.018].n_starts, by[0.018].n_expected, by[0.018].complete) == (1, 10, False)
    assert by[0.10].complete and by[0.15].complete
    assert c.status() == "running" and c.provisional
    assert c.starts_progress() == (21, 40)


def test_mle_and_interval_ignore_partial_values(tmp_path):
    """0.018's single start has the highest loglik so far, but a partial value
    is only a lower bound; the MLE comes from the complete values."""
    store = _store(tmp_path)
    _mid_run(store)
    _invocation(store, state=RUNNING)
    c = load_profile(store, BASE)
    assert c is not None and c.mle is not None
    assert c.mle.coords == (0.15,)


def test_spread_of_finished_starts_is_reported(tmp_path):
    store = _store(tmp_path)
    _mid_run(store)
    _invocation(store, state=RUNNING)
    by = {p.coords[0]: p for p in load_profile(store, BASE).points}
    assert by[0.10].loglik_spread == pytest.approx(0.9)
    assert by[0.018].loglik_spread is None  # one start


def test_stale_heartbeat_reads_as_stalled_with_its_finished_starts(tmp_path):
    store = _store(tmp_path)
    _mid_run(store)
    _invocation(store, state=RUNNING, age=3600)
    c = load_profile(store, BASE)
    assert c.status() == "stalled"
    assert len(c.points) == 3 and len(c.pending) == 1


def test_profile_with_only_pending_values_is_listed(tmp_path):
    store = _store(tmp_path)
    _invocation(store, state={"running": {"phase": "profiling", "step": 0, "total": 40}})
    [c] = discover_profiles(store)
    assert c.points == [] and len(c.pending) == 4
    assert c.params == ("leave_s",) and c.mle is None and c.status() == "running"


def test_finished_profile_is_complete_and_done(tmp_path):
    store = _store(tmp_path)
    for v in GRID:
        for k in range(10):
            _start(store, v, k, -4630.0 + v * 10 - 0.01 * k)
    _invocation(store, state="done")
    c = load_profile(store, BASE)
    assert c.status() == "done" and not c.provisional and c.pending == []
    assert all(p.complete for p in c.points)
    assert c.starts_progress() == (40, 40)


def test_profile_without_an_invocation_record_reads_as_before(tmp_path):
    store = _store(tmp_path)
    _start(store, 0.10, 0, -10.0)
    _start(store, 0.15, 0, -9.0)
    c = load_profile(store, BASE)
    assert c.status() == "unknown" and not c.provisional
    assert c.starts_progress() is None
    assert all(p.complete and p.n_expected is None for p in c.points)
    assert c.mle.coords == (0.15,)


def test_invocations_dir_is_not_a_grid_value(tmp_path):
    store = _store(tmp_path)
    _mid_run(store)
    _invocation(store, state="done")
    c = load_profile(store, BASE)
    assert {p.coords[0] for p in c.points} == {0.018, 0.10, 0.15}


def test_unknown_invocation_schema_is_skipped(tmp_path):
    store = _store(tmp_path)
    _start(store, 0.10, 0, -10.0)
    d = _base(store) / "invocations" / "x"
    d.mkdir(parents=True)
    (d / "invocation.json").write_text(json.dumps({"schema": "camdl.profile-invocation/v9"}))
    c = load_profile(store, BASE)
    assert c.invocations == [] and c.status() == "unknown"


def test_endpoints_carry_status_counts_pending_and_partial(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _mid_run(store)
    _invocation(store, state=RUNNING)
    monkeypatch.setenv("CAMDL_SCOPE_STORE", str(store))
    from camdl_scope.api.app import app

    c = TestClient(app)
    [s] = c.get("/api/profiles").json()
    assert s["status"] == "running" and s["n_points"] == 4
    assert (s["n_starts_finished"], s["n_starts_expected"]) == (21, 40)
    d = c.get(f"/api/profiles/{BASE}").json()
    assert d["provisional"] is True and d["status"] == "running"
    assert d["pending"] == [{"coords": [0.05], "n_expected": 10}]
    partial = next(p for p in d["points"] if p["coords"] == [0.018])
    assert partial["complete"] is False and partial["n_expected"] == 10
    assert d["mle_coords"] == [0.15]
    assert d["n_estimate_points"] == 2  # 0.10 and 0.15; the partial 0.018 is not one
    assert d["heartbeat_at"] is not None


def test_no_interval_is_drawn_from_partial_values_alone(tmp_path):
    """Two values with 1 of 10 starts each, none complete: the best so far is
    reported, but no interval is computed from two lower bounds."""
    store = _store(tmp_path)
    _start(store, 0.018, 0, -4620.0)
    _start(store, 0.15, 0, -4630.0)
    _invocation(store, state=RUNNING)
    c = load_profile(store, BASE)
    assert c.estimate_points == []
    assert c.mle.coords == (0.018,) and c.provisional
    assert c.ci_bounds_1d() == (None, None)
