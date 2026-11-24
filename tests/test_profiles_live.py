"""A profile command in progress: pending grid values, partial values, starts
progress and liveness, read from its invocation record (``.invocations/<seq>.json``),
its points' ``run.json`` ``invocation``, its owner's beat file and its
progress snapshot (``.invocations/<seq>.progress.json``)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from camdl_scope.profiles import discover_profiles, load_profile
from tests.storekit import StoreKit

GRID = [0.018, 0.05, 0.10, 0.15]
SWEEP = [{"name": "leave_s", "grid": {"list": GRID}}]
N_STARTS = 10


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


def _progress(age: float = 0.0, step: int = 21) -> dict:
    return {"at": _iso(time.time() - age), "phase": "profiling", "step": step, "total": 40}


def _start(v: float, loglik: float) -> dict:
    return {"focal": {"leave_s": v}, "loglik": loglik, "mle": {"beta": 0.9}}


def _mid_run_points() -> list[dict]:
    """0.05 has no finished start; 0.018 has 1 of 10 (and the highest loglik so
    far); 0.10 and 0.15 are complete."""
    return ([_start(0.018, -4620.0)]
            + [_start(0.10, -4631.8 - 0.1 * k) for k in range(N_STARTS)]
            + [_start(0.15, -4628.2 - 0.01 * k) for k in range(N_STARTS)])


def _mid_run(kit: StoreKit, **kw) -> int:
    kw.setdefault("progress", _progress())
    return kit.profile(SWEEP, _mid_run_points(), n_starts=N_STARTS, status="running",
                       label="m4l_profile", **kw)


def test_unstarted_value_is_pending_and_partial_value_is_marked(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = _mid_run(kit)
    c = load_profile(kit.root, str(seq))
    assert c is not None
    assert [p.coords for p in c.pending] == [(0.05,)]
    assert c.pending[0].n_expected == N_STARTS
    by = {p.coords[0]: p for p in c.points}
    assert (by[0.018].n_starts, by[0.018].n_expected, by[0.018].complete) == (1, 10, False)
    assert by[0.10].complete and by[0.15].complete
    assert c.status() == "running" and c.provisional
    assert c.starts_progress() == (21, 40)


def test_mle_and_interval_ignore_partial_values(tmp_path):
    """0.018's single start has the highest loglik so far, but a partial value
    is only a lower bound; the MLE comes from the complete values."""
    kit = StoreKit.create(tmp_path / "camdl-store")
    c = load_profile(kit.root, str(_mid_run(kit)))
    assert [p.coords for p in c.estimate_points] == [(0.10,), (0.15,)]
    assert c.mle.coords == (0.15,)
    # The interval is read off 0.10 and 0.15 only: 0.10 lies 3.6 below the
    # MLE, so the lower limit is interpolated on [0.10, 0.15]; the upper side
    # is open because the grid's complete values end at the MLE.
    thresh = -4628.2 - 1.9207
    lo, hi = c.ci_bounds_1d()
    assert lo == pytest.approx(0.10 + (thresh + 4631.8) / (3.6) * 0.05, abs=1e-9)
    assert hi is None


def test_spread_of_finished_starts_is_reported(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    by = {p.coords[0]: p for p in load_profile(kit.root, str(_mid_run(kit))).points}
    assert by[0.10].loglik_spread == pytest.approx(0.9)
    assert by[0.15].loglik_spread == pytest.approx(0.09)
    assert by[0.018].loglik_spread is None  # one start


def test_seeds_multiply_the_expected_starts(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = kit.profile(SWEEP, [_start(0.10, -1.0), _start(0.10, -2.0)], n_starts=2,
                      seeds=[1, 2, 3], status="running", progress=_progress())
    c = load_profile(kit.root, str(seq))
    [p] = c.points
    assert (p.n_starts, p.n_expected, p.complete) == (2, 6, False)
    assert [q.n_expected for q in c.pending] == [6, 6, 6]
    assert c.starts_progress() == (2, 24)


def test_dead_owner_reads_as_stalled_with_its_finished_starts(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    c = load_profile(kit.root, str(_mid_run(kit, alive=False)))
    assert c.status() == "stalled"
    assert len(c.points) == 3 and len(c.pending) == 1
    assert c.starts_progress() == (21, 40)


def test_interrupted_command_reads_as_stalled(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = kit.profile(SWEEP, _mid_run_points(), n_starts=N_STARTS, status="interrupted")
    c = load_profile(kit.root, str(seq))
    assert c.status() == "stalled" and c.provisional
    assert {p.coords[0] for p in c.points} == {0.018, 0.10, 0.15}


def _move_owner_to_another_host(kit: StoreKit, seq: int) -> None:
    """Rewrite the command's owner as a process on another host, with its beat
    file present: this reader cannot see that process, so the progress
    snapshot's freshness decides. (StoreKit's owners are all on this host.)"""
    path = kit.root / ".invocations" / f"{seq}.json"
    rec = json.loads(path.read_text())
    rec["owner"] = {"host": "elsewhere.example", "pid": 4242, "started": "2026-11-24T00:00:00.000Z"}
    path.write_text(json.dumps(rec, indent=1))
    (kit.root / ".owners" / "elsewhere.example-4242.json").write_text(
        json.dumps({"format": 1, **rec["owner"], "beat": 1}))


@pytest.mark.parametrize(("age", "status"), [(1.0, "running"), (3600.0, "stalled")])
def test_owner_on_another_host_is_judged_by_its_progress(tmp_path, age, status):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = _mid_run(kit, progress=_progress(age=age))
    _move_owner_to_another_host(kit, seq)
    assert load_profile(kit.root, str(seq)).status() == status


def test_profile_with_only_pending_values_is_listed(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    kit.profile(SWEEP, [], n_starts=N_STARTS, status="running", progress=_progress(step=0))
    [c] = discover_profiles(kit.root)
    assert c.points == [] and [p.coords for p in c.pending] == [(v,) for v in GRID]
    assert c.params == ("leave_s",) and c.mle is None and c.status() == "running"
    assert c.starts_progress() == (0, 40)


def test_finished_profile_is_complete_and_done(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = kit.profile(SWEEP, [_start(v, -4630.0 + v * 10 - 0.01 * k)
                              for v in GRID for k in range(N_STARTS)], n_starts=N_STARTS)
    c = load_profile(kit.root, str(seq))
    assert c.status() == "done" and not c.provisional and c.pending == []
    assert all(p.complete for p in c.points)
    assert c.starts_progress() == (40, 40)
    assert c.heartbeat_at is None
    assert c.mle.coords == (0.15,)


def test_ended_command_with_partial_cells_is_done_but_provisional(tmp_path):
    """A command that ended before every start finished (camdl stopped early)
    is done, yet its incomplete cells still keep the estimate provisional."""
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = kit.profile(SWEEP, _mid_run_points(), n_starts=N_STARTS)
    c = load_profile(kit.root, str(seq))
    assert c.status() == "done" and c.provisional
    assert c.mle.coords == (0.15,)


def test_heartbeat_is_the_progress_snapshots_time(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = _mid_run(kit, progress={**_progress(), "at": "2026-11-24T00:00:05.000Z"})
    c = load_profile(kit.root, str(seq))
    assert c.heartbeat_at == datetime(2026, 11, 24, 0, 0, 5, tzinfo=timezone.utc).timestamp()


def test_no_interval_is_drawn_from_partial_values_alone(tmp_path):
    """Two values with 1 of 10 starts each, none complete: the best so far is
    reported, but no interval is computed from two lower bounds."""
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = kit.profile(SWEEP, [_start(0.018, -4620.0), _start(0.15, -4630.0)],
                      n_starts=N_STARTS, status="running", progress=_progress())
    c = load_profile(kit.root, str(seq))
    assert c.estimate_points == []
    assert c.mle.coords == (0.018,) and c.provisional
    assert c.ci_bounds_1d() == (None, None)


def test_starts_of_an_earlier_command_do_not_count_as_this_ones(tmp_path):
    """An earlier command left 10 starts at 0.10; this running command has
    finished 2 of its 10 there. Its cell is partial, and its log-likelihood is
    the best of its own two starts."""
    kit = StoreKit.create(tmp_path / "camdl-store")
    kit.profile(SWEEP, [_start(0.10, -4600.0) for _ in range(N_STARTS)], n_starts=N_STARTS)
    seq = kit.profile(SWEEP, [_start(0.10, -4631.8), _start(0.10, -4632.0)],
                      n_starts=N_STARTS, status="running", progress=_progress())
    p = next(p for p in load_profile(kit.root, str(seq)).points if p.coords == (0.10,))
    assert (p.n_starts, p.n_expected, p.complete) == (2, 10, False)
    assert p.loglik == -4631.8


def test_endpoints_carry_status_counts_pending_and_partial(tmp_path, monkeypatch):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = _mid_run(kit)
    monkeypatch.setenv("CAMDL_STORE", str(kit.root))
    from camdl_scope.api.app import app

    c = TestClient(app)
    [s] = c.get("/api/profiles").json()
    assert s["base_id"] == str(seq) and s["label"] == "m4l_profile"
    assert s["status"] == "running" and s["n_points"] == 4
    assert (s["n_starts_finished"], s["n_starts_expected"]) == (21, 40)
    r = c.get(f"/api/profiles/{seq}")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["provisional"] is True and d["status"] == "running"
    assert d["pending"] == [{"coords": [0.05], "n_expected": 10}]
    partial = next(p for p in d["points"] if p["coords"] == [0.018])
    assert partial["complete"] is False and partial["n_expected"] == 10
    assert d["mle_coords"] == [0.15]
    assert d["n_estimate_points"] == 2  # 0.10 and 0.15; the partial 0.018 is not one
    assert d["heartbeat_at"] == pytest.approx(load_profile(kit.root, str(seq)).heartbeat_at)
