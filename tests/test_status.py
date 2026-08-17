"""Status classification: liveness (the sampler `.lock` PID / heartbeat), not
trace mtime.

A slow / write-buffered sampler (PGAS on a spatial model can take minutes per
sweep) legitimately goes a long time between trace writes. Status must come from
whether the process is alive, never from file staleness — otherwise a live run
is mislabeled "done". The classification core is :func:`camdl_watch.assembly`,
exercised here against synthetic on-disk stores the way the API assembles them.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from camdl_watch import assembly, ingest
from camdl_watch.state import RunState


def _mkrun(
    store: Path, name: str, *, rows: int, lock_pid: int, draws: bool = True
) -> Path:
    seed = store / name / "01-posterior-aaaa1111" / "seed_1-bbbb2222"
    (seed / "chain_1").mkdir(parents=True)
    tp = seed / "chain_1" / "trace.tsv"
    if rows:
        tp.write_text(
            "step\tlog_posterior\tR0\n"
            + "".join(f"{i}\t-5.0\t1.{i}\n" for i in range(rows))
        )
        # A finished stage writes the pooled posterior; a killed one (draws=False)
        # leaves only the partial per-chain trace above.
        if draws:
            (seed / "draws.tsv").write_text(
                "R0\n" + "".join(f"1.{i}\n" for i in range(rows))
            )
    else:
        tp.write_text("")  # burn-in: empty
    (seed / ".lock").write_text(str(lock_pid))
    (store / name / "fit.meta.json").write_text(
        json.dumps({"estimated": ["R0"], "resolved_priors": []})
    )
    (store / name / "fit.toml.original").write_text(
        '[stages.posterior]\nalgorithm="pgas"\nbackend="chain_binomial"\nburn_in=2000\n'
    )
    return tp


_DEAD_PID = 2147483646  # almost certainly not a live process


def _registry(store: Path) -> dict[str, RunState]:
    """Discover + assemble every run the way the API does — the v2 equivalent of
    the old Shiny registry."""
    return {
        m.run_id: assembly.build_run_state(m)
        for m in ingest.discover_runs(store, include_warming=True)
    }


def test_live_slow_sampler_with_ancient_trace_is_running(tmp_path):
    store = tmp_path / "fits"
    tp = _mkrun(store, "slow-aaaa", rows=72, lock_pid=os.getpid())
    ancient = time.time() - 3600  # 1 hour since last write — the bug trigger
    os.utime(tp, (ancient, ancient))
    assert _registry(store)["slow-aaaa"].status.value == "running"


def test_finished_run_with_draws_is_done(tmp_path):
    store = tmp_path / "fits"
    _mkrun(store, "fin-bbbb", rows=72, lock_pid=_DEAD_PID)  # writes draws.tsv
    assert _registry(store)["fin-bbbb"].status.value == "done"


def test_killed_partial_run_is_stalled_not_done(tmp_path):
    # No heartbeat, dead PID, trace rows present but NO draws.tsv: a stage that
    # was killed / crashed / OOM'd mid-sampling. It must NOT read as done —
    # there is no pooled posterior to predict from.
    store = tmp_path / "fits"
    _mkrun(store, "killed-eeee", rows=72, lock_pid=_DEAD_PID, draws=False)
    assert _registry(store)["killed-eeee"].status.value == "stalled"


def _seed(store: Path, name: str) -> Path:
    return store / name / "01-posterior-aaaa1111" / "seed_1-bbbb2222"


def _write_progress(seed: Path, state, *, updated_at: float, pid: int = 12345) -> None:
    seed.mkdir(parents=True, exist_ok=True)
    (seed / "progress.json").write_text(
        json.dumps({"updated_at": int(updated_at), "pid": pid, "state": state})
    )


def _classify_one(store: Path, name: str) -> RunState:
    return _registry(store)[name]


def test_heartbeat_burn_in_is_warming(tmp_path):
    # Heartbeat wins over the .lock: dead PID but a FRESH burn-in heartbeat.
    store = tmp_path / "fits"
    _mkrun(store, "hb-warm", rows=0, lock_pid=_DEAD_PID)
    _write_progress(_seed(store, "hb-warm"),
                    {"running": {"phase": "burn_in", "step": 130, "total": 1000}},
                    updated_at=time.time())
    rs = _classify_one(store, "hb-warm")
    assert rs.status.value == "warming"
    assert (rs.progress.step, rs.progress.total) == (130, 1000)


def test_heartbeat_sampling_is_running(tmp_path):
    store = tmp_path / "fits"
    _mkrun(store, "hb-run", rows=50, lock_pid=_DEAD_PID)
    _write_progress(_seed(store, "hb-run"),
                    {"running": {"phase": "sampling", "step": 600, "total": 1000}},
                    updated_at=time.time())
    assert _classify_one(store, "hb-run").status.value == "running"


def test_heartbeat_stale_running_is_stalled(tmp_path):
    # Stale heartbeat -> presumed dead, even though the .lock PID is alive.
    store = tmp_path / "fits"
    _mkrun(store, "hb-stale", rows=50, lock_pid=os.getpid())
    _write_progress(_seed(store, "hb-stale"),
                    {"running": {"phase": "sampling", "step": 600, "total": 1000}},
                    updated_at=time.time() - 120)
    assert _classify_one(store, "hb-stale").status.value == "stalled"


def test_heartbeat_terminal_done_and_failed(tmp_path):
    # Terminal states win regardless of a live .lock / freshness.
    store = tmp_path / "fits"
    _mkrun(store, "hb-done", rows=50, lock_pid=os.getpid())
    _write_progress(_seed(store, "hb-done"), "done", updated_at=time.time())
    assert _classify_one(store, "hb-done").status.value == "done"

    store2 = tmp_path / "fits2"
    _mkrun(store2, "hb-fail", rows=50, lock_pid=os.getpid())
    _write_progress(_seed(store2, "hb-fail"),
                    {"failed": {"reason": "nan in likelihood"}}, updated_at=time.time())
    rs = _classify_one(store2, "hb-fail")
    assert rs.status.value == "failed"
    assert rs.progress.reason == "nan in likelihood"


def test_burn_in_death_is_hidden_not_zombied(tmp_path):
    # No heartbeat (e.g. mh/ode): a run seen live in burn-in, then its sampler
    # dies with zero draws. v2 hides the empty dead stub (no clutter), rather
    # than the v1 behavior of keeping it as a "stalled" zombie.
    store = tmp_path / "fits"
    _mkrun(store, "die-aaaa", rows=0, lock_pid=os.getpid())
    assert _registry(store)["die-aaaa"].status.value == "warming"
    (_seed(store, "die-aaaa") / ".lock").write_text(str(_DEAD_PID))  # sampler dies
    assert "die-aaaa" not in _registry(store)  # empty + dead -> hidden


def test_deleted_run_dir_is_dropped_not_zombied(tmp_path):
    # A live run whose dir is pruned off disk must leave the listing entirely.
    store = tmp_path / "fits"
    _mkrun(store, "prune-aaaa", rows=72, lock_pid=os.getpid())
    assert _registry(store)["prune-aaaa"].status.value == "running"
    import shutil

    shutil.rmtree(store / "prune-aaaa")
    assert "prune-aaaa" not in _registry(store)


def test_pick_prefers_live_stage_among_empty_siblings(tmp_path):
    # Two empty stage dirs (a relaunch): even when the dead one sorts first by
    # mtime, the live one is surfaced.
    run = tmp_path / "cfg-deadbeef"
    dead = run / "02-posterior-bbbb2222" / "seed_1-bbbb"
    live = run / "01-posterior-aaaa1111" / "seed_1-aaaa"
    for sd, pid in ((dead, _DEAD_PID), (live, os.getpid())):
        (sd / "chain_1").mkdir(parents=True)
        (sd / "chain_1" / "trace.tsv").write_text("")
        (sd / ".lock").write_text(str(pid))
    newer = time.time() + 100
    os.utime(dead / "chain_1" / "trace.tsv", (newer, newer))
    picked = ingest._pick_posterior_dir(run, include_warming=True)
    assert picked is not None and picked[1] == live  # the live seed dir


def test_pick_prefers_completed_stage_over_newer_stub(tmp_path):
    # A crashed relaunch leaves a stub with as many started chains as the real
    # run and a *newer* mtime. Without the completion key it would win the
    # (chains, mtime) tie-break and mask the finished stage; the completed stage
    # (draws.tsv) must still be the one surfaced.
    run = tmp_path / "cfg-feedface"
    done_stage = run / "01-posterior-aaaa1111" / "seed_1-aaaa"
    stub_stage = run / "02-posterior-bbbb2222" / "seed_1-bbbb"
    trace = (
        "step\tlog_posterior\tR0\n"
        + "".join(f"{i}\t-5.0\t1.{i}\n" for i in range(50))
    )
    for sd in (done_stage, stub_stage):
        (sd / "chain_1").mkdir(parents=True)
        (sd / "chain_1" / "trace.tsv").write_text(trace)
    (done_stage / "draws.tsv").write_text("R0\n1.0\n1.1\n")
    newer = time.time() + 100  # the stub is more recent
    os.utime(stub_stage / "chain_1" / "trace.tsv", (newer, newer))
    picked = ingest._pick_posterior_dir(run)
    assert picked is not None and picked[1] == done_stage


def test_live_burn_in_is_warming_dead_burn_in_hidden(tmp_path):
    store = tmp_path / "fits"
    _mkrun(store, "warm-cccc", rows=0, lock_pid=os.getpid())   # live, no draws
    _mkrun(store, "dead-dddd", rows=0, lock_pid=_DEAD_PID)     # dead, no draws
    reg = _registry(store)
    assert reg["warm-cccc"].status.value == "warming"
    assert "dead-dddd" not in reg  # empty + dead stays hidden
