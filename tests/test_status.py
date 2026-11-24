"""Status classification: the run's ``run.json`` record and its owner's
liveness, not trace mtime.

A slow / write-buffered sampler (PGAS on a spatial model can take minutes per
sweep) legitimately goes a long time between trace writes. Status must come from
the run's record and whether the process that owns it is alive, never from file
staleness — otherwise a live run is mislabeled "done". The classification core
is :func:`camdl_scope.assembly.classify_from`, exercised here against stores
built with :class:`tests.storekit.StoreKit` the way the API assembles them.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from camdl_scope import assembly, ingest
from camdl_scope.state import RunState
from tests.storekit import StoreKit


def _trace(rows: int) -> str:
    return (
        "step\tlog_posterior\tR0\n"
        + "".join(f"{i}\t-5.0\t1.{i}\n" for i in range(rows))
    )


def _write_trace(leaf: Path, rows: int) -> Path:
    """One chain's trace in ``leaf``; ``rows=0`` is a burn-in chain that has
    created its trace but written nothing to it."""
    tp = leaf / "chain_0" / "trace.tsv"
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(_trace(rows) if rows else "")
    return tp


def _run(
    kit: StoreKit, problem: Path, *, rows: int, status: str = "completed",
    alive: bool = True,
) -> Path:
    """A PGAS fit run of ``problem`` with one chain of ``rows`` trace rows."""
    leaf = kit.fit_run(
        problem, algorithm="pgas", status=status, alive=alive,
        method={"backend": "chain_binomial", "burn_in": 2000},
        estimated=["R0"], meta={"estimated": ["R0"], "resolved_priors": []},
    )
    _write_trace(leaf, rows)
    return leaf


def _mkrun(kit: StoreKit, name: str, **kw) -> Path:
    """A problem folder named ``name`` holding one run (see :func:`_run`)."""
    return _run(kit, kit.problem(name=name), **kw)


def _kill(kit: StoreKit, leaf: Path) -> None:
    """The run's owner dies without ending its claim: ``run.json`` keeps
    ``running`` but names a process that has exited, as a killed camdl leaves it."""
    rec = json.loads((leaf / "run.json").read_text())
    rec["owner"] = kit.owner(alive=False)
    (leaf / "run.json").write_text(json.dumps(rec))


def _move_owner_to_another_host(kit: StoreKit, leaf: Path) -> None:
    """Give the run an owner on another machine, with its beat file present.
    This reader cannot see that process, so the run's heartbeat decides.
    (StoreKit only makes owners on this host.)"""
    owner = {"host": "elsewhere-node", "pid": 4242, "started": "2026-11-24T00:00:00.000Z"}
    (kit.root / ".owners" / "elsewhere-node-4242.json").write_text(
        json.dumps({"format": 1, **owner, "beat": 7}))
    rec = json.loads((leaf / "run.json").read_text())
    rec["owner"] = owner
    (leaf / "run.json").write_text(json.dumps(rec))


def _failed_beat(leaf: Path, reason: str) -> None:
    """A terminal ``failed`` heartbeat with its reason (StoreKit's ``beat``
    writes only ``running`` or a bare unit state)."""
    (leaf / "progress.json").write_text(json.dumps({
        "format": 1, "at": "2026-11-24T00:00:05.000Z",
        "snapshot": {"updated_at": int(time.time()), "pid": os.getpid(),
                     "state": {"failed": {"reason": reason}}},
    }))


def _registry(root: Path) -> dict[str, RunState]:
    """Discover + assemble every run the way the API does."""
    return {
        m.run_id: assembly.build_run_state(m)
        for m in ingest.discover_runs(root, include_warming=True)
    }


def _by_folder(root: Path, folder: str) -> list[RunState]:
    """Every entry of one problem folder, its default first."""
    return [rs for rs in _registry(root).values() if rs.meta.problem_dir.name == folder]


@pytest.fixture
def kit(tmp_path: Path) -> StoreKit:
    return StoreKit.create(tmp_path / "camdl-store")


def test_live_slow_sampler_with_ancient_trace_is_running(kit):
    leaf = _mkrun(kit, "slow-aaaa", rows=72, status="running")
    ancient = time.time() - 3600  # 1 hour since last write — the bug trigger
    os.utime(leaf / "chain_0" / "trace.tsv", (ancient, ancient))
    assert _registry(kit.root)["slow-aaaa"].status.value == "running"


def test_live_owner_with_stale_heartbeat_is_still_running(kit):
    # On this host the owner's process is visible, so a heartbeat that has not
    # been rewritten for minutes (a long sweep) does not make the run stalled.
    leaf = _mkrun(kit, "slowbeat-aaaa", rows=50, status="running")
    kit.beat(leaf, phase="sampling", step=600, total=1000, age=600)
    assert _registry(kit.root)["slowbeat-aaaa"].status.value == "running"


def test_finished_run_is_done(kit):
    # Completed by its record, whatever its owner is doing now.
    _mkrun(kit, "fin-bbbb", rows=72, status="completed", alive=False)
    rs = _registry(kit.root)["fin-bbbb"]
    assert rs.status.value == "done"
    assert sum(b.n for b in rs.chains.values()) == 72


def test_killed_partial_run_is_stalled_not_done(kit):
    # The record still says running, the owner is dead, trace rows present: a
    # run killed / crashed / OOM'd mid-sampling. It must NOT read as done.
    _mkrun(kit, "killed-eeee", rows=72, status="running", alive=False)
    assert _registry(kit.root)["killed-eeee"].status.value == "stalled"


def test_heartbeat_burn_in_is_warming(kit):
    leaf = _mkrun(kit, "hb-warm", rows=0, status="running")
    kit.beat(leaf, phase="burn_in", step=130, total=1000)
    rs = _registry(kit.root)["hb-warm"]
    assert rs.status.value == "warming"
    assert (rs.progress.step, rs.progress.total) == (130, 1000)


def test_heartbeat_sampling_is_running(kit):
    leaf = _mkrun(kit, "hb-run", rows=50, status="running")
    kit.beat(leaf, phase="sampling", step=600, total=1000)
    assert _registry(kit.root)["hb-run"].status.value == "running"


def test_owner_on_another_host_is_judged_by_its_heartbeat(kit):
    # The owner's process cannot be seen from here: a fresh running heartbeat
    # counts as alive, a stale one as stalled.
    fresh = _mkrun(kit, "remote-fresh", rows=50, status="running")
    _move_owner_to_another_host(kit, fresh)
    kit.beat(fresh, phase="sampling", step=600, total=1000)
    stale = _mkrun(kit, "remote-stale", rows=50, status="running")
    _move_owner_to_another_host(kit, stale)
    kit.beat(stale, phase="sampling", step=600, total=1000,
             age=ingest.PROGRESS_STALE_S + 90)
    reg = _registry(kit.root)
    assert reg["remote-fresh"].meta.owner_alive is None
    assert reg["remote-fresh"].status.value == "running"
    assert reg["remote-stale"].status.value == "stalled"


def test_record_terminal_states_win_over_a_live_owner_and_heartbeat(kit):
    done = _mkrun(kit, "hb-done", rows=50, status="completed")
    kit.beat(done, phase="sampling", step=600, total=1000)  # a leftover beat
    failed = _mkrun(kit, "hb-fail", rows=50, status="failed", alive=True)
    _failed_beat(failed, "nan in likelihood")
    reg = _registry(kit.root)
    assert reg["hb-done"].status.value == "done"
    assert reg["hb-fail"].status.value == "failed"
    assert reg["hb-fail"].progress.reason == "nan in likelihood"


def test_burn_in_death_is_hidden_not_zombied(kit):
    # A run seen live in burn-in, then its owner dies with zero draws: the
    # empty dead run is hidden rather than kept as a "stalled" zombie.
    leaf = _mkrun(kit, "die-aaaa", rows=0, status="running")
    kit.beat(leaf, phase="burn_in", step=40, total=1000)
    assert _registry(kit.root)["die-aaaa"].status.value == "warming"
    _kill(kit, leaf)
    assert "die-aaaa" not in _registry(kit.root)


def test_deleted_run_dir_is_dropped_not_zombied(kit):
    # A live run whose folder is pruned off disk must leave the listing entirely.
    leaf = _mkrun(kit, "prune-aaaa", rows=72, status="running")
    assert _registry(kit.root)["prune-aaaa"].status.value == "running"
    shutil.rmtree(leaf)
    assert "prune-aaaa" not in _registry(kit.root)


def test_live_empty_run_is_listed_and_its_dead_sibling_hidden(kit):
    # Two empty runs of one problem (a relaunch beside a killed attempt): even
    # with the dead one's trace newer, the live one is the folder's entry.
    problem = kit.problem(name="cfg-deadbeef")
    dead = _run(kit, problem, rows=0, status="running", alive=False)
    live = _run(kit, problem, rows=0, status="running")
    for leaf in (dead, live):
        kit.beat(leaf, phase="burn_in", step=40, total=1000)
    newer = time.time() + 100
    os.utime(dead / "chain_0" / "trace.tsv", (newer, newer))
    reg = _registry(kit.root)
    assert list(reg) == ["cfg-deadbeef"]  # one run listed: the folder's name
    assert reg["cfg-deadbeef"].meta.leaf_dir == live
    assert reg["cfg-deadbeef"].status.value == "warming"


def test_completed_run_is_default_over_a_newer_dead_stub(kit):
    # A crashed relaunch leaves a stub with as many rows as the real run and a
    # *newer* mtime. The completed run is still the one the folder opens.
    problem = kit.problem(name="cfg-feedface")
    done = _run(kit, problem, rows=50, status="completed")
    stub = _run(kit, problem, rows=50, status="running", alive=False)
    newer = time.time() + 100
    os.utime(stub / "chain_0" / "trace.tsv", (newer, newer))
    default, other = _by_folder(kit.root, "cfg-feedface")
    assert default.meta.folder_default and default.meta.leaf_dir == done
    assert default.status.value == "done"
    assert other.meta.leaf_dir == stub and other.status.value == "stalled"
    metas = ingest.discover_runs(kit.root)
    assert ingest.resolve_run(metas, "cfg-feedface").leaf_dir == done


def test_relaunch_beside_completed_run_reads_live_not_done(kit):
    # A re-run adds a run next to the finished one and is still in burn-in.
    # Each is its own entry. The folder opens the finished one, whose draws
    # are the ones to show, and the re-run is listed beside it as warming.
    problem = kit.problem(name="relaunch-aaaa")
    finished = _run(kit, problem, rows=72)
    kit.beat(finished, state="done")
    new = _run(kit, problem, rows=0, status="running")
    kit.beat(new, phase="burn_in", step=319, total=40000)
    default, rerun = _by_folder(kit.root, "relaunch-aaaa")
    assert default.meta.folder_default and default.meta.leaf_dir == finished
    assert default.status.value == "done"
    assert sum(b.n for b in default.chains.values()) == 72
    assert rerun.meta.leaf_dir == new and not rerun.meta.folder_default
    assert rerun.status.value == "warming"
    assert (rerun.progress.step, rerun.progress.total) == (319, 40000)
    metas = ingest.discover_runs(kit.root, include_warming=True)
    assert ingest.resolve_run(metas, "relaunch-aaaa").leaf_dir == finished
    # Without include_warming the burn-in re-run is not listed, so the folder
    # holds one entry and keeps its plain name.
    assert [m.run_id for m in ingest.discover_runs(kit.root)] == ["relaunch-aaaa"]


def test_relaunch_with_rows_is_the_run_we_read(kit):
    # Once the live run has rows it is what the folder opens: reporting
    # "running" over the superseded run's traces would pass off the old fit's
    # sweep count and mixing as the new sampler's.
    problem = kit.problem(name="relaunch-bbbb")
    finished = _run(kit, problem, rows=72)
    new = _run(kit, problem, rows=9, status="running")
    kit.beat(new, phase="sampling", step=9, total=40000)
    default, older = _by_folder(kit.root, "relaunch-bbbb")
    assert default.meta.leaf_dir == new
    assert default.status.value == "running"
    assert sum(b.n for b in default.chains.values()) == 9
    assert older.meta.leaf_dir == finished and older.status.value == "done"


def test_dead_relaunch_stub_keeps_the_completed_run(kit):
    # The mirror image: the newer run's owner died partway with rows written
    # and a stale heartbeat. A dead stub speaks for neither the draws nor the
    # status of the folder.
    problem = kit.problem(name="relaunch-cccc")
    finished = _run(kit, problem, rows=72)
    dead = _run(kit, problem, rows=40, status="running", alive=False)
    kit.beat(dead, phase="burn_in", step=319, total=40000, age=600)
    metas = ingest.discover_runs(kit.root, include_warming=True)
    rs = assembly.build_run_state(ingest.resolve_run(metas, "relaunch-cccc"))
    assert rs.status.value == "done"
    assert rs.meta.leaf_dir == finished
    assert sum(b.n for b in rs.chains.values()) == 72
    stub = next(m for m in metas if m.leaf_dir == dead)
    assert assembly.build_run_state(stub).status.value == "stalled"


def test_live_burn_in_is_warming_dead_burn_in_hidden(kit):
    warm = _mkrun(kit, "warm-cccc", rows=0, status="running")             # live, no draws
    dead = _mkrun(kit, "dead-dddd", rows=0, status="running", alive=False)  # dead, no draws
    for leaf in (warm, dead):
        kit.beat(leaf, phase="burn_in", step=40, total=1000)
    reg = _registry(kit.root)
    assert reg["warm-cccc"].status.value == "warming"
    assert "dead-dddd" not in reg  # empty + dead stays hidden


def test_overview_agrees_with_the_full_state_it_replaces(tmp_path):
    """The run list is built from a cheap projection instead of a parsed
    RunState — 11 s per poll on a real store, since it re-read every trace to
    learn four facts. It must reach the SAME facts, or the list quietly
    disagrees with the tab you open from it."""
    from tests.fixtures.make_golden_store import RUN_DIR, build

    build(tmp_path)
    meta = next(
        m for m in ingest.discover_runs(tmp_path, include_warming=True)
        if m.run_id == RUN_DIR
    )
    ov = assembly.build_run_overview(meta)
    rs = assembly.build_run_state(meta)

    assert ov.status == rs.status
    assert ov.max_iter == rs.max_iter()
    assert ov.chain_ids == sorted(rs.chains)
    assert ov.updated_at == pytest.approx(rs.updated_at)


def test_last_iter_is_read_from_the_tail_not_the_whole_file(tmp_path):
    """`read_last_iter` must find the final row without parsing what precedes
    it, and must say None — not 0 — for a header-only chain, since 'produced
    nothing' and 'is at iteration 0' classify differently."""
    header = "sweep\tlog_posterior\tbeta\n"

    full = tmp_path / "trace.tsv"
    full.write_text(header + "".join(f"{i}\t-1.0\t0.5\n" for i in range(5000)))
    assert ingest.read_last_iter(full) == 4999

    empty = tmp_path / "empty.tsv"
    empty.write_text(header)
    assert ingest.read_last_iter(empty) is None

    # A torn final append (the sampler is mid-write) falls back to the last
    # COMPLETE row rather than mis-parsing the fragment.
    torn = tmp_path / "torn.tsv"
    torn.write_text(header + "1\t-1.0\t0.5\n2\t-1.0\t0.5\n3\t-1.0")
    assert ingest.read_last_iter(torn) == 2
