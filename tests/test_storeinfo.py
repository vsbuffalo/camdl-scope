"""Store-walk tests. Synthetic trees with byte counts chosen so every total is
checkable by hand — the whole point of this module is arithmetic nobody can
eyeball on a real 36 GB store."""

from __future__ import annotations

import os
from pathlib import Path

from camdl_scope import storeinfo


def _file(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n)


def _tree(root: Path) -> Path:
    """`<root>/results/{fits,sims}` with known sizes; returns the fits store.

    `sims` is deliberately larger than `fits` by more than a block, so the
    ordering assertions hold on APPARENT and ALLOCATED size alike — with
    kilobyte files both kinds round to the same number of blocks and the sort
    order becomes a tie nobody meant to assert on.
    """
    _file(root / "results/fits/run_a-1111/trace.tsv", 1000)
    _file(root / "results/fits/run_a-1111/run.json", 10)
    _file(root / "results/fits/run_b-2222/trace.tsv", 500)
    _file(root / "results/sims/s1/traj.tsv", 200_000)
    return root / "results/fits"


# --- scan root ---------------------------------------------------------------


def test_run_kind_store_scans_its_parent(tmp_path):
    # The bytes live in the SIBLING kinds, so a fits-only view answers the wrong
    # question; `results/fits` must widen to `results`.
    store = _tree(tmp_path)
    assert storeinfo.resolve_scan_root(store) == (tmp_path / "results").resolve()


def test_non_run_kind_store_is_its_own_root(tmp_path):
    # Widening an arbitrary directory to its parent could walk a home dir.
    d = tmp_path / "somewhere"
    d.mkdir()
    assert storeinfo.resolve_scan_root(d) == d.resolve()


# --- roll-ups ----------------------------------------------------------------


def test_totals_reconcile_with_the_parts(tmp_path):
    snap = storeinfo.scan(_tree(tmp_path))
    assert snap.total_bytes == 201_510
    assert snap.total_files == 4
    assert sum(c.bytes for c in snap.children) == snap.total_bytes
    assert sum(k.bytes for k in snap.kinds) == snap.total_bytes
    assert sum(k.files for k in snap.kinds) == snap.total_files


def test_children_are_per_run_kind_biggest_first(tmp_path):
    snap = storeinfo.scan(_tree(tmp_path))
    assert [c.name for c in snap.children] == ["sims", "fits"]
    assert dict((c.name, c.bytes) for c in snap.children) == {
        "sims": 200_000,
        "fits": 1510,
    }


def test_kinds_roll_up_by_filename_across_the_whole_tree(tmp_path):
    # trace.tsv appears in two runs; traj.tsv only under sims. Rolling up by
    # EXTENSION would merge them into one meaningless ".tsv" bucket.
    snap = storeinfo.scan(_tree(tmp_path))
    kinds = dict((k.name, (k.bytes, k.files)) for k in snap.kinds)
    assert kinds["trace.tsv"] == (1500, 2)
    assert kinds["traj.tsv"] == (200_000, 1)


def test_runs_cover_the_store_only_and_sort_by_size(tmp_path):
    # `sims` has millions of directories; a per-run row for each is unshippable,
    # so run detail is the watcher's own store and nothing else.
    snap = storeinfo.scan(_tree(tmp_path))
    assert [r.run_id for r in snap.runs] == ["run_a-1111", "run_b-2222"]
    assert [r.bytes for r in snap.runs] == [1010, 500]


def test_allocated_size_is_reported_and_never_below_apparent(tmp_path):
    # The gap between the two IS the finding on a tree of small files; a build
    # that reported one number would contradict `du` or hide the slack.
    snap = storeinfo.scan(_tree(tmp_path))
    assert snap.total_alloc >= snap.total_bytes
    for c in snap.children:
        assert c.alloc_bytes >= c.bytes
    for r in snap.runs:
        assert r.alloc_bytes >= r.bytes


def test_empty_store_scans_without_error(tmp_path):
    root = tmp_path / "results"
    (root / "fits").mkdir(parents=True)
    snap = storeinfo.scan(root / "fits")
    assert snap.total_bytes == 0
    assert snap.runs == []


def test_unreadable_directory_is_counted_not_swallowed(tmp_path):
    # A silently skipped subtree under-reports the total and makes a store look
    # tidy; the count is how a reader knows the number is short.
    store = _tree(tmp_path)
    locked = tmp_path / "results/fits/run_c-3333"
    locked.mkdir(parents=True)
    _file(locked / "trace.tsv", 100)
    os.chmod(locked, 0o000)
    try:
        snap = storeinfo.scan(store)
        assert snap.unreadable >= 1
    finally:
        os.chmod(locked, 0o755)


def test_progress_advances_and_settles_on_done(tmp_path):
    p = storeinfo.ScanProgress()
    storeinfo.scan(_tree(tmp_path), p)
    assert p.state == "done"
    assert p.files == 4
    assert p.total_bytes == 201_510


# --- the scanner ------------------------------------------------------------


def test_scanner_runs_in_background_and_publishes_a_snapshot(tmp_path):
    sc = storeinfo.StoreScanner(_tree(tmp_path))
    assert sc.snapshot is None
    assert sc.start() is True
    sc._thread.join(timeout=30)  # type: ignore[union-attr]
    assert sc.snapshot is not None
    assert sc.snapshot.total_bytes == 201_510
    assert sc.progress.state == "done"


def test_scanner_is_single_flight(tmp_path):
    # Two concurrent walks of one tree only halve each other's throughput.
    sc = storeinfo.StoreScanner(_tree(tmp_path))
    sc.start()
    second = sc.start() if sc.scanning else None
    if second is not None:
        assert second is False
    sc._thread.join(timeout=30)  # type: ignore[union-attr]
    assert sc.snapshot is not None
