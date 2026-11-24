"""What is on disk under a camdl store, and what it is costing.

The scan covers the whole store: every build's kind directories (``fit``,
``sim``, ``profile_point``, ``compiled_model``…) and the store's own entries
(``.invocations``, ``.index``…). A size view scoped to the fits the app can
open would answer the wrong question: on the ebola project the simulations held
25 GB of a 36 GB store. Run-level detail is reported for the fit problems, which
are what the app opens.

Two properties this walk has to have, both forced by that 36 GB:

* **It is not request-shaped.** A cold walk of the full tree took 3m48s
  (``du -sh``, measured). Nothing here is called from a handler; a caller runs
  :func:`scan` on a background thread and serves the last completed snapshot.
* **It reports as it goes.** :class:`ScanProgress` is mutated in place so the
  UI can show a running count rather than an unexplained four-minute wait.

**Two sizes, always.** ``bytes`` is apparent size (``st_size``) — what the data
costs to copy or ship. ``alloc_bytes`` is what it occupies (``st_blocks × 512``)
— what it costs on this disk. On a fits tree they agree to a few percent, but on
``sims`` they do not: 5.1 million files averaging 1,891 bytes each round up to a
block apiece, so 9.4 GB of data occupies ~25 GB. Reporting one number would
either contradict ``du`` or hide the 15 GB of slack, and the slack is the
finding — it is an argument about how sims are written, not about how many
there are.

Pure reader. No dependency on ingest — the run-status join is done by the
caller, which already owns discovery.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import store as store_mod

#: Filenames whose roll-up is worth naming in the UI. Everything else lands in
#: its own bucket by name anyway; this only pins the reading order for the ones
#: that dominate (trace/draws/trajectories are 97% of a fits tree).
_TOP_KINDS = 14


@dataclass
class ScanProgress:
    """Live counters for an in-flight walk, mutated by the scanning thread.

    Read without a lock: every field is a plain int/str rebind, which CPython
    makes atomic, and a torn read here costs a slightly stale progress number —
    not correctness. Locking a counter updated millions of times to protect a
    display value would cost more than the value is worth.
    """

    state: str = "idle"  # idle | scanning | done | failed
    files: int = 0
    dirs: int = 0
    total_bytes: int = 0
    total_alloc: int = 0
    started_at: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class FileKind:
    """One filename rolled up across the tree — ``trace.tsv``, ``run.json``.

    By *name*, not extension: a fits tree is 98% ``.tsv`` by extension, which
    says nothing, while by name it is 64% ``trace.tsv`` and 16%
    ``trajectories.tsv``, which says where to look.
    """

    name: str
    bytes: int
    alloc_bytes: int
    files: int


@dataclass(frozen=True)
class DirRollup:
    """One part of the store: a build's kind directory
    (``0.1.0_de896b47c-8fd5625dd2f8/sim``) or one of the store's own entries
    (``.invocations``)."""

    name: str
    bytes: int
    alloc_bytes: int
    files: int
    dirs: int


@dataclass(frozen=True)
class RunRollup:
    """One fit problem folder (``<build>/fit/<model>-<12 hex>``).

    ``run_id`` is the folder's name. Whether it is a run the app can *open* is
    the caller's join — a folder here with no matching run is the interesting
    case (runs that never finished, which the app does not list), and
    conflating "not a run" with "status unknown" would hide it.
    """

    run_id: str
    bytes: int
    alloc_bytes: int
    files: int
    mtime: float


@dataclass(frozen=True)
class StoreSnapshot:
    """A completed walk. Immutable, so a reader never sees a half-updated one."""

    root: str
    store: str
    scanned_at: float
    duration_s: float
    total_bytes: int
    total_alloc: int
    total_files: int
    total_dirs: int
    children: list[DirRollup] = field(default_factory=list)
    kinds: list[FileKind] = field(default_factory=list)
    runs: list[RunRollup] = field(default_factory=list)
    #: Paths the walk could not read (permissions, or a directory deleted mid
    #: walk). Reported rather than swallowed: a scan that silently skipped half
    #: the tree would under-report the total and look like a tidy store.
    unreadable: int = 0


def _walk(
    root: Path,
    progress: ScanProgress | None,
    kinds: dict[str, list[int]],
    base: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[int, int, int, int, int]:
    """Recursive roll-up. Returns ``(bytes, alloc, files, dirs, unreadable)``.

    Iterative rather than recursive: ``sims`` nests deep enough that a recursive
    walk risks the interpreter's stack limit, and this has no reason to care how
    deep the tree goes.

    ``kinds`` buckets are ``[bytes, alloc, files]``, accumulated in place — a
    dict of small lists rather than of dataclasses because this inner loop runs
    five million times on a real store.

    ``base`` is ``(files, dirs, bytes, alloc)`` already counted by earlier calls, so the
    reported progress is the running total across the whole scan. Without it the
    counters restart at every top-level child and the UI shows the last child's
    tally — which on a store where one child dominates looks plausible and is
    wrong.
    """
    total = alloc = files = dirs = bad = 0
    stack = [str(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            dirs += 1
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            st = e.stat(follow_symlinks=False)
                            size = st.st_size
                            # st_blocks is 512-byte units by POSIX convention,
                            # independent of the filesystem's own block size.
                            on_disk = st.st_blocks * 512
                            files += 1
                            total += size
                            alloc += on_disk
                            bucket = kinds.get(e.name)
                            if bucket is None:
                                kinds[e.name] = [size, on_disk, 1]
                            else:
                                bucket[0] += size
                                bucket[1] += on_disk
                                bucket[2] += 1
                    except OSError:
                        bad += 1
        except OSError:
            bad += 1
            continue
        if progress is not None:
            progress.files = base[0] + files
            progress.dirs = base[1] + dirs
            progress.total_bytes = base[2] + total
            progress.total_alloc = base[3] + alloc
    return total, alloc, files, dirs, bad


def _parts(root: Path) -> list[tuple[str, Path]]:
    """The store's parts, as ``(name, path)``: each build's kind directories,
    then each of the store's own entries."""
    out: list[tuple[str, Path]] = []
    for e in sorted(os.scandir(root), key=lambda e: e.name):
        if not e.is_dir(follow_symlinks=False):
            continue
        if e.name.startswith("."):
            out.append((e.name, Path(e.path)))
            continue
        try:
            kinds = [k for k in os.scandir(e.path) if k.is_dir(follow_symlinks=False)]
        except OSError:
            kinds = []
        out.extend((f"{e.name}/{k.name}", Path(k.path)) for k in sorted(kinds, key=lambda k: k.name))
    return out


def scan(
    store: Path, progress: ScanProgress | None = None, top_kinds: int = _TOP_KINDS
) -> StoreSnapshot:
    """Walk the store and roll it up. Minutes on a large store — call from a
    background thread, never from a request handler."""
    root = Path(store).resolve()
    t0 = time.time()
    if progress is not None:
        progress.state = "scanning"
        progress.started_at = t0
        progress.files = progress.dirs = progress.total_bytes = 0
        progress.total_alloc = 0
        progress.error = None

    kinds: dict[str, list[int]] = {}
    children: list[DirRollup] = []
    total = alloc = files = dirs = bad = 0
    try:
        parts = _parts(root)
    except OSError as exc:
        if progress is not None:
            progress.state = "failed"
            progress.error = str(exc)
        raise

    # One pass per part, so a part's own roll-up is a by-product of the total
    # rather than a second walk.
    for name, path in parts:
        cb, ca, f, d, u = _walk(path, progress, kinds, (files, dirs, total, alloc))
        children.append(DirRollup(name=name, bytes=cb, alloc_bytes=ca, files=f, dirs=d + 1))
        total += cb
        alloc += ca
        files += f
        dirs += d + 1
        bad += u

    # Run-level detail for the fit problems only. The other kinds are one line
    # each: a simulation sweep has millions of folders, and a row for each would
    # be a table nobody can read and a payload nobody should ship.
    runs: list[RunRollup] = []
    for kd in store_mod.kind_dirs(root, "fit"):
        for problem in store_mod.walk(kd, 1):
            # No progress: a re-walk of a subtree already counted in the totals.
            rb, ra, rf, _, _ = _walk(problem, None, {})
            try:
                mtime = problem.stat().st_mtime
            except OSError:
                mtime = 0.0
            runs.append(RunRollup(run_id=problem.name, bytes=rb, alloc_bytes=ra, files=rf, mtime=mtime))
    runs.sort(key=lambda r: r.alloc_bytes, reverse=True)

    # Ranked by what it OCCUPIES, not by apparent size, because that is the
    # number the view leads with — and the two orders genuinely differ: 640k
    # `traj.tsv` cost more disk than 385 `ensemble.tsv` while holding less data.
    # A list not sorted by its own headline column reads as a bug.
    ranked = sorted(
        (
            FileKind(name=n, bytes=v[0], alloc_bytes=v[1], files=v[2])
            for n, v in kinds.items()
        ),
        key=lambda k: k.alloc_bytes,
        reverse=True,
    )
    snapshot = StoreSnapshot(
        root=str(root),
        store=str(root),
        scanned_at=t0,
        duration_s=time.time() - t0,
        total_bytes=total,
        total_alloc=alloc,
        total_files=files,
        total_dirs=dirs,
        children=sorted(children, key=lambda c: c.alloc_bytes, reverse=True),
        kinds=ranked[:top_kinds],
        runs=runs,
        unreadable=bad,
    )
    if progress is not None:
        progress.state = "done"
    return snapshot


class StoreScanner:
    """Owns the current snapshot and the one background thread allowed to build
    it.

    Single-flight by construction: a rescan requested while one is running is a
    no-op rather than a second walk, because two concurrent four-minute walks
    over the same tree would halve each other's throughput and neither would be
    the one the caller wanted.
    """

    def __init__(self, store: Path) -> None:
        self._store = Path(store)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.progress = ScanProgress()
        self.snapshot: StoreSnapshot | None = None

    @property
    def scanning(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def start(self) -> bool:
        """Kick off a scan if none is running. True when one was started."""
        with self._lock:
            if self.scanning:
                return False
            self.progress = ScanProgress(state="scanning", started_at=time.time())
            t = threading.Thread(target=self._run, name="store-scan", daemon=True)
            self._thread = t
            t.start()
            return True

    def _run(self) -> None:
        try:
            snap = scan(self._store, self.progress)
        except Exception as exc:  # a walk failure must not kill the thread silently
            self.progress.state = "failed"
            self.progress.error = str(exc)
            return
        self.snapshot = snap
