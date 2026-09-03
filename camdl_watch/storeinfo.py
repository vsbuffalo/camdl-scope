"""What is on disk under a camdl store, and what it is costing.

The watcher indexes one run kind (``results/fits`` by default), but the disk is
spent elsewhere: on the ebola project that subtree is 4.7 GB of a 36 GB
``results/`` tree, with ``sims/`` holding 25 GB of it. A size view scoped to the
runs the app can open would therefore answer the wrong question, so the scan
root is the *store family* — the parent holding ``fits``/``sims``/``ensembles``
/``pfilters``/``surveys`` — and the run-level detail is reported for the
watcher's own store within it.

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

#: Directories a camdl store keeps its run kinds in. Used only to recognise a
#: store family — an unknown sibling is still scanned and reported, since the
#: point of the view is to find what nobody remembered was there.
RUN_KIND_DIRS: tuple[str, ...] = (
    "fits",
    "sims",
    "ensembles",
    "pfilters",
    "surveys",
)

#: Filenames whose roll-up is worth naming in the UI. Everything else lands in
#: its own bucket by name anyway; this only pins the reading order for the ones
#: that dominate (trace/draws/trajectories are 97% of a fits tree).
_TOP_KINDS = 14


def resolve_scan_root(store: Path) -> Path:
    """The store *family* to scan, given the store the watcher indexes.

    ``<results>/fits`` → ``<results>``, so the view can see the sibling run
    kinds that hold most of the bytes. A store that is not a recognised run-kind
    directory is its own root — pointing the watcher at an arbitrary directory
    must not silently walk its parent, which could be a home directory.
    """
    store = Path(store).resolve()
    return store.parent if store.name in RUN_KIND_DIRS else store


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
    """One top-level child of the scan root (``sims``, ``fits``, …)."""

    name: str
    bytes: int
    alloc_bytes: int
    files: int
    dirs: int


@dataclass(frozen=True)
class RunRollup:
    """One directory directly under the watcher's own store.

    ``run_id`` is the directory name. Whether it is a run the app can *open* is
    the caller's join — a directory here with no matching run is the interesting
    case (debris the watcher cannot read), and conflating "not a run" with
    "status unknown" would hide it.
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


def scan(
    store: Path, progress: ScanProgress | None = None, top_kinds: int = _TOP_KINDS
) -> StoreSnapshot:
    """Walk the store family and roll it up. Minutes on a large tree — call from
    a background thread, never from a request handler."""
    store = Path(store).resolve()
    root = resolve_scan_root(store)
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

    # One pass per top-level child, so a child's own roll-up is a by-product of
    # the total rather than a second walk.
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError as exc:
        if progress is not None:
            progress.state = "failed"
            progress.error = str(exc)
        raise

    for e in entries:
        try:
            if not e.is_dir(follow_symlinks=False):
                if e.is_file(follow_symlinks=False):
                    st = e.stat(follow_symlinks=False)
                    on_disk = st.st_blocks * 512
                    files += 1
                    total += st.st_size
                    alloc += on_disk
                    b = kinds.setdefault(e.name, [0, 0, 0])
                    b[0] += st.st_size
                    b[1] += on_disk
                    b[2] += 1
                continue
        except OSError:
            bad += 1
            continue
        cb, ca, f, d, u = _walk(
            Path(e.path), progress, kinds, (files, dirs, total, alloc)
        )
        children.append(
            DirRollup(name=e.name, bytes=cb, alloc_bytes=ca, files=f, dirs=d + 1)
        )
        total += cb
        alloc += ca
        files += f
        dirs += d + 1
        bad += u

    # Run-level detail for the watcher's own store only. The sibling kinds are
    # reported as one line each: `sims` has millions of directories, and a
    # per-run row for each would be a table nobody can read and a payload
    # nobody should ship.
    runs: list[RunRollup] = []
    if store.is_dir():
        try:
            run_dirs = [e for e in os.scandir(store) if e.is_dir(follow_symlinks=False)]
        except OSError:
            run_dirs = []
        for e in run_dirs:
            # No progress: this is a re-walk of a subtree already counted in the
            # totals, and feeding it back would double the reported files.
            rb, ra, rf, _, _ = _walk(Path(e.path), None, {})
            try:
                mtime = e.stat(follow_symlinks=False).st_mtime
            except OSError:
                mtime = 0.0
            runs.append(
                RunRollup(
                    run_id=e.name, bytes=rb, alloc_bytes=ra, files=rf, mtime=mtime
                )
            )
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
        store=str(store),
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
