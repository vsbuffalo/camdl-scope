"""camdl's run store, read directly — the only module that knows its records.

camdl stores every result under one root, ``camdl-store/`` (camdl
``docs/dev/proposals/2026-11-20-run-store-api.md``, "On disk")::

    <root>/
      store.json                      {"format": 1, "top_level": "by_build"}
      .invocations/<seq>.json         one per command run: argv, given, label, results
      .invocations/<seq>.progress.json  a running batch's progress (a profile's)
      .owners/<host>-<pid>-<start>.json the beat file of a process holding claims
      <build label>-<12 hex>/         everything one camdl build made
        <kind>/<group>/…/<label>-<12 hex>/   one result: identity.json, run.json, files

A result folder holds two records. ``identity.json`` is what its run id hashes
(``kind``, ``run_id``, ``identity``, and ``parent`` for a result placed inside
another); ``run.json`` is its status (``running``, ``completed`` or
``failed``), the process that owns it, and its file manifest. A completed
record is sealed. A label is not stored on a result: it is part of the
invocation record of the command that carried ``--label``, and ``@L`` names the
latest such invocation whose results all completed.

Folder names come from the first writer's labels, so nothing here reads meaning
from a name. A reader walks a kind's directory to the depth its layout has
(:func:`walk`) and identifies each folder by its ``identity.json``.

What a result's folder holds beyond the records is the kind's business, and is
read by the module for that kind (``ingest`` for fits, ``profiles``, ``sims``).
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

#: The store and record format this reader knows. camdl refuses a record of
#: another version by name, and so does this reader.
STORE_FORMAT = 1
RECORD_FORMAT = 1
TOP_LEVEL = "by_build"

STORE_FILE = "store.json"
IDENTITY_FILE = "identity.json"
RUN_FILE = "run.json"
INVOCATIONS_DIR = ".invocations"
OWNERS_DIR = ".owners"

#: Where a store is looked for, in camdl's order: ``--store``, then this
#: variable, then ``./camdl-store``.
STORE_ENV = "CAMDL_STORE"
DEFAULT_ROOT = Path("camdl-store")

RunStatus = Literal["running", "completed", "failed"]
InvocationStatus = Literal["running", "ended", "interrupted"]
#: An invocation entry's outcome: ``planned`` appears only in the record a
#: command writes when it starts; ``busy`` means another process held it.
Outcome = Literal["planned", "computed", "reused", "failed", "busy"]


class UnsupportedRecord(ValueError):
    """A record of a format or shape this reader does not know, refused by name."""


class NotAStore(UnsupportedRecord):
    """The root has no ``store.json``, or one this reader does not know."""


def locate(explicit: Path | None = None) -> Path:
    """The store root, chosen as camdl chooses it: an explicit path, then
    ``$CAMDL_STORE``, then ``./camdl-store``. Not checked; see :func:`check`."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(STORE_ENV)
    return Path(env) if env else DEFAULT_ROOT


def check(root: Path) -> None:
    """Refuse a root that is not a store this reader knows: no ``store.json``,
    an unreadable one, another format, or a top level other than one directory
    per build."""
    try:
        rec = json.loads((root / STORE_FILE).read_text())
    except FileNotFoundError:
        raise NotAStore(f"{root} is not a camdl store: it has no {STORE_FILE}") from None
    except (OSError, json.JSONDecodeError) as e:
        raise NotAStore(f"{root / STORE_FILE} cannot be read: {e}") from None
    fmt, top = rec.get("format"), rec.get("top_level")
    if fmt != STORE_FORMAT or top != TOP_LEVEL:
        raise NotAStore(
            f"{root / STORE_FILE} is format {fmt!r} with top level {top!r}; "
            f"camdl-scope reads format {STORE_FORMAT} with top level {TOP_LEVEL!r}"
        )


def is_store(root: Path) -> bool:
    try:
        check(root)
    except NotAStore:
        return False
    return True


def _subdirs(d: Path) -> list[Path]:
    """``d``'s subdirectories in name order, skipping the store's own entries
    (every name camdl reserves starts with ``.``)."""
    try:
        with os.scandir(d) as it:
            return sorted(Path(e.path) for e in it if e.is_dir() and not e.name.startswith("."))
    except OSError:
        return []


def kind_dirs(root: Path, kind: str) -> list[Path]:
    """Every build's directory for ``kind`` (``fit``, ``profile_point``, ``sim``…)."""
    return [b / kind for b in _subdirs(root) if (b / kind).is_dir()]


def walk(base: Path, depth: int) -> Iterator[Path]:
    """The folders exactly ``depth`` levels below ``base``, in path order. A
    kind's layout fixes how many group levels sit above its result folders, so
    a reader passes that depth rather than recognising a folder by its name."""
    if depth == 0:
        yield base
        return
    for d in _subdirs(base):
        yield from walk(d, depth - 1)


# ----------------------------------------------------------------------------
# A result's records
# ----------------------------------------------------------------------------


def _read_record(path: Path) -> dict | None:
    """A record as a dict, or ``None`` when absent or torn (a reader can catch
    one mid-replace). A record of another format is refused by name."""
    try:
        rec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("format") != RECORD_FORMAT:
        raise UnsupportedRecord(
            f"{path} is record format {rec.get('format')!r}; "
            f"camdl-scope reads format {RECORD_FORMAT}"
        )
    return rec


@dataclass(frozen=True)
class Identity:
    """``identity.json``: what a result is. ``body`` is the kind's typed
    identity (a fit run's ``chain`` and ``length``, a profile point's ``focal``,
    ``method`` and ``seed``), passed through for that kind's reader."""

    kind: str
    run_id: str
    body: dict
    #: The result this one is placed inside (a fit run's problem, a
    #: prediction's run), or ``None`` for a result under its kind's directory.
    parent_run_id: str | None = None


def read_identity(folder: Path) -> Identity | None:
    rec = _read_record(folder / IDENTITY_FILE)
    if rec is None:
        return None
    parent = rec.get("parent")
    body = rec.get("identity")
    return Identity(
        kind=str(rec.get("kind") or ""),
        run_id=str(rec.get("run_id") or ""),
        body=body if isinstance(body, dict) else {},
        parent_run_id=str(parent["run_id"]) if isinstance(parent, dict) and parent.get("run_id") else None,
    )


@dataclass(frozen=True)
class Owner:
    """The process that claimed a result: host, process id and process start
    time, which together survive the reuse of a process id."""

    host: str
    pid: int
    started: str

    @classmethod
    def from_json(cls, d: object) -> "Owner | None":
        if not isinstance(d, dict):
            return None
        try:
            return cls(host=str(d["host"]), pid=int(d["pid"]), started=str(d["started"]))
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class RunRecord:
    """``run.json``: a result's status and who holds it."""

    status: RunStatus
    owner: Owner | None
    #: The invocation that claimed it (``.invocations/<seq>.json``).
    invocation: int | None
    started: str | None
    finished: str | None
    #: ``failure.reason`` of a failed result; ``None`` for one dropped without a
    #: reason, and for any other status.
    failure: str | None
    #: The run's inputs as given (the parsed command line and config file).
    given: dict = field(default_factory=dict)


def read_run_record(folder: Path) -> RunRecord | None:
    rec = _read_record(folder / RUN_FILE)
    if rec is None:
        return None
    status = rec.get("status")
    if status not in ("running", "completed", "failed"):
        raise UnsupportedRecord(f"{folder / RUN_FILE} has status {status!r}")
    failure = rec.get("failure")
    inv = rec.get("invocation")
    given = rec.get("given")
    return RunRecord(
        status=status,
        owner=Owner.from_json(rec.get("owner")),
        invocation=inv if isinstance(inv, int) else None,
        started=rec.get("started"),
        finished=rec.get("finished"),
        failure=str(failure["reason"]) if isinstance(failure, dict) and failure.get("reason") else None,
        given=given if isinstance(given, dict) else {},
    )


# ----------------------------------------------------------------------------
# Liveness
# ----------------------------------------------------------------------------


def _short_host(host: str) -> str:
    return host.split(".", 1)[0].lower()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def _owner_beats(root: Path) -> set[Owner]:
    """The owners whose beat file is present. camdl removes a process's beat
    file when its command ends, so a present file is a process that has not
    ended cleanly: still running, or killed."""
    out: set[Owner] = set()
    try:
        with os.scandir(root / OWNERS_DIR) as it:
            paths = [Path(e.path) for e in it if e.name.endswith(".json")]
    except OSError:
        return out
    for p in paths:
        try:
            o = Owner.from_json(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
        if o is not None:
            out.add(o)
    return out


@dataclass(frozen=True)
class Liveness:
    """Which owners are alive, judged once per read of the store.

    On the owner's host, an owner is alive while its process exists and its
    beat file is present. On another host this reader cannot see the process;
    camdl's own rule there is to watch the beat counter over a few intervals,
    which a reader answering one request cannot do, so the owner is reported
    alive while its beat file is present and the caller's progress snapshot
    (``fresh``) decides."""

    beating: frozenset[Owner]
    host: str

    @classmethod
    def read(cls, root: Path) -> "Liveness":
        return cls(frozenset(_owner_beats(root)), _short_host(socket.gethostname()))

    def alive(self, owner: Owner | None) -> bool | None:
        """``True`` or ``False`` on this host; ``None`` for an owner on another
        host whose beat file is present (the caller decides by freshness)."""
        if owner is None or owner not in self.beating:
            return False
        if _short_host(owner.host) == self.host:
            return _pid_alive(owner.pid)
        return None


@dataclass(frozen=True)
class Progress:
    """A progress file (``progress.json`` in a run folder,
    ``.invocations/<seq>.progress.json`` for a batch): camdl wraps the
    command's own progress record as ``{"format", "at", "snapshot"}``."""

    #: When camdl wrote it, in unix seconds; ``None`` if unparseable.
    at: float | None
    snapshot: dict


def _iso_seconds(text: object) -> float | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def read_progress(path: Path) -> Progress | None:
    try:
        rec = _read_record(path)
    except UnsupportedRecord:
        return None
    snap = rec.get("snapshot") if rec is not None else None
    if not isinstance(snap, dict):
        return None
    return Progress(at=_iso_seconds(rec.get("at")), snapshot=snap)


# ----------------------------------------------------------------------------
# Invocations and labels
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class InvocationResult:
    kind: str
    run_id: str
    labels: dict
    outcome: Outcome


@dataclass(frozen=True)
class Invocation:
    """One command run, from ``.invocations/<seq>.json``.

    camdl writes it twice: when the command first calls the store (status
    ``running``, with only the results planned by then) and when it ends
    (``ended``, every result with its outcome). A batch's results are listed
    only in the second version; while it runs, its count is in
    ``<seq>.progress.json``."""

    seq: int
    status: InvocationStatus
    label: str | None
    argv: list[str]
    given: dict
    results: list[InvocationResult]
    owner: Owner | None
    started: str | None
    ended: str | None

    @property
    def command(self) -> str:
        """The subcommand, e.g. ``fit`` or ``profile`` (``argv[1]``)."""
        return self.argv[1] if len(self.argv) > 1 else ""

    def all_completed(self) -> bool:
        return self.status == "ended" and all(
            r.outcome in ("computed", "reused") for r in self.results
        )


@lru_cache(maxsize=4096)
def _parse_invocation(path: Path, _mtime_ns: int, _size: int) -> Invocation | None:
    """One invocation record, cached on its path, mtime and size: a finished
    record never changes, and a store holds thousands of them."""
    rec = _read_record(path)
    if rec is None or rec.get("record") != "invocation":
        return None
    results = [
        InvocationResult(
            kind=str(r.get("kind") or ""),
            run_id=str(r.get("run_id") or ""),
            labels=r.get("labels") if isinstance(r.get("labels"), dict) else {},
            outcome=r.get("outcome"),
        )
        for r in rec.get("results") or []
        if isinstance(r, dict)
    ]
    given = rec.get("given")
    return Invocation(
        seq=int(rec["seq"]),
        status=rec.get("status"),
        label=rec.get("label") or None,
        argv=[str(a) for a in rec.get("argv") or []],
        given=given if isinstance(given, dict) else {},
        results=results,
        owner=Owner.from_json(rec.get("owner")),
        started=rec.get("started"),
        ended=rec.get("ended"),
    )


def read_invocations(root: Path) -> list[Invocation]:
    """Every invocation record, in sequence order."""
    out: list[Invocation] = []
    try:
        with os.scandir(root / INVOCATIONS_DIR) as it:
            entries = [e for e in it if e.name.endswith(".json") and e.name[:-5].isdigit()]
    except OSError:
        return out
    for e in entries:
        try:
            st = e.stat()
        except OSError:
            continue
        inv = _parse_invocation(Path(e.path), st.st_mtime_ns, st.st_size)
        if inv is not None:
            out.append(inv)
    return sorted(out, key=lambda i: i.seq)


def invocation_progress(root: Path, seq: int) -> Progress | None:
    """A running batch's progress (``.invocations/<seq>.progress.json``)."""
    return read_progress(root / INVOCATIONS_DIR / f"{seq}.progress.json")


def result_labels(invocations: list[Invocation]) -> dict[str, dict]:
    """Each run id's labels, as the latest invocation that listed it gave them
    (a problem's ``model``, a fit run's ``method`` and ``seed``). camdl keeps a
    result's display names only here: folder names are not read."""
    out: dict[str, dict] = {}
    for inv in invocations:
        for r in inv.results:
            if r.labels:
                out[r.run_id] = r.labels
    return out


def folders_by_id(root: Path, kind: str, depth: int) -> dict[str, Path]:
    """Every result folder of ``kind`` (``depth`` levels below each build's kind
    directory), keyed by the run id its ``identity.json`` holds."""
    out: dict[str, Path] = {}
    for kd in kind_dirs(root, kind):
        for folder in walk(kd, depth):
            ident = read_identity(folder)
            if ident is not None and ident.kind == kind:
                out[ident.run_id] = folder
    return out


def current_labels(invocations: list[Invocation]) -> dict[str, str]:
    """Each run id's current label: the label of the latest invocation, among
    those labelled ``L`` whose results all completed, that lists it — camdl's
    rule for ``@L``. A run named by two labels takes the later invocation's."""
    holders: dict[str, Invocation] = {}
    for inv in invocations:
        if inv.label and inv.all_completed():
            holders[inv.label] = inv  # sequence order: the latest wins
    out: dict[str, tuple[int, str]] = {}
    for label, inv in holders.items():
        for r in inv.results:
            if r.run_id not in out or out[r.run_id][0] < inv.seq:
                out[r.run_id] = (inv.seq, label)
    return {rid: label for rid, (_seq, label) in out.items()}
