"""Run-state assembly + status classification — the shiny-free core the API and
the tests share.

``build_run_state(meta)`` tail-reads a run's chains and attaches its priors /
progress / authoritative summary, then ``classify(rs, now)`` tags it
``running | warming | done | failed | stalled``. The run's ``run.json`` decides:
``completed`` is done and ``failed`` failed; a ``running`` record is live while
its owner is, and stalled once the owner is gone (killed, crashed or OOM'd
without ending its claim).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import ingest
from .state import ChainBuffer, RunMeta, RunProgress, RunState, Status


def classify(rs: RunState, now: float) -> Status:
    """Status for an assembled run — see :func:`classify_from`."""
    return classify_from(rs.meta, rs.progress, now)


def classify_from(meta: RunMeta, prog: RunProgress | None, now: float) -> Status:
    """Status from the run's record, and for a running one its owner's
    liveness and heartbeat phase.

    An owner on another host cannot be seen from here (``meta.owner_alive`` is
    ``None``); a fresh ``running`` heartbeat then counts as alive."""
    if meta.record_status == "completed":
        return Status.DONE
    if meta.record_status == "failed":
        return Status.FAILED
    alive = meta.owner_alive
    if alive is None:
        alive = prog is not None and prog.state == "running" and ingest.progress_is_fresh(prog, now)
    if not alive:
        return Status.STALLED
    return Status.WARMING if prog is not None and prog.phase == "burn_in" else Status.RUNNING


@dataclass(frozen=True)
class RunOverview:
    """What the run LIST needs, gathered without parsing a single trace row.

    Every field here is available from metadata, ``progress.json``, a stat, or
    the last line of a trace. Building a full :class:`RunState` to obtain them
    re-parsed the whole store on every poll."""

    status: Status
    chain_ids: list[int]
    max_iter: int | None
    updated_at: float
    progress: RunProgress | None


#: The files that date an optimizer leaf: its point estimate, its restart table
#: (appended as starts finish), its heartbeat (rewritten every few seconds while
#: it runs) and its run record.
_MLE_DATING = ("mle_params.toml", "chain_results.tsv", "chain_evaluations.tsv",
               "progress.json", "run.json")


def _mle_mtime(meta: RunMeta) -> float:
    """When an optimizer leaf last wrote anything, so a running optimizer sorts
    by real recency in the run list rather than to the bottom."""
    out = 0.0
    for name in _MLE_DATING:
        try:
            out = max(out, (meta.leaf_dir / name).stat().st_mtime)
        except OSError:
            continue
    return out


def build_run_overview(meta: RunMeta, now: float | None = None) -> RunOverview:
    """The cheap projection of a run, for the list.

    Chain ids come from the discovered paths, recency from their mtimes, and
    how far each chain has got from :func:`ingest.read_last_iter`, which reads
    one block from the end of each trace rather than all of it."""
    max_iter: int | None = None
    max_mtime = 0.0
    for path in meta.chain_paths.values():
        last = ingest.read_last_iter(path)
        if last is not None and (max_iter is None or last > max_iter):
            max_iter = last
        try:
            max_mtime = max(max_mtime, path.stat().st_mtime)
        except OSError:
            pass
    if meta.fit_kind == "mle":
        max_mtime = _mle_mtime(meta)
    prog = ingest.read_progress(meta.leaf_dir)
    return RunOverview(
        status=classify_from(meta, prog, now or time.time()),
        chain_ids=sorted(meta.chain_paths),
        max_iter=max_iter,
        updated_at=max_mtime,
        progress=prog,
    )


def build_run_state(meta: RunMeta) -> RunState:
    """Assemble a full :class:`RunState` for one run: tail-read every chain from
    offset 0, attach priors / progress / authoritative summary, and classify."""
    rs = RunState(meta=meta)
    max_mtime = 0.0
    for cid, path in meta.chain_paths.items():
        buf = ChainBuffer(cid=cid, path=path)
        ingest.tail_chain(buf)  # full read from offset 0
        rs.chains[cid] = buf
        try:
            max_mtime = max(max_mtime, path.stat().st_mtime)
        except OSError:
            pass
    # An MLE fit has no chains; date it by its own files (see _mle_mtime).
    if meta.fit_kind == "mle":
        max_mtime = _mle_mtime(meta)
    rs.priors = ingest.extract_priors(meta)
    rs.progress = ingest.read_progress(meta.leaf_dir)
    rs.summary = ingest.read_chain_summary(meta.leaf_dir)
    rs.updated_at = max_mtime
    rs.status = classify(rs, time.time())
    return rs
