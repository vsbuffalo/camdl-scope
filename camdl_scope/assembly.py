"""Run-state assembly + status classification — the shiny-free core the API and
the tests share.

``build_run_state(meta)`` tail-reads a run's chains and attaches its priors /
progress / authoritative summary, then ``classify(rs, now)`` tags it
``running | warming | done | failed | stalled`` from camdl's ``progress.json``
heartbeat (terminal states win; a fresh ``running`` beat is live) or, absent a
heartbeat, the seed ``.lock`` PID plus whether the stage wrote its pooled
``draws.tsv`` — a dead process with only partial per-chain traces and no
``draws.tsv`` is ``stalled`` (killed mid-run), not ``done``.

The heartbeat is read from ``meta.status_dir``, which is the stage that is
sampling now rather than the stage whose draws we read: a re-run under new
settings leaves a finished stage and a live one side by side in the same run
dir, and the run is live if *any* of its stages is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import ingest
from .state import ChainBuffer, RunMeta, RunProgress, RunState, Status


def classify(rs: RunState, now: float) -> Status:
    """Status for an assembled run — see :func:`classify_from`, which holds the
    policy so the run list can reach the same verdict without parsing traces."""
    return classify_from(
        rs.meta, rs.progress, any(buf.n for buf in rs.chains.values()), now
    )


def classify_from(
    meta: RunMeta, prog: RunProgress | None, has_rows: bool, now: float
) -> Status:
    """Status from camdl's ``progress.json`` heartbeat when present (terminal
    states win regardless of freshness; a fresh ``running`` beat is live), else
    the ``.lock`` PID + presence of draws.

    Takes primitives rather than a ``RunState`` because the run list must reach
    this verdict for every run on every poll, and assembling a RunState to get
    one boolean (``has_rows``) meant parsing every chain of every fit."""
    # An MLE ('scout') fit has no chains; it's discovered only once its optimizer
    # wrote the point estimate, so it's a completed fit.
    if meta.fit_kind == "mle":
        return Status.DONE
    if prog is not None:
        if prog.state == "done":
            return Status.DONE
        if prog.state == "failed":
            return Status.FAILED
        if prog.state == "running":
            if not ingest.progress_is_fresh(prog, now):
                return Status.STALLED
            return Status.WARMING if prog.phase == "burn_in" else Status.RUNNING
        return Status.DONE
    live = ingest.stage_is_live(meta.status_dir)
    if has_rows:
        if live:
            return Status.RUNNING
        # A dead process with trace rows is NOT proof of completion: a killed,
        # crashed, or OOM'd stage leaves partial per-chain traces but never
        # writes the pooled ``draws.tsv``. Only call it done if that completion
        # artifact exists; otherwise it terminated without finishing.
        if ingest.stage_completed(meta.posterior_dir):
            return Status.DONE
        return Status.STALLED
    return Status.WARMING if live else Status.STALLED


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
        try:
            max_mtime = (meta.posterior_dir / "mle_params.toml").stat().st_mtime
        except OSError:
            pass
    prog = ingest.read_progress(meta.status_dir)
    return RunOverview(
        status=classify_from(meta, prog, max_iter is not None, now or time.time()),
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
    # camdl's declared diagnostic columns, so a sampler column the watcher has
    # never heard of is still classified as a diagnostic rather than mistaken
    # for a parameter (see RunMeta.column_roles).
    diag_cols = frozenset(
        c for c, role in meta.column_roles.items() if role == "diagnostic"
    )
    for cid, path in meta.chain_paths.items():
        buf = ChainBuffer(cid=cid, path=path, diagnostic_cols=diag_cols)
        ingest.tail_chain(buf)  # full read from offset 0
        rs.chains[cid] = buf
        try:
            max_mtime = max(max_mtime, path.stat().st_mtime)
        except OSError:
            pass
    # An MLE fit has no chains — date it by its point-estimate artifact so it
    # sorts by real recency in the run list, not to the bottom on a 0 timestamp.
    if meta.fit_kind == "mle":
        try:
            max_mtime = (meta.posterior_dir / "mle_params.toml").stat().st_mtime
        except OSError:
            pass
    rs.priors = ingest.extract_priors(meta)
    # Progress comes from the stage that is sampling; the summary from the stage
    # we read — camdl's R̂/ESS describe the draws on screen, not the new sampler.
    rs.progress = ingest.read_progress(meta.status_dir)
    rs.summary = ingest.read_chain_summary(meta.posterior_dir)
    rs.updated_at = max_mtime
    rs.status = classify(rs, time.time())
    return rs
