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

from . import ingest
from .state import ChainBuffer, RunMeta, RunState, Status


def classify(rs: RunState, now: float) -> Status:
    """Status from camdl's ``progress.json`` heartbeat when present (terminal
    states win regardless of freshness; a fresh ``running`` beat is live), else
    the ``.lock`` PID + presence of draws."""
    # An MLE ('scout') fit has no chains; it's discovered only once its optimizer
    # wrote the point estimate, so it's a completed fit.
    if rs.meta.fit_kind == "mle":
        return Status.DONE
    prog = rs.progress
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
    live = ingest.stage_is_live(rs.meta.status_dir)
    has_rows = any(buf.n for buf in rs.chains.values())
    if has_rows:
        if live:
            return Status.RUNNING
        # A dead process with trace rows is NOT proof of completion: a killed,
        # crashed, or OOM'd stage leaves partial per-chain traces but never
        # writes the pooled ``draws.tsv``. Only call it done if that completion
        # artifact exists; otherwise it terminated without finishing.
        if ingest.stage_completed(rs.meta.posterior_dir):
            return Status.DONE
        return Status.STALLED
    return Status.WARMING if live else Status.STALLED


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
