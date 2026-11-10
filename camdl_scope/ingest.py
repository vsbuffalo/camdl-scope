"""Ingest — the only module that knows the camdl run-store layout.

Three jobs:
  * ``discover_runs(store)``      -> [RunMeta]
  * ``tail_chain(buf)``           -> rows appended since last call (tail-safe)
  * ``extract_priors(meta)``      -> {param: PriorSpec}

Store layout (content-addressed, nested)::

    <store>/<stem>-<fit hash>/         # a "segment": one problem, one config
        fit.meta.json
        fit.toml.original
        predict/<run_id12>[-<opts8>]/  # `fit predict` output, one per run (and
                                       #   option set); no seed_* children
        <method dir>/                  # pgas-<hash>, if2-<hash>, nl-sbplx-<hash>
            seed_<n>-<hash>/
                draws.tsv              # the pooled posterior, written at the end
                mle_params.toml        # …or a point estimate, for an optimizer
                chain_<k>/
                    trace.tsv          # appended live by the sampler
                    trajectories/

Two layouts coexist. A fit.toml used to declare a ``[stages]`` map of named
stages run in order, and the directory was ``<NN>-<stage name>-<hash>``
(``01-posterior-89a2e6fc``). It now declares one ``[method]`` table, and the
directory is ``<algorithm>-<hash>`` (``pgas-79d6825a``) — no ordinal, no
user-chosen name. camdl does not rewrite segments written before the split, so
a store carries both and this module reads a leaf by what it contains rather
than by what its directory is called.

Edge cases handled:
  * Several method dirs in one segment (a resume leaves an empty stub; a re-run
    under changed sampler settings adds a sibling with the same algorithm
    label) — pick the one whose chains have non-empty trace.tsv.
  * A re-run reuses the segment and adds a *new* method dir beside the finished
    one. Which leaf we read (the one holding a posterior) and which leaf answers
    "is this run sampling right now" are then different questions, so discovery
    resolves both.
  * A point-estimate leaf beside a posterior leaf in one segment — classified by
    artifact (``draws.tsv``/``chain_*/trace.tsv`` vs ``mle_params.toml``).
  * Torn final line — the sampler appends, so the last line may be partial.
    We read only up to the last newline and remember that byte offset.
  * MH vs PGAS column schema — normalized to a ``draw`` iteration column
    plus recognized aux columns; everything else is an estimated parameter.

This module is deliberately isolated: if camdl grows a sanctioned
``camdl watch`` API later, swap this file and nothing downstream changes.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import tomllib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

import time

from .docs import ModelDocs
from .schema import ObsSchema
from . import fitmeta
from .verdict import UnsupportedSchema, parse_number
from .state import (
    AUX_COLUMNS,
    ITER_COL,
    Backend,
    ChainBuffer,
    ChainSummary,
    Finding,
    PriorFamily,
    PriorSpec,
    RunMeta,
    RunProgress,
    Severity,
)

# A run's heartbeat (progress.json) is written every 5–10 s; allow a few missed
# beats before a still-"running" heartbeat is treated as stale (presumed dead).
PROGRESS_STALE_S = 30.0

# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------


def _chain_id(p: Path) -> int | None:
    m = re.fullmatch(r"chain_(\d+)", p.name)
    return int(m.group(1)) if m else None


def _seed_chain_paths(seed_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for cd in sorted(seed_dir.glob("chain_*")):
        cid = _chain_id(cd)
        if cid is None:
            continue
        tp = cd / "trace.tsv"
        if tp.exists():
            out[cid] = tp
    return out


def _nonempty(paths: dict[int, Path]) -> dict[int, Path]:
    return {c: p for c, p in paths.items() if p.exists() and p.stat().st_size > 0}


def _pid_alive(pid: int) -> bool:
    """True if a process with this PID currently exists on the local host."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by another user
    return True


def read_progress(seed_dir: Path) -> RunProgress | None:
    """Parse a stage's ``progress.json`` heartbeat (gh#278), or ``None`` if the
    run predates the heartbeat / the file is absent or unreadable.

    The ``state`` field is an externally-tagged ADT: the string ``"done"``, or
    ``{"running": {phase, step, total}}`` / ``{"failed": {reason}}``."""
    try:
        raw = json.loads((seed_dir / "progress.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    state = raw.get("state")
    phase = step = total = reason = None
    if isinstance(state, str):
        name = state  # unit variant, e.g. "done"
    elif isinstance(state, dict) and state:
        name = next(iter(state))  # "running" | "failed"
        body = state[name]
        if isinstance(body, dict):
            phase, step, total = body.get("phase"), body.get("step"), body.get("total")
            reason = body.get("reason")
    else:
        return None
    return RunProgress(
        state=name, updated_at=raw.get("updated_at"), pid=raw.get("pid"),
        phase=phase, step=step, total=total, reason=reason,
    )


def progress_is_fresh(prog: RunProgress, now: float | None = None) -> bool:
    """A ``running`` heartbeat whose ``updated_at`` is within the staleness
    window — the cross-host liveness signal (camdl's ``liveness()`` policy)."""
    if prog.updated_at is None:
        return True
    return (now or time.time()) - prog.updated_at <= PROGRESS_STALE_S


def stage_is_live(seed_dir: Path) -> bool:
    """True if the sampler owning this seed stage is still going.

    Prefers the ``progress.json`` heartbeat (gh#278): live iff its state is
    ``running`` and the heartbeat is fresh — cross-host and PID-reuse-proof.
    Falls back to the seed dir's ``.lock`` PID for runs that predate the
    heartbeat (same-host only; a stale PID marks an abandoned stub).
    """
    prog = read_progress(seed_dir)
    if prog is not None:
        return prog.state == "running" and progress_is_fresh(prog)
    lock = seed_dir / ".lock"
    try:
        pid = int(lock.read_text().strip())
    except (OSError, ValueError):
        return False
    return _pid_alive(pid)


def stage_completed(seed_dir: Path) -> bool:
    """True if the stage finished and wrote its pooled posterior (``draws.tsv``).

    A posterior stage writes its completion artifacts only at the end —
    ``draws.tsv`` (the pooled, thinned posterior, and the input ``camdl fit
    predict`` re-simulates from), alongside ``fit_state.toml`` and the summary
    JSONs. A stage that was killed, crashed, or OOM'd leaves partial per-chain
    ``trace.tsv`` files and *none* of these. ``draws.tsv`` is the load-bearing
    one, so we key on it: no draws.tsv means there is no posterior to predict
    from, whatever the trace files contain.
    """
    draws = seed_dir / "draws.tsv"
    try:
        return draws.stat().st_size > 0
    except OSError:
        return False


# ----------------------------------------------------------------------------
# Authoritative end-of-stage diagnostics (camdl's own summary + findings)
# ----------------------------------------------------------------------------

# camdl's diagnostics.json severity strings -> our enum.
_SEVERITY_MAP = {
    "error": Severity.ERROR,
    "warning": Severity.WARN,
    "warn": Severity.WARN,
    "info": Severity.INFO,
}


def _read_findings(seed_dir: Path) -> list[Finding]:
    """Parse ``diagnostics.json`` — a list of typed, severity-tagged findings —
    or ``[]`` if absent/unreadable. Each entry's ``kind`` is an internally-tagged
    object ``{"type": ..., <payload>}``; we split the tag from the payload.

    A payload ``chain_id`` is kept as written. camdl counts chains from 0
    everywhere (its message, ``chain_id`` here and in ``events.jsonl``,
    ``chain_starts.tsv`` and the ``chain_N/`` directories), so it already names
    a chain the way the directories do."""
    try:
        raw = json.loads((seed_dir / "diagnostics.json").read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[Finding] = []
    for d in raw:
        if not isinstance(d, dict):
            continue
        kind = d.get("kind")
        if isinstance(kind, dict):
            ktype = str(kind.get("type", "unknown"))
            param = kind.get("param")
            detail = {k: v for k, v in kind.items() if k != "type"}
        else:
            ktype, param, detail = str(kind or "unknown"), None, {}
        out.append(
            Finding(
                kind=ktype,
                severity=_SEVERITY_MAP.get(str(d.get("severity", "")).lower(), Severity.WARN),
                message=str(d.get("message", "")),
                param=str(param) if param is not None else None,
                detail=detail,
            )
        )
    return out


def _read_summary_json(seed_dir: Path) -> dict | None:
    """The leaf's authoritative summary — ``<algorithm>_summary.json`` — or
    ``None``. Every sampler writes the same ``rhat`` / ``ess`` / ``thin`` keys.

    Matched by suffix rather than against a list of algorithm names. camdl names
    the file after whichever sampler wrote it, and a list only covers the ones it
    was written with: an ``mh_summary.json`` present in a store today reaches the
    viewer with no R̂, no ESS and no findings, because ``mh`` was not on the
    list."""
    for p in sorted(seed_dir.glob("*_summary.json")):
        try:
            payload = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _normalize_acceptance(summary: dict) -> list[list[float]] | None:
    """camdl writes PGAS acceptance as ``acceptance_rates`` (``[chain][param]``)
    and PMMH as ``acceptance_rate`` (per-chain scalar). Normalize both to
    ``[chain][*]`` so a single reducer covers them."""
    rates = summary.get("acceptance_rates")
    if isinstance(rates, list) and rates:
        try:
            return [[float(x) for x in row] for row in rates]
        except (TypeError, ValueError):
            return None
    scalar = summary.get("acceptance_rate")
    if isinstance(scalar, list) and scalar:
        try:
            return [[float(x)] for x in scalar]
        except (TypeError, ValueError):
            return None
    if isinstance(scalar, (int, float)):
        return [[float(scalar)]]
    return None


def read_chain_summary(seed_dir: Path) -> ChainSummary | None:
    """camdl's authoritative end-of-stage diagnostics, or ``None`` if neither a
    ``*_summary.json`` nor ``diagnostics.json`` is present yet (i.e. the stage
    is still live and only the streamed trace exists).

    Merges the numeric summary (R̂, ESS combined + per-chain, acceptance, PMMH
    MAP) with the typed findings list — camdl's own verdict, computed with its
    own thresholds, so the watcher and ``camdl fit summary`` agree."""
    found = _read_summary_json(seed_dir)
    findings = _read_findings(seed_dir)
    if found is None:
        if not findings:
            return None
        # Findings without a numeric summary (unusual) — still surface the verdict.
        return ChainSummary(stage="", n_chains=0, rhat={}, ess={},
                            ess_per_chain={}, findings=findings)

    s = found

    # camdl's number grammar: a number, "NaN", "inf", "-inf", or null for
    # not computed (verdict.parse_number). A non-finite value is kept as the
    # float it names; a null is absent, never a number.
    def _fdict(key: str) -> dict[str, float]:
        d = s.get(key) or {}
        if not isinstance(d, dict):
            return {}
        return {k: x for k, v in d.items() if (x := parse_number(v)) is not None}

    ess_raw = s.get("ess") or {}
    ess = {k: parse_number(v) for k, v in ess_raw.items()} \
        if isinstance(ess_raw, dict) else {}
    epc_raw = s.get("ess_per_chain") or {}
    ess_per_chain = (
        {k: [float(x) for x in v] for k, v in epc_raw.items() if isinstance(v, list)}
        if isinstance(epc_raw, dict) else {}
    )
    map_params = _fdict("map_params") or None
    vintage = (
        "rank_bulk"
        if s.get("schema") or "rhat_bulk" in s or "rhat_classic" in s
        else "classic"
    )
    refused_raw = s.get("rhat_not_reported") or {}
    refused = (
        {
            str(k): (v if isinstance(v, str) else str((v or {}).get("type", v)))
            for k, v in refused_raw.items()
        }
        if isinstance(refused_raw, dict) else {}
    )
    return ChainSummary(
        # camdl's current summaries name the method, not a stage.
        stage=str(s.get("stage") or s.get("method") or ""),
        stats_vintage=vintage,
        rhat_refused=refused,
        n_chains=int(s.get("n_chains", 0) or 0),
        rhat=_fdict("rhat"),
        ess=ess,
        ess_per_chain=ess_per_chain,
        acceptance_rates=_normalize_acceptance(s),
        map_params=map_params,
        map_loglik=(float(s["map_loglik"]) if s.get("map_loglik") is not None else None),
        map_chain=(int(s["map_chain"]) if s.get("map_chain") is not None else None),
        findings=findings,
        thin=int(s.get("thin", 1) or 1),
        wall_time_secs=_read_wall_time_secs(seed_dir),
    )


def _read_wall_time_secs(seed_dir: Path) -> float | None:
    """Stage wall-clock from the stage leaf's ``run.json``
    (``inputs.wall_time_seconds``) — the denominator for ESS/second. ``None`` on
    older runs that predate the field or when ``run.json`` is absent/unreadable."""
    p = seed_dir / "run.json"
    try:
        rec = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    inputs = rec.get("inputs") if isinstance(rec, dict) else None
    if not isinstance(inputs, dict):
        return None
    v = inputs.get("wall_time_seconds")
    return float(v) if isinstance(v, (int, float)) else None


def read_column_roles(seed_dir: Path) -> dict[str, str]:
    """camdl's declared role for each trace column, from the stage's
    ``run.json`` (``output_schema["chain_{n}/trace.tsv"].columns``): e.g.
    ``sweep -> iteration``, ``n_divergent -> diagnostic``, ``r_eff ->
    param_estimated``.

    This is the contract that lets the watcher classify a column it has never
    heard of. ``{}`` when the file is absent or predates the schema — the
    caller then falls back to :data:`camdl_scope.state.AUX_COLUMNS`."""
    try:
        rec = json.loads((seed_dir / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    schema = rec.get("output_schema") if isinstance(rec, dict) else None
    if not isinstance(schema, dict):
        return {}
    out: dict[str, str] = {}
    for entry in schema.values():
        cols = entry.get("columns") if isinstance(entry, dict) else None
        if not isinstance(cols, list):
            continue
        for c in cols:
            if isinstance(c, dict) and isinstance(c.get("name"), str):
                role = c.get("role")
                if isinstance(role, str):
                    out[c["name"]] = role
    return out


def _read_algorithm_config(seed_dir: Path) -> dict[str, str | float | int]:
    """The sampler's static configuration from the stage leaf's ``run.json``
    (``inputs.algorithm``) — particle count, sweep budget, optimizer tolerance.

    The keys differ by algorithm (PGAS records ``particles``/``sweeps``, MH
    ``iterations``, an NLopt scout ``max_evals``/``tolerance``), so we pass
    through whatever camdl wrote instead of naming a fixed set: a sampler that
    starts recording a new knob surfaces without a change here. Empty on runs
    that predate the field and when ``run.json`` is absent/unreadable."""
    p = seed_dir / "run.json"
    try:
        rec = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    inputs = rec.get("inputs") if isinstance(rec, dict) else None
    if not isinstance(inputs, dict):
        return {}
    cfg = inputs.get("algorithm")
    if not isinstance(cfg, dict):
        return {}
    out: dict[str, str | float | int] = {}
    for k, v in cfg.items():
        if isinstance(v, bool):
            # A JSON bool would validate against the numeric arm of the wire
            # union and reach the UI as a meaningless 1/0; keep it a word.
            out[str(k)] = str(v).lower()
        elif isinstance(v, (str, int, float)):
            out[str(k)] = v
        # Anything structured (a list, a nested block) has no compact rendering
        # in a one-line header strip, so it is dropped rather than stringified.
    return out


def _method_seed_dirs(run_dir: Path) -> list[Path]:
    """Every ``<method dir>/seed_*`` leaf of a segment, in directory then seed
    order.

    A segment accumulates method dirs: a resume, a crash-relaunch, or a re-run
    under changed sampler settings each add one (the layout hashes the settings
    into the directory name, so a changed setting is a new dir rather than an
    overwrite), and a second algorithm run on the same problem lands here too.

    The directory name is not parsed. A method dir is one holding ``seed_*``
    children; the segment's other children (``predict/``, ``prior_predictive/``)
    hold none. That reads
    both the current ``<algorithm>-<hash>`` and the older
    ``<NN>-<stage name>-<hash>``, and it also surfaces stages whose user-chosen
    name was never ``posterior`` — ``01-post-*``, ``01-fit-*``, both present in
    real stores — which the old name glob skipped without saying so."""
    out: list[Path] = []
    try:
        with os.scandir(run_dir) as entries:
            method_dirs = sorted(
                (e for e in entries if e.is_dir()), key=lambda e: e.name
            )
    except OSError:
        return out
    for entry in method_dirs:
        out.extend(sorted(Path(entry.path).glob("seed_*")))
    return out


def _mle_seed_dir(run_dir: Path) -> Path | None:
    """The first method leaf carrying ``mle_params.toml`` — camdl's marker for a
    point estimate rather than a posterior — or ``None``.

    Classified by that artifact, not by the directory name: an optimizer used to
    live in a ``01-scout-<hash>`` / ``01-mle-<hash>`` stage, and now sits in an
    ``if2-<hash>`` or ``nl-sbplx-<hash>`` method dir which may be a sibling of a
    posterior leaf under one segment. A posterior leaf never writes
    ``mle_params.toml``, so nothing has to be excluded by name."""
    for seed_dir in _method_seed_dirs(run_dir):
        if (seed_dir / "mle_params.toml").is_file():
            return seed_dir
    return None


#: A pre-split method dir, ``<ordinal>-<stage name>-<hash>``. The hash is the
#: last dash-free segment, so the stage name may itself contain dashes.
_LEGACY_STAGE_DIR = re.compile(r"\d+-(?P<name>.+)-[^-]+")


def _legacy_stage_name(dir_name: str) -> str:
    """The stage name a pre-split method dir was given in ``[stages]``, or ``""``
    for a current ``<algorithm>-<hash>`` dir, which has none."""
    m = _LEGACY_STAGE_DIR.fullmatch(dir_name)
    return m.group("name") if m else ""


def _method_dir_algorithm(dir_name: str) -> str:
    """The algorithm a current method dir is named for — ``pgas-79d6825a`` ->
    ``pgas``, ``nl-sbplx-4fd60eb3`` -> ``nl-sbplx`` — or ``""`` for a pre-split
    dir, whose middle is a user-chosen stage name and not an algorithm.

    Derived rather than matched against the eight algorithm labels camdl has
    today, so a sampler added upstream is named correctly here without a
    release."""
    body = dir_name.rsplit("-", 1)[0] if "-" in dir_name else ""
    if not body or body.split("-", 1)[0].isdigit():
        return ""
    return body


def _method_table(toml: dict, method_dir_name: str = "") -> dict:
    """The sampler settings for a fit: the ``[method]`` table current camdl
    writes, else the pre-split ``[stages.<name>]`` entry.

    One config declares one method, so ``[method]`` needs no name to look up.
    The ``[stages]`` map needs one, and the leaf's own directory carries it
    (``01-scout-9f2a`` -> ``[stages.scout]``). ``posterior`` is the fallback for
    a leaf whose directory names no declared stage — which is what the reader
    assumed unconditionally before, and why a stage a user called ``post`` or
    ``fit`` came through with no algorithm, no backend and no sweep target."""
    method = toml.get("method")
    if isinstance(method, dict) and method:
        return method
    stages = toml.get("stages")
    if not isinstance(stages, dict) or not stages:
        return {}
    for key in (_legacy_stage_name(method_dir_name), "posterior"):
        entry = stages.get(key) if key else None
        if isinstance(entry, dict):
            return entry
    return {}


def _live_stage_dir(run_dir: Path) -> Path | None:
    """The seed dir of the stage sampling *right now*, or ``None`` if no stage
    is. Freshest heartbeat wins if two stages claim to be running.

    Liveness is decided only by a fresh ``running`` ``progress.json`` (camdl's
    staleness contract, gh#278), never by the seed ``.lock``: a lock file is a
    same-host PID that outlives the abandoned stub that wrote it, and letting an
    abandoned stub speak for the run would drag a finished fit back to
    ``stalled``. A stage with no heartbeat therefore can never claim liveness
    away from the stage we read — which is what keeps pre-heartbeat runs
    classifying exactly as they did before.
    """
    best: tuple[float, Path] | None = None
    for seed_dir in _method_seed_dirs(run_dir):
        prog = read_progress(seed_dir)
        if prog is None or prog.state != "running" or not progress_is_fresh(prog):
            continue
        beat = float(prog.updated_at or 0.0)
        if best is None or beat > best[0]:
            best = (beat, seed_dir)
    return best[1] if best is not None else None


#: Bytes read from the end of a trace to find its last complete row. Comfortably
#: more than one row of the widest trace seen (20 params ≈ 500 B).
_TAIL_PROBE_BYTES = 16_384


def read_last_iter(path: Path) -> int | None:
    """The last iteration number in a trace, read from its TAIL.

    The run list needs only how far a chain has got, and parsing the whole file
    to learn it costs the same as parsing it to plot it: on a 3.6 GB store that
    was ~11 s per poll, every poll. Read the header to locate the iteration
    column, then one block from the end.

    ``None`` when the file has no complete data row (header only, or a torn
    first append), which the caller reads as "this chain has produced nothing"
    — the same conclusion a full parse would reach."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            header = fh.readline().decode("utf-8", "replace")
            if not header:
                return None
            cols = _normalize_header(header.rstrip("\n").split("\t"))
            try:
                idx = cols.index(ITER_COL)
            except ValueError:
                return None
            start = max(fh.tell(), size - _TAIL_PROBE_BYTES)
            fh.seek(start)
            chunk = fh.read()
    except OSError:
        return None
    text = chunk.decode("utf-8", "replace")
    lines = text.split("\n")
    # The sampler appends, so a final partial line is normal; drop it unless the
    # block ended cleanly. Anything before the first newline may also be a
    # fragment of a row that started before `start`.
    if lines and not text.endswith("\n"):
        lines.pop()
    for line in reversed(lines):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) <= idx:
            continue
        try:
            return int(float(fields[idx]))
        except ValueError:
            continue
    return None


def _pick_posterior_dir(
    run_dir: Path, *, include_warming: bool = False
) -> tuple[Path, Path, dict[int, Path], bool] | None:
    """Return (posterior_dir, seed_dir, chain_paths, has_draws) for the stage
    whose draws we read, or ``None`` if there's nothing worth surfacing.

    Prefer a stage that is *not abandoned* — one that wrote ``draws.tsv``, or
    one that is sampling right now (:func:`_live_stage_dir`) — over any partial
    stub, then the stage with the most non-empty chains, then the
    most-recently-modified. A resume/relaunch leaves an empty ``trace.tsv`` stub
    — those score zero non-empty chains and lose. A *crashed* relaunch can leave
    a stub with as many started chains as the real run and a newer mtime; the
    not-abandoned key keeps the finished stage from being masked by it.

    A re-run under new settings is the same shape as that crashed stub — same
    chain count, newer mtime — but it is alive, and its rows are the ones the
    watcher exists to show. Reading the superseded stage while the status line
    reads ``running`` would report the *old* fit's sweep count and traces as if
    they were the live run's. So a live stage that has rows wins; a live stage
    still in burn-in has nothing to show and yields to the completed sibling.

    When no stage has any draws yet (every ``trace.tsv`` is empty, as during
    burn-in), the run is hidden by default. With ``include_warming=True`` it is
    surfaced *only if the sampler is still live* (``stage_is_live``), with
    ``has_draws=False`` — that's a run warming up, not a killed stub. The
    returned ``chain_paths`` are the (currently empty) trace files, so the
    normal tail picks up rows the moment burn-in clears.
    """
    live_dir = _live_stage_dir(run_dir)
    candidates: list[tuple[bool, int, float, Path, Path, dict[int, Path]]] = []
    for seed_dir in _method_seed_dirs(run_dir):
        paths = _seed_chain_paths(seed_dir)
        ne = _nonempty(paths)
        if not paths:
            continue
        mtime = max((p.stat().st_mtime for p in paths.values()), default=0.0)
        # Rank not-abandoned first, then by #non-empty chains, then recency.
        # Use the full path set for display when none are non-empty yet.
        candidates.append(
            (stage_completed(seed_dir) or seed_dir == live_dir,
             len(ne), mtime, seed_dir.parent, seed_dir, ne or paths)
        )
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    _kept, n_ne, _mtime, pdir, seed_dir, paths = candidates[0]
    if n_ne > 0:
        return pdir, seed_dir, paths, True
    # No draws anywhere (burn-in, or a killed stub). A run can have several
    # empty stage dirs from relaunches/resumes — surface the *live* one, not
    # just the most-recent by mtime, so a relaunched fit isn't read off a dead
    # sibling stub.
    if include_warming:
        for _c, _n, _mt, pd_, sd_, pth_ in candidates:
            if stage_is_live(sd_):
                return pd_, sd_, pth_, False
    return None


def _read_fit_toml(run_dir: Path) -> dict:
    tp = run_dir / "fit.toml.original"
    if not tp.exists():
        return {}
    try:
        with tp.open("rb") as fh:
            return tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def _read_meta(run_dir: Path) -> dict:
    mp = run_dir / "fit.meta.json"
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _fit_toml_stem(meta: dict, run_dir: Path) -> str:
    """The config stem: ``fit.meta.json`` records the originating
    config path (``source_config``, or ``fit_toml_path`` in a version 1 record; e.g. ``natbc_dens_hierk_nc_pgas_long.toml``), the most
    reliable source. Fall back to the run-dir name prefix before the hash."""
    ftp = fitmeta.config_ref(meta)
    if ftp:
        return Path(ftp).stem
    name = run_dir.name
    return name.rsplit("-", 1)[0] if "-" in name else name


class LeafIdentity(NamedTuple):
    """What a leaf's ``run.json`` says about which run it is."""

    run_id: str | None  # camdl's run_id; names the run's predict/ directory
    label: str | None  # provenance.label, the run's own label


def _read_leaf_identity(seed_dir: Path) -> LeafIdentity:
    """The run's ``run_id`` and label, from the leaf's ``run.json``.

    camdl (gh#896, 0.2.0) gives a label to exactly one run and records it on
    that run's leaf (``provenance.label``), set by ``fit run --label L`` or
    ``camdl label <run_id> L``. A problem folder holding several runs can hold
    several labels, so the label is read from the leaf this entry shows, never
    from the folder. Labels an older camdl kept on the folder
    (``fit.meta.json``) are not read: camdl itself no longer reads them, and
    showing one would name a run that ``@L`` does not resolve to.

    Both fields are ``None`` when ``run.json`` is absent or unreadable (a leaf
    still warming up); ``label`` is ``None`` for an unlabelled run."""
    try:
        rec = json.loads((seed_dir / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        return LeafIdentity(None, None)
    if not isinstance(rec, dict):
        return LeafIdentity(None, None)
    run_id = rec.get("run_id")
    prov = rec.get("provenance")
    label = prov.get("label") if isinstance(prov, dict) else None
    return LeafIdentity(
        run_id if isinstance(run_id, str) and run_id else None,
        label if isinstance(label, str) and label else None,
    )


def _mle_run_meta(run_dir: Path, mle_seed: Path) -> RunMeta:
    """A ``RunMeta`` for a pure MLE (optimization) run. ``posterior_dir`` is
    reused to point at the optimizer's seed leaf (for status/progress);
    ``chain_paths`` is empty since there are no draws."""
    meta = _read_meta(run_dir)
    toml = _read_fit_toml(run_dir)
    model_path = fitmeta.model_ref(meta)
    model = Path(model_path).stem if model_path else run_dir.name

    config = _read_algorithm_config(mle_seed)
    algorithm, backend, _settings = _resolve_method(toml, config, mle_seed.parent.name)
    ident = _read_leaf_identity(mle_seed)

    return RunMeta(
        run_id=run_dir.name,
        run_dir=run_dir,
        posterior_dir=mle_seed,
        chain_paths={},
        model=model,
        algorithm=algorithm,
        backend=backend,
        estimated=list(meta.get("estimated", [])),
        target_sweeps=None,
        declared_burn_in=None,
        thin=1,
        fit_toml_stem=_fit_toml_stem(meta, run_dir),
        user_label=ident.label,
        leaf_run_id=ident.run_id,
        docs=ModelDocs.from_meta(meta),
        schema=ObsSchema.from_meta(meta),
        fit_kind="mle",
        algorithm_config=config,
    )


def _resolve_method(
    toml: dict, config: dict, method_dir_name: str
) -> tuple[str, Backend, dict]:
    """What produced this leaf, and the settings that may be attributed to it:
    ``(algorithm, backend, settings)``.

    Three sources carry this, and only two of them are about *this* leaf. The
    leaf's own ``run.json`` records the sampler and the settings that keyed the
    run, but camdl writes that block when the leaf finishes — it is ``null`` for
    the run you are watching. The method directory's name carries the algorithm
    and is there from the first sweep. The segment's ``fit.toml.original``
    carries what neither does (``burn_in``, ``thin``) but belongs to the
    *segment*, not the leaf: one segment holds every leaf of one problem, and
    the file on disk is whichever config was written into it last.

    So the config is read only when it declares the algorithm this leaf actually
    ran. Re-running one file after switching its algorithm leaves a PGAS leaf and
    an IF2 leaf side by side under a fit.toml describing only the second — and
    reporting that file's iteration budget and burn-in against the first fit
    would be wrong in a way nothing on screen could contradict."""
    leaf_algorithm = (
        str(config.get("algorithm") or "") or _method_dir_algorithm(method_dir_name)
    )
    declared = _method_table(toml, method_dir_name)
    if leaf_algorithm and str(declared.get("algorithm") or "") not in (
        "", leaf_algorithm
    ):
        declared = {}
    # The leaf's own record wins wherever the two overlap.
    settings = {**declared, **config}
    algorithm = leaf_algorithm or str(declared.get("algorithm") or "") or "unknown"
    try:
        backend = Backend(str(settings.get("backend") or "unknown"))
    except ValueError:
        backend = Backend.UNKNOWN
    return algorithm, backend, settings


#: Separates a problem folder's name from a leaf's in a seed run's id. Not a
#: character camdl writes in a directory name, and safe in a URL path.
LEAF_SEP = "~"

_SEED_RE = re.compile(r"^seed_(\d+)")


def _leaf_seed(seed_dir: Path) -> int | None:
    m = _SEED_RE.match(seed_dir.name)
    return int(m.group(1)) if m else None


class _Leaf(NamedTuple):
    pdir: Path
    seed_dir: Path
    chain_paths: dict[int, Path]


def _posterior_leaves(run_dir: Path, *, include_warming: bool) -> list[_Leaf]:
    """Every posterior run of a problem folder worth its own entry, the one a
    bare folder name opens first.

    The default is :func:`_pick_posterior_dir`'s choice, which keeps a live
    re-run ahead of the finished stage it supersedes and an abandoned stub
    behind both. Among finished replicate seeds of one method directory, which
    finished last is an accident of scheduling, so the default is the lowest
    seed instead. Every other leaf that finished (``draws.tsv``) or is live is
    listed after it, by method directory then seed; abandoned stubs are not."""
    picked = _pick_posterior_dir(run_dir, include_warming=include_warming)
    if picked is None:
        return []
    pdir, seed_dir, paths, _has_draws = picked
    default = _Leaf(pdir, seed_dir, paths)
    live = _live_stage_dir(run_dir)
    others: list[_Leaf] = []
    for sd in _method_seed_dirs(run_dir):
        if sd == seed_dir:
            continue
        all_paths = _seed_chain_paths(sd)
        ne = _nonempty(all_paths)
        if stage_completed(sd) and ne:
            others.append(_Leaf(sd.parent, sd, ne))
        elif sd == live and (ne or include_warming):
            others.append(_Leaf(sd.parent, sd, ne or all_paths))
    if stage_completed(seed_dir) and seed_dir != live:
        replicates = [o for o in others if o.pdir == pdir and stage_completed(o.seed_dir)
                      and len(o.chain_paths) >= len(paths)]
        lowest = min([default, *replicates],
                     key=lambda lf: (_leaf_seed(lf.seed_dir) is None, _leaf_seed(lf.seed_dir) or 0))
        if lowest is not default:
            others = [default, *(o for o in others if o is not lowest)]
            default = lowest
    others.sort(key=lambda lf: (lf.pdir.name, _leaf_seed(lf.seed_dir) or 0, lf.seed_dir.name))
    return [default, *others]


def _leaf_ids(run_dir: Path, leaves: list[_Leaf]) -> list[str]:
    """Entry ids for a folder's leaves: the folder name when it holds one run,
    else ``<folder>~<seed dir>``, with the method dir added only when two
    leaves share a seed dir name (the same seed under two method settings)."""
    if len(leaves) == 1:
        return [run_dir.name]
    names = [lf.seed_dir.name for lf in leaves]
    return [
        LEAF_SEP.join([run_dir.name, lf.seed_dir.name]) if names.count(lf.seed_dir.name) == 1
        else LEAF_SEP.join([run_dir.name, lf.pdir.name, lf.seed_dir.name])
        for lf in leaves
    ]


def _posterior_run_meta(
    run_dir: Path, leaf: _Leaf, run_id: str, *, folder_default: bool,
    live_stage_dir: Path | None,
) -> RunMeta:
    pdir, seed_dir, chain_paths = leaf
    meta = _read_meta(run_dir)
    toml = _read_fit_toml(run_dir)

    model_path = fitmeta.model_ref(meta)
    model = Path(model_path).stem if model_path else run_dir.name
    estimated = list(meta.get("estimated", []))

    config = _read_algorithm_config(seed_dir)
    algorithm, backend, settings = _resolve_method(toml, config, pdir.name)
    # Each sampler budgets its run in its own unit: PGAS counts ``sweeps``,
    # PMMH/MH/IF2 ``iterations``, NUTS ``samples`` drawn after ``warmup``.
    # They are the same number to a progress bar — how many iterations this
    # run is aiming for — so they collapse to one field.
    target = (
        settings.get("sweeps")
        or settings.get("iterations")
        or settings.get("samples")
    )
    burn_in = settings.get("burn_in")
    if burn_in is None:
        burn_in = settings.get("warmup")
    thin = settings.get("thin")
    ident = _read_leaf_identity(seed_dir)

    return RunMeta(
        run_id=run_id,
        run_dir=run_dir,
        posterior_dir=seed_dir,
        live_stage_dir=live_stage_dir,
        chain_paths=chain_paths,
        model=model,
        algorithm=algorithm,
        backend=backend,
        estimated=estimated,
        target_sweeps=int(target) if target is not None else None,
        declared_burn_in=int(burn_in) if burn_in is not None else None,
        thin=int(thin) if thin is not None else 1,
        fit_toml_stem=_fit_toml_stem(meta, run_dir),
        user_label=ident.label,
        leaf_run_id=ident.run_id,
        seed=_leaf_seed(seed_dir),
        folder_default=folder_default,
        docs=ModelDocs.from_meta(meta),
        schema=ObsSchema.from_meta(meta),
        column_roles=read_column_roles(seed_dir),
        algorithm_config=config,
    )


def resolve_run(metas: list[RunMeta], run_id: str) -> RunMeta | None:
    """The entry ``run_id`` names: an exact entry id, or a problem folder's
    name, which opens that folder's default run. A link written before a
    folder held several runs keeps working, and opens what the folder now
    defaults to."""
    exact = next((m for m in metas if m.run_id == run_id), None)
    if exact is not None:
        return exact
    return next(
        (m for m in metas if m.folder_default and m.run_dir.name == run_id), None
    )


def discover_runs(store: Path, *, include_warming: bool = False) -> list[RunMeta]:
    """Scan ``store`` for runs that have at least one non-empty chain trace —
    one entry per run, so a problem folder holding several (fit seeds, re-runs
    under changed settings) yields several, its default first.

    With ``include_warming=True``, also surface runs whose latest posterior
    stage has started but written no draws yet *and* whose sampler process is
    still alive — i.e. live runs in burn-in. Killed-mid-burn-in stubs (same
    empty files, dead PID) stay hidden.
    """
    store = Path(store)
    out: list[RunMeta] = []
    if not store.is_dir():
        return out
    for run_dir in sorted(store.iterdir()):
        if not run_dir.is_dir():
            continue
        leaves = _posterior_leaves(run_dir, include_warming=include_warming)
        if not leaves:
            # No posterior draws — but the segment may hold an optimizer leaf,
            # which has a point estimate instead of chains.
            mle_seed = _mle_seed_dir(run_dir)
            if mle_seed is not None:
                out.append(_mle_run_meta(run_dir, mle_seed))
            continue
        # A live stage listed as its own entry speaks for itself; otherwise
        # (a re-run still in burn-in, not surfaced) the default entry reports
        # it, so the folder does not read as idle while it samples.
        live = _live_stage_dir(run_dir)
        live_listed = any(lf.seed_dir == live for lf in leaves)
        for i, (leaf, rid) in enumerate(zip(leaves, _leaf_ids(run_dir, leaves))):
            out.append(_posterior_run_meta(
                run_dir, leaf, rid, folder_default=(i == 0),
                live_stage_dir=(_live_stage_dir(run_dir) if i == 0 and not live_listed else None),
            ))
    return out


# ----------------------------------------------------------------------------
# Tail-safe incremental read
# ----------------------------------------------------------------------------


def _normalize_header(cols: list[str]) -> list[str]:
    """Rename the iteration column (``sweep``/``step``) to the canonical
    ``draw``; leave everything else."""
    return [ITER_COL if c in ("sweep", "step") else c for c in cols]


def _parse_block(text: str, header: list[str]) -> pl.DataFrame | None:
    """Parse a block of TSV *data* rows (no header) given a known header.

    ``truncate_ragged_lines`` guards against a torn last row that slips
    through; we also pre-trim to the last newline in ``tail_chain`` so this
    is belt-and-suspenders."""
    if not text.strip():
        return None
    buf = io.StringIO("\t".join(header) + "\n" + text)
    try:
        df = pl.read_csv(
            buf,
            separator="\t",
            has_header=True,
            truncate_ragged_lines=True,
            infer_schema_length=10000,
        )
    except Exception:
        return None
    # Drop any row that came out ragged (nulls in the iteration col).
    if ITER_COL in df.columns:
        df = df.filter(pl.col(ITER_COL).is_not_null())
    return df if df.height else None


def tail_chain(buf: ChainBuffer) -> int:
    """Read rows appended to ``buf.path`` since ``buf.byte_offset``.

    Tail-safe: reads only up to the final newline, so a half-written last
    line is left for the next poll. Updates ``buf`` in place (iters, values,
    aux, header, byte_offset). Returns the number of new rows ingested.

    Handles file truncation/replacement (offset past EOF) by resetting.
    """
    path = buf.path
    if not path.exists():
        return 0
    size = path.stat().st_size
    if size < buf.byte_offset:
        # File shrank/was replaced -> re-read from scratch.
        buf.byte_offset = 0
        buf.iters = np.empty(0, dtype=np.int64)
        buf.values = {}
        buf.aux = {}
        buf.header = None

    with path.open("rb") as fh:
        # Ensure we have the header before reading data offsets.
        if buf.header is None:
            first = fh.readline()
            if not first.endswith(b"\n"):
                # Header itself not yet complete.
                return 0
            raw_header = first.decode("utf-8", "replace").rstrip("\n").split("\t")
            buf.header = _normalize_header(raw_header)
            buf.byte_offset = fh.tell()

        fh.seek(buf.byte_offset)
        chunk = fh.read()

    if not chunk:
        return 0

    # Trim to the last newline; keep the partial tail for next time.
    last_nl = chunk.rfind(b"\n")
    if last_nl == -1:
        return 0  # no complete line yet
    complete = chunk[: last_nl + 1]
    new_offset = buf.byte_offset + last_nl + 1

    text = complete.decode("utf-8", "replace")
    df = _parse_block(text, buf.header)
    buf.byte_offset = new_offset
    if df is None or df.height == 0:
        return 0

    _append_df(buf, df)
    return df.height


def _append_df(buf: ChainBuffer, df: pl.DataFrame) -> None:
    """Append a parsed block into the chain's numpy buffers."""
    cols = df.columns
    if ITER_COL not in cols:
        return
    new_iters = df[ITER_COL].cast(pl.Int64, strict=False).to_numpy()
    buf.iters = np.concatenate([buf.iters, new_iters])

    for col in cols:
        if col == ITER_COL:
            continue
        arr = df[col].cast(pl.Float64, strict=False).to_numpy()
        target = buf.aux if buf.is_diagnostic(col) else buf.values
        if col in target:
            target[col] = np.concatenate([target[col], arr])
        else:
            # Back-fill with NaN if this column appeared late (shouldn't, but
            # keeps arrays aligned).
            if buf.n > arr.shape[0]:
                pad = np.full(buf.n - arr.shape[0], np.nan)
                target[col] = np.concatenate([pad, arr])
            else:
                target[col] = arr


# ----------------------------------------------------------------------------
# Prior extraction
# ----------------------------------------------------------------------------

# Map camdl distribution keys -> (family, arg-rename map). One entry per family
# camdl can declare; the authority is `Density::kind_str` in camdl's
# rust/crates/sim/src/inference/prior.rs. A family missing from this table does
# not raise — it degrades to FLAT on the declared bounds — so the table falling
# behind camdl is a silent substitution of a uniform for whatever was written.
# That is what `_ParsedPrior.unsupported` records.
_TOML_FAMILY = {
    "normal": (PriorFamily.NORMAL, {"mu": "mu", "sigma": "sigma",
                                    "mean": "mu", "sd": "sigma"}),
    "log_normal": (PriorFamily.LOGNORMAL, {"mu": "mu", "sigma": "sigma"}),
    "half_normal": (PriorFamily.HALFNORMAL, {"sigma": "sigma"}),
    "beta": (PriorFamily.BETA, {"alpha": "alpha", "beta": "beta"}),
    "gamma": (PriorFamily.GAMMA, {"alpha": "alpha", "beta": "beta",
                                  "shape": "alpha", "rate": "beta"}),
    # rate, not scale: camdl writes `exponential(rate = R)` with rate = 1/mean
    # (language spec §prior families). Reading it as a scale would put the mass
    # at R instead of 1/R — for `rate = 5.99`, a mean of 6 rather than 0.17.
    "exponential": (PriorFamily.EXPONENTIAL, {"rate": "rate",
                                              "lambda": "rate"}),
    # camdl's canonical spelling is `lower = L, upper = U` (language spec
    # §prior families); the other aliases are kept for older TOML variants.
    "uniform": (PriorFamily.UNIFORM, {"lower": "lo", "upper": "hi",
                                      "lo": "lo", "hi": "hi",
                                      "low": "lo", "high": "hi",
                                      "min": "lo", "max": "hi"}),
    "log_uniform": (PriorFamily.LOGUNIFORM, {"lower": "lo", "upper": "hi",
                                             "lo": "lo", "hi": "hi"}),
    # In a .camdl the truncation points come from the parameter's `in [lo, hi]`
    # box, so the model spelling carries mean/sd only; the fit TOML may repeat
    # them as lower/upper, which camdl requires to equal the resolved bounds.
    "truncated_normal": (PriorFamily.TRUNCNORMAL, {"mean": "mu", "mu": "mu",
                                                   "sd": "sigma", "sigma": "sigma",
                                                   "lower": "lo", "upper": "hi"}),
}

# Same families, model-IR (.camdl) spelling: `~ log_normal(mu=, sigma=)` etc.
# The IR uses the same distribution names, so we share the table.
_IR_FAMILY = _TOML_FAMILY


class _ParsedPrior(NamedTuple):
    """One parsed prior declaration, before it is bound to a parameter name.

    ``unsupported`` is the distribution name as written when it is not in
    :data:`_TOML_FAMILY`. The parse still succeeds — ``family`` is FLAT and the
    bounds are kept, because the box is real information even when the density
    is lost — but the name travels with it so the loss can be reported."""

    family: PriorFamily
    args: dict[str, float]
    bounds: tuple[float, float] | None
    unsupported: str | None = None


def _prior_from_toml_entry(entry: dict) -> _ParsedPrior:
    """Parse one ``[estimate]`` TOML entry.

    The TOML shape is ``prior = { <family> = { <args> } }`` (camdl run spec
    §estimate). A single unrecognised key there is a declared prior this reader
    cannot draw, not an absent one, and is reported as such."""
    bounds = None
    if isinstance(entry.get("bounds"), list) and len(entry["bounds"]) == 2:
        bounds = (float(entry["bounds"][0]), float(entry["bounds"][1]))
    prior = entry.get("prior")
    if not isinstance(prior, dict):
        return _ParsedPrior(PriorFamily.FLAT, {}, bounds)
    for key, (family, rename) in _TOML_FAMILY.items():
        if key in prior and isinstance(prior[key], dict):
            raw = prior[key]
            args = {rename[k]: float(v) for k, v in raw.items() if k in rename}
            return _ParsedPrior(family, args, bounds)
    # `flat` is a family camdl writes deliberately, so it is not a gap; any
    # other name is one this table does not carry.
    declared = next((k for k in prior if k != "flat"), None)
    return _ParsedPrior(PriorFamily.FLAT, {}, bounds, declared)


# Model-IR (.camdl) parameter block:
#   name : type in [lo, hi] ~ dist(arg = val, ...)   # comment
#   k_raw[patch] : real in [-5.0, 5.0] ~ normal(mu = 0.0, sigma = 1.0)
_IR_PARAM_RE = re.compile(
    r"""^\s*
        (?P<name>[A-Za-z_]\w*)
        (?:\[(?P<dim>[^\]]*)\])?          # optional [patch] indexing
        \s*:\s*[^~\[]*?                    # type spec (positive/real/...)
        (?:in\s*\[\s*(?P<lo>[-+0-9.eE]+)\s*,\s*(?P<hi>[-+0-9.eE]+)\s*\])?
        \s*
        (?:~\s*(?P<dist>[A-Za-z_]\w*)\s*\((?P<args>[^)]*)\))?
        \s*(?:\#.*)?$
    """,
    re.VERBOSE,
)


def _parse_ir_params(model_path: Path) -> dict[str, _ParsedPrior]:
    """Parse the ``parameters { ... }`` block of a .camdl model file.

    Returns ``{base_name: _ParsedPrior}``. Indexed params (``k_raw``) are keyed
    by their base name; per-patch coordinates (``k_raw_Bo``) inherit that prior
    via :func:`_resolve_ir_for_param`.
    """
    out: dict[str, _ParsedPrior] = {}
    # is_file (not exists): an empty/blank model_path resolves to Path('.'),
    # which *exists* (it's the cwd) — read_text on a directory would crash.
    if not model_path.is_file():
        return out
    text = model_path.read_text()
    # Strip comments before the block match: a `}` inside a doc comment (e.g. the
    # `e^{−μ}` in a mortality param's `#'` line) would otherwise truncate the
    # non-greedy `parameters{...}` capture, dropping every param after it.
    text = re.sub(r"#.*", "", text)
    m = re.search(r"parameters\s*\{(.*?)\}", text, re.DOTALL)
    block = m.group(1) if m else text
    for line in block.splitlines():
        mm = _IR_PARAM_RE.match(line)
        if not mm or not mm.group("name"):
            continue
        name = mm.group("name")
        bounds = None
        if mm.group("lo") and mm.group("hi"):
            bounds = (float(mm.group("lo")), float(mm.group("hi")))
        dist = mm.group("dist")
        if dist is None:
            # No `~` at all: the parameter declares a box and nothing more,
            # which is a flat prior on that box and not a gap in this table.
            out[name] = _ParsedPrior(PriorFamily.FLAT, {}, bounds)
            continue
        fam_entry = _IR_FAMILY.get(dist.lower())
        if fam_entry is None:
            out[name] = _ParsedPrior(PriorFamily.FLAT, {}, bounds, dist.lower())
            continue
        family, rename = fam_entry
        args: dict[str, float] = {}
        for kv in mm.group("args").split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            try:
                fv = float(v.strip())
            except ValueError:
                continue
            if k in rename:
                args[rename[k]] = fv
        out[name] = _ParsedPrior(family, args, bounds)
    return out


def _resolve_ir_for_param(
    param: str, ir: dict[str, _ParsedPrior]
) -> _ParsedPrior | None:
    """Match an estimated coordinate to an IR entry, allowing for the
    ``<base>_<Patch>`` expansion of indexed model params (``k_raw[patch]``
    -> ``k_raw_Bo``)."""
    if param in ir:
        return ir[param]
    # Longest matching base prefix `<base>_...`.
    best = None
    for base in ir:
        if param.startswith(base + "_"):
            if best is None or len(base) > len(best):
                best = base
    return ir[best] if best is not None else None


def _model_file(run_dir: Path, meta_json: dict) -> Path:
    """The model ``.camdl`` to parse for IR priors: the leaf-archived copy
    (``model.camdl.original``, gh#353 — path-independent and always present on
    recent fits) when available, else the live source the record names, resolved
    by the rule of its ``schema_version`` (:mod:`camdl_scope.fitmeta`). Reading a
    recorded relative path as-is fails from any other CWD, which silently
    dropped every model-declared prior to Flat."""
    archived = run_dir / "model.camdl.original"
    if archived.is_file():
        return archived
    try:
        live = fitmeta.source_model_path(meta_json, run_dir, run_dir.parent)
    except UnsupportedSchema:
        live = None
    return live if live is not None else Path("")


def extract_priors(meta: RunMeta) -> dict[str, PriorSpec]:
    """Resolve the prior family+args for every estimated coordinate.

    Strategy (most reliable first):
      1. ``fit.meta.json`` ``resolved_priors`` tells us the *source* per param
         (``fit_toml`` | ``model_ir``).
      2. For ``fit_toml`` params, parse the ``[estimate]`` block of
         ``fit.toml.original``.
      3. For ``model_ir`` params, parse the ``parameters {}`` block of the
         model ``.camdl`` file (expanding indexed params).
      4. Fall back to a FLAT prior on the declared bounds, else Normal(0,1).
    """
    run_dir = meta.run_dir
    toml = _read_fit_toml(run_dir)
    meta_json = _read_meta(run_dir)
    estimate = (toml.get("estimate", {}) or {})

    sources = {
        d["param"]: d.get("source", "")
        for d in meta_json.get("resolved_priors", [])
        if isinstance(d, dict) and "param" in d
    }

    ir = _parse_ir_params(_model_file(run_dir, meta_json))

    out: dict[str, PriorSpec] = {}
    for param in meta.estimated:
        source = sources.get(param, "")
        parsed: _ParsedPrior | None = None
        src_label = "default"

        # Preferred source per meta.json.
        if source == "fit_toml" and param in estimate:
            parsed, src_label = _prior_from_toml_entry(estimate[param]), "fit_toml"
        elif source == "model_ir":
            res = _resolve_ir_for_param(param, ir)
            if res is not None:
                parsed, src_label = res, "model_ir"

        # Fallbacks regardless of declared source.
        if parsed is None and param in estimate:
            parsed, src_label = _prior_from_toml_entry(estimate[param]), "fit_toml"
        if parsed is None:
            res = _resolve_ir_for_param(param, ir)
            if res is not None:
                parsed, src_label = res, "model_ir"
        if parsed is None:
            # Last resort: a bounds-only flat, else unit normal.
            b = None
            if param in estimate and isinstance(estimate[param].get("bounds"), list):
                bb = estimate[param]["bounds"]
                if len(bb) == 2:
                    b = (float(bb[0]), float(bb[1]))
            parsed = (
                _ParsedPrior(PriorFamily.FLAT, {}, b)
                if b
                else _ParsedPrior(PriorFamily.NORMAL, {"mu": 0.0, "sigma": 1.0}, None)
            )
            src_label = "default"

        out[param] = PriorSpec(
            param=param,
            family=parsed.family,
            args=parsed.args,
            source=src_label,
            bounds=parsed.bounds,
            unsupported=parsed.unsupported,
        )
    return out


# ----------------------------------------------------------------------------
# Prior sampling / density (used by plots; pure given a PriorSpec)
# ----------------------------------------------------------------------------


def trunc_support(spec: PriorSpec) -> tuple[float, float]:
    """The truncation points of a TruncatedNormal, in the order camdl resolves
    them: the prior's own ``lower``/``upper`` when the fit TOML spells them,
    otherwise the parameter's declared box. A ``.camdl`` writes
    ``truncated_normal(mean =, sd =)`` and takes the interval from ``in [lo,
    hi]``, so without the box fallback the density would normalise over the
    whole line and understate itself everywhere."""
    lo = spec.args.get("lo")
    hi = spec.args.get("hi")
    box = spec.bounds or (-math.inf, math.inf)
    return (box[0] if lo is None else lo, box[1] if hi is None else hi)


def sample_prior(spec: PriorSpec, n: int = 10_000, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw ``n`` samples from a resolved prior, truncated to its bounds.

    FLAT priors sample uniformly on their bounds. A FLAT prior with no bounds —
    or with a bound at infinity, which the fit TOML permits as ``[0, inf]`` —
    returns an empty array: a uniform on an infinite interval is improper, so
    there is nothing sensible to draw. (A *proper* prior truncated to
    ``[0, inf]`` is unaffected; it samples from its family and the truncation
    below simply keeps everything.)"""
    rng = rng or np.random.default_rng(0)
    f = spec.family
    a = spec.args
    if f is PriorFamily.NORMAL:
        x = rng.normal(a.get("mu", 0.0), a.get("sigma", 1.0), n)
    elif f is PriorFamily.LOGNORMAL:
        x = rng.lognormal(a.get("mu", 0.0), a.get("sigma", 1.0), n)
    elif f is PriorFamily.HALFNORMAL:
        x = np.abs(rng.normal(0.0, a.get("sigma", 1.0), n))
    elif f is PriorFamily.BETA:
        x = rng.beta(a.get("alpha", 1.0), a.get("beta", 1.0), n)
    elif f is PriorFamily.GAMMA:
        # args are (alpha=shape, beta=rate); numpy uses scale = 1/rate.
        rate = a.get("beta", 1.0)
        x = rng.gamma(a.get("alpha", 1.0), 1.0 / rate if rate else 1.0, n)
    elif f is PriorFamily.EXPONENTIAL:
        # numpy parameterises by scale = 1/rate; camdl declares the rate.
        rate = a.get("rate", 1.0)
        x = rng.exponential(1.0 / rate if rate > 0 else 1.0, n)
    elif f is PriorFamily.UNIFORM:
        lo, hi = a.get("lo", 0.0), a.get("hi", 1.0)
        x = rng.uniform(lo, hi, n)
    elif f is PriorFamily.LOGUNIFORM:
        lo, hi = a.get("lo", 1e-9), a.get("hi", 1.0)
        x = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    elif f is PriorFamily.TRUNCNORMAL:
        from scipy import stats  # local import keeps ingest import-light

        mu, sigma = a.get("mu", 0.0), a.get("sigma", 1.0)
        lo, hi = trunc_support(spec)
        if sigma <= 0 or not (hi > lo):
            return np.empty(0)
        # Sampling from the truncation directly, rather than drawing a normal
        # and letting the bounds filter below do it: for a box in the tail
        # (camdl requires lower/upper to equal the declared bounds) rejection
        # would keep almost nothing and the moments would be built from a
        # handful of draws.
        x = stats.truncnorm.rvs(
            (lo - mu) / sigma, (hi - mu) / sigma,
            loc=mu, scale=sigma, size=n, random_state=rng,
        )
    else:  # FLAT
        if spec.bounds is None or not all(np.isfinite(b) for b in spec.bounds):
            return np.empty(0)
        lo, hi = spec.bounds
        x = rng.uniform(lo, hi, n)
    if spec.bounds is not None:
        lo, hi = spec.bounds
        x = x[(x >= lo) & (x <= hi)]
    return x


def log_prior_density(spec: PriorSpec, x: np.ndarray) -> np.ndarray:
    """Elementwise log prior density at ``x`` (used to reconstruct
    ``log_posterior`` when the trace lacks it). FLAT priors contribute 0
    (improper constant) inside bounds, ``-inf`` outside."""
    from scipy import stats  # local import keeps ingest import-light

    f = spec.family
    a = spec.args
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if f is PriorFamily.NORMAL:
            lp = stats.norm.logpdf(x, a.get("mu", 0.0), a.get("sigma", 1.0))
        elif f is PriorFamily.LOGNORMAL:
            lp = stats.lognorm.logpdf(x, s=a.get("sigma", 1.0), scale=math.exp(a.get("mu", 0.0)))
        elif f is PriorFamily.HALFNORMAL:
            lp = stats.halfnorm.logpdf(x, scale=a.get("sigma", 1.0))
        elif f is PriorFamily.BETA:
            lp = stats.beta.logpdf(x, a.get("alpha", 1.0), a.get("beta", 1.0))
        elif f is PriorFamily.GAMMA:
            rate = a.get("beta", 1.0)
            lp = stats.gamma.logpdf(x, a.get("alpha", 1.0), scale=1.0 / rate if rate else 1.0)
        elif f is PriorFamily.EXPONENTIAL:
            rate = a.get("rate", 1.0)
            lp = stats.expon.logpdf(x, scale=1.0 / rate if rate > 0 else 1.0)
        elif f is PriorFamily.TRUNCNORMAL:
            mu, sigma = a.get("mu", 0.0), a.get("sigma", 1.0)
            lo, hi = trunc_support(spec)
            lp = stats.truncnorm.logpdf(
                x, (lo - mu) / sigma, (hi - mu) / sigma, loc=mu, scale=sigma
            )
        elif f is PriorFamily.UNIFORM:
            lo, hi = a.get("lo", 0.0), a.get("hi", 1.0)
            lp = np.where((x >= lo) & (x <= hi), -math.log(hi - lo), -np.inf)
        elif f is PriorFamily.LOGUNIFORM:
            # density 1/(x * log(hi/lo)) on [lo, hi]
            lo, hi = a.get("lo", 1e-9), a.get("hi", 1.0)
            lp = np.where(
                (x >= lo) & (x <= hi),
                -np.log(x) - math.log(math.log(hi / lo)),
                -np.inf,
            )
        else:  # FLAT
            lp = np.zeros_like(x)
    if spec.bounds is not None:
        lo, hi = spec.bounds
        lp = np.where((x >= lo) & (x <= hi), lp, -np.inf)
    return lp
