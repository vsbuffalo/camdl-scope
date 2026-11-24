"""Ingest — fits in camdl's run store: discovery, the trace tail, priors.

Three jobs:
  * ``discover_runs(root)``       -> [RunMeta]
  * ``tail_chain(buf)``           -> rows appended since last call (tail-safe)
  * ``extract_priors(meta)``      -> {param: PriorSpec}

A fit is two kinds of result (camdl run spec; the records are read by
:mod:`camdl_scope.store`)::

    <build>/fit/<model>-<12 hex>/              # a FitProblem: one problem
        problem.toml  model.camdl  inputs/     #   (data, partition, model source)
        runs/<method>-<12 hex>/                # one method's settings
            seed_<n>-<12 hex>/                 # a FitRun: one run of that method
                identity.json  run.json        #   records: what it is, its status
                method.toml  fit.toml          #   the [method] table, the fit file
                fit.meta.json                  #   resolved partition, priors, schema
                draws.tsv | mle_params.toml    #   a posterior, or a point estimate
                chain_<k>/trace.tsv            #   per-chain draws
                progress.json  events.jsonl    #   heartbeat and event log
                predict/<mode>-<12 hex>/       #   predictions from this run

Each FitRun is one entry. Its status is its record's: ``completed`` and
``failed`` are final, and ``running`` is live while the owning process is
(:class:`camdl_scope.store.Liveness`). A problem folder holding several runs
(fit seeds, two methods, a resume) lists them all, its default first.

This module is deliberately isolated: if camdl grows a sanctioned
``camdl watch`` API later, swap this file and nothing downstream changes.
"""

from __future__ import annotations

import io
import json
import math
import re
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from . import mle as mle_mod
from . import store
from .docs import ModelDocs
from .schema import ObsSchema
from .verdict import UnsupportedSchema, parse_number, read_events
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

#: A heartbeat (``progress.json``) is rewritten every 5–10 s; allow a few missed
#: beats before a ``running`` one is treated as stale. Used only for an owner on
#: another host, whose process this reader cannot see.
PROGRESS_STALE_S = 30.0

#: The kind directory fits live under, and the depth of a run below its
#: problem's ``runs/`` (the method level, then the run).
FIT_KIND = "fit"
RUNS_DIR = "runs"
RUN_DEPTH = 2

# ----------------------------------------------------------------------------
# A fit run's own files
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


def read_progress(seed_dir: Path) -> RunProgress | None:
    """A run's heartbeat (gh#278): the ``snapshot`` of its ``progress.json``,
    or ``None`` when absent or unreadable.

    The ``state`` field is an externally-tagged ADT: the string ``"done"``, or
    ``{"running": {phase, step, total}}`` / ``{"failed": {reason}}``."""
    prog = store.read_progress(seed_dir / "progress.json")
    if prog is None:
        return None
    snap = prog.snapshot
    state = snap.get("state")
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
        state=name, updated_at=snap.get("updated_at"), pid=snap.get("pid"),
        phase=phase, step=step, total=total, reason=reason,
    )


def progress_is_fresh(prog: RunProgress, now: float | None = None) -> bool:
    """A ``running`` heartbeat whose ``updated_at`` is within the staleness
    window."""
    if prog.updated_at is None:
        return True
    return (now or time.time()) - prog.updated_at <= PROGRESS_STALE_S


def is_point_estimate_algorithm(algorithm: str) -> bool:
    """Whether ``algorithm`` produces a point estimate rather than draws: IF2 and
    the NLopt optimizers (``nl-sbplx``, ``nl-bobyqa``, ``nl-lbfgs``, …). The
    ``nl-`` prefix is camdl's naming for every NLopt method."""
    return algorithm == "if2" or algorithm.startswith("nl-")


#: A run that scores a fixed point with the particle filter: a score for
#: ``camdl compare``, not a fit to show (:mod:`camdl_scope.compare` lists it).
SCORING_ALGORITHMS = frozenset({"pfilter"})


def _read_json(path: Path) -> dict:
    try:
        rec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return rec if isinstance(rec, dict) else {}


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def read_method(seed_dir: Path) -> dict:
    """The run's ``[method]`` table, from its own ``method.toml`` (written when
    the run starts)."""
    method = _read_toml(seed_dir / "method.toml").get("method")
    return method if isinstance(method, dict) else {}


def read_expected_starts(seed_dir: Path) -> int | None:
    """How many starts (chains) the run was configured with — ``[method]``
    ``chains`` — or ``None``. A hint, not a promise: ``starts = "single"``
    collapses a run to one start."""
    v = read_method(seed_dir).get("chains")
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def _read_algorithm_config(seed_dir: Path) -> dict[str, str | float | int]:
    """The sampler's static configuration as camdl resolved it
    (``run_inputs.json`` ``algorithm``): particle count, sweep budget, optimizer
    tolerance. Keys differ by algorithm and are passed through, so a knob camdl
    starts recording surfaces without a change here. Empty until the run has
    written the file."""
    cfg = _read_json(seed_dir / "run_inputs.json").get("algorithm")
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
        # Anything structured has no compact rendering in a one-line header
        # strip, so it is dropped rather than stringified.
    return out


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
        rhat_band=_read_rhat_bands(s, seed_dir),
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


def _read_rhat_bands(summary: dict, seed_dir: Path) -> dict[str, str]:
    """camdl's per-parameter R-hat verdict (camdl#1199): the summary's
    ``rhat_band`` map, else ``method_result.diagnostics.per_param.<p>.rhat_band``
    in the end-of-run report in ``events.jsonl``. Empty when camdl saved
    neither (a leaf written before the field), or the report is unreadable."""
    raw = summary.get("rhat_band")
    if isinstance(raw, dict) and raw:
        return {str(k): v for k, v in raw.items() if isinstance(v, str)}
    try:
        ev = read_events(seed_dir)
    except (ValueError, UnsupportedSchema):
        return {}
    rep = ev.method_report if ev is not None else None
    node: object = rep
    for key in ("report", "method_result", "diagnostics", "per_param"):
        node = node.get(key) if isinstance(node, dict) else None
    if not isinstance(node, dict):
        return {}
    return {
        str(p): d["rhat_band"] for p, d in node.items()
        if isinstance(d, dict) and isinstance(d.get("rhat_band"), str)
    }




def _read_wall_time_secs(seed_dir: Path) -> float | None:
    """The run's wall-clock, the denominator for ESS/second: ``execution.json``
    ``wall_time_seconds`` (schema ``camdl.execution/v1``). ``None`` until the
    run has written it."""
    v = _read_json(seed_dir / "execution.json").get("wall_time_seconds")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------

#: Separates a problem folder's name from a run's in an entry id. Not a
#: character camdl writes in a folder name, and safe in a URL path.
LEAF_SEP = "~"

#: Where a fit problem's compiled model is: one level below each build's
#: ``compiled_model/`` directory.
COMPILED_MODEL_KIND = "compiled_model"


@dataclass(frozen=True)
class _StoreView:
    """What every fit entry reads from the store as a whole, read once per
    discovery: owner liveness, labels, display names, compiled models."""

    liveness: store.Liveness
    #: run id -> its current ``--label``.
    labels: dict[str, str]
    #: run id -> the labels camdl named it with (a problem's ``model``).
    names: dict[str, dict]
    #: compiled model run id -> its folder (``model.ir.json``, ``docs.json``).
    models: dict[str, Path]

    @classmethod
    def read(cls, root: Path) -> "_StoreView":
        invocations = store.read_invocations(root)
        return cls(
            liveness=store.Liveness.read(root),
            labels=store.current_labels(invocations),
            names=store.result_labels(invocations),
            models=store.folders_by_id(root, COMPILED_MODEL_KIND, 1),
        )


@dataclass(frozen=True)
class _Run:
    """One FitRun folder, as discovery sees it."""

    leaf: Path
    identity: store.Identity
    record: store.RunRecord
    #: The owner is alive: ``True``/``False`` on this host, ``None`` for an
    #: owner on another host (decided by heartbeat freshness).
    owner_alive: bool | None
    algorithm: str
    #: The fit seed, as typed (the identity's ``chain.seed.seed``).
    seed: int | None
    #: Every ``chain_<k>/trace.tsv`` present, empty ones included.
    chain_paths: dict[int, Path]
    #: Whether the run has written results to show: a non-empty trace, or a
    #: finished optimiser start.
    has_rows: bool

    @property
    def completed(self) -> bool:
        return self.record.status == "completed"

    @property
    def live(self) -> bool:
        return self.record.status == "running" and self.owner_alive is not False

    @property
    def point_estimate(self) -> bool:
        return is_point_estimate_algorithm(self.algorithm)

    def default_rank(self) -> tuple:
        """Which run a folder (or a method directory) opens by default: a live
        run with rows to watch, then a completed one, a posterior before a point
        estimate, the lowest seed first. Among completed replicate seeds, which
        finished last is an accident of scheduling, so seed order decides."""
        return (
            not (self.live and self.has_rows),
            not self.completed,
            self.point_estimate,
            self.seed if self.seed is not None else math.inf,
        )


def _chain_identity(ident: store.Identity) -> dict:
    chain = ident.body.get("chain")
    return chain if isinstance(chain, dict) else {}


def _read_run(leaf: Path, view: _StoreView) -> _Run | None:
    """The run in ``leaf``, or ``None`` when its records are not there yet (a
    claim being made) or it is not a fit run."""
    ident = store.read_identity(leaf)
    record = store.read_run_record(leaf)
    if ident is None or record is None or ident.kind != "fit_run":
        return None
    chain = _chain_identity(ident)
    method = chain.get("method")
    algorithm = str(method.get("algorithm") or "") if isinstance(method, dict) else ""
    seed_block = chain.get("seed")
    seed = seed_block.get("seed") if isinstance(seed_block, dict) else None
    paths = _seed_chain_paths(leaf)
    if is_point_estimate_algorithm(algorithm):
        has_rows = mle_mod.has_restart_rows(leaf)
    else:
        has_rows = bool(_nonempty(paths))
    return _Run(
        leaf=leaf, identity=ident, record=record,
        owner_alive=view.liveness.alive(record.owner),
        algorithm=algorithm,
        seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        chain_paths=paths, has_rows=has_rows,
    )


def _listed(run: _Run, *, include_warming: bool) -> bool:
    """Whether a run gets an entry: completed; or with results written (running,
    stalled or failed partway — finished optimiser starts and trace rows are
    results); or, with ``include_warming``, running with nothing written yet. A
    particle-filter scoring run is listed by Compare, not here."""
    if run.algorithm in SCORING_ALGORITHMS:
        return False
    if run.completed or run.has_rows:
        return True
    return include_warming and run.live


def _entry_ids(problem: Path, runs: list[_Run]) -> list[str]:
    """Entry ids for a problem's runs: the folder name when it holds one, else
    ``<folder>~<seed dir>``, with the method dir added only when two runs share
    a seed dir name (the same seed under two method settings)."""
    if len(runs) == 1:
        return [problem.name]
    names = [r.leaf.name for r in runs]
    return [
        LEAF_SEP.join([problem.name, r.leaf.name]) if names.count(r.leaf.name) == 1
        else LEAF_SEP.join([problem.name, r.leaf.parent.name, r.leaf.name])
        for r in runs
    ]


def _run_meta(
    problem: store.Identity | None, run: _Run, entry_id: str, view: _StoreView, *,
    problem_dir: Path, folder_default: bool, method_default: bool,
) -> RunMeta:
    leaf = run.leaf
    meta = _read_json(leaf / "fit.meta.json")
    method = read_method(leaf)
    # Each sampler budgets its run in its own unit: PGAS counts ``sweeps``,
    # PMMH/MH/IF2 ``iterations``, NUTS ``samples`` drawn after ``warmup``. They
    # are the same number to a progress bar, so they collapse to one field.
    target = method.get("sweeps") or method.get("iterations") or method.get("samples")
    burn_in = method.get("burn_in", method.get("warmup"))
    thin = method.get("thin")
    try:
        backend = Backend(str(method.get("backend") or "unknown"))
    except ValueError:
        backend = Backend.UNKNOWN
    config = run.record.given.get("config")
    config_path = config.get("path") if isinstance(config, dict) else None
    model_ref = problem.body.get("model") if problem is not None else None
    model_dir = view.models.get(model_ref.get("run_id")) if isinstance(model_ref, dict) else None
    problem_names = view.names.get(problem.run_id, {}) if problem is not None else {}
    return RunMeta(
        run_id=entry_id,
        problem_dir=problem_dir,
        leaf_dir=leaf,
        chain_paths=_nonempty(run.chain_paths) or run.chain_paths,
        model=str(problem_names.get("model") or ""),
        algorithm=run.algorithm or str(method.get("algorithm") or "unknown"),
        backend=backend,
        estimated=list(meta.get("estimated", [])),
        target_sweeps=int(target) if target is not None else None,
        declared_burn_in=int(burn_in) if burn_in is not None else None,
        thin=int(thin) if thin is not None else 1,
        config_stem=Path(config_path).stem if config_path else "",
        user_label=view.labels.get(run.identity.run_id),
        leaf_run_id=run.identity.run_id,
        seed=run.seed,
        folder_default=folder_default,
        method_default=method_default,
        fit_kind="mle" if run.point_estimate else "posterior",
        docs=ModelDocs.from_meta({"docs": _read_json(model_dir / "docs.json")}) if model_dir else ModelDocs(),
        schema=ObsSchema.from_meta(meta),
        algorithm_config=_read_algorithm_config(leaf),
        record_status=run.record.status,
        owner_alive=run.owner_alive,
        model_ir=model_dir / "model.ir.json" if model_dir else None,
    )


def _problem_entries(problem_dir: Path, view: _StoreView, *, include_warming: bool) -> list[RunMeta]:
    runs = [
        r for leaf in store.walk(problem_dir / RUNS_DIR, RUN_DEPTH)
        if (r := _read_run(leaf, view)) is not None and _listed(r, include_warming=include_warming)
    ]
    if not runs:
        return []
    default = min(runs, key=_Run.default_rank)
    runs = [default, *(r for r in runs if r is not default)]
    method_defaults = {
        md: min((r for r in runs if r.leaf.parent == md), key=_Run.default_rank)
        for md in {r.leaf.parent for r in runs}
    }
    problem = store.read_identity(problem_dir)
    return [
        _run_meta(
            problem, r, rid, view, problem_dir=problem_dir,
            folder_default=(r is default), method_default=(method_defaults[r.leaf.parent] is r),
        )
        for r, rid in zip(runs, _entry_ids(problem_dir, runs))
    ]


def discover_runs(root: Path, *, include_warming: bool = False) -> list[RunMeta]:
    """Every fit run in the store at ``root``, one entry per run; a problem
    folder holding several (fit seeds, two methods, a resume) yields several,
    its default first. Empty when ``root`` is not a store.

    With ``include_warming=True``, a run that is live but has written nothing
    to show yet (a sampler in burn-in, an optimiser before its first finished
    start) is listed too."""
    root = Path(root)
    if not store.is_store(root):
        return []
    view = _StoreView.read(root)
    out: list[RunMeta] = []
    for kind_dir in store.kind_dirs(root, FIT_KIND):
        for problem_dir in store.walk(kind_dir, 1):
            out.extend(_problem_entries(problem_dir, view, include_warming=include_warming))
    return out


def resolve_run(metas: list[RunMeta], run_id: str) -> RunMeta | None:
    """The entry ``run_id`` names: an exact entry id; ``<folder>~<method dir>``,
    which opens that method's default seed; or a problem folder's name, which
    opens that folder's default run."""
    exact = next((m for m in metas if m.run_id == run_id), None)
    if exact is not None:
        return exact
    folder, sep, method = run_id.partition(LEAF_SEP)
    if sep:
        return next(
            (m for m in metas if m.method_default and m.problem_dir.name == folder
             and m.method == method), None,
        )
    return next(
        (m for m in metas if m.folder_default and m.problem_dir.name == run_id), None
    )


# ----------------------------------------------------------------------------
# Tail-safe incremental read
# ----------------------------------------------------------------------------


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


def extract_priors(meta: RunMeta) -> dict[str, PriorSpec]:
    """Resolve the prior family+args for every estimated coordinate.

    Strategy (most reliable first):
      1. ``fit.meta.json`` ``resolved_priors`` tells us the *source* per param
         (``fit_toml`` | ``model_ir``).
      2. For ``fit_toml`` params, parse the ``[estimate]`` block of the run's
         ``fit.toml``.
      3. For ``model_ir`` params, parse the ``parameters {}`` block of the
         problem's ``model.camdl`` (expanding indexed params).
      4. Fall back to a FLAT prior on the declared bounds, else Normal(0,1).
    """
    toml = _read_toml(meta.leaf_dir / "fit.toml")
    meta_json = _read_json(meta.leaf_dir / "fit.meta.json")
    estimate = (toml.get("estimate", {}) or {})

    sources = {
        d["param"]: d.get("source", "")
        for d in meta_json.get("resolved_priors", [])
        if isinstance(d, dict) and "param" in d
    }

    ir = _parse_ir_params(meta.problem_dir / "model.camdl")

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
