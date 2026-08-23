"""Ingest — the only module that knows the camdl run-store layout.

Three jobs:
  * ``discover_runs(store)``      -> [RunMeta]
  * ``tail_chain(buf)``           -> rows appended since last call (tail-safe)
  * ``extract_priors(meta)``      -> {param: PriorSpec}

Store layout (content-addressed, nested)::

    <store>/<run>-<hash>/
        fit.meta.json
        fit.toml.original
        <NN>-posterior-<hash>/
            seed_<n>-<hash>/
                chain_<k>/
                    trace.tsv          # appended live by the sampler
                    trajectories/

Edge cases handled:
  * Multiple ``NN-posterior-*`` dirs (a resume leaves an empty stub) — pick
    the one whose chains have non-empty trace.tsv.
  * A re-run under new sampler settings reuses the run dir and adds a *new*
    stage beside the finished one. Which stage we read (the one holding a
    posterior) and which stage answers "is this run sampling right now" are
    then different questions, so discovery resolves both.
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
import shutil
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

import time

from . import mle
from .docs import ModelDocs
from .schema import ObsSchema
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
    object ``{"type": ..., <payload>}``; we split the tag from the payload."""
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


def _read_summary_json(seed_dir: Path) -> tuple[str, dict] | None:
    """The stage's authoritative summary (``pgas_summary.json`` /
    ``pmmh_summary.json`` / ``nuts_summary.json``), as ``(name, payload)``, or
    ``None``. All three share the ``rhat`` / ``ess`` / ``thin`` keys."""
    for name in ("pgas_summary.json", "pmmh_summary.json", "nuts_summary.json"):
        p = seed_dir / name
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return name, payload
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

    _name, s = found

    def _fdict(key: str) -> dict[str, float]:
        d = s.get(key) or {}
        return {k: float(v) for k, v in d.items() if v is not None} if isinstance(d, dict) else {}

    ess_raw = s.get("ess") or {}
    ess = {k: (float(v) if v is not None else None) for k, v in ess_raw.items()} \
        if isinstance(ess_raw, dict) else {}
    epc_raw = s.get("ess_per_chain") or {}
    ess_per_chain = (
        {k: [float(x) for x in v] for k, v in epc_raw.items() if isinstance(v, list)}
        if isinstance(epc_raw, dict) else {}
    )
    map_params = _fdict("map_params") or None
    return ChainSummary(
        stage=str(s.get("stage", "")),
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


def _stage_seed_dirs(run_dir: Path) -> list[Path]:
    """Every ``NN-posterior-*/seed_*`` leaf of a run, in stage then seed order.

    A run dir accumulates stages: a resume, a crash-relaunch, or a re-run under
    new sampler settings each add one (camdl's content-addressed layout hashes
    the settings into the stage name, so a changed setting is a new dir rather
    than an overwrite)."""
    return [
        seed_dir
        for pdir in sorted(run_dir.glob("[0-9]*-posterior-*"))
        for seed_dir in sorted(pdir.glob("seed_*"))
    ]


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
    for seed_dir in _stage_seed_dirs(run_dir):
        prog = read_progress(seed_dir)
        if prog is None or prog.state != "running" or not progress_is_fresh(prog):
            continue
        beat = float(prog.updated_at or 0.0)
        if best is None or beat > best[0]:
            best = (beat, seed_dir)
    return best[1] if best is not None else None


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
    for seed_dir in _stage_seed_dirs(run_dir):
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
    ``fit_toml_path`` (e.g. ``natbc_dens_hierk_nc_pgas_long.toml``), the most
    reliable source. Fall back to the run-dir name prefix before the hash."""
    ftp = meta.get("fit_toml_path", "")
    if ftp:
        return Path(ftp).stem
    name = run_dir.name
    return name.rsplit("-", 1)[0] if "-" in name else name


def _native_labels(store: Path) -> dict[str, str]:
    """Best-effort map ``{run_dir_name: user_label}`` from camdl's own index.

    Reads ``camdl list --kind fit --format json`` (JSONL: one object per fit).
    Each row carries ``path`` (e.g. ``results/fits/<run>-<hash>``) and ``label``
    (the user-display label set via ``camdl label <hash> "..."``, else null).

    Native labeling: ``camdl label`` writes the label into the run's leaf
    ``run.json`` and surfaces it here. We key on the run-dir *name* so it lines
    up with our ``run_id``. Returns ``{}`` if the binary is absent or errors —
    the caller then falls back to the derived label.
    """
    exe = shutil.which("camdl")
    if exe is None:
        return {}
    # The store is ``<root>/fits``; camdl's --root is that parent.
    root = store.parent if store.name == "fits" else store
    try:
        proc = subprocess.run(
            [exe, "list", "--root", str(root), "--kind", "fit",
             "--format", "json", "--all"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    out: dict[str, str] = {}

    def _ingest_row(row: object) -> None:
        if not isinstance(row, dict):
            return
        label = row.get("label")
        path = row.get("path")
        if label and path:
            out[Path(str(path)).name] = str(label)

    # camdl emits one JSON object per fit (JSONL), plus a trailing empty-array
    # sentinel line. Tolerate both: a dict line is a row; a list line is a batch
    # of rows. Also try a whole-output array parse as a last resort.
    parsed_any = False
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        parsed_any = True
        if isinstance(obj, list):
            for r in obj:
                _ingest_row(r)
        else:
            _ingest_row(obj)
    if not parsed_any:
        try:
            obj = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return out
        if isinstance(obj, list):
            for r in obj:
                _ingest_row(r)
        else:
            _ingest_row(obj)
    return out


def _mle_run_meta(run_dir: Path, mle_seed: Path, labels: dict[str, str]) -> RunMeta:
    """A ``RunMeta`` for a pure MLE (optimization) run. ``posterior_dir`` is
    reused to point at the scout seed leaf (for status/progress); ``chain_paths``
    is empty since there are no draws. Algorithm/backend come from the fit.toml
    stage matching the seed's parent dir (``01-scout-<hash>`` → ``[stages.scout]``)."""
    meta = _read_meta(run_dir)
    toml = _read_fit_toml(run_dir)
    model_path = meta.get("model_path", "")
    model = Path(model_path).stem if model_path else run_dir.name

    stage_dir = mle_seed.parent.name
    stage_name = (
        stage_dir.split("-", 1)[1].rsplit("-", 1)[0] if "-" in stage_dir else stage_dir
    )
    stage = (toml.get("stages", {}) or {}).get(stage_name, {}) or {}
    algorithm = stage.get("algorithm", "unknown")
    try:
        backend = Backend(stage.get("backend", "unknown"))
    except ValueError:
        backend = Backend.UNKNOWN

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
        user_label=labels.get(run_dir.name),
        docs=ModelDocs.from_meta(meta),
        schema=ObsSchema.from_meta(meta),
        fit_kind="mle",
    )


def discover_runs(store: Path, *, include_warming: bool = False) -> list[RunMeta]:
    """Scan ``store`` for runs that have at least one non-empty chain trace.

    With ``include_warming=True``, also surface runs whose latest posterior
    stage has started but written no draws yet *and* whose sampler process is
    still alive — i.e. live runs in burn-in. Killed-mid-burn-in stubs (same
    empty files, dead PID) stay hidden. Default ``False`` keeps existing
    callers byte-for-byte unchanged.
    """
    store = Path(store)
    out: list[RunMeta] = []
    if not store.is_dir():
        return out
    labels = _native_labels(store)
    for run_dir in sorted(store.iterdir()):
        if not run_dir.is_dir():
            continue
        picked = _pick_posterior_dir(run_dir, include_warming=include_warming)
        if picked is None:
            # No posterior draws — but the run may be a pure MLE (optimization
            # 'scout' stage) fit, which has a point estimate instead of chains.
            mle_seed = mle.find_mle_seed(run_dir)
            if mle_seed is not None:
                out.append(_mle_run_meta(run_dir, mle_seed, labels))
            continue
        pdir, seed_dir, chain_paths, _has_draws = picked
        meta = _read_meta(run_dir)
        toml = _read_fit_toml(run_dir)

        model_path = meta.get("model_path", "")
        model = Path(model_path).stem if model_path else run_dir.name
        estimated = list(meta.get("estimated", []))

        stage = (toml.get("stages", {}) or {}).get("posterior", {}) or {}
        algorithm = stage.get("algorithm", "unknown")
        backend_str = stage.get("backend", "unknown")
        try:
            backend = Backend(backend_str)
        except ValueError:
            backend = Backend.UNKNOWN
        target = stage.get("sweeps") or stage.get("iterations")
        burn_in = stage.get("burn_in")
        thin = stage.get("thin")

        out.append(
            RunMeta(
                run_id=run_dir.name,
                run_dir=run_dir,
                posterior_dir=seed_dir,
                live_stage_dir=_live_stage_dir(run_dir),
                chain_paths=chain_paths,
                model=model,
                algorithm=algorithm,
                backend=backend,
                estimated=estimated,
                target_sweeps=int(target) if target is not None else None,
                declared_burn_in=int(burn_in) if burn_in is not None else None,
                thin=int(thin) if thin is not None else 1,
                fit_toml_stem=_fit_toml_stem(meta, run_dir),
                user_label=labels.get(run_dir.name),
                docs=ModelDocs.from_meta(meta),
                schema=ObsSchema.from_meta(meta),
            )
        )
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
        target = buf.aux if col in AUX_COLUMNS else buf.values
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

# Map TOML prior keys -> (family, arg-rename map). camdl TOML uses snake_case
# distribution keys: log_normal, half_normal, normal, beta, gamma, uniform.
_TOML_FAMILY = {
    "normal": (PriorFamily.NORMAL, {"mu": "mu", "sigma": "sigma"}),
    "log_normal": (PriorFamily.LOGNORMAL, {"mu": "mu", "sigma": "sigma"}),
    "half_normal": (PriorFamily.HALFNORMAL, {"sigma": "sigma"}),
    "beta": (PriorFamily.BETA, {"alpha": "alpha", "beta": "beta"}),
    "gamma": (PriorFamily.GAMMA, {"alpha": "alpha", "beta": "beta",
                                  "shape": "alpha", "rate": "beta"}),
    "uniform": (PriorFamily.UNIFORM, {"lo": "lo", "hi": "hi",
                                      "low": "lo", "high": "hi",
                                      "min": "lo", "max": "hi"}),
}

# Same families, model-IR (.camdl) spelling: `~ log_normal(mu=, sigma=)` etc.
# The IR uses the same distribution names, so we share the table.
_IR_FAMILY = _TOML_FAMILY


def _prior_from_toml_entry(entry: dict) -> tuple[PriorFamily, dict, tuple | None] | None:
    """Parse one ``[estimate]`` TOML entry -> (family, args, bounds)."""
    bounds = None
    if isinstance(entry.get("bounds"), list) and len(entry["bounds"]) == 2:
        bounds = (float(entry["bounds"][0]), float(entry["bounds"][1]))
    prior = entry.get("prior")
    if not isinstance(prior, dict):
        return PriorFamily.FLAT, {}, bounds
    for key, (family, rename) in _TOML_FAMILY.items():
        if key in prior and isinstance(prior[key], dict):
            raw = prior[key]
            args = {rename[k]: float(v) for k, v in raw.items() if k in rename}
            return family, args, bounds
    return PriorFamily.FLAT, {}, bounds


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


def _parse_ir_params(model_path: Path) -> dict[str, tuple[PriorFamily, dict, tuple | None]]:
    """Parse the ``parameters { ... }`` block of a .camdl model file.

    Returns ``{base_name: (family, args, bounds)}``. Indexed params (``k_raw``)
    are keyed by their base name; per-patch coordinates (``k_raw_Bo``) inherit
    that prior via :func:`_resolve_ir_for_param`.
    """
    out: dict[str, tuple[PriorFamily, dict, tuple | None]] = {}
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
            out[name] = (PriorFamily.FLAT, {}, bounds)
            continue
        fam_entry = _IR_FAMILY.get(dist.lower())
        if fam_entry is None:
            out[name] = (PriorFamily.FLAT, {}, bounds)
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
        out[name] = (family, args, bounds)
    return out


def _resolve_ir_for_param(
    param: str, ir: dict[str, tuple[PriorFamily, dict, tuple | None]]
) -> tuple[PriorFamily, dict, tuple | None] | None:
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
    recent fits) when available, else the recorded ``model_path`` resolved
    against the project root. The raw ``model_path`` is stored *relative* (e.g.
    ``fits/../models/m.camdl``), so reading it as-is fails from any other CWD —
    which silently dropped every model-declared prior to Flat."""
    archived = run_dir / "model.camdl.original"
    if archived.is_file():
        return archived
    mp = Path(meta_json.get("model_path", ""))
    if not str(mp) or mp.is_absolute():
        return mp
    store = run_dir.parent
    root = store.parent.parent if store.name == "fits" else store.parent
    return root / mp


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
        family = args = bounds = None

        # Preferred source per meta.json.
        if source == "fit_toml" and param in estimate:
            family, args, bounds = _prior_from_toml_entry(estimate[param])
            src_label = "fit_toml"
        elif source == "model_ir":
            res = _resolve_ir_for_param(param, ir)
            if res is not None:
                family, args, bounds = res
                src_label = "model_ir"

        # Fallbacks regardless of declared source.
        if family is None and param in estimate:
            family, args, bounds = _prior_from_toml_entry(estimate[param])
            src_label = "fit_toml"
        if family is None:
            res = _resolve_ir_for_param(param, ir)
            if res is not None:
                family, args, bounds = res
                src_label = "model_ir"
        if family is None:
            # Last resort: a bounds-only flat, else unit normal.
            b = None
            if param in estimate and isinstance(estimate[param].get("bounds"), list):
                bb = estimate[param]["bounds"]
                if len(bb) == 2:
                    b = (float(bb[0]), float(bb[1]))
            family, args, bounds = (
                (PriorFamily.FLAT, {}, b) if b else (PriorFamily.NORMAL, {"mu": 0.0, "sigma": 1.0}, None)
            )
            src_label = "default"

        out[param] = PriorSpec(
            param=param, family=family, args=args or {}, source=src_label, bounds=bounds
        )
    return out


# ----------------------------------------------------------------------------
# Prior sampling / density (used by plots; pure given a PriorSpec)
# ----------------------------------------------------------------------------


def sample_prior(spec: PriorSpec, n: int = 10_000, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw ``n`` samples from a resolved prior, truncated to its bounds.

    FLAT priors sample uniformly on their bounds. A FLAT prior with no
    bounds returns an empty array (nothing sensible to draw)."""
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
    elif f is PriorFamily.UNIFORM:
        lo, hi = a.get("lo", 0.0), a.get("hi", 1.0)
        x = rng.uniform(lo, hi, n)
    else:  # FLAT
        if spec.bounds is not None:
            lo, hi = spec.bounds
            x = rng.uniform(lo, hi, n)
        else:
            return np.empty(0)
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
        elif f is PriorFamily.UNIFORM:
            lo, hi = a.get("lo", 0.0), a.get("hi", 1.0)
            lp = np.where((x >= lo) & (x <= hi), -math.log(hi - lo), -np.inf)
        else:  # FLAT
            lp = np.zeros_like(x)
    if spec.bounds is not None:
        lo, hi = spec.bounds
        lp = np.where((x >= lo) & (x <= hi), lp, -np.inf)
    return lp
