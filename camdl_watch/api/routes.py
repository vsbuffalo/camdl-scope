"""Read-only API routes — the typed projection of the run store as JSON.

Each request resolves the store fresh via ``current_store()`` (so the CLI and
tests can repoint it), discovers runs through :mod:`camdl_watch.ingest`, builds
a :class:`~camdl_watch.state.RunState` server-side, and serializes the
diagnostics / docs / schema the core already computed. No statistic is computed
in the browser; every number here is produced in Python (the proposal's
correctness guardrail).

A run state is rebuilt per request (a full tail-read of each chain). That is
fine for finished fits polled infrequently; a signature-keyed cache is a clean
later optimization, not a correctness requirement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from .. import compare as compare_mod
from .. import diagnostics as diag_mod
from .. import ingest
from .. import predictive
from .. import quantities as quantities_mod
from ..assembly import build_run_state
from ..grouping import group_params
from ..highlight import HIGHLIGHT_CSS, highlight_camdl, highlight_toml
from ..state import (
    AUX_COLUMNS,
    PriorFamily,
    PriorSpec,
    RunMeta,
    RunState,
)
from .models import (
    ChainMixing,
    CompareResponse,
    CompareRow,
    DiagnosticsResponse,
    DimensionInfo,
    DrawsResponse,
    FindingGroup,
    ObservedPoint,
    ParamDiagnostic,
    ParamFamily,
    ParamGroups,
    ParamPosterior,
    ParamTrace,
    PosteriorResponse,
    PredictivePoint,
    PredictiveResponse,
    PriorCurve,
    ProgressInfo,
    QuantityBandPoint,
    QuantityInfo,
    QuantityScalarRow,
    QuantityScalarsResponse,
    QuantitySeriesResponse,
    RunDetail,
    RunSummary,
    SourceFile,
    SourceResponse,
    StreamInfo,
    TraceSeries,
    TracesResponse,
)

router = APIRouter(prefix="/api")

_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _store() -> Path:
    """The store to read, resolved fresh per request. Imported lazily from the
    app module to keep the import acyclic (app.py imports this module to mount
    the router)."""
    from .app import current_store

    return current_store()


# ---------------------------------------------------------------------------
# Run-state assembly (server-side; never shipped raw)
# ---------------------------------------------------------------------------


def _warmup_cutoff(rs: RunState, warmup_pct: int) -> int:
    """Sweep index that splits warm-up from the retained tail (app.py's rule)."""
    lo = rs.min_iter() or 0
    hi = rs.max_iter() or lo
    return int(lo + (hi - lo) * warmup_pct / 100.0)


# ---------------------------------------------------------------------------
# Formatting / projection helpers
# ---------------------------------------------------------------------------


def _g(x: float) -> str:
    """Compact number for a prior label: drops trailing zeros (``0.0`` -> ``0``,
    ``-0.6`` -> ``-0.6``)."""
    return f"{float(x):g}"


def _format_prior(spec: PriorSpec | None) -> str | None:
    """A resolved prior as a human label, e.g. ``LogNormal(μ=-0.6, σ=0.4)``,
    ``Beta(α=3, β=6)``, ``Uniform(0, 1)``, ``Flat[-5, 5]``. ``None`` when there
    is no prior to render."""
    if spec is None:
        return None
    a = spec.args
    f = spec.family
    if f is PriorFamily.NORMAL:
        return f"Normal(μ={_g(a.get('mu', 0.0))}, σ={_g(a.get('sigma', 1.0))})"
    if f is PriorFamily.LOGNORMAL:
        return f"LogNormal(μ={_g(a.get('mu', 0.0))}, σ={_g(a.get('sigma', 1.0))})"
    if f is PriorFamily.HALFNORMAL:
        return f"HalfNormal(σ={_g(a.get('sigma', 1.0))})"
    if f is PriorFamily.BETA:
        return f"Beta(α={_g(a.get('alpha', 1.0))}, β={_g(a.get('beta', 1.0))})"
    if f is PriorFamily.GAMMA:
        return f"Gamma(α={_g(a.get('alpha', 1.0))}, β={_g(a.get('beta', 1.0))})"
    if f is PriorFamily.UNIFORM:
        return f"Uniform({_g(a.get('lo', 0.0))}, {_g(a.get('hi', 1.0))})"
    # FLAT: a bounds-only / improper prior.
    if spec.bounds is not None:
        lo, hi = spec.bounds
        return f"Flat[{_g(lo)}, {_g(hi)}]"
    return "Flat"


def _finite_or_none(x: float | None) -> float | None:
    """A diagnostic value for the wire: ``None`` unless it is a finite float
    (Starlette serializes with ``allow_nan=False``, so NaN/inf cannot ship)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _progress_info(rs: RunState) -> ProgressInfo | None:
    """Project camdl's ``progress.json`` heartbeat onto the wire, deriving a
    completion ``pct`` when step/total are known. ``None`` when the run has no
    heartbeat (older runs, or finished fits that never wrote one)."""
    p = rs.progress
    if p is None:
        return None
    pct: int | None = None
    if p.step is not None and p.total:
        pct = max(0, min(100, round(100.0 * p.step / p.total)))
    return ProgressInfo(
        state=p.state, phase=p.phase, step=p.step, total=p.total,
        pct=pct, reason=p.reason,
        updated_at=float(p.updated_at) if p.updated_at is not None else None,
    )


def _run_summary(meta: RunMeta, rs: RunState) -> RunSummary:
    return RunSummary(
        run_id=meta.run_id,
        label=meta.display_label,
        model=meta.model,
        algorithm=meta.algorithm,
        backend=meta.backend.value,
        status=rs.status.value,
        n_chains=len(rs.chains),
        n_params=len(meta.estimated),
        has_docs=not meta.docs.is_empty(),
        has_prequential=compare_mod.find_prequential(meta.run_dir) is not None,
        progress=_progress_info(rs),
        max_iter=rs.max_iter(),
        updated_at=rs.updated_at,
    )


def _quantity_info(q: quantities_mod.QuantityMeta, meta: RunMeta) -> QuantityInfo:
    """A logical quantity for the wire, joined with its ``#'`` docs (symbol /
    description / citation) when the model carries them — exactly like params."""
    db = meta.docs.for_quantity(q.name)
    return QuantityInfo(
        name=q.name, shape=q.shape, source=q.source, index_dims=q.index_dims,
        reduce=q.reduce, unit=q.unit, censorable=q.censorable,
        symbol=(db.symbol if db else None),
        description=(db.text if db else None),
        reference=(db.reference if db else None),
    )


def _run_detail(meta: RunMeta, rs: RunState) -> RunDetail:
    schema = meta.schema
    streams = [
        StreamInfo(
            name=s.name,
            index_dims=list(s.index_dims),
            value_kind=s.value_kind,
            likelihood=s.likelihood,
        )
        for s in (schema.streams if schema else [])
    ]
    dimensions = [
        DimensionInfo(name=d.name, levels=list(d.levels))
        for d in (schema.dimensions.values() if schema else [])
    ]
    findings: list[FindingGroup] = []
    if rs.summary is not None:
        for g in diag_mod.summarize_findings(rs.summary.findings):
            findings.append(
                FindingGroup(
                    kind=g.kind,
                    severity=g.severity.value,
                    headline=g.headline,
                    params=list(g.params),
                )
            )
    pg = group_params(list(meta.estimated))
    groups = ParamGroups(
        scalars=pg.scalars,
        families=[ParamFamily(base=b, members=ms) for b, ms in pg.families.items()],
        default_selection=pg.default_selection(),
    )
    _quantity_manifest = quantities_mod.read_manifest(meta.run_dir)
    return RunDetail(
        run_id=meta.run_id,
        label=meta.display_label,
        model=meta.model,
        algorithm=meta.algorithm,
        backend=meta.backend.value,
        status=rs.status.value,
        n_chains=len(rs.chains),
        max_iter=rs.max_iter(),
        target_sweeps=meta.target_sweeps,
        estimated=list(meta.estimated),
        groups=groups,
        streams=streams,
        dimensions=dimensions,
        findings=findings,
        # camdl writes predictive/observed at the FIT (run) dir level, not the
        # seed dir — read there.
        available_streams=predictive.discover_streams(meta.run_dir),
        available_quantities=[
            _quantity_info(q, meta) for q in _quantity_manifest.quantities
        ],
        quantity_scenarios=_quantity_manifest.scenarios,
    )


def _param_posterior(
    meta: RunMeta, rs: RunState, diag, cutoff: int, param: str
) -> ParamPosterior | None:
    """Project one estimated coordinate onto the wire: pooled post-warmup
    quantiles (computed here), the resolved doc block + prior, and the effective
    R̂/ESS. ``None`` when the coordinate has no finite post-warmup draws."""
    parts = [
        buf.values[param][buf.iters >= cutoff]
        for buf in rs.chains.values()
        if param in buf.values
    ]
    vals = np.concatenate(parts) if parts else np.empty(0)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    q05, q25, q50, q75, q95 = (float(x) for x in np.quantile(vals, _QUANTILES))
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if vals.size >= 2 else 0.0

    block = meta.docs.for_param(param)
    spec = rs.priors.get(param)
    rhat_v, _ = diag_mod.effective_rhat(diag, rs.summary, param)
    ess_v, _ = diag_mod.effective_ess(diag, rs.summary, param)
    return ParamPosterior(
        name=param,
        symbol=block.symbol if block else None,
        description=block.text if block else None,
        reference=block.reference if block else None,
        source=spec.source if spec else "unknown",
        prior=_format_prior(spec),
        bounds=spec.bounds if spec else None,
        mean=mean,
        sd=sd,
        q05=q05,
        q25=q25,
        q50=q50,
        q75=q75,
        q95=q95,
        rhat=_finite_or_none(rhat_v),
        ess=_finite_or_none(ess_v),
    )


def _build_draws(
    meta: RunMeta, rs: RunState, cutoff: int, max_draws: int
) -> tuple[list[int], dict[str, np.ndarray], list[str]]:
    """Row-aligned, pooled, thinned post-warmup draws (params + objectives).

    Within a chain the i-th retained sweep is the same joint sample across
    columns; chains are concatenated (carrying a chain id per row). The objective
    aux columns (``log_posterior`` / ``log_likelihood``), when present in every
    chain, are pooled alongside the params so they can be paired against them
    (Stan's lp__). Rows where any column is non-finite are dropped so every
    column stays aligned and JSON-serializable, then thinned to ``max_draws`` by
    an even stride. Returns ``(chain, cols, objectives)``."""
    params = list(meta.estimated)
    objectives = [
        c for c in ("log_posterior", "log_likelihood")
        if c in AUX_COLUMNS and rs.chains and all(c in b.aux for b in rs.chains.values())
    ]
    wanted = params + objectives
    chain_parts: list[np.ndarray] = []
    col_parts: dict[str, list[np.ndarray]] = {p: [] for p in wanted}
    for cid, buf in sorted(rs.chains.items()):
        idx = np.where(buf.iters >= cutoff)[0]
        if idx.size == 0:
            continue
        chain_parts.append(np.full(idx.size, cid, dtype=np.int64))
        for p in params:
            col_parts[p].append(
                buf.values[p][idx] if p in buf.values else np.full(idx.size, np.nan)
            )
        for o in objectives:
            col_parts[o].append(
                buf.aux[o][idx] if o in buf.aux else np.full(idx.size, np.nan)
            )
    if not chain_parts:
        return [], {p: np.empty(0) for p in wanted}, objectives

    chain = np.concatenate(chain_parts)
    cols = {p: np.concatenate(col_parts[p]) for p in wanted}
    finite = np.ones(chain.size, dtype=bool)
    for p in wanted:
        finite &= np.isfinite(cols[p])
    chain = chain[finite]
    cols = {p: cols[p][finite] for p in wanted}

    total = chain.size
    if total > max_draws:
        sel = np.unique(np.linspace(0, total - 1, max_draws).astype(int))
        chain = chain[sel]
        cols = {p: cols[p][sel] for p in wanted}
    return chain.tolist(), cols, objectives


def _find_meta(store: Path, run_id: str) -> RunMeta | None:
    for meta in ingest.discover_runs(store, include_warming=True):
        if meta.run_id == run_id:
            return meta
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    """Every discoverable run, newest first by last-written chain mtime."""
    store = _store()
    summaries = [
        _run_summary(meta, build_run_state(meta))
        for meta in ingest.discover_runs(store, include_warming=True)
    ]
    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str) -> RunDetail:
    """One run's metadata, schema, and authoritative verdict."""
    store = _store()
    meta = _find_meta(store, run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return _run_detail(meta, build_run_state(meta))


@router.get("/runs/{run_id}/posterior", response_model=PosteriorResponse)
def get_posterior(
    run_id: str, warmup_pct: int = Query(default=50, ge=0, le=100)
) -> PosteriorResponse:
    """Doc-labelled posterior summary (the forest-plot payload). Params are in
    the model's estimated order; a run with no draws yet returns ``params=[]``
    and ``n_tail=0`` rather than erroring."""
    store = _store()
    meta = _find_meta(store, run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    rs = build_run_state(meta)
    cutoff = _warmup_cutoff(rs, warmup_pct)
    if rs.max_iter() is None:  # warming up — no draws to summarize
        return PosteriorResponse(
            run_id=run_id, warmup_pct=warmup_pct, warmup_cutoff=cutoff,
            n_tail=0, params=[],
        )
    diag = diag_mod.compute_diagnostics(rs, cutoff, params=rs.params)
    params = [
        pp
        for p in meta.estimated
        if (pp := _param_posterior(meta, rs, diag, cutoff, p)) is not None
    ]
    return PosteriorResponse(
        run_id=run_id, warmup_pct=warmup_pct, warmup_cutoff=cutoff,
        n_tail=diag.n_tail, params=params,
    )


def _sample_priors(rs: RunState, params: list[str], n: int = 2000) -> dict[str, list[float]]:
    """Marginal prior samples per param (for the pair-plot diagonal overlay).

    Drawn from each resolved :class:`PriorSpec` with a fixed seed (deterministic
    per request), truncated to bounds by the sampler. A param with no usable
    prior (e.g. an unbounded flat) yields an empty list."""
    rng = np.random.default_rng(0)
    out: dict[str, list[float]] = {}
    for p in params:
        spec = rs.priors.get(p)
        if spec is None:
            out[p] = []
            continue
        s = ingest.sample_prior(spec, n=n, rng=rng)
        out[p] = s[np.isfinite(s)].tolist()
    return out


def _prior_curves(
    rs: RunState,
    params: list[str],
    cols: dict[str, np.ndarray],
    prior_samples: dict[str, list[float]],
    n_grid: int = 160,
) -> dict[str, PriorCurve]:
    """Smooth analytic prior density per param, evaluated over the union of the
    posterior window and the prior's central 99% interval. The posterior part
    overlays the diagonal in "fit to posterior" mode; the wider part gives "show
    prior breadth" mode a curve to draw when the axis zooms out to the prior's
    scale (the posterior then reads as a spike). A binned histogram of clipped
    prior samples reads as noise; the analytic density is exact and smooth.
    Flat/unbounded priors (no informative shape) are skipped."""
    out: dict[str, PriorCurve] = {}
    for p in params:
        spec = rs.priors.get(p)
        if spec is None or spec.family is PriorFamily.FLAT:
            continue
        arr = cols.get(p)
        if arr is None or arr.size < 2:
            continue
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        # Widen to the prior's central 99% so the breadth view has a curve to
        # draw beyond the (tight) posterior window.
        ps = np.asarray(prior_samples.get(p, []), dtype=float)
        ps = ps[np.isfinite(ps)]
        if ps.size:
            lo = min(lo, float(np.quantile(ps, 0.005)))
            hi = max(hi, float(np.quantile(ps, 0.995)))
        pad = (hi - lo) * 0.04
        grid = np.linspace(lo - pad, hi + pad, n_grid)
        dens = np.exp(ingest.log_prior_density(spec, grid))
        dens = np.where(np.isfinite(dens), dens, 0.0)
        if not np.any(dens > 0):
            continue
        out[p] = PriorCurve(x=grid.tolist(), y=dens.tolist())
    return out


@router.get("/runs/{run_id}/draws", response_model=DrawsResponse)
def get_draws(
    run_id: str,
    warmup_pct: int = Query(default=50, ge=0, le=100),
    max_draws: int = Query(default=1200, ge=50, le=5000),
) -> DrawsResponse:
    """Row-aligned post-warmup draws (plus marginal prior samples) for the
    marginal densities and the pair plot. Pooled across chains, thinned to
    ``max_draws``; ``params`` in estimated order. A run with no draws yet returns
    empty columns and ``n_draws=0`` (priors are still sampled)."""
    store = _store()
    meta = _find_meta(store, run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    rs = build_run_state(meta)
    cutoff = _warmup_cutoff(rs, warmup_pct)
    params = list(meta.estimated)
    prior = _sample_priors(rs, params)
    if rs.max_iter() is None:
        return DrawsResponse(
            run_id=run_id, warmup_pct=warmup_pct, warmup_cutoff=cutoff,
            n_draws=0, params=params, objectives=[], chain=[],
            draws={p: [] for p in params}, prior=prior, prior_density={},
        )
    chain, cols, objectives = _build_draws(meta, rs, cutoff, max_draws)
    return DrawsResponse(
        run_id=run_id, warmup_pct=warmup_pct, warmup_cutoff=cutoff,
        n_draws=len(chain), params=params, objectives=objectives,
        chain=chain, draws={k: v.tolist() for k, v in cols.items()},
        prior=prior, prior_density=_prior_curves(rs, params, cols, prior),
    )


# ---------------------------------------------------------------------------
# Source tab
# ---------------------------------------------------------------------------


def _project_root(store: Path) -> Path:
    """The camdl project root a fit's *relative* paths resolve against. The store
    is ``<root>/results/fits``, so the project root is two levels up."""
    return store.parent.parent if store.name == "fits" else store.parent


def _read_model_source(run_dir: Path, store: Path, model_path: str) -> SourceFile:
    """The model source, syntax-highlighted.

    Prefer the copy archived in the fit run leaf (``model.camdl.original``,
    gh#353): it's self-contained and path-independent, so it resolves from a
    viewer launched anywhere. Older fits (and ``.ir.json`` models, which archive
    no ``.camdl``) have no such copy — fall back to reading live from the
    recorded ``model_path``, resolved relative to the project root (newer fits
    store it relative; older ones absolute)."""
    archived = run_dir / "model.camdl.original"
    if archived.is_file():
        try:
            text = archived.read_text()
        except OSError:
            pass
        else:
            return SourceFile(
                path=model_path or "model.camdl", present=True, origin="leaf",
                html=highlight_camdl(text), text=text,
            )
    if not model_path:
        return SourceFile(path=None, present=False, origin="live")
    p = Path(model_path)
    if not p.is_absolute():
        p = _project_root(store) / p
    if not p.is_file():
        return SourceFile(path=model_path, present=False, origin="live")
    try:
        text = p.read_text()
    except OSError:
        return SourceFile(path=model_path, present=False, origin="live")
    return SourceFile(
        path=model_path, present=True, origin="live",
        html=highlight_camdl(text), text=text,
    )


@router.get("/runs/{run_id}/source", response_model=SourceResponse)
def get_source(run_id: str) -> SourceResponse:
    """The fit's sources: the highlighted ``.camdl`` model (the copy archived in
    the run leaf when present, else read live from the recorded path) and the
    ``fit.toml`` (always archived in the run leaf)."""
    store = _store()
    meta = _find_meta(store, run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    try:
        meta_json = json.loads((meta.run_dir / "fit.meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        meta_json = {}
    model = _read_model_source(
        meta.run_dir, store, str(meta_json.get("model_path", ""))
    )

    toml_path = meta.run_dir / "fit.toml.original"
    if toml_path.is_file():
        try:
            ttext = toml_path.read_text()
            fit_toml = SourceFile(
                path="fit.toml", present=True, origin="leaf",
                html=highlight_toml(ttext), text=ttext,
            )
        except OSError:
            fit_toml = SourceFile(path="fit.toml", present=False, origin="leaf")
    else:
        fit_toml = SourceFile(path="fit.toml", present=False, origin="leaf")

    return SourceResponse(
        run_id=run_id, model=model,
        model_identity=meta_json.get("model_identity"),
        fit_toml=fit_toml, highlight_css=HIGHLIGHT_CSS,
    )


# ---------------------------------------------------------------------------
# Predictive tab
# ---------------------------------------------------------------------------


def _stream_index_dims(meta: RunMeta, stream: str) -> list[str]:
    if meta.schema is None:
        return []
    for s in meta.schema.streams:
        if s.name == stream:
            return list(s.index_dims)
    return []


def _fnum(v: object) -> float:
    """A finite float for the wire (Starlette can't serialize NaN/inf)."""
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


@router.get("/runs/{run_id}/predictive/{stream}", response_model=PredictiveResponse)
def get_predictive(run_id: str, stream: str) -> PredictiveResponse:
    """One stream's posterior-predictive ribbons (``camdl fit predict`` output)
    plus the observed series. 404 if the stream has no predictive artifact."""
    store = _store()
    meta = _find_meta(store, run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    ps = predictive.read_predictive(meta.run_dir, stream)
    if ps is None:
        raise HTTPException(status_code=404, detail=f"no predictive artifact for stream: {stream}")
    obs = predictive.read_observed(meta.run_dir, stream)
    index_dims = _stream_index_dims(meta, stream)

    def stratum(row: dict) -> dict[str, str]:
        return {d: str(row[d]) for d in index_dims if row.get(d) is not None}

    horizons: set[str] = set()
    treatments: set[str] = set()
    scenarios: list[str] = []
    pred_points: list[PredictivePoint] = []
    for r in ps.table.to_dicts():
        h, t = str(r.get("horizon") or ""), str(r.get("treatment") or "")
        sc = str(r.get("scenario") or "as_fitted")
        horizons.add(h)
        treatments.add(t)
        if sc not in scenarios:
            scenarios.append(sc)
        pred_points.append(
            PredictivePoint(
                time=_fnum(r.get("time")), stratum=stratum(r),
                scenario=sc, horizon=h, treatment=t,
                q05=_fnum(r.get("q05")), q25=_fnum(r.get("q25")), q50=_fnum(r.get("q50")),
                q75=_fnum(r.get("q75")), q95=_fnum(r.get("q95")),
            )
        )
    obs_points: list[ObservedPoint] = []
    if obs is not None:
        for r in obs.table.to_dicts():
            v = r.get("value")
            obs_points.append(
                ObservedPoint(
                    time=_fnum(r.get("time")), stratum=stratum(r),
                    value=(_fnum(v) if v is not None else None),
                )
            )
    return PredictiveResponse(
        run_id=run_id, stream=stream, index_dims=index_dims,
        scenarios=scenarios, horizons=sorted(horizons), treatments=sorted(treatments),
        predictive=pred_points, observed=obs_points,
    )


# ---------------------------------------------------------------------------
# Quantities tab (generated quantities — camdl fit predict's quantities/)
# ---------------------------------------------------------------------------


def _band_cell(v: object) -> float | None:
    """A band quantile for the wire: ``None`` for an empty cell (a fully-censored
    scalar writes blank q*), else the finite float."""
    if v is None or v == "":
        return None
    return _finite_or_none(_fnum(v))


def _stratum_of(row: dict, dims: list[str]) -> dict[str, str]:
    return {d: str(row[d]) for d in dims if row.get(d) is not None}


@router.get(
    "/runs/{run_id}/quantity-series/{name}", response_model=QuantitySeriesResponse
)
def get_quantity_series(run_id: str, name: str) -> QuantitySeriesResponse:
    """One series quantity's banded trajectory (a ribbon). 404 if the run has no
    such series quantity in its manifest, or its TSV is missing."""
    store = _store()
    meta = _find_meta(store, run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    manifest = quantities_mod.read_manifest(meta.run_dir)
    qm = next(
        (q for q in manifest.quantities if q.name == name and q.shape == "series"),
        None,
    )
    if qm is None:
        raise HTTPException(status_code=404, detail=f"no series quantity: {name}")
    df = quantities_mod.read_quantity(meta.run_dir, name)
    if df is None:
        raise HTTPException(status_code=404, detail=f"no data for quantity: {name}")
    points = [
        QuantityBandPoint(
            scenario=str(r.get("scenario") or "as_fitted"),
            time=_fnum(r.get("time")),
            stratum=_stratum_of(r, qm.index_dims),
            q05=_fnum(r.get("q05")), q25=_fnum(r.get("q25")), q50=_fnum(r.get("q50")),
            q75=_fnum(r.get("q75")), q95=_fnum(r.get("q95")),
        )
        for r in df.iter_rows(named=True)
    ]
    return QuantitySeriesResponse(
        run_id=run_id, name=name, index_dims=qm.index_dims,
        scenarios=manifest.scenarios, points=points,
    )


@router.get("/runs/{run_id}/quantity-scalars", response_model=QuantityScalarsResponse)
def get_quantity_scalars(run_id: str) -> QuantityScalarsResponse:
    """Every scalar quantity, one row per stratum cell — the quantities table.
    Manifest-driven (stale orphan TSVs are ignored)."""
    store = _store()
    meta = _find_meta(store, run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    manifest = quantities_mod.read_manifest(meta.run_dir)
    rows: list[QuantityScalarRow] = []
    for qm in manifest.quantities:
        if qm.shape != "scalar":
            continue
        df = quantities_mod.read_quantity(meta.run_dir, qm.name)
        if df is None:
            continue
        for r in df.iter_rows(named=True):
            pc = r.get("p_censored")
            rows.append(
                QuantityScalarRow(
                    name=qm.name,
                    scenario=str(r.get("scenario") or "as_fitted"),
                    reduce=qm.reduce, source=qm.source,
                    stratum=_stratum_of(r, qm.index_dims),
                    n_draws=int(r.get("n_draws") or 0),
                    p_censored=(_band_cell(pc) if pc is not None else None),
                    q05=_band_cell(r.get("q05")), q25=_band_cell(r.get("q25")),
                    q50=_band_cell(r.get("q50")), q75=_band_cell(r.get("q75")),
                    q95=_band_cell(r.get("q95")),
                )
            )
    return QuantityScalarsResponse(
        run_id=run_id, scenarios=manifest.scenarios, rows=rows
    )


# ---------------------------------------------------------------------------
# Traces tab
# ---------------------------------------------------------------------------


@router.get("/runs/{run_id}/traces", response_model=TracesResponse)
def get_traces(
    run_id: str,
    warmup_pct: int = Query(default=50, ge=0, le=100),
    max_points: int = Query(default=600, ge=50, le=4000),
) -> TracesResponse:
    """Per-parameter, per-chain iteration traces (thinned) for the trace grid.
    Includes the estimated coordinates plus any present objective aux columns
    (``log_posterior`` / ``log_likelihood``) — the first thing to eyeball for
    mixing."""
    store = _store()
    meta = _find_meta(store, run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    rs = build_run_state(meta)
    cutoff = _warmup_cutoff(rs, warmup_pct)

    objectives = [
        c for c in ("log_posterior", "log_likelihood")
        if c in AUX_COLUMNS and any(c in b.aux for b in rs.chains.values())
    ]
    traces: list[ParamTrace] = []
    for p in list(meta.estimated) + objectives:
        series: list[TraceSeries] = []
        for cid, buf in sorted(rs.chains.items()):
            arr = buf.values.get(p)
            if arr is None:
                arr = buf.aux.get(p)
            if arr is None or buf.iters.size == 0:
                continue
            m = min(buf.iters.size, arr.size)
            it, vv = buf.iters[:m], arr[:m]
            # Trim the burn-in off the left so the trace tab's slider actually
            # removes the messy initial transient (and the y-scale rescales to
            # the stationary region). At warmup_pct=0 the cutoff is the first
            # sweep, so nothing is dropped.
            keep = it >= cutoff
            it, vv = it[keep], vv[keep]
            if it.size > max_points:
                sel = np.unique(np.linspace(0, it.size - 1, max_points).astype(int))
                it, vv = it[sel], vv[sel]
            finite = np.isfinite(vv)
            series.append(
                TraceSeries(
                    chain=int(cid),
                    iters=it[finite].astype(np.int64).tolist(),
                    values=vv[finite].tolist(),
                )
            )
        if series:
            traces.append(ParamTrace(param=p, series=series))
    return TracesResponse(
        run_id=run_id, warmup_cutoff=cutoff,
        params=[t.param for t in traces], traces=traces,
    )


# ---------------------------------------------------------------------------
# Diagnostics tab
# ---------------------------------------------------------------------------

_SEV_RANK = {"error": 0, "warn": 1, "info": 2}


def _warning_kind(message: str) -> str:
    """Collapse a live warning message to a finding ``kind`` for grouping."""
    if "ESS" in message:
        return "ess_low"
    if "separated" in message:
        return "chain_separation"
    if "plateaued" in message:
        return "loglik_not_plateaued"
    if "divergent" in message:
        return "divergent"
    if "Too few" in message:
        return "insufficient_draws"
    if "No warnings" in message:
        return "ok"
    if ">" in message:  # the R̂ = x > thresh message
        return "rhat_high"
    return "diagnostic"


def _live_findings(diag: diag_mod.Diagnostics, rs: RunState) -> list[FindingGroup]:
    """Synthesize the verdict for a still-sampling run (no authoritative stage
    summary) from the watcher's *live* diagnostics — so a running fit with bad
    R̂/ESS/plateau shows real warnings instead of a falsely-green "no findings".
    Mirrors :func:`summarize_findings`' one-line-per-kind collapse."""
    warnings = diag_mod.derive_warnings(diag, rs, summary=None)
    real = [w for w in warnings if "No warnings" not in w.message]
    use = real if real else warnings
    by_kind: dict[str, list] = {}
    for w in use:
        by_kind.setdefault(_warning_kind(w.message), []).append(w)
    groups: list[FindingGroup] = []
    for kind, ws in by_kind.items():
        sev = min((w.severity.value for w in ws), key=lambda s: _SEV_RANK.get(s, 3))
        headline = ws[0].message + (f"  (+{len(ws) - 1} more)" if len(ws) > 1 else "")
        groups.append(
            FindingGroup(
                kind=kind, severity=sev, headline=headline,
                params=[w.param for w in ws if w.param],
            )
        )
    groups.sort(key=lambda g: _SEV_RANK.get(g.severity, 3))
    return groups


@router.get("/runs/{run_id}/diagnostics", response_model=DiagnosticsResponse)
def get_diagnostics(
    run_id: str, warmup_pct: int = Query(default=50, ge=0, le=100)
) -> DiagnosticsResponse:
    """Convergence diagnostics: camdl's authoritative verdict (findings) and
    R̂/ESS where a stage summary exists, else the watcher's live arviz estimate;
    plus per-chain mixing (acceptance / trajectory renewal) and the PMMH MAP."""
    store = _store()
    meta = _find_meta(store, run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    rs = build_run_state(meta)
    cutoff = _warmup_cutoff(rs, warmup_pct)
    summ = rs.summary
    base = dict(
        run_id=run_id, warmup_pct=warmup_pct, warmup_cutoff=cutoff,
        n_chains=len(rs.chains),
        stage=(summ.stage if summ is not None and summ.stage else None),
        logpost_label=meta.backend.logpost_label,
    )
    if rs.max_iter() is None:
        return DiagnosticsResponse(
            **base, n_tail=0, source="live", findings=[], params=[],
        )

    diag = diag_mod.compute_diagnostics(rs, cutoff, params=rs.params)

    findings: list[FindingGroup] = []
    if summ is not None:
        for g in diag_mod.summarize_findings(summ.findings):
            findings.append(FindingGroup(
                kind=g.kind, severity=g.severity.value,
                headline=g.headline, params=list(g.params),
            ))
    else:
        # No authoritative summary yet (still sampling) — synthesize the verdict
        # from live diagnostics so the strip isn't falsely green.
        findings = _live_findings(diag, rs)

    params_out: list[ParamDiagnostic] = []
    for p in meta.estimated:
        pd = diag.per_param.get(p)
        if pd is None:
            continue
        rhat_v, _ = diag_mod.effective_rhat(diag, summ, p)
        ess_v, _ = diag_mod.effective_ess(diag, summ, p)
        block = meta.docs.for_param(p)
        epc = summ.ess_per_chain.get(p, []) if summ is not None else []
        params_out.append(ParamDiagnostic(
            name=p, symbol=(block.symbol if block else None),
            rhat=_finite_or_none(rhat_v), ess_bulk=_finite_or_none(ess_v),
            ess_tail=_finite_or_none(pd.tail_ess), mcse=_finite_or_none(pd.mcse),
            mean=_fnum(pd.mean), sd=_fnum(pd.sd),
            sep=_finite_or_none(diag.chain_separation.get(p)),
            ess_per_chain=[_fnum(x) for x in epc],
        ))

    mixing = None
    mix = diag_mod.per_chain_mixing(rs, cutoff)
    if mix is not None:
        label, values, _labels, band = mix
        mixing = ChainMixing(
            label=label, values=[_fnum(v) for v in values],
            band=((float(band[0]), float(band[1])) if band and len(band) == 2 else None),
        )

    source = "camdl" if (summ is not None and summ.rhat) else "live"
    return DiagnosticsResponse(
        **base, n_tail=diag.n_tail, source=source,
        findings=findings, params=params_out, mixing=mixing,
        map_loglik=(_finite_or_none(summ.map_loglik) if summ is not None else None),
        map_chain=(summ.map_chain if summ is not None else None),
    )


# ---------------------------------------------------------------------------
# Compare workspace
# ---------------------------------------------------------------------------


@router.get("/compare", response_model=CompareResponse)
def compare(
    runs: list[str] = Query(..., description="run ids to compare (≥2)"),
    baseline: str | None = Query(default=None),
    allow_mismatched_horizon: bool = Query(default=False),
) -> CompareResponse:
    """Prequential model comparison via the authoritative ``camdl compare``.

    Resolves each run's ``prequential.json``, shells out (single source of truth
    for the elpd / Δelpd math and the evidence scale), and projects the result.
    Runs lacking a score artifact are dropped (reported in
    ``missing_prequential``). When the surviving models were scored on different
    horizons, camdl's commensurability guard trips: Δ columns come back ``None``
    and ``commensurable`` is false (the caller may still pass
    ``allow_mismatched_horizon`` to acknowledge it explicitly)."""
    store = _store()
    if not compare_mod.camdl_available():
        raise HTTPException(
            status_code=503,
            detail="camdl binary not found on PATH — model comparison needs it.",
        )

    specs: list[compare_mod.CompareSpec] = []
    labels: dict[str, str] = {}
    missing: list[str] = []
    for rid in runs:
        meta = _find_meta(store, rid)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"run not found: {rid}")
        labels[rid] = meta.display_label
        pq = compare_mod.find_prequential(meta.run_dir)
        if pq is None:
            missing.append(rid)
            continue
        specs.append(compare_mod.CompareSpec(name=rid, path=pq))

    if len(specs) < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                "need ≥2 runs with a prequential.json to compare; "
                f"have {len(specs)} (missing: {missing})"
            ),
        )
    if baseline is not None and baseline not in {s.name for s in specs}:
        raise HTTPException(
            status_code=422,
            detail=f"baseline '{baseline}' has no prequential.json among the selected runs",
        )

    try:
        data, commensurable, notes = compare_mod.run_compare(
            specs, baseline=baseline, allow_mismatched=allow_mismatched_horizon
        )
    except compare_mod.CompareError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    base_name = data.get("baseline", "")
    rows: list[CompareRow] = []
    for r in data.get("rows", []):
        name = r["name"]
        d = r.get("delta_elpd")
        se = r.get("se_delta_elpd")
        pit = r.get("pit_cov90")
        rows.append(
            CompareRow(
                run_id=name,
                label=labels.get(name, name),
                t_score=int(r["t_score"]),
                elpd=_fnum(r.get("elpd")),
                delta_elpd=_finite_or_none(d),
                delta_elpd_db=_finite_or_none(r.get("delta_elpd_db")),
                evidence_label=r.get("evidence_label"),
                e_t=_finite_or_none(r.get("e_t")),
                se_delta_elpd=_finite_or_none(se),
                mean_crps=_finite_or_none(r.get("mean_crps")),
                delta_mean_crps=_finite_or_none(r.get("delta_mean_crps")),
                pit_cov90=_finite_or_none(pit),
                is_baseline=(name == base_name),
                gap_is_real=(
                    d is not None and se not in (None, 0) and abs(d) > 2 * se
                ),
                overconfident=(pit is not None and pit < 0.70),
            )
        )

    # `camdl compare --format json` emits rows in input order (only the table/md
    # renderers sort). Present best-first by absolute elpd — the winner on top.
    rows.sort(key=lambda r: r.elpd, reverse=True)

    return CompareResponse(
        baseline=base_name,
        metrics=list(data.get("metrics", [])),
        commensurable=commensurable,
        notes=notes,
        rows=rows,
        missing_prequential=missing,
    )
