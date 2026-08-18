"""Diagnostics core — pure functions, arviz-backed.

``compute_diagnostics(run, warmup) -> Diagnostics`` and
``derive_warnings(diag, run) -> [Warning_]``.

We do NOT hand-roll R̂/ESS/MCSE — arviz owns those (rank-normalized split-R̂,
bulk/tail-ESS, MCSE). We add the things arviz doesn't: acceptance rate, a
log-likelihood plateau test, and a chain-separation measure.

The post-warmup tail is built by selecting rows whose iteration index is
``>= warmup_cutoff`` per chain, then truncating all chains to the common
minimum length (arviz wants a rectangular ``(chain, draw)`` array).
"""

from __future__ import annotations

import warnings as _pywarnings

import arviz as az
import numpy as np

import re

from .state import (
    ChainBuffer,
    ChainSummary,
    Diagnostics,
    Finding,
    FindingGroup,
    ParamDiag,
    RunState,
    Severity,
    Warning_,
)

_SEV_ORDER = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}


def _tail_arrays(run: RunState, param: str, warmup: int) -> np.ndarray | None:
    """Build a ``(n_chains, n_draws)`` array for ``param`` from the post-warmup
    tail, truncated to the common minimum draw count. ``None`` if a value
    column is missing or there are too few post-warmup draws.

    Reads from ``values`` (params) or ``aux`` (e.g. log_posterior)."""
    per_chain: list[np.ndarray] = []
    for cid in sorted(run.chains):
        buf = run.chains[cid]
        src = buf.values if param in buf.values else (buf.aux if param in buf.aux else None)
        if src is None or buf.n == 0:
            return None
        mask = buf.iters >= warmup
        vals = src[param][mask]
        # Keep only finite tail (early -inf ll etc. would poison arviz).
        per_chain.append(vals)
    if not per_chain:
        return None
    m = min(len(v) for v in per_chain)
    if m < 4:  # arviz needs a handful of draws to be meaningful
        return None
    # Take the last `m` of each chain (align on the most recent draws).
    arr = np.stack([v[-m:] for v in per_chain])
    return arr


def _az_safe(fn, arr: np.ndarray) -> float:
    """Run an arviz scalar diagnostic on a (chain, draw) array, swallowing the
    warnings arviz emits for short/degenerate inputs and returning NaN on
    failure."""
    try:
        with _pywarnings.catch_warnings():
            _pywarnings.simplefilter("ignore")
            return float(fn(arr))
    except Exception:
        return float("nan")


# Cap on the pooled points handed to Theil–Sen. scipy's ``theilslopes``
# materializes ALL pairwise differences — n×n float64 intermediates, several of
# them — so memory is quadratic: 12k pooled points (4 chains × a 6k-sweep PGAS
# trace) allocated ~7 GB *per request* and took two watchers into system swap.
# The plateau test only asks "is the trailing slope ≈ 0 relative to the ll
# scale"; an evenly-strided subsample answers that just as well, and the cap
# makes the cost constant (2000² ≈ 32 MB per intermediate).
_PLATEAU_MAX_POINTS = 2000


def _plateau_test(
    run: RunState, window_frac: float = 0.5, min_pts: int = 20
) -> tuple[bool | None, float | None]:
    """Robust test of whether the pooled log-likelihood has plateaued.

    Pool all chains' ``log_likelihood`` over the trailing ``window_frac`` of
    sweeps, fit a Theil–Sen (median-of-slopes, outlier-robust) line of ll vs a
    normalized sweep coordinate in [0,1], and call it plateaued if the slope is
    small relative to the ll scale. Returns ``(plateaued, slope)``.

    ``slope`` is in ll-units per unit-normalized-sweep; we threshold it against
    the trailing ll standard deviation, so it's scale-aware. Pooled input is
    thinned to ``_PLATEAU_MAX_POINTS`` (see note above) — Theil–Sen's pairwise
    memory is quadratic in n."""
    from scipy import stats as sstats

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for buf in run.chains.values():
        # Plateau the *data-fit* series: obs_ll for PGAS (the complete-data
        # log_likelihood is path-dominated and isn't always written); the bare
        # log_likelihood for MH/PMMH, where it already is p(y|θ).
        col = "obs_ll" if "obs_ll" in buf.aux else "log_likelihood"
        if col not in buf.aux or buf.n == 0:
            continue
        ll = buf.aux[col]
        it = buf.iters.astype(float)
        fin = np.isfinite(ll)
        if fin.sum() < min_pts:
            continue
        ll, it = ll[fin], it[fin]
        cut = it.min() + (1 - window_frac) * (it.max() - it.min())
        sel = it >= cut
        if sel.sum() < min_pts:
            sel = np.ones_like(it, dtype=bool)
        xs.append(it[sel])
        ys.append(ll[sel])
    if not xs:
        return None, None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    # Evenly-strided thin across the pooled (per-chain-concatenated) series —
    # ordering doesn't matter to Theil–Sen, which pairs on x values.
    if x.size > _PLATEAU_MAX_POINTS:
        idx = np.linspace(0, x.size - 1, _PLATEAU_MAX_POINTS).astype(np.intp)
        x, y = x[idx], y[idx]
    if x.max() - x.min() < 1e-9:
        return True, 0.0
    xn = (x - x.min()) / (x.max() - x.min())  # in [0,1]
    try:
        slope, *_ = sstats.theilslopes(y, xn)
    except Exception:
        return None, None
    sd = float(np.std(y)) or 1.0
    # Slope < ~0.5 sd over the whole window -> effectively flat.
    plateaued = abs(slope) < 0.5 * sd
    return bool(plateaued), float(slope)


def _chain_separation(run: RunState, param: str, warmup: int) -> float:
    """Spread of per-chain means relative to the pooled within-chain sd.

    >~1 means the chains disagree more than their internal scatter — a
    separation/multimodality red flag. NaN if not computable."""
    means: list[float] = []
    within: list[float] = []
    for buf in run.chains.values():
        src = buf.values if param in buf.values else (buf.aux if param in buf.aux else None)
        if src is None or buf.n == 0:
            continue
        v = src[param][buf.iters >= warmup]
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        means.append(float(np.mean(v)))
        within.append(float(np.std(v)))
    if len(means) < 2:
        return float("nan")
    between = float(np.std(means))
    win = float(np.mean(within)) or 1e-12
    return between / win


def compute_diagnostics(
    run: RunState, warmup: int, params: list[str] | None = None
) -> Diagnostics:
    """Compute per-parameter R̂/ESS/MCSE on the post-warmup tail, plus
    acceptance, plateau, and chain separation.

    ``params`` restricts which estimated coordinates are summarized (the UI
    passes the user-selected subset so the table matches the plots); ``None``
    falls back to all of ``run.params``."""
    params = run.params if params is None else list(params)
    per_param: dict[str, ParamDiag] = {}
    n_tail = 10**9

    for p in params:
        arr = _tail_arrays(run, p, warmup)
        if arr is None:
            per_param[p] = ParamDiag(
                rhat=float("nan"), bulk_ess=float("nan"), tail_ess=float("nan"),
                mcse=float("nan"), mean=float("nan"), sd=float("nan"),
            )
            continue
        n_tail = min(n_tail, arr.shape[1])
        finite = arr[np.isfinite(arr)]
        per_param[p] = ParamDiag(
            rhat=_az_safe(az.rhat, arr),  # method="rank" (rank-normalized split) by default
            bulk_ess=_az_safe(lambda a: az.ess(a, method="bulk"), arr),
            # Tail-ESS over the standard 5%/95% quantiles (arviz>=1.2 requires
            # the explicit `prob` argument).
            tail_ess=_az_safe(lambda a: az.ess(a, method="tail", prob=(0.05, 0.95)), arr),
            mcse=_az_safe(az.mcse, arr),
            mean=float(np.mean(finite)) if finite.size else float("nan"),
            sd=float(np.std(finite)) if finite.size else float("nan"),
        )
    if n_tail == 10**9:
        n_tail = 0

    # Acceptance: mean of the `accepted` column over the post-warmup tail (MH).
    # PGAS has no accept/reject; its mixing analog is the trajectory-renewal rate.
    acceptance = _aux_tail_mean(run, warmup, "accepted")
    renewal = _aux_tail_mean(run, warmup, "trajectory_renewal")

    # Divergences: not in the trace for these samplers; left None (would come
    # from a log if a log path is provided — out of scope for trace-only v1).
    n_divergent: int | None = None

    plateaued, slope = _plateau_test(run)
    chain_sep = {p: _chain_separation(run, p, warmup) for p in params}

    return Diagnostics(
        per_param=per_param,
        acceptance=acceptance,
        n_divergent=n_divergent,
        plateaued=plateaued,
        plateau_slope=slope,
        chain_separation=chain_sep,
        warmup_cutoff=warmup,
        n_tail=n_tail,
        logpost_label=run.meta.backend.logpost_label,
        renewal=renewal,
    )


def _aux_tail_mean(run: RunState, warmup: int, col: str) -> float | None:
    """Per-chain mean of an aux column over the post-warmup tail, averaged
    across chains. ``None`` if no chain carries the column."""
    vals: list[float] = []
    for buf in run.chains.values():
        if col not in buf.aux or buf.n == 0:
            continue
        a = buf.aux[col][buf.iters >= warmup]
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(float(np.mean(a)))
    if not vals:
        return None
    return float(np.mean(vals))


def derive_warnings(
    diag: Diagnostics,
    run: RunState,
    rhat_thresh: float = 1.1,
    bulk_ess_thresh: float = 400.0,
    sep_thresh: float = 1.0,
    summary: ChainSummary | None = None,
) -> list[Warning_]:
    """Translate diagnostics into a ranked list of warnings.

    When ``summary`` (camdl's authoritative end-of-stage diagnostics) is present
    its findings own the R̂/ESS verdict, so we drop the watcher's *live* R̂/ESS
    warnings here to avoid double-reporting — they're shown via the verdict
    strip. The watcher-only signals (plateau, chain separation) always stand."""
    out: list[Warning_] = []

    if diag.n_tail < 4:
        out.append(Warning_(Severity.INFO, "Too few post-warmup draws for stable diagnostics."))
        return out

    if summary is None:
        for p, d in diag.per_param.items():
            if np.isfinite(d.rhat) and d.rhat > rhat_thresh:
                out.append(Warning_(Severity.ERROR, f"R̂ = {d.rhat:.3f} > {rhat_thresh}", param=p))
        for p, d in diag.per_param.items():
            if np.isfinite(d.bulk_ess) and d.bulk_ess < bulk_ess_thresh:
                sev = Severity.ERROR if d.bulk_ess < bulk_ess_thresh / 4 else Severity.WARN
                out.append(Warning_(sev, f"bulk-ESS = {d.bulk_ess:.0f} < {bulk_ess_thresh:.0f}", param=p))
    for p, s in diag.chain_separation.items():
        if np.isfinite(s) and s > sep_thresh:
            out.append(Warning_(Severity.WARN, f"chains separated (between/within = {s:.2f})", param=p))

    if diag.plateaued is False:
        sl = f" (slope {diag.plateau_slope:.3g})" if diag.plateau_slope is not None else ""
        out.append(Warning_(Severity.WARN, f"log-likelihood not plateaued{sl}"))

    if diag.n_divergent:
        out.append(Warning_(Severity.ERROR, f"{diag.n_divergent} divergent transitions"))

    if not out:
        out.append(Warning_(Severity.INFO, "No warnings — diagnostics within thresholds."))
    # Sort: error > warn > info.
    out.sort(key=lambda w: _SEV_ORDER[w.severity])
    return out


# ---------------------------------------------------------------------------
# camdl summary: aggregate findings + best-available per-chain mixing
# ---------------------------------------------------------------------------


def _parse_band(message: str) -> str:
    """Pull the healthy band out of an acceptance message
    (``…outside healthy range [15%, 50%].``) -> ``"healthy 15–50%"``; ``""`` if
    not present."""
    m = re.search(r"\[\s*([\d.]+)\s*%?\s*,\s*([\d.]+)\s*%?\s*\]", message)
    if not m:
        return ""
    return f"healthy {float(m.group(1)):g}–{float(m.group(2)):g}%"


def _headline_rhat_high(fs: list[Finding]) -> tuple[str, list[str]]:
    by_param: dict[str, float] = {}
    for f in fs:
        if f.param is None:
            continue
        r = f.detail.get("rhat")
        if r is not None:
            by_param[f.param] = max(by_param.get(f.param, 0.0), float(r))
    ranked = sorted(by_param.items(), key=lambda kv: kv[1], reverse=True)
    shown = ranked[:3]
    body = " · ".join(f"{p} {r:.2f}" for p, r in shown)
    if len(ranked) > 3:
        body += f"  (+{len(ranked) - 3} more)"
    return f"R̂ high: {body}", [p for p, _ in ranked]


def _headline_acceptance(fs: list[Finding]) -> tuple[str, list[str]]:
    rates = sorted({round(float(f.detail["rate"]), 4) for f in fs if "rate" in f.detail})
    band = next((_parse_band(f.message) for f in fs if _parse_band(f.message)), "")
    if rates:
        span = f"{rates[0]:.0%}" if len(rates) == 1 else f"{rates[0]:.0%}–{rates[-1]:.0%}"
        head = f"acceptance unhealthy: {len(rates)} chain rate(s) {span}"
    else:
        head = "acceptance unhealthy"
    if band:
        head += f"  ({band})"
    return head, []


def _headline_tree_depth(fs: list[Finding]) -> tuple[str, list[str]]:
    f = fs[0]
    d = f.detail
    if {"n_hits", "n_sweeps", "max_depth"} <= set(d):
        pct = d.get("pct")
        pct_s = f" ({float(pct):.0f}%)" if pct is not None else ""
        return (f"tree depth: {int(d['n_hits'])}/{int(d['n_sweeps'])} sweeps{pct_s} "
                f"hit max depth {int(d['max_depth'])}"), []
    return f.message or "max tree depth hit", []


_HEADLINERS = {
    "rhat_high": _headline_rhat_high,
    "acceptance_rate_unhealthy": _headline_acceptance,
    "max_tree_depth_hits": _headline_tree_depth,
}


def summarize_findings(findings: list[Finding]) -> list[FindingGroup]:
    """Collapse camdl's repetitive findings into one line per ``kind``, ranked
    error→warn→info. Known kinds get a hand-tuned aggregate headline; unknown
    kinds fall back to their (deduplicated) messages."""
    by_kind: dict[str, list[Finding]] = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(f)
    groups: list[FindingGroup] = []
    for kind, fs in by_kind.items():
        sev = min((f.severity for f in fs), key=lambda s: _SEV_ORDER[s])
        headliner = _HEADLINERS.get(kind)
        if headliner is not None:
            headline, params = headliner(fs)
        else:
            msgs = list(dict.fromkeys(f.message for f in fs if f.message))
            headline = (msgs[0] if msgs else kind) + (
                f"  (+{len(msgs) - 1} more)" if len(msgs) > 1 else "")
            params = [f.param for f in fs if f.param]
        groups.append(FindingGroup(kind=kind, severity=sev, headline=headline, params=params))
    groups.sort(key=lambda g: _SEV_ORDER[g.severity])
    return groups


def per_chain_mixing(
    run: RunState, warmup: int
) -> tuple[str, list[float], list[str], tuple[float, float] | None] | None:
    """``(label, values, chain_labels, band)`` for a per-chain mixing bar,
    best-source-first: camdl's authoritative acceptance; else live acceptance
    from the MH ``accepted`` column; else live PGAS ``trajectory_renewal``
    (no universal healthy band). ``None`` if nothing is available."""
    summ = run.summary
    if summ is not None:
        acc = summ.per_chain_acceptance
        if acc:
            return "acceptance", acc, [f"c{i}" for i in range(len(acc))], (0.15, 0.50)

    def _live(col: str) -> tuple[list[float], list[str]]:
        vals: list[float] = []
        labs: list[str] = []
        for cid in sorted(run.chains):
            buf = run.chains[cid]
            if col not in buf.aux or buf.n == 0:
                continue
            a = buf.aux[col][buf.iters >= warmup]
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(float(np.mean(a)))
                labs.append(f"c{cid}")
        return vals, labs

    vals, labs = _live("accepted")
    if vals:
        return "acceptance", vals, labs, (0.15, 0.50)
    vals, labs = _live("trajectory_renewal")
    if vals:
        return "trajectory renewal", vals, labs, None
    return None


def effective_rhat(
    diag: Diagnostics, summary: ChainSummary | None, param: str
) -> tuple[float, str]:
    """R̂ for ``param``, camdl-authoritative when available else the live arviz
    estimate, tagged with its source (``"camdl"`` | ``"live"``)."""
    if summary is not None and param in summary.rhat:
        return summary.rhat[param], "camdl"
    d = diag.per_param.get(param)
    return (d.rhat if d is not None else float("nan")), "live"


def effective_ess(
    diag: Diagnostics, summary: ChainSummary | None, param: str
) -> tuple[float | None, str]:
    """Combined ESS for ``param`` — camdl-authoritative (may be ``None`` when
    camdl judges it not estimable) else the live bulk-ESS estimate."""
    if summary is not None and param in summary.ess:
        return summary.ess[param], "camdl"
    d = diag.per_param.get(param)
    return (d.bulk_ess if d is not None else float("nan")), "live"
