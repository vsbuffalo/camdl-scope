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
from dataclasses import dataclass

import arviz as az
import numpy as np

import re

from . import ingest
from .state import (
    DIAGNOSTIC_META,
    SAMPLER_PANEL_EXCLUDE,
    ChainBuffer,
    ChainSummary,
    Diagnostics,
    Finding,
    FindingGroup,
    ParamDiag,
    PriorSpec,
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


def _headline_bad_init(fs: list[Finding]) -> tuple[str, list[str]]:
    """camdl emits one ``bad_init`` per skipped chain; the default aggregation
    would show the first message and hide the count, which is the number that
    matters — half a fit's chains can be missing while the run reports done.
    Lead with how many and which."""
    ids = sorted(
        {
            int(c)
            for f in fs
            if (c := f.detail.get("chain_id")) is not None
        }
    )
    who = f" (chains {', '.join(str(i) for i in ids)})" if ids else ""
    n = len(ids) or len(fs)
    reason = next((str(f.detail.get("reason")) for f in fs if f.detail.get("reason")), "")
    # The reason is one long sentence; its head names the actual pathology
    # (typically a non-finite initial log-posterior).
    head = reason.split(",")[0].strip() if reason else ""
    return (
        f"{n} chain{'s' if n != 1 else ''} never sampled — skipped at "
        f"initialisation{who}" + (f": {head}" if head else ""),
        [],
    )


_HEADLINERS = {
    "rhat_high": _headline_rhat_high,
    "acceptance_rate_unhealthy": _headline_acceptance,
    "max_tree_depth_hits": _headline_tree_depth,
    "bad_init": _headline_bad_init,
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


def chain_ids_for(run: RunState, n: int) -> list[int]:
    """The chain ids behind ``n`` positionally-ordered per-chain values.

    camdl names chains ``chain_1 … chain_n`` on disk, so ids are 1-based and a
    positional array index is NOT a chain id — labelling a summary-derived
    array by position produced a `c0` that names no chain and disagrees with
    every other chain control in the app. Prefer the ids actually discovered in
    the run; fall back to camdl's 1-based convention when the counts disagree
    (a summary written for chains whose trace dirs are not all present yet)."""
    ids = sorted(run.chains)
    if len(ids) == n:
        return ids
    return list(range(1, n + 1))


def per_chain_mixing(
    run: RunState, warmup: int
) -> tuple[str, list[float], list[int], tuple[float, float] | None] | None:
    """``(label, values, chain_ids, band)`` for a per-chain mixing bar,
    best-source-first: camdl's authoritative acceptance; else live acceptance
    from the MH ``accepted`` column; else live PGAS ``trajectory_renewal``
    (no universal healthy band). ``None`` if nothing is available.

    ``chain_ids`` is parallel to ``values`` — the consumer labels with these,
    never with the array position."""
    summ = run.summary
    if summ is not None:
        acc = summ.per_chain_acceptance
        if acc:
            return "acceptance", acc, chain_ids_for(run, len(acc)), (0.15, 0.50)

    def _live(col: str) -> tuple[list[float], list[int]]:
        vals: list[float] = []
        ids: list[int] = []
        for cid in sorted(run.chains):
            buf = run.chains[cid]
            if col not in buf.aux or buf.n == 0:
                continue
            a = buf.aux[col][buf.iters >= warmup]
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(float(np.mean(a)))
                ids.append(cid)
        return vals, ids

    vals, ids = _live("accepted")
    if vals:
        return "acceptance", vals, ids, (0.15, 0.50)
    vals, ids = _live("trajectory_renewal")
    if vals:
        return "trajectory renewal", vals, ids, None
    return None


@dataclass(frozen=True)
class PriorPosterior:
    """What the data did to one parameter's prior.

    The three numbers of the prior→posterior half of a Bayesian workflow
    (Gelman et al. 2020, *Bayesian Workflow*, §6; Betancourt, *Towards a
    Principled Bayesian Workflow*):

    * ``contraction`` = 1 − σ²_post / σ²_prior. How much the data narrowed the
      parameter. Near 1, the likelihood determines it; near 0, the posterior is
      the prior restated and any "estimate" is an assumption. Negative means the
      posterior is WIDER than the prior, which is a modelling error worth
      seeing rather than clamping away.
    * ``z`` = (μ_post − μ_prior) / σ_prior. How far the posterior moved, in
      prior standard deviations — prior/data conflict. Read WITH contraction:
      high contraction + small |z| is the healthy case; large |z| says the data
      pulled hard against where the prior was centred; low contraction + large
      |z| means the prior is fighting the likelihood and neither wins.
    * ``bound_pressure`` — the fraction of posterior draws sitting within 1% of
      a declared bound. A posterior pinned to its box is not an estimate; the
      constraint is doing the work, and the interval is meaningless.

    ``prior_mean``/``prior_sd`` are Monte-Carlo estimates from the resolved
    prior (deterministic seed), which is what makes this work uniformly across
    families and truncations rather than needing a closed form per family.
    """

    param: str
    symbol: str | None
    prior_label: str | None
    prior_mean: float | None
    prior_sd: float | None
    post_mean: float | None
    post_sd: float | None
    contraction: float | None
    z: float | None
    bound_pressure: float | None


#: Draws used to estimate a prior's moments. Large enough that the reported
#: contraction is stable in its 3rd digit; small enough to stay instant.
_PRIOR_MOMENT_DRAWS = 20_000


def prior_posterior(
    run: RunState, warmup: int, priors: dict[str, PriorSpec] | None = None
) -> list[PriorPosterior]:
    """Per-parameter prior→posterior comparison over the post-warm-up draws.

    Parameters with no resolved prior, or an improper FLAT prior with no bounds
    (nothing to draw from, so no prior scale exists), report ``None`` for the
    prior-relative columns rather than a fabricated number — "we cannot say" is
    a different statement from "no shrinkage"."""
    specs = priors if priors is not None else run.priors
    rng = np.random.default_rng(0)
    out: list[PriorPosterior] = []
    for name in run.meta.estimated:
        post = _tail_arrays(run, name, warmup)
        post_mean = post_sd = None
        if post is not None:
            flat = post[np.isfinite(post)]
            if flat.size:
                post_mean = float(flat.mean())
                post_sd = float(flat.std(ddof=1)) if flat.size > 1 else 0.0

        spec = (specs or {}).get(name)
        prior_mean = prior_sd = None
        if spec is not None:
            draws = ingest.sample_prior(spec, _PRIOR_MOMENT_DRAWS, rng)
            draws = draws[np.isfinite(draws)]
            if draws.size > 1:
                prior_mean = float(draws.mean())
                prior_sd = float(draws.std(ddof=1))

        contraction = z = None
        if prior_sd and prior_sd > 0 and post_sd is not None:
            contraction = 1.0 - (post_sd / prior_sd) ** 2
            if post_mean is not None and prior_mean is not None:
                z = (post_mean - prior_mean) / prior_sd

        bound_pressure = None
        if spec is not None and spec.bounds is not None and post is not None:
            lo, hi = spec.bounds
            flat = post[np.isfinite(post)]
            if flat.size and hi > lo:
                edge = 0.01 * (hi - lo)
                near = (flat <= lo + edge) | (flat >= hi - edge)
                bound_pressure = float(near.mean())

        block = run.meta.docs.for_param(name)
        out.append(
            PriorPosterior(
                param=name,
                symbol=(block.symbol if block else None),
                prior_label=_prior_label(spec),
                prior_mean=prior_mean,
                prior_sd=prior_sd,
                post_mean=post_mean,
                post_sd=post_sd,
                contraction=contraction,
                z=z,
                bound_pressure=bound_pressure,
            )
        )
    return out


def _prior_label(spec: PriorSpec | None) -> str | None:
    """``LogNormal(mu=-0.6, sigma=0.4)`` — the prior as written, so the reader
    can see what the contraction is relative to."""
    if spec is None:
        return None
    if spec.args:
        args = ", ".join(f"{k}={_fmt_num(v)}" for k, v in spec.args.items())
        return f"{spec.family.value}({args})"
    if spec.bounds is not None:
        lo, hi = spec.bounds
        return f"{spec.family.value}[{_fmt_num(lo)}, {_fmt_num(hi)}]"
    return spec.family.value


def _fmt_num(v: float) -> str:
    return f"{v:g}"


@dataclass(frozen=True)
class PanelColumn:
    """One column of a sampler panel: the metric, what it means, and the
    healthy band if the metric has a conventional one."""

    key: str
    label: str
    note: str | None = None
    band: tuple[float, float] | None = None


@dataclass(frozen=True)
class SamplerPanel:
    """A chain × metric table of sampler-mechanism diagnostics.

    One shape covers both panels the samplers produce — per-parameter block
    acceptance (columns are parameters) and per-chain sampler telemetry
    (columns are metrics) — because both are "a number per chain per column,
    some of which fall outside a healthy band". Adding a sampler means adding
    rows to a table, not a new component.
    """

    id: str
    title: str
    note: str | None
    rows: list[int]  # chain ids
    columns: list[PanelColumn]
    values: list[list[float | None]]  # [row][column]


def _block_acceptance_panel(run: RunState) -> SamplerPanel | None:
    """PGAS writes acceptance PER PARAMETER BLOCK (``acceptance_rates`` is
    ``[chain][param]``); the mixing bar collapses that to a chain mean, which
    hides the case this panel is for — one badly-tuned block stuck at 0.99 or
    0.02 while the chain average looks healthy."""
    summ = run.summary
    if summ is None or not summ.acceptance_rates:
        return None
    rates = summ.acceptance_rates
    width = max((len(r) for r in rates), default=0)
    # A single column per chain is PMMH's per-chain scalar — already the mixing
    # bar, nothing per-parameter to add.
    if width < 2:
        return None
    # Nor is there anything to add when a sampler replicates one chain-level
    # rate across the parameter axis (PGAS does this today): every column would
    # be identical and the grid would restate the mixing bar in 11 columns.
    # The panel appears when the per-block detail genuinely exists.
    if all(
        max(row) - min(row) < 1e-9
        for row in rates
        if row and all(np.isfinite(v) for v in row)
    ):
        return None
    # camdl orders the inner axis by the stage's estimated parameters.
    names = list(run.meta.estimated)[:width]
    if len(names) < width:
        names += [f"block{i}" for i in range(len(names), width)]
    band = (0.15, 0.50)
    return SamplerPanel(
        id="block-acceptance",
        title="block acceptance",
        note=(
            "Metropolis acceptance for each parameter's update block, per "
            "chain. The per-chain mixing bar is the average of a row — a "
            "single block outside the band is invisible there."
        ),
        rows=chain_ids_for(run, len(rates)),
        columns=[PanelColumn(key=n, label=n, band=band) for n in names],
        values=[
            [float(v) if np.isfinite(v) else None for v in row[:width]]
            + [None] * (width - len(row))
            for row in rates
        ],
    )


def _sampler_telemetry_panel(run: RunState, warmup: int) -> SamplerPanel | None:
    """Per-chain summary of every diagnostic column the trace declares.

    Driven by camdl's declared column roles rather than a fixed list, so a
    sampler that starts writing a new diagnostic surfaces it without a watcher
    change; :data:`DIAGNOSTIC_META` only adds a label, a reduction and a
    reading for the columns we can say something useful about. Counters (e.g.
    ``n_divergent``) sum — averaging an event count over draws hides it."""
    cols: list[str] = []
    for buf in run.chains.values():
        for c in buf.aux:
            if c not in cols and c not in SAMPLER_PANEL_EXCLUDE:
                cols.append(c)
    if not cols:
        return None
    cols.sort(key=lambda c: (c not in DIAGNOSTIC_META, c))

    columns: list[PanelColumn] = []
    for c in cols:
        meta = DIAGNOSTIC_META.get(c)
        columns.append(
            PanelColumn(
                key=c,
                label=meta[0] if meta else c,
                note=meta[2] if meta else None,
                band=meta[3] if meta else None,
            )
        )

    rows = sorted(run.chains)
    values: list[list[float | None]] = []
    for cid in rows:
        buf = run.chains[cid]
        row: list[float | None] = []
        for c in cols:
            arr = buf.aux.get(c)
            if arr is None or buf.n == 0:
                row.append(None)
                continue
            a = arr[buf.iters >= warmup]
            a = a[np.isfinite(a)]
            if a.size == 0:
                row.append(None)
                continue
            how = DIAGNOSTIC_META.get(c, (None, "mean", None, None))[1]
            row.append(float(a.sum()) if how == "sum" else float(a.mean()))
        values.append(row)

    return SamplerPanel(
        id="sampler-telemetry",
        title="sampler diagnostics",
        note=(
            "Per-chain summary of the sampler's own telemetry, over the "
            "retained draws. Counts are sums; everything else is a mean."
        ),
        rows=rows,
        columns=columns,
        values=values,
    )


def sampler_panels(run: RunState, warmup: int) -> list[SamplerPanel]:
    """Every method-specific diagnostic panel this run can support, in reading
    order. Empty when the run's sampler exposes none — the UI then shows
    nothing rather than an empty frame."""
    panels = [
        _block_acceptance_panel(run),
        _sampler_telemetry_panel(run, warmup),
    ]
    return [p for p in panels if p is not None]


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
