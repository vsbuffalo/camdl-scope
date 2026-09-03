"""Which term of the sampled target separates the chains.

A PGAS fit samples ``(θ, X)`` against a target that is a sum of four things::

    log p(θ, X | y) = obs_ll + transition_ll + initial_state_ll + log prior
                      └ data ┘ └── the latent path ──┘          └ priors ┘

When chains disagree, the useful question is not *how much* they disagree but
*about what*. A fit whose chains differ by 130 nats of trajectory density and
10 nats of data fit is one where the data cannot adjudicate between the paths
the sampler found; a fit where the data term dominates is a different and more
alarming problem. Neither is visible in R̂, which reports per parameter.

**Shares, not ranges.** The natural first instinct is to plot each term's range
across chain means, and it misleads: ``max−min`` of a sum is not the sum of
``max−min``. On one real fit the four ranges sum to 159.3 nats against a total
range of 131.0, so the parts appear to over-explain the whole. The variance
identity ``Var(T) = Σᵢ Cov(Xᵢ, T)`` *is* exact for ``T = Σᵢ Xᵢ``, so shares are
what the bars carry and the range in nats rides along as a readable magnitude.
A share can be negative — a term anti-correlated with the total is pulling the
chains *together*, which is the honest answer to "is a prior overruling the
data?" and cannot be expressed by a range at all.

Measured over this project's store: the trajectory term explains more than the
observation term in 51 of 61 decomposable fits, and the prior's share has
median +0.002 (range −0.06 to +0.16). So the ranking alone is nearly constant
and is not the finding; the proportions, and the minority where observation
dominates, are.

Pure functions over a :class:`RunState`. No I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import RunState

#: The additive terms, in the order they read on screen (data first — it is the
#: one a modeller is accountable for).
TERM_LABELS: dict[str, str] = {
    "obs_ll": "observation likelihood",
    "transition_ll": "trajectory probability",
    "initial_state_ll": "initial state",
    "prior": "prior (implied)",
}

#: One-line readings, shown beside each bar.
TERM_NOTES: dict[str, str] = {
    "obs_ll": (
        "How well the sampled path reproduces the observed data. A large share "
        "means the chains disagree about something the data can see — the "
        "serious case, and the one to follow into the per-stream breakdown."
    ),
    "transition_ll": (
        "The process density of the conditioned latent path. PGAS samples the "
        "path, so chains hold different paths by construction and some share "
        "here is expected; a dominant share means the target is mostly path "
        "mass the data cannot adjudicate."
    ),
    "initial_state_ll": (
        "Density of the initial state under the model's init laws. Zero for a "
        "deterministic init; a large share means the chains disagree mainly "
        "about where the epidemic started."
    ),
    "prior": (
        "Implied as log posterior − complete-data log-likelihood. A negative "
        "share means the prior pulls the chains together rather than apart — "
        "it is not the thing separating them."
    ),
}

#: Terms that must be present (or derivable) for a decomposition to exist.
_REQUIRED = ("log_posterior", "log_complete_data_ll", "obs_ll", "transition_ll")

#: The identity is written to the trace at limited precision; anything above
#: this is a real disagreement, not rounding. Observed max across this project's
#: store: 1e-4.
RESIDUAL_TOL = 1e-2


@dataclass(frozen=True)
class TermSpread:
    """One additive term, summarised across chains."""

    key: str
    label: str
    note: str
    #: Cov(term, total) / Var(total) over chain means. Sums to 1 across terms.
    share: float
    #: max − min across chain means, in nats. Readable, but NOT additive.
    range_nats: float
    sd_nats: float
    chain_means: list[float]


@dataclass(frozen=True)
class TargetDecomposition:
    """The sampled target broken into its terms, or the reason there is none."""

    available: bool
    #: Why not, when ``available`` is false — shown instead of an empty chart.
    reason: str | None = None
    chain_ids: list[int] = field(default_factory=list)
    terms: list[TermSpread] = field(default_factory=list)
    total_range: float = 0.0
    total_sd: float = 0.0
    #: Largest |complete_data − (obs + transition + initial)| over retained
    #: draws. Non-zero beyond tolerance means camdl grew a term this does not
    #: know about, which would otherwise be silently absorbed into "prior".
    residual_max: float = 0.0
    #: True when ``initial_state_ll`` was not a column and was recovered by
    #: subtraction — correct, but worth saying, since a run predating the
    #: column has no independent check on it.
    initial_state_derived: bool = False


@dataclass(frozen=True)
class StreamSpread:
    """One observation stream's part of the data term's chain disagreement."""

    stream: str
    #: Cov(stream, obs_ll) / Var(obs_ll) over chain means; sums to 1.
    share: float
    range_nats: float
    chain_means: list[float]


def _chain_means(
    rs: RunState, col: str, cutoff: int, nonfinite: set[int] | None = None
) -> dict[int, float] | None:
    """Post-warm-up mean of ``col`` per chain, or ``None`` when no chain has one.

    Chains whose retained draws are all non-finite are recorded in ``nonfinite``
    rather than merely omitted: "this chain has no draws" and "this chain's
    density is -inf" are different problems, and reporting the second as the
    first sends the reader to the sampler instead of to the densities.
    """
    out: dict[int, float] = {}
    for cid, buf in sorted(rs.chains.items()):
        arr = buf.aux.get(col)
        if arr is None or buf.iters.size == 0:
            continue
        n = min(buf.iters.size, arr.size)
        keep = buf.iters[:n] >= cutoff
        vals = arr[:n][keep]
        if vals.size and not np.isfinite(vals).any():
            if nonfinite is not None:
                nonfinite.add(cid)
            continue
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[cid] = float(vals.mean())
    return out or None


def _shares(
    parts: dict[str, np.ndarray], total: np.ndarray
) -> dict[str, float] | None:
    """Cov(part, total)/Var(total) per term — exact because total is their sum."""
    var = float(np.var(total, ddof=1)) if total.size > 1 else 0.0
    if not np.isfinite(var) or var <= 0:
        return None
    return {
        k: float(np.cov(v, total, ddof=1)[0, 1] / var) for k, v in parts.items()
    }


def decompose_target(rs: RunState, cutoff: int) -> TargetDecomposition:
    """Split the between-chain spread in the sampler's target into its terms.

    Unavailable — with a stated reason rather than an empty panel — for a fit
    whose sampler writes no complete-data split (PMMH/MH/NUTS carry a scalar
    ``log_likelihood`` only), for a single chain, or when the terms are
    non-finite.
    """
    if len(rs.chains) < 2:
        return TargetDecomposition(
            available=False,
            reason="a decomposition compares chains — this run has one.",
        )
    have = {c for buf in rs.chains.values() for c in buf.aux}
    missing = [c for c in _REQUIRED if c not in have]
    if missing:
        return TargetDecomposition(
            available=False,
            reason=(
                "this sampler does not write a complete-data split — only "
                "marginal samplers' log_likelihood is present, which has no "
                "path or prior term to separate."
            ),
        )

    nonfinite: set[int] = set()
    per_term: dict[str, dict[int, float]] = {}
    for col in ("log_posterior", "log_complete_data_ll", "obs_ll", "transition_ll"):
        m = _chain_means(rs, col, cutoff, nonfinite)
        if m is None:
            return TargetDecomposition(
                available=False,
                reason=f"no finite post-warm-up draws for {col}.",
            )
        per_term[col] = m

    init = _chain_means(rs, "initial_state_ll", cutoff, nonfinite)
    derived = init is None
    chain_ids = sorted(
        set(per_term["log_posterior"])
        & set(per_term["log_complete_data_ll"])
        & set(per_term["obs_ll"])
        & set(per_term["transition_ll"])
        & (set(init) if init is not None else set(per_term["obs_ll"]))
    )
    if len(chain_ids) < 2:
        if nonfinite:
            ids = ", ".join(f"c{c}" for c in sorted(nonfinite))
            return TargetDecomposition(
                available=False,
                reason=(
                    f"non-finite target terms in {ids}, leaving fewer than two "
                    f"comparable chains. That is the finding — check the trace "
                    f"for -inf log-densities rather than for missing draws."
                ),
            )
        return TargetDecomposition(
            available=False,
            reason="fewer than two chains have post-warm-up draws for every term.",
        )

    def vec(m: dict[int, float]) -> np.ndarray:
        return np.array([m[c] for c in chain_ids], dtype=float)

    total = vec(per_term["log_posterior"])
    complete = vec(per_term["log_complete_data_ll"])
    obs = vec(per_term["obs_ll"])
    trans = vec(per_term["transition_ll"])
    # Recover the init term by subtraction when the column predates the run.
    # camdl writes it explicitly precisely because a component visible only by
    # subtraction is hard to find; derived here, it is flagged as such.
    initial = vec(init) if init is not None else complete - obs - trans
    prior = total - complete

    parts = {
        "obs_ll": obs,
        "transition_ll": trans,
        "initial_state_ll": initial,
        "prior": prior,
    }
    if not all(np.isfinite(v).all() for v in parts.values()) or not np.isfinite(
        total
    ).all():
        return TargetDecomposition(
            available=False,
            reason=(
                "at least one chain has a non-finite target term, so the "
                "shares are undefined. That is itself the finding — check the "
                "trace for -inf log-densities."
            ),
        )

    shares = _shares(parts, total)
    if shares is None:
        return TargetDecomposition(
            available=False,
            reason="the chains' mean log posterior is identical — nothing to split.",
        )

    # The identity guard, over the raw draws rather than the chain means: a term
    # camdl added since would otherwise be absorbed silently into "prior".
    resid = 0.0
    if init is not None:
        for cid, buf in rs.chains.items():
            cols = [
                buf.aux.get(c)
                for c in ("log_complete_data_ll", "obs_ll", "transition_ll", "initial_state_ll")
            ]
            if any(a is None for a in cols) or buf.iters.size == 0:
                continue
            n = min([buf.iters.size] + [a.size for a in cols])  # type: ignore[union-attr]
            keep = buf.iters[:n] >= cutoff
            c0, c1, c2, c3 = (a[:n][keep] for a in cols)  # type: ignore[index]
            r = np.abs(c0 - (c1 + c2 + c3))
            r = r[np.isfinite(r)]
            if r.size:
                resid = max(resid, float(r.max()))

    terms = [
        TermSpread(
            key=k,
            label=TERM_LABELS[k],
            note=TERM_NOTES[k],
            share=shares[k],
            range_nats=float(v.max() - v.min()),
            sd_nats=float(np.std(v, ddof=1)),
            chain_means=[float(x) for x in v],
        )
        for k, v in parts.items()
    ]
    # Biggest explainer first — the reason the panel exists is to name it.
    terms.sort(key=lambda t: t.share, reverse=True)
    return TargetDecomposition(
        available=True,
        chain_ids=chain_ids,
        terms=terms,
        total_range=float(total.max() - total.min()),
        total_sd=float(np.std(total, ddof=1)),
        residual_max=resid,
        initial_state_derived=derived,
    )


def decompose_streams(rs: RunState, cutoff: int) -> list[StreamSpread]:
    """Split the observation term's chain disagreement by stream.

    The drill-down from a large ``obs_ll`` share: which stream do the chains
    actually disagree about? Empty unless camdl wrote per-stream ``obs_ll_*``
    columns (PGAS, gh#742).
    """
    prefix = "obs_ll_"
    names = sorted(
        {c[len(prefix) :] for buf in rs.chains.values() for c in buf.aux if c.startswith(prefix)}
    )
    if not names:
        return []
    obs = _chain_means(rs, "obs_ll", cutoff)
    if obs is None:
        return []
    per: dict[str, dict[int, float]] = {}
    for s in names:
        m = _chain_means(rs, prefix + s, cutoff)
        if m is not None:
            per[s] = m
    if not per:
        return []
    chain_ids = sorted(set(obs).intersection(*(set(m) for m in per.values())))
    if len(chain_ids) < 2:
        return []
    total = np.array([obs[c] for c in chain_ids], dtype=float)
    parts = {
        s: np.array([m[c] for c in chain_ids], dtype=float) for s, m in per.items()
    }
    if not np.isfinite(total).all() or not all(
        np.isfinite(v).all() for v in parts.values()
    ):
        return []
    shares = _shares(parts, total)
    if shares is None:
        return []
    out = [
        StreamSpread(
            stream=s,
            share=shares[s],
            range_nats=float(v.max() - v.min()),
            chain_means=[float(x) for x in v],
        )
        for s, v in parts.items()
    ]
    out.sort(key=lambda s: s.share, reverse=True)
    return out
