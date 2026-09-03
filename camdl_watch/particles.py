"""Particle-filter health, per sampler.

The two samplers fail in different ways and expose different evidence, so this
reports different things for each rather than a lowest-common-denominator table.

**PGAS** runs a conditional SMC sweep and keeps one conditioned path. Its
failure is a path that stops being replaced: the chain still moves in θ while
the latent trajectory is frozen, which looks like sampling and is not. The
evidence is the renewal rate, and — more usefully — *where* along the
trajectory renewal happens. camdl writes ``renewal_b0…b9``, renewal over each
tenth of the substeps; averaged over post-burn-in sweeps that profile is the
update-rate-against-time diagnostic Lindsten, Jordan and Schön (2014, JMLR
15:2145–2184, fig. 1) and Chopin and Singh (2015, Bernoulli 21:1855–1883)
recommend in place of any rule for choosing a particle count.

The ancestor-sampling counters say *why* a profile is flat, and the distinction
matters because the two causes have opposite fixes. Quoting camdl's own writer
(gh#718): ``as_opportunity`` "is how many substeps in the sweep drew an
ancestry, which is the only place an ancestor move is legal… a low renewal with
``as_opportunity`` near zero is a schedule with nowhere to renew, not a sampler
that will not." So: low renewal with opportunities available means the filter is
degenerate and wants more particles; low renewal with no opportunities means the
observation schedule leaves nowhere to move and more particles will not help.

**PMMH** stores no path, so none of the above exists. Its characteristic
failure is *sticking*: an unluckily-high log-likelihood estimate is accepted and
then nothing can beat it, so the chain freezes for a long run of rejections.
That is visible in the ``accepted`` column as the longest rejection run, which
is reported here alongside the plain acceptance rate — a healthy-looking 20%
acceptance can still hide a single stall covering a third of the chain.

Pure functions over a :class:`RunState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import RunState

#: camdl writes renewal in ten equal bins over the substeps (gh#688).
RENEWAL_BINS = 10
RENEWAL_BIN_COLUMNS = tuple(f"renewal_b{i}" for i in range(RENEWAL_BINS))


@dataclass(frozen=True)
class HealthStat:
    """One scalar, with the reading that makes it actionable."""

    key: str
    label: str
    value: float | None
    #: "" | "fraction" | "per sweep" | "sweeps". A ``fraction`` is a proportion
    #: in [0, 1]; the value stays native and the reader renders it as a percent,
    #: which is how a rate is read out loud.
    unit: str
    note: str
    #: Conventional healthy range, when one exists. Absent where no threshold
    #: is defensible — an invented band is worse than none.
    band: tuple[float, float] | None = None
    severity: str = "none"  # none | ok | warn


@dataclass(frozen=True)
class ParticleHealth:
    """What this fit's sampler exposes about its particle filter."""

    method: str  # "pgas" | "pmmh" | "" (unknown)
    available: bool
    reason: str | None = None
    particles: int | None = None
    stats: list[HealthStat] = field(default_factory=list)
    #: Mean renewal in each tenth of the trajectory, post-warm-up, pooled over
    #: chains. Empty for PMMH and for PGAS runs predating gh#688.
    renewal_profile: list[float] = field(default_factory=list)
    #: Per-chain overall renewal, so one frozen chain is visible among healthy
    #: ones rather than averaged away.
    renewal_by_chain: dict[int, float] = field(default_factory=dict)
    #: Longest run of consecutive rejections, per chain (PMMH).
    longest_stall_by_chain: dict[int, int] = field(default_factory=dict)


def _pool(rs: RunState, col: str, cutoff: int) -> np.ndarray:
    """Post-warm-up values of ``col`` pooled over chains, finite only."""
    out: list[np.ndarray] = []
    for buf in rs.chains.values():
        arr = buf.aux.get(col)
        if arr is None or buf.iters.size == 0:
            continue
        n = min(buf.iters.size, arr.size)
        v = arr[:n][buf.iters[:n] >= cutoff]
        out.append(v[np.isfinite(v)])
    return np.concatenate(out) if out else np.empty(0)


def _per_chain_mean(rs: RunState, col: str, cutoff: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for cid, buf in sorted(rs.chains.items()):
        arr = buf.aux.get(col)
        if arr is None or buf.iters.size == 0:
            continue
        n = min(buf.iters.size, arr.size)
        v = arr[:n][buf.iters[:n] >= cutoff]
        v = v[np.isfinite(v)]
        if v.size:
            out[cid] = float(v.mean())
    return out


def _longest_false_run(flags: np.ndarray) -> int:
    """Longest run of zeros — for ``accepted``, the longest stall."""
    best = cur = 0
    for x in flags:
        if x:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def _mean_or_none(a: np.ndarray) -> float | None:
    return float(a.mean()) if a.size else None


def _pgas(rs: RunState, cutoff: int, particles: int | None) -> ParticleHealth:
    renewal = _pool(rs, "trajectory_renewal", cutoff)
    as_accept = _pool(rs, "as_accept", cutoff)
    as_opp = _pool(rs, "as_opportunity", cutoff)
    as_prop = _pool(rs, "as_proposed", cutoff)

    stats: list[HealthStat] = []
    r = _mean_or_none(renewal)
    if r is not None:
        stats.append(
            HealthStat(
                key="trajectory_renewal",
                label="renewal",
                value=r,
                unit="fraction",
                note=(
                    "Fraction of the conditioned path replaced per sweep. Near "
                    "zero means the latent trajectory is frozen while θ keeps "
                    "moving — the chain looks like it is sampling and is not."
                ),
                # No conventional threshold exists; the cited work recommends
                # reading the PROFILE rather than testing a scalar, so this
                # carries no band on purpose.
                band=None,
                severity="warn" if r < 0.01 else "ok",
            )
        )
    a = _mean_or_none(as_accept)
    if a is not None:
        stats.append(
            HealthStat(
                key="as_accept",
                label="ancestor accept",
                value=a,
                unit="fraction",
                note=(
                    "Of the ancestor moves proposed, the fraction the Metropolis "
                    "step accepted. Low with opportunities available is a "
                    "degenerate filter — the candidate ancestors are all worse "
                    "than the conditioned path."
                ),
                severity="warn" if a < 0.01 else "ok",
            )
        )
    o = _mean_or_none(as_opp)
    if o is not None:
        stats.append(
            HealthStat(
                key="as_opportunity",
                label="ancestry draws",
                value=o,
                unit="per sweep",
                note=(
                    "Substeps per sweep that drew an ancestry — the only places "
                    "an ancestor move is legal, and the denominator behind the "
                    "proposals. Not the observation count. Near zero means the "
                    "schedule leaves nowhere to renew, so more particles will "
                    "not help."
                ),
                severity="warn" if o < 1 else "ok",
            )
        )
    p = _mean_or_none(as_prop)
    if p is not None:
        stats.append(
            HealthStat(
                key="as_proposed",
                label="ancestor moves proposed",
                value=p,
                unit="per sweep",
                note="Ancestor moves offered per sweep, out of the ancestry draws.",
            )
        )
    if particles is not None:
        stats.append(
            HealthStat(
                key="particles",
                label="particles",
                value=float(particles),
                unit="",
                note=(
                    "Particle count per sweep, from the fit config. Raising it "
                    "helps a degenerate filter; it does not help a schedule "
                    "with no ancestry draws."
                ),
            )
        )

    profile: list[float] = []
    bins = [_pool(rs, c, cutoff) for c in RENEWAL_BIN_COLUMNS]
    if all(b.size for b in bins):
        profile = [float(b.mean()) for b in bins]

    return ParticleHealth(
        method="pgas",
        available=bool(stats or profile),
        reason=None if (stats or profile) else "this PGAS stage wrote no filter diagnostics.",
        particles=particles,
        stats=stats,
        renewal_profile=profile,
        renewal_by_chain=_per_chain_mean(rs, "trajectory_renewal", cutoff),
    )


def _pmmh(rs: RunState, cutoff: int, particles: int | None) -> ParticleHealth:
    acc = _pool(rs, "accepted", cutoff)
    stats: list[HealthStat] = []
    stalls: dict[int, int] = {}
    for cid, buf in sorted(rs.chains.items()):
        arr = buf.aux.get("accepted")
        if arr is None or buf.iters.size == 0:
            continue
        n = min(buf.iters.size, arr.size)
        v = arr[:n][buf.iters[:n] >= cutoff]
        v = v[np.isfinite(v)]
        if v.size:
            stalls[cid] = _longest_false_run(v > 0)

    a = _mean_or_none(acc)
    if a is not None:
        stats.append(
            HealthStat(
                key="accepted",
                label="acceptance",
                value=a,
                unit="fraction",
                note=(
                    "Metropolis acceptance over retained draws. For PMMH this "
                    "is necessary but not sufficient: a healthy average can "
                    "hide a single long stall."
                ),
                band=(0.10, 0.40),
                severity="warn" if a < 0.05 else "ok",
            )
        )
    if stalls:
        worst = max(stalls.values())
        n_tail = max(
            (int((buf.iters >= cutoff).sum()) for buf in rs.chains.values()), default=0
        )
        stats.append(
            HealthStat(
                key="longest_stall",
                label="longest stall",
                value=float(worst),
                unit="sweeps",
                note=(
                    "Longest run of consecutive rejections in any chain. PMMH "
                    "sticks when an over-estimated log-likelihood is accepted "
                    "and nothing can beat it; a stall covering a large share of "
                    "the chain means the estimator variance is too high, which "
                    "is fixed with more particles, not more iterations."
                ),
                severity=(
                    "warn" if n_tail and worst > 0.10 * n_tail else "ok"
                ),
            )
        )
    if particles is not None:
        stats.append(
            HealthStat(
                key="particles",
                label="particles",
                value=float(particles),
                unit="",
                note=(
                    "Particle count per likelihood evaluation, from the fit "
                    "config. The lever on estimator variance, and so on sticking."
                ),
            )
        )

    return ParticleHealth(
        method="pmmh",
        available=bool(stats),
        reason=None if stats else "this PMMH stage wrote no acceptance column.",
        particles=particles,
        stats=stats,
        longest_stall_by_chain=stalls,
    )


def particle_health(rs: RunState, cutoff: int) -> ParticleHealth:
    """Filter health for whichever sampler ran, or a stated reason there is none."""
    meta = rs.meta
    algo = (getattr(meta, "algorithm", "") or "").lower()
    cfg = getattr(meta, "algorithm_config", {}) or {}
    raw = cfg.get("particles")
    particles = int(raw) if isinstance(raw, (int, float)) else None

    if algo == "pgas":
        return _pgas(rs, cutoff, particles)
    if algo in ("pmmh", "mh"):
        return _pmmh(rs, cutoff, particles)
    return ParticleHealth(
        method=algo,
        available=False,
        reason=(
            f"'{algo or 'this'}' is not a particle method — there is no filter "
            f"to report on."
        ),
        particles=particles,
    )
