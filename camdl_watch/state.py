"""Core data types for camdl-watch.

The middle layer. Everything downstream (diagnostics, plots, UI) is a
projection of these. Ingest builds them; diagnostics consumes them; the
UI renders them.

Design notes
------------
* A ``RunState`` is one fit (one ``<run>-<hash>/`` dir, one posterior
  stage, one seed). Its chains are keyed by integer chain id.
* A ``ChainBuffer`` holds the *raw per-sweep* trace as numpy arrays, plus
  the byte offset of the last fully-read line so the next tail can resume.
  We keep auxiliary columns (``log_likelihood``, ``log_posterior``, and
  PGAS-only ``trajectory_renewal``/``transition_ll``/``obs_ll``) separate
  from the estimated-parameter columns.
* Sweep/step is normalized onto ``draws`` (the iteration index column).
  MH ``step`` can start at a nonzero offset (e.g. 10000) — we preserve the
  raw values; warm-up cutoffs are expressed in the same units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .docs import ModelDocs
from .schema import ObsSchema


# Auxiliary (non-parameter) columns we recognize and pull out of the trace.
# ``log_complete_data_ll`` is PGAS's joint log p(x,y|θ) along the resampled path
# (newer trace schema); like the others it's diagnostics, not an estimated
# coordinate, so it must never leak into ``values`` as a fake parameter.
AUX_COLUMNS: tuple[str, ...] = (
    "log_likelihood",
    "log_complete_data_ll",
    "log_posterior",
    "trajectory_renewal",
    "transition_ll",
    "obs_ll",
    "accepted",
)
"""Fallback list of non-parameter trace columns.

Only a fallback: camdl declares a ``role`` for every trace column in the stage's
``run.json`` (``output_schema``), and :func:`camdl_watch.ingest.read_column_roles`
prefers that. A hardcoded list silently misfiles whatever a sampler adds — the
PGAS parameter-block columns (``n_divergent``, ``step_size``, ``tree_depth``,
``n_leapfrog``, ``accept_stat``, ``energy``) all landed in the *parameter*
bucket, so divergences were read off disk and never surfaced. Use this only for
runs whose ``run.json`` predates the schema.
"""

# How to reduce a diagnostic column over a chain's post-warm-up draws, and what
# it means. A COUNTER sums (divergences are events, and a mean over draws hides
# them); everything else averages. `band` is the healthy range where one is
# conventional. Columns absent here still render — mean, no note — so a new
# camdl diagnostic appears in the UI the day it is written rather than waiting
# for this table to be updated.
DIAGNOSTIC_META: dict[str, tuple[str, str, str, tuple[float, float] | None]] = {
    # column: (label, reduce, note, band)
    "n_divergent": (
        "divergences",
        "sum",
        "Divergent transitions: the integrator failed where the posterior's "
        "geometry is too sharp. Any nonzero count biases the region it "
        "clusters in — reparameterise rather than sample longer.",
        None,
    ),
    "accept_stat": (
        "accept",
        "mean",
        "Average Metropolis acceptance of the parameter block. Far below "
        "target wastes proposals; far above means steps too small to explore.",
        (0.6, 0.95),
    ),
    "step_size": (
        "step size",
        "mean",
        "Adapted leapfrog step. Chains that settle on very different step "
        "sizes are exploring different geometry.",
        None,
    ),
    "tree_depth": (
        "tree depth",
        "mean",
        "NUTS doubling depth. Saturating at the maximum means trajectories "
        "are being truncated before the U-turn.",
        None,
    ),
    "n_leapfrog": (
        "leapfrog",
        "mean",
        "Gradient evaluations per iteration — the cost side of ESS/second.",
        None,
    ),
    "energy": (
        "energy",
        "mean",
        "Hamiltonian energy; its marginal vs transition spread (E-BFMI) "
        "detects a heavy-tailed posterior the sampler cannot traverse.",
        None,
    ),
    "trajectory_renewal": (
        "renewal",
        "mean",
        "Fraction of the conditional trajectory replaced per PGAS sweep — the "
        "particle filter's mixing. Near zero means the ancestor path is stuck "
        "and the latent state is barely updating.",
        None,
    ),
    "accepted": (
        "accept",
        "mean",
        "Metropolis acceptance rate.",
        (0.15, 0.50),
    ),
}

# Log-density columns are trajectory quantities shown in the Traces tab; they
# are diagnostics but not *sampler-mechanism* diagnostics, so the sampler panel
# leaves them out rather than repeating them as per-chain means.
SAMPLER_PANEL_EXCLUDE: frozenset[str] = frozenset(
    {
        "log_likelihood",
        "log_complete_data_ll",
        "log_posterior",
        "transition_ll",
        "obs_ll",
    }
)

# The iteration-index column, after normalization.
ITER_COL = "draw"


class Status(str, Enum):
    RUNNING = "running"      # heartbeat fresh, sampling
    WARMING = "warming"      # heartbeat fresh, burn-in (no draws yet)
    DONE = "done"            # clean terminal
    FAILED = "failed"        # clean terminal failure (carries a reason)
    STALLED = "stalled"      # heartbeat went stale -> presumed dead / hung


@dataclass(frozen=True)
class RunProgress:
    """Parsed ``progress.json`` — camdl's per-run heartbeat (gh#278).

    ``state`` is the RunState tag (``running`` | ``done`` | ``failed``);
    ``phase`` (``burn_in`` | ``sampling`` | ``optimizing`` | ``profiling``) and
    ``step``/``total`` are present only while running; ``reason`` only on
    failure. ``updated_at`` is unix seconds — its freshness is the liveness
    signal (cross-host, PID-reuse-proof)."""

    state: str
    updated_at: int | None = None
    pid: int | None = None
    phase: str | None = None
    step: int | None = None
    total: int | None = None
    reason: str | None = None


class Backend(str, Enum):
    """Backend determines whether the trace log-posterior is complete-data
    (latent + obs, large-magnitude) or marginal (integrated)."""

    CHAIN_BINOMIAL = "chain_binomial"
    ODE = "ode"
    UNKNOWN = "unknown"

    @property
    def logpost_label(self) -> str:
        if self is Backend.ODE:
            return "marginal"
        if self is Backend.CHAIN_BINOMIAL:
            return "complete-data"
        return "log-posterior"


class PriorFamily(str, Enum):
    NORMAL = "Normal"
    LOGNORMAL = "LogNormal"
    HALFNORMAL = "HalfNormal"
    BETA = "Beta"
    GAMMA = "Gamma"
    UNIFORM = "Uniform"
    FLAT = "Flat"  # bounds-only / improper -> rendered as a uniform band on bounds


@dataclass(frozen=True)
class PriorSpec:
    """A resolved prior for one estimated coordinate.

    ``args`` carries the family-specific parameters
    (e.g. ``{"mu": 7.6, "sigma": 0.6}`` for LogNormal, ``{"alpha": 4.0,
    "beta": 2.0}`` for Beta). ``bounds`` is the optional ``(lo, hi)`` box
    constraint, used both to render FLAT priors as a band and to truncate
    sampled prior draws to the feasible region.
    """

    param: str
    family: PriorFamily
    args: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"  # fit_toml | model_ir | default
    bounds: tuple[float, float] | None = None


@dataclass
class ChainBuffer:
    """One chain's streamed trace, accumulated incrementally."""

    cid: int
    path: Path
    byte_offset: int = 0
    iters: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    values: dict[str, np.ndarray] = field(default_factory=dict)  # param -> array
    aux: dict[str, np.ndarray] = field(default_factory=dict)  # aux col -> array
    header: list[str] | None = None  # raw column names (cached after first read)
    # camdl's declared diagnostic columns for this run (from run.json). Empty
    # means "not declared" — the reader then falls back to AUX_COLUMNS. Carried
    # per buffer so the streaming appender can classify without the run meta.
    diagnostic_cols: frozenset[str] = frozenset()

    def is_diagnostic(self, col: str) -> bool:
        return (
            col in self.diagnostic_cols
            if self.diagnostic_cols
            else col in AUX_COLUMNS
        )

    @property
    def n(self) -> int:
        return int(self.iters.shape[0])

    def max_iter(self) -> int | None:
        return int(self.iters[-1]) if self.n else None


@dataclass
class RunMeta:
    """Lightweight discovery record — enough to identify and label a run
    without parsing the whole trace."""

    run_id: str  # dir name, e.g. natbc_dens_hierk_nc_pgas_long-754f3fe8
    run_dir: Path  # the <run>-<hash> directory
    posterior_dir: Path  # the stage we read: a non-empty NN-posterior-*/seed_*/
    chain_paths: dict[int, Path]  # cid -> trace.tsv
    model: str  # model file stem
    algorithm: str  # pgas | mh | pmmh | ...
    backend: Backend
    estimated: list[str]
    target_sweeps: int | None  # sweeps (pgas) or iterations (mh), from fit.toml
    declared_burn_in: int | None  # burn_in from fit.toml
    thin: int = 1  # thinning factor from fit.toml — denominator for a live ESS/iter
    fit_toml_stem: str = ""  # config stem, e.g. natbc_dens_hierk_nc_pgas_long
    user_label: str | None = None  # camdl-native user-display label, if set
    # The stage whose heartbeat is running *now*, when the run has one. A re-run
    # under new sampler settings adds a stage beside the finished one, and until
    # that stage writes rows we still read the finished stage's draws — so the
    # stage holding the posterior and the stage doing the sampling can differ.
    # None when they coincide (the ordinary single-stage case).
    live_stage_dir: Path | None = None
    # How the fit summarizes: "posterior" (chains/draws — the default path) or
    # "mle" (a point estimate from an optimization 'scout' stage; no draws).
    fit_kind: str = "posterior"
    # Per-fit sidecar metadata (fit.meta.json). ``docs`` is the model's ``#'``
    # documentation dictionary (empty when undocumented); ``schema`` is the
    # observation/dimension schema (None when the sidecar carried no model).
    docs: ModelDocs = field(default_factory=ModelDocs)
    schema: ObsSchema | None = None
    # Trace column -> camdl's declared role ("iteration" | "diagnostic" |
    # "param_estimated" | …), read from the stage's run.json output_schema.
    # Empty for runs that predate the schema, which then fall back to
    # AUX_COLUMNS. This is what lets a sampler add a diagnostic column and have
    # it classified correctly without a watcher release.
    column_roles: dict[str, str] = field(default_factory=dict)
    # The sampler's static configuration as camdl recorded it (run.json
    # ``inputs.algorithm``): particle count, sweep budget, tolerance. Keys are
    # algorithm-specific and deliberately not enumerated. Empty for runs whose
    # leaf predates the field.
    algorithm_config: dict[str, str | float | int] = field(default_factory=dict)

    def is_diagnostic_col(self, col: str) -> bool:
        """Whether ``col`` is a non-parameter diagnostic, per camdl's declared
        role when available, else the legacy hardcoded list."""
        role = self.column_roles.get(col)
        if role is not None:
            return role == "diagnostic"
        return col in AUX_COLUMNS

    @property
    def hash(self) -> str:
        """The content-hash suffix of the run dir name (after the last '-')."""
        return self.run_id.rsplit("-", 1)[-1] if "-" in self.run_id else ""

    @property
    def status_dir(self) -> Path:
        """The stage that answers "is this run doing anything *right now*" — the
        live stage when a re-run is in flight beside a finished one, else the
        stage we read. Status is a fact about the run, not about the stage whose
        draws happen to be the best ones to show."""
        return self.live_stage_dir or self.posterior_dir

    @property
    def derived_label(self) -> str:
        """A readable label derived from metadata, independent of camdl's
        native label. Form: ``<config-stem> · <algorithm>/<backend>``."""
        stem = self.fit_toml_stem or self.run_id.rsplit("-", 1)[0]
        return f"{stem} · {self.algorithm}/{self.backend.value}"

    @property
    def display_label(self) -> str:
        """The label to show in the UI: the camdl-native user label if set,
        else the derived one."""
        return self.user_label or self.derived_label


@dataclass
class RunState:
    """Full live state for one run."""

    meta: RunMeta
    chains: dict[int, ChainBuffer] = field(default_factory=dict)
    priors: dict[str, PriorSpec] = field(default_factory=dict)
    status: Status = Status.RUNNING
    updated_at: float = 0.0  # max chain-file mtime seen
    last_growth_at: float = 0.0  # wall-clock when we last saw new rows
    progress: RunProgress | None = None  # latest progress.json, if any
    summary: "ChainSummary | None" = None  # camdl's authoritative end-of-stage diagnostics

    @property
    def params(self) -> list[str]:
        return list(self.meta.estimated)

    def max_iter(self) -> int | None:
        ms = [c.max_iter() for c in self.chains.values() if c.max_iter() is not None]
        return max(ms) if ms else None

    def min_iter(self) -> int | None:
        ms = [int(c.iters[0]) for c in self.chains.values() if c.n]
        return min(ms) if ms else None


@dataclass(frozen=True)
class ParamDiag:
    rhat: float
    bulk_ess: float
    tail_ess: float
    mcse: float
    mean: float
    sd: float


@dataclass
class Diagnostics:
    per_param: dict[str, ParamDiag]
    acceptance: float | None  # mean of `accepted` over post-warmup tail, if present
    n_divergent: int | None  # if a divergence column/log is available, else None
    plateaued: bool | None  # ll plateau test result (None if too few samples)
    plateau_slope: float | None  # robust slope of ll over trailing window
    chain_separation: dict[str, float]  # param -> between/within spread ratio
    warmup_cutoff: int
    n_tail: int  # post-warmup draws per chain (min across chains)
    logpost_label: str = "log-posterior"
    # PGAS has no MH accept/reject; its mixing analog is the trajectory-renewal
    # rate (fraction of the reference path renewed per sweep). None if absent.
    renewal: float | None = None


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Warning_:
    severity: Severity
    message: str
    param: str | None = None


# ---------------------------------------------------------------------------
# camdl's authoritative end-of-stage telemetry (pgas_summary.json /
# pmmh_summary.json + diagnostics.json). Present only once a stage finishes;
# while a run is live the watcher's own arviz Diagnostics stand in.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One typed diagnostic finding from camdl's ``diagnostics.json``.

    ``kind`` is camdl's tag (``rhat_high`` | ``acceptance_rate_unhealthy`` |
    ``max_tree_depth_hits`` | …); ``detail`` is the rest of the ``kind`` payload
    (``rhat``, ``rate``, ``threshold``, ``pct``, ``max_depth``, …) so views can
    aggregate without re-parsing the message string."""

    kind: str
    severity: Severity
    message: str
    param: str | None = None
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FindingGroup:
    """Findings of one ``kind`` collapsed to a single human line — the raw file
    repeats e.g. ``acceptance_rate_unhealthy`` per param×chain, so the verdict
    strip shows the aggregate, not the firehose."""

    kind: str
    severity: Severity
    headline: str
    params: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChainSummary:
    """camdl's authoritative per-stage diagnostics.

    R̂ and combined ``ess`` are per-param (ESS may be ``None`` when not
    estimable). ``ess_per_chain`` maps param -> a list in CHAIN ORDER: the
    index is an array position, NOT a chain id — camdl names chains
    ``chain_1 … chain_n`` on disk, so position ``i`` is chain ``i + 1``. Label
    via :func:`camdl_watch.diagnostics.chain_ids_for`; treating the position as
    an id produced a ``c0`` that names no chain. ``acceptance_rates`` is normalized to
    ``[chain][*]``: PGAS keeps its per-parameter block-MH rates (constant within
    a chain); PMMH's per-chain scalar is wrapped as a singleton — reduce via
    :attr:`per_chain_acceptance`. ``map_*`` are PMMH-only (a concrete MAP point).
    """

    stage: str
    n_chains: int
    rhat: dict[str, float]
    ess: dict[str, float | None]
    ess_per_chain: dict[str, list[float]]
    acceptance_rates: list[list[float]] | None = None
    map_params: dict[str, float] | None = None
    map_loglik: float | None = None
    map_chain: int | None = None
    findings: list[Finding] = field(default_factory=list)
    # Thinning factor + stage wall-clock — the two primitives (written by every
    # sampler's ``*_summary.json`` / ``run.json``) behind the thinning-invariant
    # efficiency metrics: ESS/iteration = min_ess / (n_samples × thin),
    # ESS/second = min_ess / wall_time_secs. ``thin`` defaults to 1 (unthinned)
    # on runs predating the field; ``wall_time_secs`` is None when unrecorded.
    thin: int = 1
    wall_time_secs: float | None = None

    @property
    def per_chain_acceptance(self) -> list[float] | None:
        """One acceptance rate per chain (mean over the stored block, which is
        constant for PGAS and a singleton for PMMH)."""
        if not self.acceptance_rates:
            return None
        return [float(np.mean(row)) for row in self.acceptance_rates if len(row)]
