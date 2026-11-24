"""Core data types for camdl-scope.

The middle layer. Everything downstream (diagnostics, plots, UI) is a
projection of these. Ingest builds them; diagnostics consumes them; the
UI renders them.

Design notes
------------
* A ``RunState`` is one fit run (one camdl FitRun folder: one method, one
  seed). Its chains are keyed by integer chain id.
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
    # HMC/NUTS parameter-block telemetry. A sampler that writes these is
    # reporting on ITSELF, never on the model, so none of them is ever an
    # estimated coordinate.
    "n_divergent",
    "step_size",
    "tree_depth",
    "n_leapfrog",
    "accept_stat",
    "energy",
)
"""The non-parameter trace columns: what a sampler writes about itself.

camdl's run store records no role for a trace column, so this list is how a
sampler's own columns are told from the model's coordinates.

A hardcoded list silently misfiles whatever a sampler adds, and the cost is
invisible rather than loud: a misfiled diagnostic lands in the *parameter*
bucket, where nothing reads it (``RunState.params`` comes from the fit's
declared estimands, not the trace header), so the column is parsed off disk
every refresh and then dropped. Divergences, step size and tree depth simply
never appear — the run reads as clean because the evidence was filed where no
consumer looks. Anything added here must be a column a sampler writes ABOUT
ITSELF; model coordinates come from the fit metadata.
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
        None,
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
        None,
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
        "initial_state_ll",
    }
)

#: Diagnostic columns that are PROPORTIONS in [0, 1]. Declared here so the
#: percent rendering has one source of truth beside the label and the band,
#: rather than a list of column names duplicated in the frontend. Counts
#: (divergences, leapfrog steps), step sizes and energies are excluded: a
#: percentage of a count is meaningless.
FRACTION_COLUMNS: frozenset[str] = frozenset(
    {"accept_stat", "accepted", "trajectory_renewal"}
)

#: Column-name prefixes excluded from the sampler panel for the same reason as
#: :data:`SAMPLER_PANEL_EXCLUDE`, but open-ended: camdl writes one
#: ``obs_ll_<stream>`` per declared stream, so an eight-stream fit put eight
#: unlabelled columns of data-fit density into a table about the sampler.
SAMPLER_PANEL_EXCLUDE_PREFIXES: tuple[str, ...] = ("obs_ll_",)

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
    """The prior families this reader can draw from and evaluate.

    One member per family camdl can declare, so a model's prior survives the
    trip to a plot. The camdl side is ``Density::kind_str`` in
    ``rust/crates/sim/src/inference/prior.rs``: flat, uniform, normal,
    log_normal, half_normal, beta, gamma, exponential, log_uniform,
    truncated_normal. A family camdl gains and this enum lacks does not fail —
    it degrades to :attr:`FLAT`, which is why the degradation has to be
    recorded (see :attr:`PriorSpec.unsupported`) rather than left silent.
    """

    NORMAL = "Normal"
    LOGNORMAL = "LogNormal"
    HALFNORMAL = "HalfNormal"
    BETA = "Beta"
    GAMMA = "Gamma"
    EXPONENTIAL = "Exponential"
    UNIFORM = "Uniform"
    LOGUNIFORM = "LogUniform"
    TRUNCNORMAL = "TruncatedNormal"
    FLAT = "Flat"  # bounds-only / improper -> rendered as a uniform band on bounds


@dataclass(frozen=True)
class PriorSpec:
    """A resolved prior for one estimated coordinate.

    ``args`` carries the family-specific parameters
    (e.g. ``{"mu": 7.6, "sigma": 0.6}`` for LogNormal, ``{"alpha": 4.0,
    "beta": 2.0}`` for Beta). ``bounds`` is the optional ``(lo, hi)`` box
    constraint, used both to render FLAT priors as a band and to truncate
    sampled prior draws to the feasible region.

    ``unsupported`` names the distribution the model actually declared when
    this reader could not model it and fell back to FLAT on the declared
    bounds. That fallback keeps the box visible but replaces the declared
    density with a uniform, which inflates every prior-relative number
    downstream — contraction, z, the prior curve on the diagonal. The field
    exists so the substitution is stated wherever those numbers are read
    instead of passing as a deliberate flat prior.
    """

    param: str
    family: PriorFamily
    args: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"  # fit_toml | model_ir | default
    bounds: tuple[float, float] | None = None
    unsupported: str | None = None  # declared family this reader cannot model


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

    def is_diagnostic(self, col: str) -> bool:
        return col in AUX_COLUMNS

    @property
    def n(self) -> int:
        return int(self.iters.shape[0])

    def max_iter(self) -> int | None:
        return int(self.iters[-1]) if self.n else None


@dataclass
class RunMeta:
    """Lightweight discovery record — enough to identify and label a run
    without parsing the whole trace. One entry is one camdl FitRun."""

    #: The entry id: the problem folder's name when it holds one run, else
    #: ``<folder>~<seed dir>`` (:func:`camdl_scope.ingest.discover_runs`).
    run_id: str
    #: The FitProblem folder, ``<build>/fit/<model>-<12 hex>/``: the problem's
    #: data, partition and model source, and every run of it under ``runs/``.
    problem_dir: Path
    #: The FitRun folder, ``<problem>/runs/<method>-<12 hex>/seed_<n>-<12 hex>/``.
    leaf_dir: Path
    chain_paths: dict[int, Path]  # cid -> trace.tsv
    model: str  # the model's name, as camdl labelled the problem
    algorithm: str  # pgas | mh | pmmh | ...
    backend: Backend
    estimated: list[str]
    target_sweeps: int | None  # sweeps (pgas) or iterations (mh), from [method]
    declared_burn_in: int | None  # burn_in from [method]
    thin: int = 1  # thinning factor from [method] — denominator for a live ESS/iter
    config_stem: str = ""  # the fit file's stem, e.g. natbc_dens_hierk_nc_pgas_long
    user_label: str | None = None  # the run's current --label, if any
    #: camdl's run id of this FitRun (64 hex, from ``identity.json``). ``None``
    #: only for a hand-built entry.
    leaf_run_id: str | None = None
    # Which run of its problem folder this entry is. ``seed`` is the fit seed as
    # typed; ``folder_default`` marks the run a bare folder name opens.
    seed: int | None = None
    folder_default: bool = True
    # Whether this is the run its method directory opens by default. A method
    # directory is one method's settings for the problem, so its seed runs are
    # replicates of one another. ``folder~method`` opens this run.
    method_default: bool = True
    # How the fit summarizes: "posterior" (chains/draws) or "mle" (a point
    # estimate from an optimizer; no draws).
    fit_kind: str = "posterior"
    # The model's ``#'`` documentation (empty when undocumented) and the
    # observation/dimension schema (None until the run has written fit.meta.json).
    docs: ModelDocs = field(default_factory=ModelDocs)
    schema: ObsSchema | None = None
    # The sampler's static configuration as camdl resolved it
    # (``run_inputs.json`` ``algorithm``): particle count, sweep budget,
    # tolerance. Keys are algorithm-specific and deliberately not enumerated.
    algorithm_config: dict[str, str | float | int] = field(default_factory=dict)
    #: The run's status as its ``run.json`` records it.
    record_status: str = "completed"
    #: Whether the owning process is alive: ``True``/``False`` on this host,
    #: ``None`` for an owner on another host (then the heartbeat decides).
    owner_alive: bool | None = None
    #: The problem's compiled model (``model.ir.json``), when found.
    model_ir: Path | None = None

    @property
    def hash(self) -> str:
        """The 12-hex suffix of the problem folder's name."""
        name = self.problem_dir.name
        return name.rsplit("-", 1)[-1] if "-" in name else ""

    @property
    def method(self) -> str:
        """The method directory holding this run, e.g. ``pmmh-35a17c58a0b1``:
        one method's settings for the problem."""
        return self.leaf_dir.parent.name

    @property
    def derived_label(self) -> str:
        """A readable label derived from metadata, independent of camdl's
        native label. Form: ``<config-stem> · <algorithm>/<backend>``."""
        stem = self.config_stem or self.model or self.problem_dir.name.rsplit("-", 1)[0]
        return f"{stem} · {self.algorithm}/{self.backend.value}"

    @property
    def display_label(self) -> str:
        """The label to show in the UI: the user's label if set, else the
        derived one."""
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
    #: ``(chain_id, camdl's message)`` per finding that names a chain, so a
    #: per-chain finding can be shown in camdl's own words, one line each.
    chain_messages: list[tuple[int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ChainSummary:
    """camdl's authoritative per-stage diagnostics.

    R̂ and combined ``ess`` are per-param (ESS may be ``None`` when not
    estimable). ``ess_per_chain`` maps param -> a list in CHAIN ORDER: the
    index is an array position, NOT a chain id — camdl names chains
    ``chain_0 … chain_{n-1}`` on disk. Label via
    :func:`camdl_scope.diagnostics.chain_ids_for` rather than by position: a
    chain that never started leaves a gap, so position and id can differ. ``acceptance_rates`` is normalized to
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
    # Which statistics ``rhat`` and ``ess`` hold. camdl changed them under the
    # same keys (camdl 4e418b17, 2026-08-22): "rank_bulk" is max(rank-normalized,
    # folded) split-R-hat and bulk-ESS (Vehtari et al. 2021), never withheld for
    # a high R-hat; "classic" is Gelman-Rubin R-hat and summed per-chain ESS,
    # left blank when chains disagreed. Read from the summary's ``schema`` tag,
    # else from the keys only the newer writer emits.
    stats_vintage: str = "rank_bulk"
    # camdl's reason a parameter has no R-hat (``rhat_not_reported``), e.g.
    # ``non_finite_rhat``. Empty on summaries that predate the field.
    rhat_refused: dict[str, str] = field(default_factory=dict)
    # camdl's convergence verdict for each parameter's R-hat (camdl#1199):
    # ``converged`` | ``not_converged`` | ``severe`` | ``not_assessed``, the
    # band behind the mark ``fit summary`` prints. Empty on leaves written
    # before camdl saved it.
    rhat_band: dict[str, str] = field(default_factory=dict)

    @property
    def per_chain_acceptance(self) -> list[float] | None:
        """One acceptance rate per chain (mean over the stored block, which is
        constant for PGAS and a singleton for PMMH)."""
        if not self.acceptance_rates:
            return None
        return [float(np.mean(row)) for row in self.acceptance_rates if len(row)]
