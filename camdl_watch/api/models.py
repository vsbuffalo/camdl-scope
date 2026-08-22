"""Pydantic wire models — the typed JSON contract the browser depends on.

These re-project the Python core's ADTs (``state.py`` / ``docs.py`` /
``schema.py``) onto the HTTP boundary. FastAPI turns them into the OpenAPI
schema, which ``openapi-typescript`` turns into ``web/src/api/types.ts`` — one
source of truth, typed on both ends. Nothing here computes anything: the routes
fill these in from numbers the core already produced (R̂, ESS, quantiles), so a
field carries a value, never a recipe for one.

Floats on the wire are always finite. R̂/ESS that the core could not estimate
arrive as ``None`` (not ``NaN``) because Starlette serializes with
``allow_nan=False``; the routes are responsible for that conversion.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ParamPosterior(BaseModel):
    """One estimated coordinate, doc-labelled and summarized — the unit a forest
    plot draws: a point (``q50``) with an interval (``q05``…``q95``), a human
    label (``symbol`` / ``description`` / ``reference``), and its prior."""

    name: str
    symbol: str | None = None
    description: str | None = None
    reference: str | None = None
    source: str  # prior provenance: fit_toml | model_ir | default
    prior: str | None = None  # human-formatted prior, e.g. "LogNormal(μ=-0.6, σ=0.4)"
    bounds: tuple[float, float] | None = None
    mean: float
    sd: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    rhat: float | None = None
    ess: float | None = None
    # True for a pooled objective (log_posterior / log_likelihood) appended after
    # the estimated params — a fit summary, not an estimand, with no prior/docs.
    is_objective: bool = False


class PosteriorResponse(BaseModel):
    """Doc-labelled posterior summary for one run — what the first frontend
    screen (the forest plot) consumes. ``params`` is in the model's estimated
    order; empty when the run has no draws yet."""

    run_id: str
    warmup_pct: int
    warmup_cutoff: int
    n_tail: int
    params: list[ParamPosterior]


class PriorCurve(BaseModel):
    """A smooth analytic prior density: ``y`` evaluated at grid ``x`` (same length)."""

    x: list[float]
    y: list[float]


class DrawsResponse(BaseModel):
    """Row-aligned post-warmup posterior draws — the substrate for proper
    statistical graphics (marginal densities, the pair/corner plot). Row ``i`` is
    one joint sample: ``draws[param][i]`` for every param, drawn by chain
    ``chain[i]``. Pooled across chains and thinned to at most ``max_draws`` rows;
    every value is finite (non-finite rows are dropped to keep alignment)."""

    run_id: str
    warmup_pct: int
    warmup_cutoff: int
    n_draws: int
    params: list[str]
    # Objective columns (log_posterior / log_likelihood) present in the trace,
    # included row-aligned in `draws` so they can be paired against parameters
    # (Stan's lp__). Listed separately from `params` — they're diagnostics, not
    # estimated coordinates, and carry no prior.
    objectives: list[str] = []
    chain: list[int]
    draws: dict[str, list[float]]
    # Marginal prior samples per param (NOT row-aligned; truncated to bounds, may
    # be shorter/empty). Retained for compatibility; the diagonals now overlay
    # ``prior_density`` instead.
    prior: dict[str, list[float]] = {}
    # Smooth ANALYTIC prior density per param: ``{param: {x: [...], y: [...]}}``
    # over the param's posterior window — a clean curve for the pair-plot
    # diagonals (a binned histogram of clipped samples reads as noise).
    prior_density: dict[str, PriorCurve] = {}


class StreamInfo(BaseModel):
    """One observation stream's structure (from the fit's ``schema``)."""

    name: str
    index_dims: list[str]
    value_kind: str | None = None
    likelihood: str | None = None


class DimensionInfo(BaseModel):
    """One indexing dimension and its ordered levels (from the fit's ``schema``)."""

    name: str
    levels: list[str]


class FindingGroup(BaseModel):
    """One ``kind`` of camdl diagnostic finding, collapsed to a single line."""

    kind: str
    severity: str
    headline: str
    params: list[str]


class ProgressInfo(BaseModel):
    """camdl's per-run progress heartbeat (``progress.json``). ``phase`` /
    ``step`` / ``total`` are present only while running; ``reason`` only on
    failure; ``pct`` is the derived completion fraction (0–100) when step/total
    are known. ``updated_at`` is unix seconds — its freshness is the liveness
    signal."""

    state: str
    phase: str | None = None
    step: int | None = None
    total: int | None = None
    pct: int | None = None
    reason: str | None = None
    updated_at: float | None = None


class RunSummary(BaseModel):
    """A run as it appears in the selector list — enough to identify, label, and
    badge it without fetching its draws."""

    run_id: str
    label: str
    model: str
    algorithm: str
    backend: str
    status: str
    # How the fit summarizes, and thus which display it gets: "posterior" (chains
    # → the forest/traces/diagnostics view) or "mle" (a point estimate → the
    # estimate/restarts view). Drives the kind-specific tab set in the UI.
    fit_kind: str = "posterior"
    n_chains: int
    # The run's actual chain ids, ascending (camdl numbers them from 1, not 0).
    # The authority for the chain selector so its labels/colours and its
    # include-list match the real chains — never synthesize 0..n-1.
    chain_ids: list[int]
    n_params: int
    has_docs: bool
    # camdl's live progress heartbeat, when present (burn-in/sweep step, or a
    # failure reason) — drives the live progress blurb in the Explore header.
    progress: ProgressInfo | None = None
    # Whether the run has a prequential.json (a pfilter score artifact) — the
    # gate for inclusion in the Compare workspace's model comparison.
    has_prequential: bool = False
    max_iter: int | None = None
    # Target total sweeps/iterations (from fit.toml). With ``max_iter`` this gives
    # a completion fraction for runs that emit no ``progress.json`` heartbeat —
    # the fallback behind the run bar's progress bar.
    target_sweeps: int | None = None
    updated_at: float


class ParamFamily(BaseModel):
    """An indexed parameter family — a base name expanded per stratum, e.g.
    ``k_raw`` → ``[k_raw_Bo, k_raw_Bombali, …]``. The UI toggles these as a group."""

    base: str
    members: list[str]


class ParamGroups(BaseModel):
    """Estimated coordinates partitioned for selection UIs: ungrouped ``scalars``
    plus indexed ``families`` (≥2 members). ``default_selection`` is the
    recommended visible set (scalars + hyperparameters; family leaves hidden) so
    a 20-parameter hierarchical fit doesn't open as a wall of panels."""

    scalars: list[str]
    families: list[ParamFamily]
    default_selection: list[str]


class QuantityInfo(BaseModel):
    """A generated quantity's identity + shape, from the manifest — enough to
    decide its rendering (``series`` → ribbon, ``scalar`` → table row) without
    reading its TSV. ``censorable`` flags a scalar whose reduction can fail to
    fire (a time-to-event), whose band is conditional on firing. ``unit`` is
    reserved upstream but currently always null."""

    name: str
    shape: str  # "series" | "scalar"
    source: str  # "state" | "observations" | "derived"
    index_dims: list[str]
    reduce: str | None = None
    unit: str | None = None
    censorable: bool = False
    # `#'` docs joined from the fit's docs.quantities, when the model carries them.
    symbol: str | None = None
    description: str | None = None
    reference: str | None = None


class Calendar(BaseModel):
    """The fit's time-axis calendar. A numeric ``time`` value maps to the date
    ``origin + time × days_per_unit`` days, so the viewer can render real dates
    instead of raw day-indices on every time axis (predictive ribbons, quantity
    trajectories). ``None`` on a fit that declares no calendar (relative time —
    the axis stays numeric)."""

    origin: str  # ISO date the time axis counts from, e.g. "1910-01-01"
    time_unit: str = "days"
    days_per_unit: float = 1.0


class RunDetail(BaseModel):
    """A run's metadata, schema, and verdict — everything but the draws."""

    run_id: str
    label: str
    model: str
    algorithm: str
    backend: str
    status: str
    fit_kind: str = "posterior"  # "posterior" | "mle" — see RunSummary.fit_kind
    n_chains: int
    max_iter: int | None = None
    target_sweeps: int | None = None
    estimated: list[str]
    groups: ParamGroups
    streams: list[StreamInfo]
    dimensions: list[DimensionInfo]
    findings: list[FindingGroup]
    available_streams: list[str]
    # Generated quantities the fit's predict produced (manifest-driven, deduped to
    # logical quantities); empty when `camdl fit predict` was never run, or the
    # model has no quantities block.
    available_quantities: list[QuantityInfo] = []
    # The scenario set the predict overlaid (e.g. baseline / no_sia / strong_sia);
    # empty for a scenario-less (older) predict.
    quantity_scenarios: list[str] = []
    # The fit-level time calendar (origin epoch + unit) from the artifact
    # sidecar, so time axes render as dates. None for a relative-time fit.
    calendar: Calendar | None = None
    # Whether the run carries a ``model.render.json`` (structured model math) —
    # gates the Model tab's equations view. False for runs predating it.
    has_model_render: bool = False
    # Whether the run carries a ``model.graph.json`` (compartmental flow graph) —
    # gates the Model tab's diagram view. False for runs predating it.
    has_model_graph: bool = False
    # The sampler's static configuration exactly as camdl recorded it — e.g.
    # ``{"particles": 1200, "sweeps": 40000}`` for PGAS, ``{"iterations": …}``
    # for MH. The key set is the algorithm's, not ours: a new knob reaches the
    # UI without a schema change. Empty for runs predating the field.
    algorithm_config: dict[str, str | float | int] = {}


# --- Source tab --------------------------------------------------------------


class SourceFile(BaseModel):
    """One source artifact: syntax-highlighted ``html`` to render and raw
    ``text`` for the copy button. ``present`` is false when the file couldn't be
    read (e.g. the model moved since the fit). ``origin`` records where the bytes
    came from: ``leaf`` = archived in the self-contained fit run leaf,
    ``live`` = read live from the recorded checkout path."""

    path: str | None = None
    present: bool
    origin: Literal["leaf", "live"] | None = None
    html: str = ""
    text: str = ""


class SourceResponse(BaseModel):
    """The fit's sources: the ``.camdl`` model (preferably the copy archived in
    the run leaf; older fits fall back to the recorded checkout path) and the
    ``fit.toml`` (always archived in the run leaf). ``highlight_css`` is the
    Pygments token stylesheet to inject once."""

    run_id: str
    model: SourceFile
    model_identity: str | None = None
    fit_toml: SourceFile
    highlight_css: str


# --- Rendered model (model.render.json) --------------------------------------
# Every ``*_tex``/``symbol``/``rate`` string is a standalone KaTeX-renderable
# expression — the viewer renders leaves client-side and owns the layout.


class RenderDimension(BaseModel):
    """One indexing dimension of the model and its ordered levels."""

    name: str
    levels: list[str] = []


class RenderParameter(BaseModel):
    """A model parameter for the glossary: ``symbol`` is KaTeX (e.g. ``\\beta``);
    ``description`` is prose."""

    name: str
    symbol: str
    description: str | None = None


class RenderDefinition(BaseModel):
    """A named auxiliary definition, ``tex`` a full KaTeX expression
    (e.g. ``N = S + I + R``)."""

    name: str
    tex: str


class RenderTransition(BaseModel):
    """One reaction, kept split so the viewer chooses table vs inline-arrow:
    ``reactants``/``products`` are KaTeX state expressions, ``rate`` the KaTeX
    rate law."""

    name: str
    reactants: str
    products: str
    rate: str


class RenderDynamic(BaseModel):
    """One state's ODE, ``tex`` the full KaTeX expression
    (e.g. ``\\dot{S} = -\\frac{\\beta S I}{N}``)."""

    state: str
    tex: str


class ModelRender(BaseModel):
    """Structured model math for display (``model.render.json``). Every math
    string is KaTeX-safe; the viewer renders leaves and lays them out. Present
    for any run (fit or sim). Optional sections default to empty so the contract
    stays forward-compatible."""

    model: str
    mode: str
    states: list[str] = []
    dimensions: list[RenderDimension] = []
    parameters: list[RenderParameter] = []
    definitions: list[RenderDefinition] = []
    transitions: list[RenderTransition] = []
    dynamics: list[RenderDynamic] = []


# --- Model flow graph (model.graph.json) -------------------------------------
# A compartmental node-link diagram: base compartments + stratifying plates +
# transition edges (KaTeX rate) + mean-field couplings. Model-pure, so it is
# byte-identical across runs of the same model. The viewer lays it out; nothing
# here is derived server-side.


class GraphNode(BaseModel):
    """One base compartment. ``label`` is a KaTeX string (e.g. ``S_{naive}``);
    ``id`` is the plain identifier edges reference."""

    id: str
    label: str


class GraphPlate(BaseModel):
    """One stratifying dimension the compartments are replicated over (e.g.
    ``age`` with its ordered ``levels``). Drawn as an enclosing annotation, never
    one node per level — a stratified model can be thousands of cells."""

    name: str
    levels: list[str] = []


class GraphEdge(BaseModel):
    """One transition between compartments. ``rate`` is a KaTeX string.
    ``source``/``target`` are compartment ids; ``None`` marks an exogenous flow
    (``source=None`` → inflow/birth, ``target=None`` → outflow/death), and the
    literal ``"c"`` is the compartment iterator (applies to every node — a
    plate-family edge such as aging/death). ``advances`` names the plate an edge
    steps along (e.g. ``age``), else ``None``. ``reads_pool`` flags a rate that
    reads a mean-field aggregate (its couplings carry which pools)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    # ``from`` is a Python keyword; expose the JSON key via alias, field is
    # ``source``. ``to`` is not reserved but paired with ``target`` for symmetry.
    source: str | None = Field(default=None, alias="from")
    target: str | None = Field(default=None, alias="to")
    rate: str = ""
    advances: str | None = None
    reads_pool: bool = False


class GraphCoupling(BaseModel):
    """A mean-field coupling: ``edge``'s rate reads the ``aggregate`` pool (its
    name, e.g. ``inf_vil``) summed ``over`` the listed plates."""

    edge: str
    aggregate: str
    over: list[str] = []


class ModelGraph(BaseModel):
    """The compartmental flow graph (``model.graph.json``) for the Model tab's
    diagram. Optional sections default to empty so the contract stays
    forward-compatible."""

    model_config = ConfigDict(populate_by_name=True)

    model: str
    nodes: list[GraphNode] = []
    plates: list[GraphPlate] = []
    edges: list[GraphEdge] = []
    couplings: list[GraphCoupling] = []


# --- Predictive tab ----------------------------------------------------------


class PredictivePoint(BaseModel):
    """One posterior-predictive ribbon point: quantiles at a time × stratum, for
    a given ``scenario`` × forecast ``horizon`` × ``treatment``. ``scenario`` is
    ``as_fitted`` for a scenario-less predict (and for the in-sample one_step
    rows, which are scenario-independent)."""

    time: float
    stratum: dict[str, str] = {}
    scenario: str = "as_fitted"
    horizon: str = ""
    treatment: str = ""
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float


class ObservedPoint(BaseModel):
    """One observed value to overlay (``value`` is null where the series has a hole)."""

    time: float
    stratum: dict[str, str] = {}
    value: float | None = None


class PredictiveResponse(BaseModel):
    """One stream's posterior-predictive ribbons + observed series. The frontend
    facets by ``stratum`` and filters by ``horizon`` (e.g. free_forward). Time is
    dated via the fit-level ``RunDetail.calendar``.

    ``rhat_max`` / ``ess_min`` / ``n_draws`` surface the artifact's convergence
    channel (worst case across rows — normally constant): the Gelman–Rubin
    summary of the stage that produced the draws the predictive replays.
    ``None`` when the stage reported no summary (upstream ``NotAssessed``,
    e.g. a single-chain stage). Upstream reports the numbers only — the
    converged/marginal/unconverged judgment is the consumer's."""

    run_id: str
    stream: str
    index_dims: list[str]
    scenarios: list[str]
    horizons: list[str]
    treatments: list[str]
    rhat_max: float | None = None
    ess_min: float | None = None
    n_draws: int | None = None
    predictive: list[PredictivePoint]
    observed: list[ObservedPoint]


# --- Quantities tab ----------------------------------------------------------


class QuantityBandPoint(BaseModel):
    """One banded snapshot of a series quantity at a scenario × time × stratum.
    ``scenario`` is ``as_fitted`` for an old (scenario-less) sidecar."""

    scenario: str = "as_fitted"
    time: float
    stratum: dict[str, str] = {}
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float


class QuantitySeriesResponse(BaseModel):
    """A series quantity's banded trajectory — the ribbon payload. Faceted by
    ``stratum`` and overlaid by ``scenario`` on the frontend."""

    run_id: str
    name: str
    index_dims: list[str]
    scenarios: list[str]
    points: list[QuantityBandPoint]


class QuantityScalarRow(BaseModel):
    """One banded scalar quantity (one row per scenario × stratum cell). A
    censorable scalar carries ``p_censored`` (fraction of draws where the event
    never fired); a fully-censored cell has ``q* = None`` (no band, only the
    count). ``scenario`` is ``as_fitted`` for an old (scenario-less) sidecar."""

    name: str
    scenario: str = "as_fitted"
    reduce: str | None = None
    source: str
    stratum: dict[str, str] = {}
    n_draws: int
    p_censored: float | None = None
    q05: float | None = None
    q25: float | None = None
    q50: float | None = None
    q75: float | None = None
    q95: float | None = None


class QuantityScalarsResponse(BaseModel):
    """Every scalar quantity, one row per scenario × stratum cell — the
    quantities table. ``scenarios`` is the distinct scenario set (``[]`` when the
    fit has no scenario axis)."""

    run_id: str
    scenarios: list[str]
    rows: list[QuantityScalarRow]


# --- Sims (forward simulations) ----------------------------------------------


class SimSummary(BaseModel):
    """A discoverable forward-simulation run (``sims/`` tree) for the run list.
    ``n_members`` is the sweep size (the overlay members)."""

    sim_id: str
    model: str
    n_members: int
    status: str
    updated_at: float


class SimMemberSeries(BaseModel):
    """One sweep member's trajectory of a compartment total: aligned
    ``time`` + ``value`` (summed over the compartment's strata), thinned."""

    member: str
    scenario: str
    time: list[float]
    value: list[float]


class SimBandPoint(BaseModel):
    """A quantile snapshot across the sweep members at one time — the ensemble
    ribbon for a large sweep."""

    time: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float


class SimSeriesResponse(BaseModel):
    """A sim's compartment trajectory across its sweep members — the overlay
    payload. ``states`` lists the selectable compartments; ``state`` is the one
    returned. ``mode`` is ``members`` for a small sweep (each thinned trajectory
    in ``members``) or ``band`` for a large one (a quantile-across-members ribbon
    in ``band`` + a handful of sample ``members`` to toggle on). ``n_members`` is
    the full sweep size the band summarises. ``calendar`` dates the axis when
    known (else the axis is numeric model-time in days)."""

    sim_id: str
    model: str
    state: str
    states: list[str]
    mode: str = "members"
    n_members: int
    members: list[SimMemberSeries] = []
    band: list[SimBandPoint] = []
    calendar: Calendar | None = None
    # The sim's FULL time domain (independent of the requested window) — the
    # bounds for the window/zoom control. Series above are re-thinned within the
    # window so a zoomed view keeps full resolution.
    t_min: float = 0.0
    t_max: float = 0.0


# --- Traces tab --------------------------------------------------------------


class TraceSeries(BaseModel):
    """One chain's thinned trace for one parameter: aligned ``iters`` + ``values``."""

    chain: int
    iters: list[int]
    values: list[float]


class ParamTrace(BaseModel):
    """One parameter's per-chain traces (estimated coordinate, or an objective
    like ``log_posterior``)."""

    param: str
    series: list[TraceSeries]


class TracesResponse(BaseModel):
    """Per-parameter, per-chain iteration traces (thinned) for the trace grid —
    the raw mixing view. ``warmup_cutoff`` marks the retained-tail boundary."""

    run_id: str
    warmup_cutoff: int
    params: list[str]
    traces: list[ParamTrace]


# --- Compare workspace -------------------------------------------------------


class CompareRow(BaseModel):
    """One model's prequential scores in a comparison, projected from ``camdl
    compare --format json``. Δ fields are ``None`` for the baseline row and when
    the models are not commensurable (``T_score`` mismatch). ``elpd`` is the
    summed out-of-sample log predictive density (higher = better); ``delta_elpd``
    is paired against the baseline with ``se_delta_elpd``; ``e_t = exp(Δelpd)`` is
    the terminal e-value / Bayes factor; ``evidence_label`` is the Jeffreys tier
    of ``delta_elpd_db`` (decibans)."""

    run_id: str
    label: str
    t_score: int
    elpd: float
    delta_elpd: float | None = None
    delta_elpd_db: float | None = None
    evidence_label: str | None = None
    e_t: float | None = None
    se_delta_elpd: float | None = None
    mean_crps: float | None = None
    delta_mean_crps: float | None = None
    pit_cov90: float | None = None
    is_baseline: bool = False
    # |Δelpd| > 2·se(Δ) — camdl's "the gap is real" rule of thumb.
    gap_is_real: bool = False
    # PIT 90%-coverage < 0.70 — the overconfidence flag.
    overconfident: bool = False


class CompareResponse(BaseModel):
    """A prequential model comparison. ``commensurable`` is false when the models
    were scored on different horizons (``T_score`` mismatch) — Δ columns are then
    meaningless and arrive ``None``. ``notes`` carries camdl's advisories (e.g.
    the in-sample / plug-in optimism caveat). ``missing_prequential`` lists
    requested runs that had no score artifact and were dropped. Rows are in
    camdl's order: ascending Δelpd, best-supported last."""

    baseline: str
    metrics: list[str]
    commensurable: bool
    notes: list[str] = []
    rows: list[CompareRow]
    missing_prequential: list[str] = []


# --- Diagnostics tab ---------------------------------------------------------


class ParamDiagnostic(BaseModel):
    """One parameter's convergence/precision diagnostics. R̂ and combined ESS are
    camdl-authoritative when a stage summary exists, else the live arviz estimate;
    tail-ESS / MCSE / sep are the live estimate. ``ess_per_chain`` is camdl's
    per-chain breakdown (empty when unavailable). None where not estimable."""

    name: str
    symbol: str | None = None
    rhat: float | None = None
    ess_bulk: float | None = None
    ess_tail: float | None = None
    mcse: float | None = None
    mean: float
    sd: float
    sep: float | None = None
    ess_per_chain: list[float] = []


class ChainMixing(BaseModel):
    """Per-chain mixing metric — MH/PMMH acceptance rate or PGAS trajectory
    renewal — with an optional healthy band ``(lo, hi)``.

    ``chains`` carries the chain id of each value, parallel to ``values``.
    camdl's chains are 1-based (``chain_1 …``) and a mixing series may cover a
    subset (chains still warming up contribute nothing), so a consumer must
    label from this list rather than from the array position."""

    label: str
    values: list[float]
    chains: list[int] = []
    band: tuple[float, float] | None = None


class PriorPosteriorRow(BaseModel):
    """One parameter's prior→posterior comparison — see
    :class:`camdl_watch.diagnostics.PriorPosterior` for what each number means
    and why a null is not a zero."""

    param: str
    symbol: str | None = None
    prior_label: str | None = None
    prior_mean: float | None = None
    prior_sd: float | None = None
    post_mean: float | None = None
    post_sd: float | None = None
    contraction: float | None = None
    z: float | None = None
    bound_pressure: float | None = None


class PriorPosteriorResponse(BaseModel):
    """The prior→posterior table for a run, over the retained draws.
    ``warmup_cutoff`` is the iteration the tail starts at, so the numbers can be
    read against the same warm-up lens the other tabs use."""

    run_id: str
    warmup_pct: int
    warmup_cutoff: int
    n_tail: int
    rows: list[PriorPosteriorRow]


class PanelColumn(BaseModel):
    """One column of a sampler panel: the metric, what it means in one line,
    and its healthy band when the metric has a conventional one."""

    key: str
    label: str
    note: str | None = None
    band: tuple[float, float] | None = None


class SamplerPanel(BaseModel):
    """A chain × metric table of method-specific sampler diagnostics.

    Deliberately one shape for every sampler: PGAS's per-parameter block
    acceptance and the per-chain telemetry (divergences, step size, tree depth,
    trajectory renewal …) are both "a value per chain per column, some outside
    a healthy band". A new sampler adds rows and columns, never a new response
    type or a new component. ``rows`` are chain ids (1-based, as camdl names
    them); ``values[r][c]`` is null where a chain has no value for a column."""

    id: str
    title: str
    note: str | None = None
    rows: list[int]
    columns: list[PanelColumn]
    values: list[list[float | None]]


class DiagnosticsResponse(BaseModel):
    """The full convergence picture for a run: camdl's verdict (findings), a
    per-parameter R̂/ESS table, per-chain mixing, and the PMMH MAP if present.
    ``source`` is ``camdl`` when an authoritative stage summary backs R̂/ESS, else
    ``live`` (the watcher's arviz estimate while a run is still sampling)."""

    run_id: str
    warmup_pct: int
    warmup_cutoff: int
    n_tail: int
    n_chains: int  # chains actually diagnosed (excludes any still warming up)
    n_chains_warming: int = 0  # chains dropped for lacking post-warm-up draws
    # Chains that produced NO draws at all. camdl skips a chain whose initial
    # complete-data log-posterior is non-finite (``bad_init``) and still
    # completes the run, so these are dead, not slow — a distinction the UI has
    # to make loudly, since half a fit's chains can be missing while the run
    # reports "done".
    n_chains_dead: int = 0
    dead_chain_ids: list[int] = []
    # Chain ids for the positional per-chain columns (``ParamDiagnostic
    # .ess_per_chain``), 1-based as camdl names them. Label from these, never
    # from the array index.
    chain_ids: list[int] = []
    # Method-specific diagnostics (per-parameter block acceptance, sampler
    # telemetry). Empty for a sampler that exposes none.
    sampler_panels: list[SamplerPanel] = []
    stage: str | None = None
    source: str
    logpost_label: str
    findings: list[FindingGroup]
    params: list[ParamDiagnostic]
    mixing: ChainMixing | None = None
    map_loglik: float | None = None
    map_chain: int | None = None
    # Run-level, thinning-invariant efficiency, computed the way camdl's
    # ``fit summary`` reports it — from the authoritative stage summary, over the
    # slowest (min-ESS) parameter and all chains, independent of the viewer's
    # warm-up / chain selection. ``None`` while a fit is still live (no summary
    # yet) or when the primitive is unrecorded (older runs).
    ess_per_iter: float | None = None  # min_ess / (n_samples × thin) — algorithm quality
    ess_per_sec: float | None = None  # min_ess / wall_time_secs — runtime (hardware-dependent)


# --- Profile tab -------------------------------------------------------------


class MleParam(BaseModel):
    """One coordinate of an MLE fit: the point estimate plus its spread across the
    converged restarts (``restart_lo``/``restart_hi`` null when only one restart —
    or none — converged). Doc labels (``symbol`` etc.) join in from the model."""

    name: str
    symbol: str | None = None
    description: str | None = None
    reference: str | None = None
    bounds: tuple[float, float] | None = None
    value: float | None
    restart_lo: float | None = None
    restart_hi: float | None = None


class MleRestart(BaseModel):
    """One multi-start restart's optimum: its log-likelihood (a huge-negative
    sentinel if it failed), the optimizer exit status, and evals spent."""

    chain: int
    loglik: float
    status: str
    n_evals: int


class MleResponse(BaseModel):
    """An MLE ('scout') fit: the point estimate θ̂ (``params``, estimated order),
    the optimum ``loglik``, and the multi-start ``restarts`` — the MLE analogue of
    convergence diagnostics (how many restarts found the mode vs failed)."""

    run_id: str
    label: str
    algorithm: str
    backend: str
    loglik: float | None = None
    n_restarts: int
    n_converged: int
    params: list[MleParam]
    restarts: list[MleRestart]


class ProfileSummary(BaseModel):
    """One profile in the selector list: the inference problem and the
    parameter(s) it profiles (1 → a curve, 2 → a likelihood surface), plus its
    MLE cell — enough to identify and label without the full grid."""

    base_id: str
    label: str
    params: list[str]  # profiled param names, e.g. ["g"] or ["g", "Cscale"]
    method: str
    n_points: int
    mle_coords: list[float]  # MLE coordinate, one value per profiled param


class ProfilePoint(BaseModel):
    """One grid cell of a profile: the profiled ``coords`` (one value per param —
    length 1 for a curve, 2 for a surface) and the best optimized log-likelihood
    there (max over restarts). ``nuisance`` is the conditional MLE of the *other*
    params at this cell (optimized with the profiled coords held fixed); empty
    when the run wrote no per-cell parameter file."""

    coords: list[float]
    loglik: float
    n_starts: int
    nuisance: dict[str, float] = {}


class ProfileResponse(BaseModel):
    """A profile likelihood over one param (a curve) or two (a surface).
    ``params`` gives the profiled names and fixes the dimensionality; each
    ``points`` cell carries ``coords`` of that length. The MLE is the argmax
    cell. ``ci_drop`` is the loglik drop for the 95% confidence set (½·χ²_df:
    1.92 for a CI, 3.00 for a 2D region). ``ci_lo``/``ci_hi`` are the 1D CI
    bracket (interpolated at the crossings, ``None`` on an unbracketed side, and
    always ``None`` for a 2D surface — a region, not an interval)."""

    base_id: str
    label: str
    params: list[str]
    method: str
    loglik_type: str
    points: list[ProfilePoint]
    mle_coords: list[float]
    mle_loglik: float
    ci_level: float
    ci_drop: float
    ci_lo: float | None = None
    ci_hi: float | None = None
