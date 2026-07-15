import type { components } from './types'

/**
 * Typed wire aliases, derived from the OpenAPI-generated schemas so the
 * frontend and the FastAPI core share one source of truth. Never widen these
 * by hand — regenerate `types.ts` from the schema instead.
 */
export type RunSummary = components['schemas']['RunSummary']
export type RunDetail = components['schemas']['RunDetail']
export type PosteriorResponse = components['schemas']['PosteriorResponse']
export type ParamPosterior = components['schemas']['ParamPosterior']
export type DrawsResponse = components['schemas']['DrawsResponse']
/** Row-aligned posterior draws — the substrate for densities and the pair plot. */
export type Draws = DrawsResponse
export type StreamInfo = components['schemas']['StreamInfo']
export type DimensionInfo = components['schemas']['DimensionInfo']
export type FindingGroup = components['schemas']['FindingGroup']
export type SourceResponse = components['schemas']['SourceResponse']
export type SourceFile = components['schemas']['SourceFile']
export type ModelRender = components['schemas']['ModelRender']
export type PredictiveResponse = components['schemas']['PredictiveResponse']
export type PredictivePoint = components['schemas']['PredictivePoint']
export type ObservedPoint = components['schemas']['ObservedPoint']
export type TracesResponse = components['schemas']['TracesResponse']
export type ParamTrace = components['schemas']['ParamTrace']
export type TraceSeries = components['schemas']['TraceSeries']
export type DiagnosticsResponse = components['schemas']['DiagnosticsResponse']
export type ParamDiagnostic = components['schemas']['ParamDiagnostic']
export type ChainMixing = components['schemas']['ChainMixing']
export type CompareResponse = components['schemas']['CompareResponse']
export type CompareRow = components['schemas']['CompareRow']
export type QuantityInfo = components['schemas']['QuantityInfo']
export type QuantitySeriesResponse = components['schemas']['QuantitySeriesResponse']
export type QuantityBandPoint = components['schemas']['QuantityBandPoint']
export type QuantityScalarsResponse = components['schemas']['QuantityScalarsResponse']
export type QuantityScalarRow = components['schemas']['QuantityScalarRow']
export type MleResponse = components['schemas']['MleResponse']
export type MleParam = components['schemas']['MleParam']
export type MleRestart = components['schemas']['MleRestart']
export type ProfileSummary = components['schemas']['ProfileSummary']
export type ProfileResponse = components['schemas']['ProfileResponse']
export type ProfilePoint = components['schemas']['ProfilePoint']

/** The closed set of run lifecycle states the UI badges. */
export type RunStatus =
  | 'running'
  | 'warming'
  | 'done'
  | 'failed'
  | 'stalled'

async function getJson<T>(path: string): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, { headers: { Accept: 'application/json' } })
  } catch (cause) {
    // Network-level failure (backend down, DNS, CORS): surface a clear message
    // rather than a cryptic TypeError.
    throw new ApiError('backend not reachable', { cause })
  }
  if (!res.ok) {
    throw new ApiError(`request failed (HTTP ${res.status})`, { status: res.status })
  }
  return (await res.json()) as T
}

/**
 * The `&chains=` query suffix for the chain selector: a comma-separated
 * include-list of chain ids. `null`/empty → no filter (server defaults to all),
 * so unfiltered requests keep their clean URLs and cache keys.
 */
function chainsParam(chains: number[] | null | undefined): string {
  return chains && chains.length ? `&chains=${chains.join(',')}` : ''
}

/** A failed `/api` call, carrying an HTTP status when one was received. */
export class ApiError extends Error {
  readonly status?: number
  constructor(message: string, opts?: { status?: number; cause?: unknown }) {
    super(message, opts?.cause ? { cause: opts.cause } : undefined)
    this.name = 'ApiError'
    this.status = opts?.status
  }
}

/** Every discoverable run, newest first. */
export function getRuns(): Promise<RunSummary[]> {
  return getJson<RunSummary[]>('/api/runs')
}

/** One run's metadata, schema, and verdict. */
export function getRun(runId: string): Promise<RunDetail> {
  return getJson<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`)
}

/**
 * Doc-labelled posterior summary — the per-row overlay/label payload. `chains`
 * restricts the pool to an include-list of chain ids (drop stuck chains); the
 * quantiles and R̂/ESS then recompute on the retained chains.
 */
export function getPosterior(
  runId: string,
  warmupPct: number,
  chains: number[] | null = null,
): Promise<PosteriorResponse> {
  const id = encodeURIComponent(runId)
  return getJson<PosteriorResponse>(
    `/api/runs/${id}/posterior?warmup_pct=${warmupPct}${chainsParam(chains)}`,
  )
}

/**
 * Row-aligned post-warmup draws — feeds the marginal densities and the
 * pair/corner plot. `max_draws` is capped server-side; the default suits the
 * Posterior tab, while the Pair tab passes a smaller cap for scatter perf.
 */
export function getDraws(
  runId: string,
  warmupPct: number,
  maxDraws = 1200,
  chains: number[] | null = null,
): Promise<DrawsResponse> {
  const id = encodeURIComponent(runId)
  return getJson<DrawsResponse>(
    `/api/runs/${id}/draws?warmup_pct=${warmupPct}&max_draws=${maxDraws}${chainsParam(chains)}`,
  )
}

/** An MLE ('scout') fit's point estimate + multi-start restarts. */
export function getMle(runId: string): Promise<MleResponse> {
  return getJson<MleResponse>(`/api/runs/${encodeURIComponent(runId)}/mle`)
}

/**
 * The fit's sources: the `.camdl` model (the copy archived in the run leaf when
 * present, else read live from its recorded path) and the leaf-archived
 * `fit.toml`, both Pygments-highlighted server-side. Carries the token
 * stylesheet to inject once.
 */
export function getSource(runId: string): Promise<SourceResponse> {
  const id = encodeURIComponent(runId)
  return getJson<SourceResponse>(`/api/runs/${id}/source`)
}

/** Structured model math (`model.render.json`) for the rendered model view. 404s
 * when the run predates the artifact. */
export function getModelRender(runId: string): Promise<ModelRender> {
  const id = encodeURIComponent(runId)
  return getJson<ModelRender>(`/api/runs/${id}/model-render`)
}

/**
 * One stream's posterior-predictive ribbons + observed series. 404s when the
 * stream has no predictive artifact (`camdl fit predict` hasn't run).
 */
export function getPredictive(
  runId: string,
  stream: string,
): Promise<PredictiveResponse> {
  const id = encodeURIComponent(runId)
  const s = encodeURIComponent(stream)
  return getJson<PredictiveResponse>(`/api/runs/${id}/predictive/${s}`)
}

/**
 * Per-parameter, per-chain iteration traces (thinned) for the trace grid — the
 * raw mixing view. `max_points` is capped server-side per chain.
 */
export function getTraces(
  runId: string,
  warmupPct: number,
  maxPoints = 600,
  chains: number[] | null = null,
): Promise<TracesResponse> {
  const id = encodeURIComponent(runId)
  return getJson<TracesResponse>(
    `/api/runs/${id}/traces?warmup_pct=${warmupPct}&max_points=${maxPoints}${chainsParam(chains)}`,
  )
}

/**
 * Convergence diagnostics for the run at a given warm-up cutoff: camdl's
 * authoritative verdict (findings) and R̂/ESS where a stage summary exists, else
 * the watcher's live arviz estimate; plus per-chain mixing and the PMMH MAP.
 */
export function getDiagnostics(
  runId: string,
  warmupPct: number,
  chains: number[] | null = null,
): Promise<DiagnosticsResponse> {
  const id = encodeURIComponent(runId)
  return getJson<DiagnosticsResponse>(
    `/api/runs/${id}/diagnostics?warmup_pct=${warmupPct}${chainsParam(chains)}`,
  )
}

/** Every scalar generated quantity for a run — the quantities table. */
export function getQuantityScalars(
  runId: string,
): Promise<QuantityScalarsResponse> {
  const id = encodeURIComponent(runId)
  return getJson<QuantityScalarsResponse>(`/api/runs/${id}/quantity-scalars`)
}

/** One series generated quantity's banded trajectory (a ribbon). */
export function getQuantitySeries(
  runId: string,
  name: string,
): Promise<QuantitySeriesResponse> {
  const id = encodeURIComponent(runId)
  const n = encodeURIComponent(name)
  return getJson<QuantitySeriesResponse>(`/api/runs/${id}/quantity-series/${n}`)
}

/**
 * Prequential model comparison (elpd / Δelpd / CRPS / PIT) over ≥2 runs that
 * carry a `prequential.json`, via the authoritative `camdl compare`. Optional
 * `baseline` run id; `allowMismatchedHorizon` acknowledges a `T_score` mismatch.
 */
export function getCompare(
  runIds: string[],
  opts: { baseline?: string; allowMismatchedHorizon?: boolean } = {},
): Promise<CompareResponse> {
  const params = new URLSearchParams()
  for (const id of runIds) params.append('runs', id)
  if (opts.baseline) params.set('baseline', opts.baseline)
  if (opts.allowMismatchedHorizon) params.set('allow_mismatched_horizon', 'true')
  return getJson<CompareResponse>(`/api/compare?${params.toString()}`)
}

/** Every profile-likelihood run under `profiles/` — the Profile selector list. */
export function getProfiles(): Promise<ProfileSummary[]> {
  return getJson<ProfileSummary[]>('/api/profiles')
}

/** One profile-likelihood curve: loglik vs the profiled value, MLE, and 95% CI. */
export function getProfile(baseId: string): Promise<ProfileResponse> {
  return getJson<ProfileResponse>(`/api/profiles/${encodeURIComponent(baseId)}`)
}
