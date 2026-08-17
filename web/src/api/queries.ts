import { useQuery } from '@tanstack/react-query'
import {
  getCompare,
  getDiagnostics,
  getDraws,
  getPosterior,
  getPredictive,
  getProfile,
  getProfiles,
  getQuantityScalars,
  getQuantitySeries,
  getModelRender,
  getModelGraph,
  getRun,
  getRuns,
  getMle,
  getSims,
  getSimSeries,
  getSource,
  getTraces,
} from './client'

/** Query-key factory — keeps cache keys consistent across the app. */
export const qk = {
  runs: ['runs'] as const,
  run: (id: string) => ['run', id] as const,
  posterior: (id: string, warmupPct: number, chains: string) =>
    ['posterior', id, warmupPct, chains] as const,
  draws: (id: string, warmupPct: number, maxDraws: number, chains: string) =>
    ['draws', id, warmupPct, maxDraws, chains] as const,
  source: (id: string) => ['source', id] as const,
  modelRender: (id: string) => ['model-render', id] as const,
  modelGraph: (id: string) => ['model-graph', id] as const,
  sims: ['sims'] as const,
  simSeries: (id: string, state: string, win: string) =>
    ['sim-series', id, state, win] as const,
  predictive: (id: string, stream: string) =>
    ['predictive', id, stream] as const,
  traces: (id: string, warmupPct: number, chains: string) =>
    ['traces', id, warmupPct, chains] as const,
  diagnostics: (id: string, warmupPct: number, chains: string) =>
    ['diagnostics', id, warmupPct, chains] as const,
}

/** Stable cache-key fragment for a chain include-list (`null` → all chains). */
function chainKey(chains: number[] | null | undefined): string {
  return chains && chains.length ? chains.join(',') : 'all'
}

/** List of runs for the selector. Refetches occasionally so new fits appear. */
export function useRuns() {
  return useQuery({
    queryKey: qk.runs,
    queryFn: getRuns,
    refetchInterval: 30_000,
  })
}

/** One run's detail (schema, findings). */
export function useRun(runId: string | undefined) {
  return useQuery({
    queryKey: qk.run(runId ?? '∅'),
    queryFn: () => getRun(runId as string),
    enabled: Boolean(runId),
  })
}

/** The doc-labelled posterior summary — overlays, labels, and numbers.
 *  `chains` (an include-list, `null` = all) recomputes it on the retained
 *  chains, shared with the Pair / Traces / Diagnostics selection. */
export function usePosterior(
  runId: string | undefined,
  warmupPct: number,
  chains: number[] | null = null,
) {
  return useQuery({
    queryKey: qk.posterior(runId ?? '∅', warmupPct, chainKey(chains)),
    queryFn: () => getPosterior(runId as string, warmupPct, chains),
    enabled: Boolean(runId),
    placeholderData: (prev) => prev,
  })
}

/**
 * Prequential model comparison over a set of runs (sorted key so selection
 * order doesn't thrash the cache). Disabled until ≥2 runs are chosen; a
 * `T_score` refusal still resolves (with `commensurable: false`), so only a
 * genuine backend error surfaces as `isError`.
 */
export function useCompare(
  runIds: string[],
  opts: { baseline?: string; allowMismatchedHorizon?: boolean } = {},
) {
  const key = [...runIds].sort()
  return useQuery({
    queryKey: ['compare', key, opts.baseline ?? null, Boolean(opts.allowMismatchedHorizon)],
    queryFn: () => getCompare(runIds, opts),
    enabled: runIds.length >= 2,
    placeholderData: (prev) => prev,
  })
}

/**
 * Row-aligned posterior draws for the marginal densities and pair plot.
 * `maxDraws` defaults to the Posterior tab's cap; the Pair tab passes a
 * smaller one to keep the scatter panels light.
 */
export function useDraws(
  runId: string | undefined,
  warmupPct: number,
  maxDraws = 1200,
  chains: number[] | null = null,
) {
  return useQuery({
    queryKey: qk.draws(runId ?? '∅', warmupPct, maxDraws, chainKey(chains)),
    queryFn: () => getDraws(runId as string, warmupPct, maxDraws, chains),
    enabled: Boolean(runId),
    placeholderData: (prev) => prev,
  })
}

/** All scalar generated quantities for a run (the quantities table). */
export function useQuantityScalars(runId: string | undefined) {
  return useQuery({
    queryKey: ['quantity-scalars', runId ?? '∅'],
    queryFn: () => getQuantityScalars(runId as string),
    enabled: Boolean(runId),
    placeholderData: (prev) => prev,
  })
}

/** One series generated quantity's banded trajectory (a ribbon). */
export function useQuantitySeries(
  runId: string | undefined,
  name: string | undefined,
) {
  return useQuery({
    queryKey: ['quantity-series', runId ?? '∅', name ?? '∅'],
    queryFn: () => getQuantitySeries(runId as string, name as string),
    enabled: Boolean(runId && name),
    placeholderData: (prev) => prev,
  })
}

/** An MLE fit's point estimate + multi-start restarts (a done fit; no polling). */
export function useMle(runId: string | undefined) {
  return useQuery({
    queryKey: ['mle', runId ?? '∅'],
    queryFn: () => getMle(runId as string),
    enabled: Boolean(runId),
  })
}

/** The fit's model + fit.toml sources, highlighted server-side. */
export function useSource(runId: string | undefined) {
  return useQuery({
    queryKey: qk.source(runId ?? '∅'),
    queryFn: () => getSource(runId as string),
    enabled: Boolean(runId),
  })
}

/** Structured model math for the rendered model view. Enabled only when the run
 * detail reports the artifact is present, so runs without it never 404. */
export function useModelRender(runId: string | undefined, enabled: boolean) {
  return useQuery({
    queryKey: qk.modelRender(runId ?? '∅'),
    queryFn: () => getModelRender(runId as string),
    enabled: Boolean(runId) && enabled,
  })
}

/** The compartmental flow graph for the Model tab's diagram. Enabled only when
 * the run detail reports the artifact is present, so runs without it never 404. */
export function useModelGraph(runId: string | undefined, enabled: boolean) {
  return useQuery({
    queryKey: qk.modelGraph(runId ?? '∅'),
    queryFn: () => getModelGraph(runId as string),
    enabled: Boolean(runId) && enabled,
  })
}

/** Every forward-simulation run (the `sims/` tree) for the Sims workspace. */
export function useSims() {
  return useQuery({ queryKey: qk.sims, queryFn: getSims })
}

/** A sim's compartment trajectory across its sweep members, optionally windowed
 * to `[tFrom, tTo]`. Disabled until a sim is chosen; keeps the previous series
 * while switching compartment/sim/window. */
export function useSimSeries(
  simId: string | undefined,
  state: string | undefined,
  window?: [number, number],
) {
  return useQuery({
    queryKey: qk.simSeries(simId ?? '∅', state ?? '', window ? window.join(':') : ''),
    queryFn: () => getSimSeries(simId as string, state, window),
    enabled: Boolean(simId),
    placeholderData: (prev) => prev,
  })
}

/**
 * One stream's posterior-predictive ribbons + observed series. Disabled until a
 * stream is chosen; 404s surface as `isError` (stream has no predictive yet).
 */
export function usePredictive(
  runId: string | undefined,
  stream: string | undefined,
) {
  return useQuery({
    queryKey: qk.predictive(runId ?? '∅', stream ?? '∅'),
    queryFn: () => getPredictive(runId as string, stream as string),
    enabled: Boolean(runId && stream),
    placeholderData: (prev) => prev,
  })
}

/** Per-parameter, per-chain iteration traces for the trace grid. */
export function useTraces(
  runId: string | undefined,
  warmupPct: number,
  chains: number[] | null = null,
) {
  return useQuery({
    queryKey: qk.traces(runId ?? '∅', warmupPct, chainKey(chains)),
    queryFn: () => getTraces(runId as string, warmupPct, 600, chains),
    enabled: Boolean(runId),
    placeholderData: (prev) => prev,
  })
}

/**
 * Convergence diagnostics at a warm-up cutoff — the verdict (findings), the
 * per-parameter R̂/ESS table, per-chain mixing, and the MAP. Recomputes when the
 * cutoff moves, so it mirrors {@link usePosterior}'s warm-up dependence.
 */
export function useDiagnostics(
  runId: string | undefined,
  warmupPct: number,
  chains: number[] | null = null,
) {
  return useQuery({
    queryKey: qk.diagnostics(runId ?? '∅', warmupPct, chainKey(chains)),
    queryFn: () => getDiagnostics(runId as string, warmupPct, chains),
    enabled: Boolean(runId),
    placeholderData: (prev) => prev,
  })
}

/** List of profile-likelihood runs for the Profile selector. Refetches
 *  occasionally so a profile that finishes appears. */
export function useProfiles() {
  return useQuery({
    queryKey: ['profiles'],
    queryFn: getProfiles,
    refetchInterval: 30_000,
  })
}

/** One profile (1D curve or 2D surface). Polls so a *running* profile's grid
 *  fills in live as camdl lands each new cell — cheap (a glob + JSON read). */
export function useProfile(baseId: string | undefined) {
  return useQuery({
    queryKey: ['profile', baseId ?? '∅'],
    queryFn: () => getProfile(baseId as string),
    enabled: Boolean(baseId),
    placeholderData: (prev) => prev,
    refetchInterval: 8_000,
  })
}
