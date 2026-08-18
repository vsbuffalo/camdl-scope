import { useMemo, useState } from 'react'
import { useDraws, usePosterior, useQuantityScalars, useRun } from '@/api/queries'
import { ForestRow } from '@/components/ForestRow'
import { ChainSelect } from '@/components/ChainSelect'
import { includedChains, type ChainControls } from '@/lib/chains'
import { WarmupControl } from '@/components/WarmupControl'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { ScalarBand } from '@/components/QuantitiesTab'
import { dayToDate } from '@/lib/calendar'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const DEFAULT_WARMUP_PCT = 50

/**
 * Scalar generated quantities (R0, Reff, …) inline on the posterior page.
 * These are posterior distributions of *derived estimands*, so they read
 * naturally as a banded section right below the sampled parameters — but they
 * are not streamed by the sampler: `camdl fit predict` evaluates them post-fit
 * (re-simulating from the posterior), so the section only exists once predict
 * has run. Shows the as-fitted scenario only; scenario overlays, series
 * ribbons, and censoring detail stay in the Quantities tab.
 */
function QuantityScalarSection({ runId }: { runId: string }) {
  const run = useRun(runId)
  const scalarInfo = useMemo(
    () => (run.data?.available_quantities ?? []).filter((q) => q.shape === 'scalar'),
    [run.data],
  )
  const { data } = useQuantityScalars(scalarInfo.length > 0 ? runId : undefined)

  // The as-fitted posterior of each quantity (old scenario-less sidecars are
  // normalized to `as_fitted` on the wire).
  const rows = useMemo(() => {
    const all = data?.rows ?? []
    const scen = all.some((r) => r.scenario === 'as_fitted')
      ? 'as_fitted'
      : (data?.scenarios[0] ?? null)
    return all.filter((r) => scen == null || r.scenario === scen)
  }, [data])

  if (rows.length === 0) return null
  const infoOf = new Map(scalarInfo.map((q) => [q.name, q]))
  const toDate = dayToDate(run.data?.calendar)

  return (
    <>
      <div className="flex items-baseline justify-between border-y border-neutral-200 bg-neutral-50 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-neutral-400">
        <span>generated quantities</span>
        <span>from camdl fit predict</span>
      </div>
      <div className="divide-y divide-neutral-100">
        {rows.map((r) => {
          const info = infoOf.get(r.name)
          const stratum = Object.entries(r.stratum)
            .map(([k, v]) => `${k}=${v}`)
            .join(' · ')
          return (
            <div
              key={`${r.name}-${JSON.stringify(r.stratum)}`}
              className="flex items-baseline justify-between gap-3 px-3 py-1.5"
            >
              <span className="flex min-w-0 items-baseline gap-1.5">
                <span className="shrink-0 font-mono text-[12px] font-semibold text-neutral-900">
                  {info?.symbol ?? r.name}
                </span>
                {stratum && (
                  <span className="shrink-0 font-mono text-[10px] text-neutral-400">
                    {stratum}
                  </span>
                )}
                {info?.description && (
                  <span
                    className="truncate text-[11px] text-neutral-500"
                    title={info.description}
                  >
                    {info.description}
                  </span>
                )}
              </span>
              <span className="shrink-0 font-mono text-[12px] tabular-nums">
                <ScalarBand q50={r.q50} q05={r.q05} q95={r.q95} unit={info?.unit} toDate={toDate} />
              </span>
            </div>
          )
        })}
      </div>
    </>
  )
}

export function PosteriorTab({
  runId,
  chainIds,
  excludedChains,
  onToggleChain,
  onResetChains,
}: { runId: string } & ChainControls) {
  const [warmupPct, setWarmupPct] = useState(DEFAULT_WARMUP_PCT)
  const chains = includedChains({ chainIds, excludedChains, onToggleChain, onResetChains })
  const { data, isPending, isError, isPlaceholderData } = usePosterior(
    runId,
    warmupPct,
    chains,
  )
  // Draws power the marginal densities; they arrive alongside the summaries and
  // the rows render a muted placeholder until they do. Both honor the same
  // dropped-chain selection so the forest and its histograms stay coherent.
  const { data: drawsData } = useDraws(runId, warmupPct, undefined, chains)

  // The posterior table is a dense readout, not a wide canvas — cap its width
  // (the app frame is wide for the pair plot) so the rows stay tight instead of
  // stranding the numerics far right of the histogram.
  return (
    <div className="max-w-4xl">
    <Card
      className={cn(
        'overflow-hidden transition-opacity',
        isPlaceholderData && 'opacity-60',
      )}
    >
      <WarmupControl
        value={warmupPct}
        onChange={setWarmupPct}
        cutoff={data?.warmup_cutoff ?? null}
        nTail={data?.n_tail ?? null}
      />

      {chainIds.length > 1 && (
        <ChainSelect
          chainIds={chainIds}
          excluded={excludedChains}
          onToggle={onToggleChain}
          onReset={onResetChains}
        />
      )}

      {isPending && <ForestSkeleton />}

      {isError && (
        <MutedNotice
          bordered={false}
          title="Couldn't load the posterior"
          detail="The backend returned an error for this run. It may still be warming up."
        />
      )}

      {data && data.params.length === 0 && (
        <MutedNotice
          bordered={false}
          title="No posterior draws yet"
          detail="This run hasn't produced post-warmup draws. Check back once it has sampled past the cutoff."
        />
      )}

      {data && data.params.length > 0 && (
        <>
          <div className="flex items-baseline justify-between border-b border-neutral-200 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-neutral-400">
            <span>parameter · marginal posterior</span>
            <span>median · 90% · R&#x0302; / ESS</span>
          </div>
          <div className="divide-y divide-neutral-100">
            {data.params
              .filter((p) => !p.is_objective)
              .map((param) => (
                <ForestRow
                  key={param.name}
                  param={param}
                  draws={drawsData?.draws[param.name] ?? []}
                />
              ))}
          </div>

          {/* Derived estimands (post-predict) sit between the sampled
              parameters and the fit objectives. */}
          <QuantityScalarSection runId={runId} />

          {/* Pooled objectives (log_posterior / log_likelihood) close the forest
              — a fit summary, set apart from the estimands (Stan's lp__). */}
          {data.params.some((p) => p.is_objective) && (
            <>
              <div className="border-y border-neutral-200 bg-neutral-50 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-neutral-400">
                fit objectives
              </div>
              <div className="divide-y divide-neutral-100">
                {data.params
                  .filter((p) => p.is_objective)
                  .map((param) => (
                    <ForestRow
                      key={param.name}
                      param={param}
                      draws={drawsData?.draws[param.name] ?? []}
                    />
                  ))}
              </div>
            </>
          )}
        </>
      )}
    </Card>
    </div>
  )
}
