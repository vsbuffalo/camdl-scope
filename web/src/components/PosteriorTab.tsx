import { useState } from 'react'
import { useDraws, usePosterior } from '@/api/queries'
import { ForestRow } from '@/components/ForestRow'
import { ChainSelect } from '@/components/ChainSelect'
import { includedChains, type ChainControls } from '@/lib/chains'
import { WarmupControl } from '@/components/WarmupControl'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const DEFAULT_WARMUP_PCT = 50

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
