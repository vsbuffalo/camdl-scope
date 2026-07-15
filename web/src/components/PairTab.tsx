import { useEffect, useMemo, useRef, useState } from 'react'
import { useDraws, usePosterior, useRun } from '@/api/queries'
import { PairPlot, type PriorXlimMode } from '@/components/PairPlot'
import { PairSettings } from '@/components/PairSettings'
import { ChainSelect } from '@/components/ChainSelect'
import { includedChains, type ChainControls } from '@/lib/chains'
import { WarmupControl } from '@/components/WarmupControl'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const DEFAULT_WARMUP_PCT = 50
// Cap draws for the pair plot: ~N²/2 scatter panels × this many points, so keep
// it lighter than the Posterior tab to stay responsive.
const PAIR_MAX_DRAWS = 800

export function PairTab({
  runId,
  chainIds,
  excludedChains,
  onToggleChain,
  onResetChains,
}: { runId: string } & ChainControls) {
  const [warmupPct, setWarmupPct] = useState(DEFAULT_WARMUP_PCT)
  const [priorXlimMode, setPriorXlimMode] = useState<PriorXlimMode>('posterior')
  const run = useRun(runId)
  const chains = includedChains({ chainIds, excludedChains, onToggleChain, onResetChains })
  const draws = useDraws(runId, warmupPct, PAIR_MAX_DRAWS, chains)
  // Posterior summaries supply the diagonal median overlays and symbol labels.
  const posterior = usePosterior(runId, warmupPct)

  const groups = run.data?.groups

  // Which params are plotted. Initialized to the run's recommended set (scalars
  // + hyperparams; family leaves hidden) and RESET whenever the run changes —
  // tracked by a ref so a same-run refetch never clobbers the user's edits.
  const [selection, setSelection] = useState<Set<string>>(() => new Set())
  const initedRun = useRef<string | null>(null)
  useEffect(() => {
    if (!groups || !draws.data || initedRun.current === runId) return
    const objs = draws.data.objectives ?? []
    const base = [...groups.default_selection]
    if (objs.includes('log_posterior')) base.push('log_posterior')
    setSelection(new Set(base))
    initedRun.current = runId
  }, [groups, draws.data, runId])

  // Render only the selected variables: estimated params first, then objectives
  // (e.g. log_posterior / log_likelihood), in that order, filtered to the
  // selection. The pair plot handles the objective columns as ordinary variables.
  const visibleParams = useMemo(() => {
    const candidates = draws.data
      ? [...draws.data.params, ...draws.data.objectives]
      : []
    return candidates.filter((p) => selection.has(p))
  }, [draws.data, selection])

  // Any visible param carrying a prior curve → the breadth toggle is meaningful.
  const anyPrior = useMemo(
    () =>
      visibleParams.some(
        (p) => (draws.data?.prior_density?.[p]?.x.length ?? 0) > 1,
      ),
    [visibleParams, draws.data],
  )

  return (
    <Card
      className={cn(
        'overflow-hidden transition-opacity',
        draws.isPlaceholderData && 'opacity-60',
      )}
    >
      <WarmupControl
        value={warmupPct}
        onChange={setWarmupPct}
        cutoff={posterior.data?.warmup_cutoff ?? draws.data?.warmup_cutoff ?? null}
        nTail={posterior.data?.n_tail ?? null}
      />

      {chainIds.length > 1 && (
        <ChainSelect
          chainIds={chainIds}
          excluded={excludedChains}
          onToggle={onToggleChain}
          onReset={onResetChains}
        />
      )}

      {groups && (
        <PairSettings
          groups={groups}
          objectives={draws.data?.objectives ?? []}
          selection={selection}
          onChange={setSelection}
        />
      )}

      {draws.isPending && <ForestSkeleton rows={3} />}

      {draws.isError && (
        <MutedNotice
          bordered={false}
          title="Couldn't load the draws"
          detail="The backend returned an error for this run. It may still be warming up."
        />
      )}

      {draws.data && draws.data.n_draws === 0 && (
        <MutedNotice
          bordered={false}
          title="No posterior draws yet"
          detail="This run hasn't produced post-warmup draws. Check back once it has sampled past the cutoff."
        />
      )}

      {draws.data && draws.data.n_draws > 0 && (
        <div className="p-3">
          {anyPrior && (
            <div className="mb-2 flex items-center gap-2">
              <span className="font-mono text-[10px] uppercase tracking-wide text-neutral-400">
                x-axis
              </span>
              <XlimToggle value={priorXlimMode} onChange={setPriorXlimMode} />
            </div>
          )}
          <PairPlot
            draws={draws.data}
            posterior={posterior.data}
            params={visibleParams}
            priorXlimMode={priorXlimMode}
          />
        </div>
      )}
    </Card>
  )
}

/** Segmented control: fit each axis to the posterior, or widen it to show the
 *  prior's breadth (the posterior then reads as a spike inside the prior). */
function XlimToggle({
  value,
  onChange,
}: {
  value: PriorXlimMode
  onChange: (m: PriorXlimMode) => void
}) {
  const opts: { v: PriorXlimMode; label: string }[] = [
    { v: 'posterior', label: 'fit to posterior' },
    { v: 'prior', label: 'show prior breadth' },
  ]
  return (
    <div className="inline-flex border border-neutral-200">
      {opts.map((o, i) => (
        <button
          key={o.v}
          type="button"
          onClick={() => onChange(o.v)}
          aria-pressed={value === o.v}
          className={cn(
            'px-2 py-0.5 font-mono text-[11px] transition-colors',
            i > 0 && 'border-l border-neutral-200',
            value === o.v
              ? 'bg-neutral-900 text-white'
              : 'text-neutral-500 hover:text-neutral-800',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}
