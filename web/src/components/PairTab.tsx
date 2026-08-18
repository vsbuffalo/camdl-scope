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
  // Objectives (log_posterior, obs_ll, …) are selected separately below.
  const [selection, setSelection] = useState<Set<string>>(() => new Set())
  // Which objective columns join the grid as extra rows/columns. Separate from
  // the param selection — they're targets to pair *against*, not estimands —
  // with their own always-visible checkbox row. Defaults to log_posterior.
  const [targets, setTargets] = useState<Set<string>>(() => new Set())
  const initedRun = useRef<string | null>(null)
  useEffect(() => {
    if (!groups || !draws.data || initedRun.current === runId) return
    setSelection(new Set(groups.default_selection))
    const objs = draws.data.objectives ?? []
    setTargets(new Set(objs.includes('log_posterior') ? ['log_posterior'] : []))
    initedRun.current = runId
  }, [groups, draws.data, runId])

  const objectives = draws.data?.objectives ?? []

  // Render the selected variables: estimated params first, then the checked
  // objective targets, in wire order. The pair plot handles objective columns
  // as ordinary variables.
  const visibleParams = useMemo(() => {
    if (!draws.data) return []
    return [
      ...draws.data.params.filter((p) => selection.has(p)),
      ...draws.data.objectives.filter((o) => targets.has(o)),
    ]
  }, [draws.data, selection, targets])

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
          selection={selection}
          onChange={setSelection}
        />
      )}

      {/* Objective targets (log_posterior, obs_ll, …) join the grid as extra
          rows — their own control, not buried in the ⚙ params panel. On a PGAS
          fit log_posterior is the COMPLETE-DATA log posterior (path-dominated);
          obs_ll is the data-fit term to pair params against. */}
      {objectives.length > 0 && (
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-neutral-100 px-3 py-2">
          <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
            Objectives
          </span>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            {objectives.map((name) => {
              const on = targets.has(name)
              return (
                <label
                  key={name}
                  className="flex cursor-pointer items-center gap-1.5 font-mono text-xs"
                >
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={() =>
                      setTargets((prev) => {
                        const next = new Set(prev)
                        if (next.has(name)) next.delete(name)
                        else next.add(name)
                        return next
                      })
                    }
                    className="size-3 accent-neutral-800"
                  />
                  <span className={on ? 'text-neutral-900' : 'text-neutral-500'}>
                    {name}
                  </span>
                </label>
              )
            })}
          </div>
        </div>
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
