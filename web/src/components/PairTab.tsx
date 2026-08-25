import { useEffect, useMemo, useRef, useState } from 'react'
import { useDraws, usePosterior, useRun } from '@/api/queries'
import { PairPlot, type PriorXlimMode } from '@/components/PairPlot'
import { PairSettings } from '@/components/PairSettings'
import { ChainSelect } from '@/components/ChainSelect'
import { includedChains, type ChainControls } from '@/lib/chains'
import { WarmupControl } from '@/components/WarmupControl'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import {
  stillValid,
  toggleInSet,
  usePersisted,
  usePersistedRunSelection,
} from '@/lib/use-persisted'
import { cn } from '@/lib/utils'

// Cap draws for the pair plot: ~N²/2 scatter panels × this many points, so keep
// it lighter than the Posterior tab to stay responsive.
const PAIR_MAX_DRAWS = 800

export function PairTab({
  runId,
  chainIds,
  excludedChains,
  onToggleChain,
  onResetChains,
  warmupPct,
  onWarmupPct,
}: { runId: string } & ChainControls) {
  const [priorXlimMode, setPriorXlimMode] = usePersisted<PriorXlimMode>(
    'pair:xlim-mode',
    'posterior',
  )
  // Splitting each bar by chain makes a stuck chain — one colour owning a
  // region the others never visit — visible without leaving the tab, which a
  // pooled marginal cannot show. Off by default so the corner plot opens on the
  // posterior's shape rather than on a diagnostic; persisted once turned on.
  const [marginalsByChain, setMarginalsByChain] = usePersisted(
    'pair:marginals-by-chain',
    false,
  )
  // Divergences answer a different question from chain mixing, so the view
  // swaps rather than stacks: turning this on drops chain colour entirely and
  // recolours the scatter clean-vs-divergent. Off by default; persisted.
  const [showDivergences, setShowDivergences] = usePersisted(
    'pair:divergences',
    false,
  )
  // Each param is drawn on the scale its MODEL declares: a LogNormal /
  // LogUniform coordinate on log, everything else linear. This is the escape
  // hatch when you want to read them all the same way — it forces linear.
  const [forceLinear, setForceLinear] = usePersisted('pair:force-linear', false)
  const run = useRun(runId)
  const chains = includedChains({ chainIds, excludedChains })
  const draws = useDraws(runId, warmupPct, PAIR_MAX_DRAWS, chains)
  // Posterior summaries supply the diagonal median overlays and symbol labels.
  const posterior = usePosterior(runId, warmupPct)

  const groups = run.data?.groups

  // Which params are plotted, and which objective columns join the grid as
  // extra rows. Both name things that exist only in THIS run, so they are
  // remembered per run and filtered on load against what the run still has —
  // choosing what to look at is work, and it should not be undone by visiting
  // another tab. Absent a stored choice, fall back to the run's recommended
  // set (scalars + hyperparams; family leaves hidden) once it arrives.
  const [storedSelection, setStoredSelection] = usePersistedRunSelection(
    runId,
    'pair:params',
  )
  const [storedTargets, setStoredTargets] = usePersistedRunSelection(
    runId,
    'pair:objectives',
  )
  const [selection, setSelectionState] = useState<Set<string>>(() => new Set())
  const [targets, setTargetsState] = useState<Set<string>>(() => new Set())
  const setSelection = (v: Set<string>) => {
    setSelectionState(v)
    setStoredSelection(v)
  }
  const setTargets = (v: Set<string>) => {
    setTargetsState(v)
    setStoredTargets(v)
  }
  const initedRun = useRef<string | null>(null)
  useEffect(() => {
    if (!groups || !draws.data || initedRun.current === runId) return
    const objs = draws.data.objectives ?? []
    const params = draws.data.params ?? []
    setSelectionState(
      stillValid(storedSelection, params) ?? new Set(groups.default_selection),
    )
    setTargetsState(
      stillValid(storedTargets, objs) ??
        new Set(objs.includes('log_posterior') ? ['log_posterior'] : []),
    )
    initedRun.current = runId
  }, [groups, draws.data, runId, storedSelection, storedTargets])

  const objectives = draws.data?.objectives ?? []
  // `null` means this sampler reports no divergence column at all — distinct
  // from 0, which means it reports one and nothing diverged.
  const nDivergent = draws.data?.n_divergent_draws ?? null
  const shownDivergent = useMemo(
    () => (draws.data?.divergent ?? []).filter(Boolean).length,
    [draws.data],
  )
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

  // Only the derived-log params actually on screen — the control is pointless
  // (and its hint misleading) when none of them is selected.
  const derivedLog = useMemo(
    () => (draws.data?.log_scale ?? []).filter((p) => visibleParams.includes(p)),
    [draws.data, visibleParams],
  )
  const logParams = forceLinear ? [] : derivedLog

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
        onChange={onWarmupPct}
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
                      setTargets(new Set(toggleInSet(targets, name)))
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
          <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-2">
            {anyPrior && (
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-wide text-neutral-400">
                  x-axis
                </span>
                <XlimToggle value={priorXlimMode} onChange={setPriorXlimMode} />
              </div>
            )}
            <label className="flex cursor-pointer items-center gap-1.5 font-mono text-[11px]">
              <input
                type="checkbox"
                checked={marginalsByChain}
                onChange={() => setMarginalsByChain(!marginalsByChain)}
                className="size-3 accent-neutral-800"
              />
              <span
                className={
                  marginalsByChain ? 'text-neutral-900' : 'text-neutral-500'
                }
              >
                marginals by chain
              </span>
            </label>
            {/* Log axes are derived, so this appears only when the derivation
                actually fired on a visible param — otherwise it would be a
                control with nothing to undo. */}
            {derivedLog.length > 0 && (
              <label className="flex cursor-pointer items-center gap-1.5 font-mono text-[11px]">
                <input
                  type="checkbox"
                  checked={forceLinear}
                  onChange={() => setForceLinear(!forceLinear)}
                  className="size-3 accent-neutral-800"
                />
                <span
                  className={forceLinear ? 'text-neutral-900' : 'text-neutral-500'}
                >
                  linear axes
                </span>
                {/* A count, not a list: the model scale applies to every
                    multiplicative coordinate, so naming them all would be a
                    paragraph on a wide fit. */}
                <span className="text-neutral-400">
                  ({derivedLog.length} of {visibleParams.length} on log)
                </span>
              </label>
            )}
            {/* Only offered when the sampler actually reports divergences: an
                MH/ODE fit writes no such column, and an absent control is
                honest where an always-empty one would read as "none". */}
            {nDivergent != null && (
              <label className="flex cursor-pointer items-center gap-1.5 font-mono text-[11px]">
                <input
                  type="checkbox"
                  checked={showDivergences}
                  onChange={() => setShowDivergences(!showDivergences)}
                  className="size-3 accent-orange-600"
                />
                <span
                  className={
                    showDivergences ? 'text-neutral-900' : 'text-neutral-500'
                  }
                >
                  divergences
                </span>
                <span className="text-neutral-400">
                  ({nDivergent.toLocaleString()})
                </span>
              </label>
            )}
          </div>
          {showDivergences && nDivergent != null && (
            <p className="mb-2 text-[10px] leading-snug text-neutral-400">
              {shownDivergent.toLocaleString()} of the{' '}
              {draws.data.n_draws.toLocaleString()} plotted draws diverged
              {draws.data.n_draws > 0 && (
                <> ({Math.round((100 * shownDivergent) / draws.data.n_draws)}%)</>
              )}
              ; {nDivergent.toLocaleString()} diverged across all retained draws.
              Both colours are thinned at the same rate, so their relative
              density is the sample's own — read whether the orange sits{' '}
              <em>somewhere the grey does not</em>, not how much of it there is.
            </p>
          )}
          <PairPlot
            draws={draws.data}
            posterior={posterior.data}
            params={visibleParams}
            priorXlimMode={priorXlimMode}
            marginalsByChain={marginalsByChain}
            showDivergences={showDivergences}
            logParams={logParams}
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
