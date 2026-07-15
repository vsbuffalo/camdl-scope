import { useEffect, useMemo, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { TraceSeries } from '@/api/client'
import { useTraces } from '@/api/queries'
import { ChainSelect } from '@/components/ChainSelect'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { includedChains, type ChainControls } from '@/lib/chains'
import { WarmupControl } from '@/components/WarmupControl'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { fmtTick } from '@/lib/format'
import { cn } from '@/lib/utils'
import { chainColor } from '@/lib/colors'

const DEFAULT_WARMUP_PCT = 0
const PANEL_HEIGHT = 80
const X_AXIS_PAD = 16 // extra bottom margin on the last panel for its shared axis
const FRAME = '#e5e5e5' // neutral-200 — hairline panel border
const AXIS = '#737373' // neutral-500 — tick labels
const MONO = 'var(--font-mono)'

// Auxiliary objective traces (Stan's lp__ and friends) that ride at the end of
// the trace list. Estimated-param panels are always shown; these are toggleable.
const OBJECTIVE_NAMES = new Set(['log_posterior', 'log_likelihood'])

type TracePoint = { iter: number; value: number; chain: number }

/**
 * One parameter's per-chain trace, drawn as overlaid lines coloured by chain.
 * Overlapping lines = mixed chains; separated lines = poor mixing. The burn-in
 * is trimmed server-side, so the x-domain auto-fits the returned (post-cutoff)
 * iterations and the y-scale rescales to the stationary part. Self-measuring
 * (seeded synchronously so it draws under headless capture). The x-axis is
 * drawn only on the last panel so the stack shares one bottom scale.
 */
function TracePanel({
  series,
  chainDomain,
  showXAxis,
}: {
  series: TraceSeries[]
  chainDomain: number[]
  showXAxis: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    setWidth(Math.round(el.getBoundingClientRect().width))
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0
      if (w > 0) setWidth(Math.round(w))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const el = ref.current
    if (!el || width <= 0) return

    const points: TracePoint[] = []
    for (const s of series) {
      const n = Math.min(s.iters.length, s.values.length)
      for (let i = 0; i < n; i++) {
        const v = s.values[i]!
        if (Number.isFinite(v)) points.push({ iter: s.iters[i]!, value: v, chain: s.chain })
      }
    }

    const node = Plot.plot({
      width,
      height: PANEL_HEIGHT,
      marginTop: 4,
      marginBottom: showXAxis ? 4 + X_AXIS_PAD : 4,
      marginLeft: 46,
      marginRight: 8,
      style: {
        background: 'transparent',
        color: AXIS,
        fontSize: '9px',
        fontFamily: MONO,
      },
      x: showXAxis
        ? { label: null, ticks: 6, tickSize: 2, tickPadding: 3 }
        : { axis: null },
      y: {
        label: null,
        ticks: 2,
        tickSize: 2,
        tickPadding: 3,
        tickFormat: (d: number) => fmtTick(d),
      },
      color: {
        type: 'categorical',
        domain: chainDomain,
        range: chainDomain.map(chainColor),
      },
      marks: [
        Plot.line(points, {
          x: 'iter',
          y: 'value',
          z: 'chain',
          stroke: 'chain',
          strokeWidth: 0.6,
          strokeOpacity: 0.8,
        }),
        Plot.frame({ stroke: FRAME, strokeWidth: 0.5 }),
      ],
    })

    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [series, chainDomain, showXAxis, width])

  return (
    <div
      ref={ref}
      className="w-full min-w-0 overflow-x-auto"
      role="img"
      aria-label="parameter trace"
    />
  )
}

/**
 * Compact chain → colour legend for decoding the overlaid traces. Colours by
 * chain **id** (`chainColor`), matching the plot's id-keyed scale, the pair
 * scatter, and the chain selector — so a chain reads the same colour
 * everywhere and doesn't shift when others are filtered out.
 */
function ChainLegend({ chains }: { chains: number[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-neutral-400">
      <span className="uppercase tracking-wider">chains</span>
      {chains.map((c) => (
        <span key={c} className="flex items-center gap-1">
          <span
            className="inline-block h-2 w-3 rounded-[1px]"
            style={{ background: chainColor(c) }}
          />
          {c}
        </span>
      ))}
    </div>
  )
}

export function TracesTab({
  runId,
  chainIds,
  excludedChains,
  onToggleChain,
  onResetChains,
}: { runId: string } & ChainControls) {
  const [warmupPct, setWarmupPct] = useState(DEFAULT_WARMUP_PCT)
  const chains = includedChains({ chainIds, excludedChains, onToggleChain, onResetChains })
  const { data, isPending, isError, isPlaceholderData } = useTraces(
    runId,
    warmupPct,
    chains,
  )

  const chainDomain = useMemo(() => {
    const set = new Set<number>()
    for (const t of data?.traces ?? [])
      for (const s of t.series) set.add(s.chain)
    return [...set].sort((a, b) => a - b)
  }, [data])

  // Objective panels (log_posterior / log_likelihood) default ON; the user can
  // toggle each off. We track the HIDDEN set so the default — empty — shows all.
  const [hiddenObjectives, setHiddenObjectives] = useState<Set<string>>(
    () => new Set(),
  )
  const toggleObjective = (name: string) =>
    setHiddenObjectives((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })

  const objectiveTraces = useMemo(
    () => data?.traces.filter((t) => OBJECTIVE_NAMES.has(t.param)) ?? [],
    [data],
  )
  // Estimated-param panels are always shown; objective panels are filtered by
  // their checkbox state.
  const visibleTraces = useMemo(
    () =>
      data?.traces.filter(
        (t) => !OBJECTIVE_NAMES.has(t.param) || !hiddenObjectives.has(t.param),
      ) ?? [],
    [data, hiddenObjectives],
  )

  // The legend + stacked trace panels are the PNG capture target.
  const figRef = useRef<HTMLDivElement>(null)

  return (
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
        nTail={null}
      />

      {chainIds.length > 1 && (
        <ChainSelect
          chainIds={chainIds}
          excluded={excludedChains}
          onToggle={onToggleChain}
          onReset={onResetChains}
        />
      )}

      {isPending && <ForestSkeleton rows={4} />}

      {isError && (
        <MutedNotice
          bordered={false}
          title="Couldn't load the traces"
          detail="The backend returned an error for this run. It may still be warming up."
        />
      )}

      {data && data.traces.length === 0 && (
        <MutedNotice
          bordered={false}
          title="No traces yet"
          detail="This run hasn't produced iteration traces. Check back once it has sampled."
        />
      )}

      {data && data.traces.length > 0 && (
        <>
          {objectiveTraces.length > 0 && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-neutral-100 px-3 py-2">
              <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
                objectives
              </span>
              {objectiveTraces.map((t) => (
                <label
                  key={t.param}
                  className="flex cursor-pointer select-none items-center gap-1.5 font-mono text-[11px] text-neutral-700"
                >
                  <input
                    type="checkbox"
                    checked={!hiddenObjectives.has(t.param)}
                    onChange={() => toggleObjective(t.param)}
                    className="h-3 w-3 accent-neutral-700"
                  />
                  <span>{t.param}</span>
                </label>
              ))}
            </div>
          )}

          <div className="group/fig relative">
            <div ref={figRef} className="bg-white">
              <div className="flex items-center justify-between gap-3 border-b border-neutral-100 px-3 py-2">
                <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
                  iteration traces · mixing
                </span>
                <ChainLegend chains={chainDomain} />
              </div>
              <div>
                {visibleTraces.map((t, i) => (
                  <div
                    key={t.param}
                    className="border-b border-neutral-100 px-3 py-1.5 last:border-b-0"
                  >
                    <div className="font-mono text-[10px] text-neutral-500">
                      {t.param}
                    </div>
                    <TracePanel
                      series={t.series}
                      chainDomain={chainDomain}
                      showXAxis={i === visibleTraces.length - 1}
                    />
                  </div>
                ))}
              </div>
            </div>
            <PlotDownloadButton targetRef={figRef} name="traces" />
          </div>
        </>
      )}
    </Card>
  )
}
