import { useEffect, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { CompareRow } from '@/api/client'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'

const INK = {
  zero: '#171717', // baseline rule at Δ=0
  baseline: '#171717',
  real: '#dc2626', // gap clears 2·se → red (decisively worse than baseline)
  muted: '#737373', // inconclusive
  axis: '#737373',
} as const

/**
 * The canonical model-comparison plot: each model's Δelpd against the baseline
 * with a ±2·se(Δ) whisker, baseline pinned at 0. A whisker that clears 0
 * (|Δelpd| > 2·se) reads red — "the gap is real"; one straddling 0 is muted.
 * Commensurable comparisons only (the caller gates on it). Self-measuring.
 */
export function DeltaElpdPlot({
  rows,
  aliasOf,
}: {
  rows: CompareRow[]
  /** Alias (M1, M2, …) per run — the short y-axis ticker, decoded by the legend. */
  aliasOf: Map<string, string>
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  // Key rows on run_id (labels/aliases are display only); short alias on the axis.
  const data = rows
    .map((r) => ({
      runId: r.run_id,
      alias: aliasOf.get(r.run_id) ?? r.run_id,
      d: r.is_baseline ? 0 : (r.delta_elpd ?? Number.NaN),
      se: r.is_baseline ? 0 : (r.se_delta_elpd ?? 0),
      real: r.gap_is_real,
      baseline: r.is_baseline,
    }))
    .filter((d) => Number.isFinite(d.d))
  const aliasFor = new Map(data.map((d) => [d.runId, d.alias]))

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
    if (!el || width <= 0 || data.length === 0) return

    // Aliases are short (M1, M2, …) so the plot keeps nearly full width.
    const longest = Math.max(...data.map((d) => d.alias.length))
    const marginLeft = Math.max(34, longest * 8 + 12)

    const node = Plot.plot({
      width,
      height: data.length * 26 + 34,
      marginTop: 8,
      marginBottom: 26,
      marginLeft,
      marginRight: 14,
      style: { background: 'transparent', color: INK.axis, fontSize: '10px' },
      x: {
        label: 'Δelpd vs baseline (nats) →',
        grid: true,
        // Let Plot pick the tick precision — forcing toFixed(0) collapses
        // adjacent fractional ticks to duplicate integer labels on small ranges.
      },
      y: {
        domain: data.map((d) => d.runId),
        tickFormat: (id: string) => aliasFor.get(id) ?? id,
        label: null,
      },
      marks: [
        Plot.ruleX([0], { stroke: INK.zero, strokeWidth: 1 }),
        Plot.ruleY(
          data.filter((d) => !d.baseline),
          {
            y: 'runId',
            x1: (d: (typeof data)[number]) => d.d - 2 * d.se,
            x2: (d: (typeof data)[number]) => d.d + 2 * d.se,
            stroke: (d: (typeof data)[number]) => (d.real ? INK.real : INK.muted),
            strokeWidth: 1.5,
          },
        ),
        Plot.dot(data, {
          y: 'runId',
          x: 'd',
          fill: (d: (typeof data)[number]) =>
            d.baseline ? INK.baseline : d.real ? INK.real : INK.muted,
          r: 3.6,
        }),
      ],
    })

    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [width, data])

  return (
    <div className="group/fig relative">
      <div ref={ref} className="w-full overflow-hidden bg-white" />
      <PlotDownloadButton targetRef={ref} name="delta-elpd" />
    </div>
  )
}
