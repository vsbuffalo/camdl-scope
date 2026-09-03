import { useEffect, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import { fmtTick } from '@/lib/format'
import { token } from '@/lib/theme'

/**
 * Ink for the marginal: faint nested credible bands behind a neutral
 * histogram, with a crisp median rule. Dark→meaningful: the bars are the data,
 * the bands are context, the median line is the point estimate.
 */
const INK = {
  ciBand: token('raised'), // neutral-100 — 90% credible region
  iqrBand: token('line'), // neutral-200 — interquartile region (nested)
  bar: token('faint'), // neutral-400 — histogram
  median: token('ink'), // neutral-900
  axis: token('muted'), // neutral-500 — tick labels
} as const

const HEIGHT = 66
const THRESHOLDS = 28

interface MarginalDensityProps {
  values: number[]
  q05: number
  q25: number
  q50: number
  q75: number
  q95: number
  /** Physical support, when the param is constrained — clamps the padded domain. */
  bounds?: [number, number] | null
}

/** Linear-interpolated quantile of an already-sorted array (display-only). */
function quantileSorted(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN
  if (sorted.length === 1) return sorted[0]!
  const idx = (sorted.length - 1) * q
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]!
  const t = idx - lo
  return sorted[lo]! * (1 - t) + sorted[hi]! * t
}

/**
 * One parameter's marginal posterior on its OWN visible, labelled x-axis.
 *
 * The axis is the point: parameters live on incomparable supports, so a shared
 * scale is meaningless — but an honest small-multiple with its own drawn scale
 * is not. A plain histogram (no KDE bandwidth to argue about) carries the
 * shape; faint q05–q95 / q25–q75 bands and a median rule carry the summary.
 */
export function MarginalDensity({
  values,
  q05,
  q25,
  q50,
  q75,
  q95,
  bounds,
}: MarginalDensityProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  // Seed width synchronously from the laid-out element so the plot draws on the
  // first paint (ResizeObserver alone never fires under headless capture).
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
    if (!el || width <= 0 || values.length < 2) return

    // Robust window: the central ~99% of the draws, so a long tail (e.g. a
    // near-zero rate) doesn't squeeze the bulk of the histogram into a corner.
    const sorted = [...values].sort((a, b) => a - b)
    const lo = quantileSorted(sorted, 0.005)
    const hi = quantileSorted(sorted, 0.995)
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return

    const pad = (hi - lo) * 0.04 || Math.abs(q50) * 0.04 || 1
    let d0 = lo - pad
    let d1 = hi + pad
    if (bounds) {
      if (d0 < bounds[0]) d0 = bounds[0]
      if (d1 > bounds[1]) d1 = bounds[1]
    }
    const domain: [number, number] = [d0, d1]
    const inWindow = values.filter((v) => v >= d0 && v <= d1)

    const node = Plot.plot({
      width,
      height: HEIGHT,
      marginTop: 6,
      marginBottom: 20,
      // Generous side margins so the end-tick labels (centered on the domain
      // edges) aren't clipped by the panel; tickPadding lifts them off the axis.
      marginLeft: 24,
      marginRight: 24,
      style: { background: 'transparent', color: INK.axis, fontSize: '9px' },
      x: {
        domain,
        ticks: [lo, q50, hi],
        tickFormat: (d: number) => fmtTick(d),
        tickSize: 2,
        tickPadding: 4,
        label: null,
      },
      y: { axis: null },
      marks: [
        // 90% credible band — faint, spans the full frame height
        Plot.rect([{}], { x1: q05, x2: q95, fill: INK.ciBand }),
        // IQR band — nested, a touch darker
        Plot.rect([{}], { x1: q25, x2: q75, fill: INK.iqrBand }),
        // histogram — the honest marginal, thin inset gaps for crisp bars
        Plot.rectY(
          inWindow,
          Plot.binX<Plot.RectYOptions>(
            { y: 'count' },
            {
              x: (d: number) => d,
              thresholds: THRESHOLDS,
              fill: INK.bar,
              fillOpacity: 0.7,
              insetLeft: 0.5,
              insetRight: 0.5,
            },
          ),
        ),
        // median — the point estimate
        Plot.ruleX([q50], { stroke: INK.median, strokeWidth: 1.25 }),
      ],
    })

    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [values, q05, q25, q50, q75, q95, bounds, width])

  return (
    <div
      ref={ref}
      className="w-full min-w-0 overflow-hidden"
      style={{ height: HEIGHT }}
      role="img"
      aria-label={`Marginal posterior: median ${q50}, 90% interval ${q05} to ${q95}`}
    />
  )
}
