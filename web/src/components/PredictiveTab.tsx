import { useEffect, useMemo, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { ObservedPoint, PredictivePoint } from '@/api/client'
import { usePredictive, useRun } from '@/api/queries'
import { ForestSkeleton, MutedNotice, NoPosteriorNotice } from '@/components/States'
import { ScenarioChecks } from '@/components/ScenarioChecks'
import { Figure } from '@/components/Figure'
import { ByIndexPlot, LevelLegend } from '@/components/ByIndexPlot'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { Segmented } from '@/components/Segmented'
import { Card } from '@/components/ui/card'
import { fmtTick } from '@/lib/format'
import { dayToDate } from '@/lib/calendar'
import {
  buildByIndexProfile,
  LEVEL_PALETTE,
  NEUTRAL,
  type IndexRecord,
} from '@/lib/byindex'
import { buildScenarioColors, referenceScenario, SCENARIO_REFERENCE } from '@/lib/scenario'
import { cn } from '@/lib/utils'

// Horizon ink (used when there is no scenario overlay): free_forward reads blue,
// one_step green, anything else neutral.
const HORIZON_MEDIAN: Record<string, string> = {
  free_forward: '#2563eb',
  one_step: '#16a34a',
}
const HORIZON_FALLBACK = '#737373'
const horizonColor = (h: string) => HORIZON_MEDIAN[h] ?? HORIZON_FALLBACK

const OBSERVED = '#171717' // neutral-900 — the data, distinct from every prediction
const AXIS = '#737373'
const PANEL_HEIGHT = 220
const MONO = 'var(--font-mono)'


/** Human label for a stratum object — `district=Bombali · age=u5`, or empty. */
function stratumLabel(stratum: Record<string, string>): string {
  return Object.entries(stratum)
    .map(([k, v]) => `${k}=${v}`)
    .join(' · ')
}

/** One overlaid ribbon (a scenario, or a horizon): its color + points. */
type OverlaySeries = { key: string; color: string; pred: PredictivePoint[] }

/** Independently toggleable layers of a predictive ribbon. */
type PredLayer = 'median' | 'p50' | 'p90'
const PRED_LAYERS: { key: PredLayer; label: string }[] = [
  { key: 'median', label: 'median' },
  { key: 'p50', label: '50%' },
  { key: 'p90', label: '90%' },
]

/**
 * One stratum's posterior-predictive check. Each overlaid arm draws its own
 * color-coded ribbon — the 90% band, the 50% band, and the median line, each
 * shown unless its layer is toggled off; the observed series is drawn once in
 * neutral-dark. Self-measuring like the other plots.
 */
function PredictivePanel({
  title,
  series,
  observed,
  dense,
  hiddenLayers,
  toDate,
  windowMode,
}: {
  title: string
  series: OverlaySeries[]
  observed: ObservedPoint[]
  dense: boolean
  hiddenLayers: ReadonlySet<PredLayer>
  toDate: ((t: number) => Date) | null
  /** 'data' clips the ribbons to the observed window (y rescales to the fit);
   *  'full' shows the whole predictive extent with a dashed rule at data end. */
  windowMode: 'data' | 'full'
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

    const obs = observed
      .filter((o) => o.value != null && Number.isFinite(o.value))
      .sort((a, b) => a.time - b.time)

    // The forecast boundary: where this stratum's observations stop. In 'data'
    // mode ribbons are clipped here (auto x/y domains then rescale to the fit
    // window); in 'full' mode a dashed rule marks it when a forecast extends
    // past it (e.g. a scenario with `simulate { to = ... }`).
    const obsEnd = obs.length ? Math.max(...obs.map((o) => o.time)) : null
    const clip = windowMode === 'data' && obsEnd != null

    // Numeric time → Date when the fit carries a calendar, so the shared x-axis
    // reads as real dates; otherwise the raw numeric index.
    const xOf = (d: { time: number }) => (toDate ? toDate(d.time) : d.time)

    // A hidden layer contributes no mark, so Plot's auto y-domain ignores its
    // (possibly wild) extent and rescales to what remains + observed.
    const showP90 = !hiddenLayers.has('p90')
    const showP50 = !hiddenLayers.has('p50')
    const showMedian = !hiddenLayers.has('median')
    const marks: Plot.Markish[] = []
    let predEnd = -Infinity
    for (const s of series) {
      let pred = [...s.pred].sort((a, b) => a.time - b.time)
      if (pred.length) predEnd = Math.max(predEnd, pred[pred.length - 1]!.time)
      if (clip) pred = pred.filter((p) => p.time <= obsEnd!)
      if (showP90) {
        marks.push(
          Plot.areaY(pred, {
            x: xOf,
            y1: 'q05',
            y2: 'q95',
            fill: s.color,
            fillOpacity: dense ? 0.16 : 0.1,
          }),
        )
      }
      if (showP50) {
        marks.push(
          Plot.areaY(pred, {
            x: xOf,
            y1: 'q25',
            y2: 'q75',
            fill: s.color,
            fillOpacity: 0.24,
          }),
        )
      }
      if (showMedian) {
        marks.push(
          Plot.line(pred, { x: xOf, y: 'q50', stroke: s.color, strokeWidth: 1.3 }),
        )
      }
    }
    // Forecast boundary rule: only when predictions actually extend past the
    // observed window (and it isn't clipped away).
    if (!clip && obsEnd != null && predEnd > obsEnd) {
      marks.push(
        Plot.ruleX([toDate ? toDate(obsEnd) : obsEnd], {
          stroke: '#a3a3a3',
          strokeWidth: 1,
          strokeDasharray: '3,3',
        }),
      )
    }
    marks.push(
      Plot.line(obs, {
        x: xOf,
        y: 'value',
        stroke: OBSERVED,
        strokeWidth: 0.75,
        strokeOpacity: 0.4,
      }),
      Plot.dot(obs, {
        x: xOf,
        y: 'value',
        fill: OBSERVED,
        r: 3,
        stroke: 'white',
        strokeWidth: 0.5,
      }),
      Plot.ruleY([0], { stroke: '#e5e5e5', strokeWidth: 0.5 }),
    )

    const node = Plot.plot({
      width,
      height: PANEL_HEIGHT,
      marginTop: 10,
      // A date axis renders two-line ticks (day over month) — reserve the
      // second line's room or the month row clips at the SVG edge.
      marginBottom: toDate ? 34 : 24,
      marginLeft: 46,
      marginRight: 12,
      style: { background: 'transparent', color: AXIS, fontSize: '10px', fontFamily: MONO },
      x: { label: null, tickSize: 2, tickPadding: 4, ticks: 6 },
      y: {
        label: null,
        tickSize: 2,
        tickPadding: 4,
        ticks: 5,
        tickFormat: (d: number) => fmtTick(d),
        grid: true,
      },
      marks,
    })

    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [series, observed, width, dense, hiddenLayers, toDate, windowMode])

  const figRef = useRef<HTMLDivElement>(null)

  return (
    <div className="group/fig relative border-t border-neutral-100 px-3 py-2">
      <div ref={figRef} className="bg-white">
        <div className="font-mono text-[10px] text-neutral-500">{title}</div>
        <div
          ref={ref}
          className="mt-1 w-full min-w-0 overflow-x-auto"
          style={{ minHeight: PANEL_HEIGHT }}
          role="img"
          aria-label={title}
        />
      </div>
      <PlotDownloadButton
        targetRef={figRef}
        name={`predictive-${title.replace(/[^\w.-]+/g, '-')}`}
      />
    </div>
  )
}

const ONE_TO_ONE = '#d4d4d4' // neutral-300 — perfect prediction / zero residual
// NEUTRAL + LEVEL_PALETTE (index-level colours) are shared with the Quantities
// by-index view — imported from '@/lib/byindex'.

/** Coefficient of determination of the predictions against the 1:1 line:
 *  1 − Σ(obs−pred)² / Σ(obs−mean(obs))². 1 is perfect; it can go negative when
 *  the predictions are worse than just predicting the observed mean. */
function rSquared(points: { obs: number; pred: number }[]): number {
  if (points.length < 2) return Number.NaN
  const mean = points.reduce((a, p) => a + p.obs, 0) / points.length
  let ssRes = 0
  let ssTot = 0
  for (const { obs, pred } of points) {
    ssRes += (obs - pred) ** 2
    ssTot += (obs - mean) ** 2
  }
  return ssTot > 0 ? 1 - ssRes / ssTot : Number.NaN
}

/** Where ``value`` falls in the predictive CDF described by quantile ``vals`` at
 *  probability ``levels`` (both ascending, same length). Piecewise-linear
 *  interpolation between stored quantiles, linearly extended past the outer
 *  quantiles using the nearest segment's slope, clamped to [0, 1]. This is the
 *  PIT — uniform under a calibrated predictive. */
function pitOf(value: number, vals: number[], levels: number[]): number {
  const n = vals.length
  if (value <= vals[0]!) {
    // Below the lowest quantile: extend the first segment's slope downward.
    const dv = vals[1]! - vals[0]!
    const p = dv > 0 ? levels[0]! + ((value - vals[0]!) / dv) * (levels[1]! - levels[0]!) : 0
    return Math.max(0, Math.min(1, p))
  }
  if (value >= vals[n - 1]!) {
    const dv = vals[n - 1]! - vals[n - 2]!
    const p = dv > 0 ? levels[n - 1]! + ((value - vals[n - 1]!) / dv) * (levels[n - 1]! - levels[n - 2]!) : 1
    return Math.max(0, Math.min(1, p))
  }
  for (let i = 1; i < n; i++) {
    if (value <= vals[i]!) {
      const dv = vals[i]! - vals[i - 1]!
      const frac = dv > 0 ? (value - vals[i - 1]!) / dv : 0
      return levels[i - 1]! + frac * (levels[i]! - levels[i - 1]!)
    }
  }
  return 1
}

/** One matched (observation, predicted-median) pair, pre-coloured + grouped. */
type ScatterPoint = {
  obs: number
  pred: number
  resid: number // obs − pred
  time: number
  color: string // resolved fill (index-level colour, or arm colour)
  group: string // colouring key (index level, or arm)
  jx: number // stable jitter seed in [−0.5, 0.5] for the by-index strip
}
type R2Line = { label: string; color: string; r2: number }

/** Window (in distinct time points) of the residuals-vs-time moving average. */
const RESID_MA_WINDOW = 5

/** Per group, the mean residual at each time smoothed by a centred moving
 *  average over ``window`` time points (shrinking at the edges). Collapses the
 *  several strata sharing a time into one mean first, so the trend reads the
 *  drift over time rather than the within-time spread. */
function residualTimeMA(
  points: ScatterPoint[],
  window: number,
): { time: number; ma: number; group: string; color: string }[] {
  const byGroup = new Map<
    string,
    { color: string; byTime: Map<number, { sum: number; n: number }> }
  >()
  for (const p of points) {
    let g = byGroup.get(p.group)
    if (!g) {
      g = { color: p.color, byTime: new Map() }
      byGroup.set(p.group, g)
    }
    let t = g.byTime.get(p.time)
    if (!t) {
      t = { sum: 0, n: 0 }
      g.byTime.set(p.time, t)
    }
    t.sum += p.resid
    t.n += 1
  }
  const half = Math.floor(window / 2)
  const out: { time: number; ma: number; group: string; color: string }[] = []
  for (const [group, g] of byGroup) {
    const times = [...g.byTime.entries()]
      .map(([time, s]) => ({ time, mean: s.sum / s.n }))
      .sort((a, b) => a.time - b.time)
    for (let i = 0; i < times.length; i++) {
      let sum = 0
      let n = 0
      for (let j = Math.max(0, i - half); j <= Math.min(times.length - 1, i + half); j++) {
        sum += times[j]!.mean
        n += 1
      }
      out.push({ time: times[i]!.time, ma: sum / n, group, color: g.color })
    }
  }
  return out
}

/**
 * Predicted-vs-observed scatter: each point pairs an observation with the
 * model's predicted median. The dashed 1:1 line is perfect prediction; R² (per
 * arm) sits in the corner. Colour is whatever the caller resolved. Square so the
 * 1:1 line reads at 45°.
 */
function PredObsScatter({ points, r2Lines }: { points: ScatterPoint[]; r2Lines: R2Line[] }) {
  return (
    <Figure
      name="pred-vs-obs"
      aria="predicted versus observed"
      deps={[points, r2Lines]}
      render={(el, width) => {
        if (points.length === 0) {
          el.replaceChildren()
          return
        }
        let lo = Infinity
        let hi = -Infinity
        for (const p of points) {
          lo = Math.min(lo, p.obs, p.pred)
          hi = Math.max(hi, p.obs, p.pred)
        }
        const pad = (hi - lo) * 0.05 || Math.abs(hi) * 0.05 || 1
        const domain: [number, number] = [lo - pad, hi + pad]
        const labelled = r2Lines.length > 1
        const node = Plot.plot({
          width,
          height: Math.min(width, 440),
          marginTop: 10,
          marginBottom: 34,
          marginLeft: 50,
          marginRight: 12,
          style: { background: 'transparent', color: AXIS, fontSize: '10px', fontFamily: MONO },
          x: { label: 'observed →', labelAnchor: 'center', domain, ticks: 6, tickFormat: (d: number) => fmtTick(d), grid: true },
          y: { label: '↑ predicted (median)', labelAnchor: 'center', domain, ticks: 6, tickFormat: (d: number) => fmtTick(d), grid: true },
          marks: [
            Plot.line([[domain[0], domain[0]], [domain[1], domain[1]]], { stroke: ONE_TO_ONE, strokeWidth: 1, strokeDasharray: '4,3' }),
            Plot.dot(points, { x: 'obs', y: 'pred', fill: 'color', r: 3, fillOpacity: 0.6, stroke: 'white', strokeWidth: 0.4 }),
            ...r2Lines
              .filter((a) => Number.isFinite(a.r2))
              .map((a, i) =>
                Plot.text([a], {
                  frameAnchor: 'top-left',
                  dx: 8,
                  dy: 10 + i * 15,
                  text: () => `${labelled ? `${a.label || '∅'}  ` : ''}R² = ${a.r2.toFixed(3)}`,
                  fill: labelled ? a.color : '#171717',
                  fontSize: 12,
                  fontWeight: 600,
                  textAnchor: 'start',
                }),
              ),
          ],
        })
        el.replaceChildren(node)
      }}
    />
  )
}

/**
 * Residuals (observed − predicted). The dashed rule at 0 is a perfect fit; a
 * coloured mark shows each group's *mean* residual — a group whose mean sits off
 * 0 is systematically over/under-predicted.
 *
 * Adapts to the colour-by choice: grouped by an index (``byIndex``) it's a
 * per-level strip — the direct "is any village/age biased?" view — with the mean
 * as a tick per level. On ``none`` it's residual-vs-fitted, where a fan-shaped
 * spread reveals heteroscedasticity the categorical view can't.
 */
function ResidualPlot({
  points,
  groupMeans,
  xMode,
  xLabel,
  toDate,
}: {
  points: ScatterPoint[]
  groupMeans: { level: string; mean: number; color: string }[]
  xMode: 'predicted' | 'time' | 'index'
  xLabel: string
  toDate: ((t: number) => Date) | null
}) {
  return (
    <Figure
      name="residuals"
      aria="residuals"
      deps={[points, groupMeans, xMode, xLabel, toDate]}
      render={(el, width) => {
        if (points.length === 0) {
          el.replaceChildren()
          return
        }
        let rlo = 0
        let rhi = 0
        for (const p of points) {
          rlo = Math.min(rlo, p.resid)
          rhi = Math.max(rhi, p.resid)
        }
        const rpad = (rhi - rlo) * 0.06 || 1
        const common = {
          width,
          height: Math.min(Math.round(width * 0.55), 320),
          marginTop: 10,
          marginBottom: 34,
          marginLeft: 50,
          marginRight: 12,
          style: { background: 'transparent', color: AXIS, fontSize: '10px', fontFamily: MONO },
          y: {
            label: '↑ residual (obs − pred)',
            labelAnchor: 'center' as const,
            domain: [rlo - rpad, rhi + rpad],
            ticks: 5,
            tickFormat: (d: number) => fmtTick(d),
          },
        }
        let node: ReturnType<typeof Plot.plot>
        if (xMode === 'index') {
          // Integer x per level + a jitter offset so the strip spreads; label the
          // integer ticks with the level names.
          const levels = groupMeans.map((g) => g.level).sort()
          const at = new Map(levels.map((l, i) => [l, i]))
          const jpts = points.map((p) => ({ ...p, xi: (at.get(p.group) ?? 0) + p.jx * 0.6 }))
          const means = groupMeans.map((g) => ({ ...g, xi: at.get(g.level) ?? 0 }))
          node = Plot.plot({
            ...common,
            x: {
              domain: [-0.5, levels.length - 0.5],
              label: `${xLabel} →`,
              ticks: levels.map((_, i) => i),
              tickFormat: (i: number) => levels[Math.round(i)] ?? '',
              tickSize: 0,
            },
            marks: [
              Plot.ruleY([0], { stroke: ONE_TO_ONE, strokeWidth: 1, strokeDasharray: '4,3' }),
              Plot.dot(jpts, { x: 'xi', y: 'resid', fill: 'color', r: 2.5, fillOpacity: 0.5, stroke: 'white', strokeWidth: 0.3 }),
              // Per-level mean residual — a bold coloured segment at the level.
              Plot.link(means, {
                x1: (d: { xi: number }) => d.xi - 0.32,
                x2: (d: { xi: number }) => d.xi + 0.32,
                y1: 'mean',
                y2: 'mean',
                stroke: 'color',
                strokeWidth: 2.5,
              }),
            ],
          })
        } else {
          // Continuous x: residuals vs predicted (heteroscedasticity) or vs time
          // (temporal structure — a run of same-sign residuals is a dynamic miss).
          const xa = xMode === 'time' ? 'time' : 'pred'
          let xlo = Infinity
          let xhi = -Infinity
          for (const p of points) {
            const v = p[xa]
            xlo = Math.min(xlo, v)
            xhi = Math.max(xhi, v)
          }
          const xpad = (xhi - xlo) * 0.05 || 1
          // In the time view, a moving average of the mean residual exposes
          // temporal drift (a run of same-sign residuals = a dynamic miss); the
          // flat overall-mean line drops back to a faint reference behind it.
          const isTime = xMode === 'time'
          const useDate = isTime && !!toDate
          const maPts = isTime ? residualTimeMA(points, RESID_MA_WINDOW) : []
          // Date x when a calendar is present (Plot infers the time scale + date
          // ticks); else the numeric axis with an explicit padded domain.
          const dotX = useDate ? (d: ScatterPoint) => toDate!(d.time) : xa
          const maX = useDate
            ? (d: { time: number }) => toDate!(d.time)
            : ('time' as const)
          node = Plot.plot({
            ...common,
            x: useDate
              ? { label: 'date →', labelAnchor: 'center', ticks: 6, grid: true }
              : {
                  label: isTime ? 'time →' : 'predicted (median) →',
                  labelAnchor: 'center',
                  domain: [xlo - xpad, xhi + xpad],
                  ticks: 6,
                  tickFormat: (d: number) => fmtTick(d),
                  grid: true,
                },
            marks: [
              Plot.ruleY([0], { stroke: ONE_TO_ONE, strokeWidth: 1, strokeDasharray: '4,3' }),
              Plot.ruleY(groupMeans, {
                y: 'mean',
                stroke: 'color',
                strokeWidth: isTime ? 1 : 1.5,
                strokeOpacity: isTime ? 0.3 : 0.8,
                strokeDasharray: isTime ? '2,3' : undefined,
              }),
              Plot.dot(points, {
                x: dotX,
                y: 'resid',
                fill: 'color',
                r: 3,
                fillOpacity: isTime ? 0.35 : 0.55,
                stroke: 'white',
                strokeWidth: 0.4,
              }),
              ...(maPts.length
                ? [
                    Plot.line(maPts, {
                      x: maX,
                      y: 'ma',
                      z: 'group',
                      stroke: 'color',
                      strokeWidth: 2,
                      strokeOpacity: 0.95,
                      curve: 'monotone-x',
                    }),
                    Plot.dot(maPts, { x: maX, y: 'ma', fill: 'color', r: 2 }),
                  ]
                : []),
            ],
          })
        }
        el.replaceChildren(node)
      }}
    />
  )
}

/** PIT calibration histogram: the distribution of each observation's position in
 *  its own predictive CDF. Under a calibrated model the bars sit at the uniform
 *  reference (dashed); a U-shape means overconfident, a central dome means
 *  underconfident, a slope means biased. */
function PitPlot({ pit }: { pit: number[] }) {
  return (
    <Figure
      name="pit-calibration"
      aria="PIT calibration histogram"
      deps={[pit]}
      render={(el, width) => {
        if (pit.length === 0) {
          el.replaceChildren()
          return
        }
        const BINS = 10
        const counts = new Array<number>(BINS).fill(0)
        for (const v of pit) {
          let b = Math.floor(v * BINS)
          if (b >= BINS) b = BINS - 1
          if (b < 0) b = 0
          counts[b]! += 1
        }
        const bars = counts.map((count, i) => ({ x0: i / BINS, x1: (i + 1) / BINS, count }))
        const expected = pit.length / BINS
        const node = Plot.plot({
          width,
          height: Math.min(Math.round(width * 0.5), 300),
          marginTop: 10,
          marginBottom: 34,
          marginLeft: 46,
          marginRight: 12,
          style: { background: 'transparent', color: AXIS, fontSize: '10px', fontFamily: MONO },
          x: {
            label: 'PIT — observation’s quantile in predictive →',
            labelAnchor: 'center',
            domain: [0, 1],
            ticks: [0, 0.25, 0.5, 0.75, 1],
          },
          y: { label: '↑ count', labelAnchor: 'center', grid: true },
          marks: [
            Plot.rectY(bars, { x1: 'x0', x2: 'x1', y: 'count', fill: NEUTRAL, fillOpacity: 0.72, inset: 0.5 }),
            Plot.ruleY([expected], { stroke: '#dc2626', strokeWidth: 1.2, strokeDasharray: '4,3' }),
            Plot.ruleY([0], { stroke: AXIS, strokeWidth: 1 }),
          ],
        })
        el.replaceChildren(node)
      }}
    />
  )
}

/** Flat, mono checkbox group — multi-select horizons that overlay in the panel. */
function HorizonChecks({
  options,
  selected,
  onToggle,
}: {
  options: readonly string[]
  selected: readonly string[]
  onToggle: (h: string) => void
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
        Horizon
      </span>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {options.map((opt) => {
          const on = selected.includes(opt)
          return (
            <label
              key={opt}
              className="flex cursor-pointer items-center gap-1.5 font-mono text-xs"
            >
              <input
                type="checkbox"
                checked={on}
                onChange={() => onToggle(opt)}
                className="size-3 accent-neutral-800"
              />
              <span className={on ? 'text-neutral-900' : 'text-neutral-500'}>
                {opt || '∅'}
              </span>
            </label>
          )
        })}
      </div>
    </div>
  )
}

/** Swatch hinting a predictive layer: a line for the median, a filled band for
 *  the 50%/90% intervals (opacity scaled so 90% reads lighter than 50%). */
function LayerSwatch({ layer, on }: { layer: PredLayer; on: boolean }) {
  const base = on ? '#525252' : '#a3a3a3'
  if (layer === 'median') {
    return <span className="inline-block h-[2px] w-3 rounded-full" style={{ background: base }} />
  }
  return (
    <span
      className="inline-block h-2 w-3 rounded-[1px]"
      style={{ background: base, opacity: layer === 'p90' ? 0.3 : 0.55 }}
    />
  )
}

/** Checkbox group toggling each predictive layer (median / 50% / 90%). Checked =
 *  layer shown; unchecking rescales the panel y-axis to what remains + observed. */
function LayerChecks({
  hidden,
  onToggle,
}: {
  hidden: ReadonlySet<PredLayer>
  onToggle: (layer: PredLayer) => void
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
        Show
      </span>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {PRED_LAYERS.map(({ key, label }) => {
          const on = !hidden.has(key)
          return (
            <label
              key={key}
              className="flex cursor-pointer items-center gap-1.5 font-mono text-xs"
            >
              <input
                type="checkbox"
                checked={on}
                onChange={() => onToggle(key)}
                className="size-3 accent-neutral-800"
              />
              <LayerSwatch layer={key} on={on} />
              <span className={on ? 'text-neutral-900' : 'text-neutral-500'}>
                {label}
              </span>
            </label>
          )
        })}
      </div>
    </div>
  )
}

/** Swatch legend mapping each overlaid arm (+ observed) to its colour. */
function Legend({ arms }: { arms: { label: string; color: string }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-neutral-400">
      {arms.map((a) => (
        <span key={a.label} className="flex items-center gap-1">
          <span
            className="inline-block h-2 w-3 rounded-[1px]"
            style={{ background: a.color }}
          />
          {a.label || '∅'}
        </span>
      ))}
      <span className="flex items-center gap-1">
        <span
          className="inline-block size-2 rounded-full"
          style={{ background: OBSERVED }}
        />
        observed
      </span>
    </div>
  )
}

export function PredictiveTab({ runId }: { runId: string }) {
  const run = useRun(runId)
  const availableStreams = useMemo(
    () => run.data?.available_streams ?? [],
    [run.data],
  )

  const [stream, setStream] = useState<string>()
  const activeStream =
    stream && availableStreams.includes(stream)
      ? stream
      : (availableStreams[0] ?? undefined)

  const { data, isPending, isError, isPlaceholderData } = usePredictive(
    runId,
    activeStream,
  )

  const [selected, setSelected] = useState<readonly string[] | null>(null)
  const [selectedScenarios, setSelectedScenarios] = useState<readonly string[] | null>(null)
  const [treatment, setTreatment] = useState<string>()
  // Inner view: the time-series ribbons (default), the predicted-vs-observed
  // scatter, the PIT calibration histogram, or the by-index profile. All read
  // the same stream/scenario/horizon selection above.
  const [view, setView] = useState<'series' | 'scatter' | 'calibration' | 'byindex'>(
    'series',
  )
  // By-index profile: which index dimension goes on x, and which (if any) facets
  // the lines by colour.
  const [byIndexX, setByIndexX] = useState<string | null>(null)
  const [byIndexFacet, setByIndexFacet] = useState<string | null>(null)
  // Which index dimension to colour the scatter/residual plots by (null = none).
  const [colorBy, setColorBy] = useState<string | null>(null)
  // Residual x-axis: vs the prediction (heteroscedasticity), vs time (temporal
  // structure — a dynamic miss), or vs the coloured index (per-level offset).
  const [residualX, setResidualX] = useState<'predicted' | 'time' | 'index'>(
    'predicted',
  )
  // Predictive layers hidden in the time-series view. Each layer (median line,
  // 50% band, 90% band) toggles independently; dropping the wild outer band
  // lets the panel's y-axis rescale to what's left + observed, so the data is
  // no longer squashed to the floor.
  const [hiddenLayers, setHiddenLayers] = useState<ReadonlySet<PredLayer>>(
    () => new Set(),
  )
  // Time window of the series view: the full predictive extent (forecasts and
  // scenario runs past the data, with a dashed rule at data end), or clipped to
  // the observed window so the axes rescale to the fit itself. Only offered
  // when some prediction actually extends past the data.
  const [windowMode, setWindowMode] = useState<'data' | 'full'>('full')

  const horizons = data?.horizons ?? []
  const scenarios = useMemo(() => data?.scenarios ?? [], [data])
  // Color by scenario once there's more than one (the comparison axis); else by
  // horizon, the original behaviour.
  const byScenario = scenarios.length > 1
  const scenarioColors = useMemo(
    () => buildScenarioColors(scenarios),
    [scenarios],
  )

  const selectedHorizons = useMemo(() => {
    const set = new Set(selected ?? horizons)
    return horizons.filter((h) => set.has(h))
  }, [selected, horizons])

  // The reference arm (fitted / as_fitted / baseline) is the posterior
  // predictive itself — pinned always-on; the checkbox selection governs only
  // the scenario overlays on top of it, so `none` never blanks the tab.
  const reference = useMemo(() => referenceScenario(scenarios), [scenarios])
  const overlayOptions = useMemo(
    () => scenarios.filter((s) => s !== reference),
    [scenarios, reference],
  )
  const activeScenarios = useMemo(() => {
    const set = new Set(selectedScenarios ?? scenarios)
    return scenarios.filter((s) => s === reference || set.has(s))
  }, [selectedScenarios, scenarios, reference])

  const toggleHorizon = (h: string) =>
    setSelected(
      selectedHorizons.includes(h)
        ? selectedHorizons.filter((x) => x !== h)
        : [...selectedHorizons, h],
    )

  const toggleScenario = (s: string) =>
    setSelectedScenarios(
      activeScenarios.includes(s)
        ? activeScenarios.filter((x) => x !== s)
        : [...activeScenarios, s],
    )

  const treatments = data?.treatments ?? []
  const needTreatment = treatments.length > 1
  const activeTreatment =
    treatment && treatments.includes(treatment)
      ? treatment
      : (treatments[0] ?? '')

  // Group the checked predictive points by stratum; within each stratum, one
  // overlaid arm per (scenario, horizon). Colored by scenario when overlaying
  // scenarios, else by horizon.
  const strata = useMemo(() => {
    if (!data) return []
    const obsByKey = new Map<string, ObservedPoint[]>()
    for (const o of data.observed) {
      const key = JSON.stringify(o.stratum)
      const arr = obsByKey.get(key)
      if (arr) arr.push(o)
      else obsByKey.set(key, [o])
    }

    const wanted = new Set(selectedHorizons)
    const wantedScenarios = new Set(activeScenarios)
    const groups = new Map<
      string,
      {
        key: string
        stratum: Record<string, string>
        byArm: Map<string, { scenario: string; horizon: string; pred: PredictivePoint[] }>
      }
    >()
    for (const p of data.predictive) {
      if (!wanted.has(p.horizon)) continue
      if (byScenario && !wantedScenarios.has(p.scenario)) continue
      if (needTreatment && p.treatment !== activeTreatment) continue
      const key = JSON.stringify(p.stratum)
      let g = groups.get(key)
      if (!g) {
        g = { key, stratum: p.stratum, byArm: new Map() }
        groups.set(key, g)
      }
      const armKey = `${p.scenario}|${p.horizon}`
      const arm = g.byArm.get(armKey)
      if (arm) arm.pred.push(p)
      else g.byArm.set(armKey, { scenario: p.scenario, horizon: p.horizon, pred: [p] })
    }

    return [...groups.values()].map((g) => ({
      key: g.key,
      stratum: g.stratum,
      series: [...g.byArm.values()].map((a): OverlaySeries => ({
        key: `${a.scenario}|${a.horizon}`,
        color: byScenario
          ? (scenarioColors.get(a.scenario) ?? SCENARIO_REFERENCE)
          : horizonColor(a.horizon),
        pred: a.pred,
      })),
      obs: obsByKey.get(g.key) ?? [],
    }))
  }, [
    data,
    selectedHorizons,
    activeScenarios,
    needTreatment,
    activeTreatment,
    byScenario,
    scenarioColors,
  ])

  // Does any prediction extend past the observed window (a free-forward
  // forecast, or a scenario with a later `simulate { to }`)? Gates the window
  // toggle — with nothing beyond the data there's nothing to clip.
  const hasForecast = useMemo(() => {
    if (!data) return false
    let obsMax = -Infinity
    for (const o of data.observed)
      if (o.value != null && Number.isFinite(o.value)) obsMax = Math.max(obsMax, o.time)
    if (!Number.isFinite(obsMax)) return false
    return data.predictive.some((p) => p.time > obsMax)
  }, [data])

  // The stream's index dimensions are the "colour by" options; guard the stored
  // choice against a stream switch that doesn't have it.
  const indexDims = useMemo(() => data?.index_dims ?? [], [data])
  const activeColorBy = colorBy && indexDims.includes(colorBy) ? colorBy : null

  // Numeric time → Date via the fit-level calendar (null for a relative-time
  // fit); feeds the shared time-series x-axis and the residuals-vs-time view.
  const toDate = useMemo(() => dayToDate(run.data?.calendar), [run.data?.calendar])

  // Flat matched (observation, predicted-median) pairs across all strata, each
  // carrying its stratum + arm so the scatter/residual plots can colour and
  // group by any index dimension.
  const matched = useMemo(() => {
    const pts: {
      obs: number; pred: number; resid: number; time: number
      armKey: string; armColor: string; stratum: Record<string, string>
    }[] = []
    for (const st of strata) {
      const byArm = st.series.map((s) => ({
        key: s.key,
        color: s.color,
        medians: new Map(s.pred.map((p) => [p.time, p.q50])),
      }))
      for (const o of st.obs) {
        if (o.value == null || !Number.isFinite(o.value)) continue
        for (const a of byArm) {
          const pred = a.medians.get(o.time)
          if (pred == null || !Number.isFinite(pred)) continue
          pts.push({
            obs: o.value, pred, resid: o.value - pred, time: o.time,
            armKey: a.key, armColor: a.color, stratum: st.stratum,
          })
        }
      }
    }
    return pts
  }, [strata])

  // Level → colour for the selected index dimension.
  const levelColor = useMemo(() => {
    const m = new Map<string, string>()
    if (activeColorBy) {
      const levels = [...new Set(matched.map((p) => p.stratum[activeColorBy] ?? ''))].sort()
      levels.forEach((l, i) => m.set(l, LEVEL_PALETTE[i % LEVEL_PALETTE.length]!))
    }
    return m
  }, [activeColorBy, matched])

  // Points with the resolved fill colour + grouping key (index level, or arm).
  const points = useMemo(
    () =>
      matched.map((p): ScatterPoint => {
        const level = activeColorBy ? (p.stratum[activeColorBy] ?? '') : ''
        return {
          obs: p.obs, pred: p.pred, resid: p.resid, time: p.time,
          color: activeColorBy ? (levelColor.get(level) ?? NEUTRAL) : p.armColor,
          group: activeColorBy ? level : p.armKey,
          jx: Math.random() - 0.5,
        }
      }),
    [matched, activeColorBy, levelColor],
  )
  const scatterHasPoints = points.length > 0

  // Per-arm R² for the scatter corner (accuracy is a property of the prediction,
  // independent of how the points are coloured).
  const r2Lines = useMemo((): R2Line[] => {
    const byArm = new Map<string, { color: string; pts: { obs: number; pred: number }[]; scenario: string; horizon: string }>()
    for (const p of matched) {
      let e = byArm.get(p.armKey)
      if (!e) {
        const [sc, hz] = p.armKey.split('|')
        e = { color: p.armColor, pts: [], scenario: sc ?? '', horizon: hz ?? '' }
        byArm.set(p.armKey, e)
      }
      e.pts.push({ obs: p.obs, pred: p.pred })
    }
    return [...byArm.values()].map((e) => ({
      label: byScenario ? e.scenario : e.horizon,
      color: e.color,
      r2: rSquared(e.pts),
    }))
  }, [matched, byScenario])

  // Per-group mean residual → the coloured mean line/tick in the residual plot.
  const groupMeans = useMemo(() => {
    const g = new Map<string, { color: string; sum: number; n: number }>()
    for (const p of points) {
      let e = g.get(p.group)
      if (!e) {
        e = { color: p.color, sum: 0, n: 0 }
        g.set(p.group, e)
      }
      e.sum += p.resid
      e.n += 1
    }
    return [...g.entries()].map(([level, e]) => ({
      level,
      mean: e.sum / e.n,
      color: e.color,
    }))
  }, [points])

  // Canonical level ordering for each index dimension, from the fit's schema.
  // Age/village levels (a1_4, a5_9, a10_14 …) don't sort lexicographically, so
  // ordinal axes must use this order, not `sort()`.
  const dimLevels = useMemo(() => {
    const m = new Map<string, string[]>()
    for (const d of run.data?.dimensions ?? []) m.set(d.name, d.levels)
    return m
  }, [run.data])

  // PIT (probability integral transform): where each observation falls in its
  // own predictive CDF, interpolated through the stored quantiles. A calibrated
  // model gives uniform PITs; a U-shape is overconfident, a dome underconfident.
  const pit = useMemo(() => {
    const levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    const vals: number[] = []
    for (const st of strata) {
      // One predictive quantile set per (arm, time); index by time within arm.
      for (const s of st.series) {
        const byTime = new Map<number, PredictivePoint>()
        for (const p of s.pred) byTime.set(p.time, p)
        for (const o of st.obs) {
          if (o.value == null || !Number.isFinite(o.value)) continue
          const p = byTime.get(o.time)
          if (!p) continue
          const q = [p.q05, p.q25, p.q50, p.q75, p.q95]
          vals.push(pitOf(o.value, q, levels))
        }
      }
    }
    return vals
  }, [strata])

  // Available index dims for the by-index profile, and the resolved x / facet.
  const activeByX =
    byIndexX && indexDims.includes(byIndexX)
      ? byIndexX
      : (indexDims[0] ?? null)
  const activeByFacet =
    byIndexFacet && byIndexFacet !== activeByX && indexDims.includes(byIndexFacet)
      ? byIndexFacet
      : null

  // By-index profile: pick one index dim for x, marginalise time (and every
  // other dim) by mean, giving predicted-median vs observed per x-level, split
  // by an optional facet dim. Fully general over a model's index tensor.
  const byIndex = useMemo(() => {
    if (!activeByX || !data) return null
    // One record per predictive-median point (pred) and per observed point
    // (obs); the shared aggregator marginalises time + other dims by mean.
    const records: IndexRecord[] = []
    for (const st of strata) {
      for (const s of st.series)
        for (const p of s.pred) records.push({ stratum: st.stratum, pred: p.q50, obs: null })
      for (const o of st.obs)
        records.push({ stratum: st.stratum, pred: null, obs: o.value ?? null })
    }
    return buildByIndexProfile(records, activeByX, activeByFacet, dimLevels)
  }, [strata, activeByX, activeByFacet, dimLevels, data])

  // Legend arms: scenarios actually shown (in canonical order) when overlaying
  // scenarios, else the checked horizons.
  const legendArms = useMemo(() => {
    if (byScenario) {
      const shown = new Set<string>()
      for (const s of strata) for (const a of s.series) shown.add(a.key.split('|')[0]!)
      return scenarios
        .filter((sc) => shown.has(sc))
        .map((sc) => ({ label: sc, color: scenarioColors.get(sc) ?? SCENARIO_REFERENCE }))
    }
    return selectedHorizons.map((h) => ({ label: h, color: horizonColor(h) }))
  }, [byScenario, strata, scenarios, scenarioColors, selectedHorizons])

  if (run.isPending) {
    return (
      <Card className="overflow-hidden">
        <ForestSkeleton rows={2} />
      </Card>
    )
  }

  if (availableStreams.length === 0) {
    return (
      <NoPosteriorNotice
        status={run.data?.status}
        whenDone={{
          title: 'No predictive artifact',
          detail: (
            <>
              Run <span className="font-mono">camdl fit predict</span> for this fit
              to generate posterior-predictive checks.
            </>
          ),
        }}
      />
    )
  }

  return (
    <Card
      className={cn(
        'overflow-hidden transition-opacity',
        isPlaceholderData && 'opacity-60',
      )}
    >
      <div className="flex flex-col gap-2 px-3 py-2.5">
        {availableStreams.length > 1 && (
          <Segmented
            label="Stream"
            options={availableStreams}
            value={activeStream ?? ''}
            onChange={(v) => {
              setStream(v)
              setSelected(null)
              setSelectedScenarios(null)
              setTreatment(undefined)
              setColorBy(null)
              setByIndexX(null)
              setByIndexFacet(null)
              setHiddenLayers(new Set())
            }}
          />
        )}
        {byScenario && (
          <ScenarioChecks
            options={overlayOptions}
            selected={activeScenarios.filter((s) => s !== reference)}
            colorOf={(s) => scenarioColors.get(s) ?? SCENARIO_REFERENCE}
            onToggle={toggleScenario}
            onSetAll={setSelectedScenarios}
            pinned={reference}
          />
        )}
        {horizons.length > 1 && (
          <HorizonChecks
            options={horizons}
            selected={selectedHorizons}
            onToggle={toggleHorizon}
          />
        )}
        {needTreatment && (
          <Segmented
            label="Treatment"
            options={treatments}
            value={activeTreatment}
            onChange={setTreatment}
          />
        )}
        {legendArms.length > 0 && <Legend arms={legendArms} />}
      </div>

      {isPending && (
        <div className="border-t border-neutral-100">
          <ForestSkeleton rows={2} />
        </div>
      )}

      {isError && (
        <div className="border-t border-neutral-100">
          <MutedNotice
            bordered={false}
            title="Couldn't load the predictive check"
            detail="The backend returned an error for this stream. The predictive artifact may be missing or still being written."
          />
        </div>
      )}

      {data && strata.length === 0 && !isPending && (
        <div className="border-t border-neutral-100">
          <MutedNotice
            bordered={false}
            title="No predictive points"
            detail="This stream's predictive artifact has no points for the selected horizon(s)."
          />
        </div>
      )}

      {strata.length > 0 && (
        <>
          <div className="flex gap-4 border-t border-neutral-200 px-3 pt-2">
            {(
              [
                ['series', 'Time series'],
                ['scatter', 'Pred vs obs'],
                ['calibration', 'Calibration'],
                ...(indexDims.length > 0
                  ? ([['byindex', 'By index']] as const)
                  : []),
              ] as const
            ).map(([v, label]) => (
              <button
                key={v}
                type="button"
                onClick={() => setView(v)}
                aria-pressed={v === view}
                className={cn(
                  '-mb-px border-b-2 py-1.5 font-mono text-xs transition-colors',
                  v === view
                    ? 'border-neutral-900 text-neutral-900'
                    : 'border-transparent text-neutral-500 hover:text-neutral-800',
                )}
              >
                {label}
              </button>
            ))}
          </div>

          {view === 'series' && (
            <>
              <div className="flex flex-col gap-2 border-t border-neutral-100 px-3 py-2">
                <LayerChecks
                  hidden={hiddenLayers}
                  onToggle={(layer) =>
                    setHiddenLayers((prev) => {
                      const next = new Set(prev)
                      if (next.has(layer)) next.delete(layer)
                      else next.add(layer)
                      return next
                    })
                  }
                />
                {hasForecast && (
                  <>
                    <Segmented
                      label="Window"
                      options={['full', 'data']}
                      value={windowMode}
                      onChange={(v) => setWindowMode(v as 'data' | 'full')}
                    />
                    {windowMode === 'full' && (
                      <span className="font-mono text-[10px] text-neutral-400">
                        dashed rule = end of observed data · beyond it is forecast
                      </span>
                    )}
                  </>
                )}
              </div>
              {strata.map((s) => {
                const lbl = stratumLabel(s.stratum)
                const title = lbl || activeStream || 'observed'
                return (
                  <PredictivePanel
                    key={s.key}
                    title={title}
                    series={s.series}
                    observed={s.obs}
                    dense={s.series.length <= 2}
                    hiddenLayers={hiddenLayers}
                    toDate={toDate}
                    windowMode={windowMode}
                  />
                )
              })}
            </>
          )}

          {view === 'scatter' &&
            (scatterHasPoints ? (
              <>
                {indexDims.length > 0 && (
                  <div className="flex flex-col gap-2 border-t border-neutral-100 px-3 py-2">
                    <Segmented
                      label="Colour by"
                      options={['none', ...indexDims]}
                      value={activeColorBy ?? 'none'}
                      onChange={(v) => setColorBy(v === 'none' ? null : v)}
                    />
                    {activeColorBy && <LevelLegend levelColor={levelColor} />}
                  </div>
                )}
                <PredObsScatter points={points} r2Lines={r2Lines} />
                <div className="flex flex-col gap-2 border-t border-neutral-100 px-3 pt-2">
                  <Segmented
                    label="Residuals vs"
                    options={[
                      'predicted',
                      'time',
                      ...(activeColorBy ? [activeColorBy] : []),
                    ]}
                    value={
                      residualX === 'index' && activeColorBy
                        ? activeColorBy
                        : residualX === 'index'
                          ? 'predicted'
                          : residualX
                    }
                    onChange={(v) =>
                      setResidualX(
                        v === 'time'
                          ? 'time'
                          : v === activeColorBy
                            ? 'index'
                            : 'predicted',
                      )
                    }
                  />
                </div>
                <ResidualPlot
                  points={points}
                  groupMeans={groupMeans}
                  xMode={
                    residualX === 'index' && !activeColorBy ? 'predicted' : residualX
                  }
                  xLabel={activeColorBy ?? ''}
                  toDate={toDate}
                />
                {residualX === 'time' && (
                  <div className="px-3 pb-2 font-mono text-[10px] text-neutral-400">
                    bold line = {RESID_MA_WINDOW}-point moving average of the mean
                    residual · dashed = overall mean. A sustained excursion from
                    zero is temporal misspecification.
                  </div>
                )}
              </>
            ) : (
              <MutedNotice
                bordered={false}
                title="No matched points"
                detail="No observations line up in time with the predicted series, so there's nothing to scatter against."
              />
            ))}

          {view === 'calibration' &&
            (pit.length > 0 ? (
              <>
                <PitPlot pit={pit} />
                <div className="px-3 pb-2 font-mono text-[10px] text-neutral-400">
                  Uniform (dashed) = calibrated · U-shape = overconfident · dome =
                  underconfident · slope = biased. n = {pit.length}.
                </div>
              </>
            ) : (
              <MutedNotice
                bordered={false}
                title="No matched points"
                detail="No observations line up in time with the predictive quantiles, so there's no PIT to compute."
              />
            ))}

          {view === 'byindex' &&
            (byIndex &&
            byIndex.facets.some((f) =>
              f.points.some((p) => p.pred != null || p.obs != null),
            ) ? (
              <>
                <div className="flex flex-col gap-2 border-t border-neutral-100 px-3 py-2">
                  <Segmented
                    label="Index (x)"
                    options={indexDims}
                    value={activeByX ?? ''}
                    onChange={setByIndexX}
                  />
                  {indexDims.length > 1 && (
                    <Segmented
                      label="Facet by"
                      options={['none', ...indexDims.filter((d) => d !== activeByX)]}
                      value={activeByFacet ?? 'none'}
                      onChange={(v) => setByIndexFacet(v === 'none' ? null : v)}
                    />
                  )}
                  {activeByFacet && (
                    <LevelLegend
                      levelColor={
                        new Map(byIndex.facets.map((f) => [f.level, f.color]))
                      }
                    />
                  )}
                  <span className="font-mono text-[10px] text-neutral-400">
                    line = predicted median · ○ = observed · marginalised over time
                  </span>
                </div>
                <ByIndexPlot
                  profile={byIndex}
                  xLabel={activeByX ?? ''}
                  yLabel={`mean ${activeStream ?? 'value'}`}
                />
              </>
            ) : (
              <MutedNotice
                bordered={false}
                title="Nothing to profile"
                detail="This stream has no index dimension with matched predicted/observed values to profile."
              />
            ))}
        </>
      )}
    </Card>
  )
}
