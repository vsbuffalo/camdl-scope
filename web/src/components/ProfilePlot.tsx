import { useEffect, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { ProfileResponse } from '@/api/client'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { fmtTick } from '@/lib/format'
import { ink, token } from '@/lib/theme'

const FRAME = token('line') // neutral-200 — hairline frame
const AXIS = token('muted') // neutral-500 — tick labels
const LINE = ink('mark-blue-deep') // blue-700 — the profile curve
const MLE = ink('mark-red') // red-600 — the MLE marker
const THRESH = ink('mark-yellow') // yellow-700 — the CI loglik threshold
const BAND = ink('mark-emerald') // emerald-500 — faint CI wash
const MONO = 'var(--font-mono)'

/**
 * A profile-likelihood curve: optimized log-likelihood (y) against the profiled
 * value (x). The MLE is the peak (red rule); the 95% CI is the band where the
 * curve stays within `ci_drop` of the max, drawn as a faint wash bracketed by
 * the threshold rule. An open bound (grid didn't bracket that side) fades to the
 * plot edge. Self-measuring so it draws on first paint (incl. headless capture).
 *
 * While a profile runs, the curve joins only *complete* grid values (every
 * start finished). A partial value is drawn hollow and left off the curve: its
 * log-likelihood is the max over the starts so far, a lower bound that can
 * only rise. A value with no finished start is an open triangle on the x axis.
 */
export function ProfilePlot({ data }: { data: ProfileResponse }) {
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

    // Drop camdl's failed/infeasible sentinel (≈ −1e100) cells — plotting them
    // would blow the y-axis out to −1e100 and flatten the real curve to nothing.
    const startsText = (n: number, of: number | null | undefined) =>
      of != null ? `${n} of ${of} starts` : `${n} start${n === 1 ? '' : 's'}`
    const pts = data.points
      .filter((p) => p.loglik > -1e99)
      .map((p) => ({
        value: p.coords[0]!,
        loglik: p.loglik,
        complete: p.complete ?? true,
        tip:
          `${data.params[0]}=${fmtTick(p.coords[0]!)}\nlogL ${p.loglik.toFixed(2)}` +
          `\n${startsText(p.n_starts, p.n_expected)}` +
          (p.complete === false ? ' (partial: a lower bound)' : '') +
          (p.loglik_spread != null ? `\nspread across starts ${p.loglik_spread.toFixed(2)}` : ''),
      }))
    const complete = pts.filter((p) => p.complete)
    const partial = pts.filter((p) => !p.complete)
    const pending = (data.pending ?? []).map((p) => ({
      value: p.coords[0]!,
      tip: `${data.params[0]}=${fmtTick(p.coords[0]!)}\n0 of ${p.n_expected} starts finished`,
    }))
    const hasMle = data.mle_loglik != null && data.mle_coords.length > 0
    const mleValue = data.mle_coords[0] ?? 0
    const mleLL = data.mle_loglik ?? 0
    const param = data.params[0]!
    const xs = [...pts.map((p) => p.value), ...pending.map((p) => p.value)]
    const xLo = Math.min(...xs)
    const xHi = Math.max(...xs)
    // Pad the x-domain so an edge point (often the MLE) isn't clipped by the frame.
    const pad = (xHi - xLo || 1) * 0.04
    const lls = pts.map((p) => p.loglik)
    const yLo = Math.min(...lls)
    const yHi = Math.max(...lls)
    const yPad = (yHi - yLo || 1) * 0.08
    const thresh = mleLL - data.ci_drop
    // CI band edges; an open side (null) runs to the last data point, reading as
    // "bounded by the grid, open beyond".
    const bandLo = data.ci_lo ?? xLo
    const bandHi = data.ci_hi ?? xHi

    const marks: Plot.Markish[] = [
      ...(hasMle
        ? [
            Plot.rectX([{ x1: bandLo, x2: bandHi }], {
              x1: 'x1',
              x2: 'x2',
              fill: BAND,
              fillOpacity: 0.08,
            }),
            // The loglik threshold that defines the CI; dashed reference.
            Plot.ruleY([thresh], {
              stroke: THRESH,
              strokeWidth: 0.75,
              strokeDasharray: '3,2',
            }),
            // The MLE — vertical marker.
            Plot.ruleX([mleValue], { stroke: MLE, strokeWidth: 0.75 }),
          ]
        : []),
      Plot.line(complete, {
        x: 'value',
        y: 'loglik',
        stroke: LINE,
        strokeWidth: 1.25,
        curve: 'monotone-x',
      }),
      Plot.dot(complete, {
        x: 'value',
        y: 'loglik',
        fill: LINE,
        r: 2.5,
        stroke: token('surface'),
        strokeWidth: 0.5,
        title: 'tip',
      }),
      // Partial values: hollow, off the curve.
      Plot.dot(partial, {
        x: 'value',
        y: 'loglik',
        fill: token('surface'),
        r: 3,
        stroke: LINE,
        strokeWidth: 1.25,
        title: 'tip',
      }),
      // Pending values: an open triangle on the x axis.
      Plot.dot(pending, {
        x: 'value',
        y: yLo - yPad,
        symbol: 'triangle',
        fill: 'none',
        r: 4,
        stroke: AXIS,
        strokeWidth: 1,
        title: 'tip',
      }),
      ...(hasMle
        ? [
            Plot.dot([{ value: mleValue, loglik: mleLL }], {
              x: 'value',
              y: 'loglik',
              fill: MLE,
              r: 4,
              stroke: token('surface'),
              strokeWidth: 0.75,
            }),
          ]
        : []),
      Plot.frame({ stroke: FRAME, strokeWidth: 0.5 }),
    ]

    const node = Plot.plot({
      width,
      height: Math.max(220, Math.round(width * 0.5)),
      marginTop: 10,
      marginBottom: 34,
      marginLeft: 56,
      marginRight: 12,
      style: {
        background: 'transparent',
        color: AXIS,
        fontSize: '10px',
        fontFamily: MONO,
      },
      x: {
        label: `${param} →`,
        labelAnchor: 'center',
        domain: [xLo - pad, xHi + pad],
        ticks: 6,
        tickSize: 3,
        tickFormat: (d: number) => fmtTick(d),
      },
      y: {
        label: '↑ profile log-likelihood',
        labelAnchor: 'center',
        domain: [yLo - yPad, yHi + yPad],
        ticks: 5,
        tickSize: 3,
        tickFormat: (d: number) => fmtTick(d),
      },
      marks,
    })

    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [data, width])

  return (
    <div className="group/fig relative">
      <div
        ref={ref}
        className="w-full min-w-0"
        role="img"
        aria-label={`profile likelihood of ${data.params[0]}`}
      />
      <PlotDownloadButton targetRef={ref} name={`profile-${data.params[0]}`} />
    </div>
  )
}
