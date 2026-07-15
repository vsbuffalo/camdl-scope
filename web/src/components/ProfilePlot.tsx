import { useEffect, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { ProfileResponse } from '@/api/client'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { fmtTick } from '@/lib/format'

const FRAME = '#e5e5e5' // neutral-200 — hairline frame
const AXIS = '#737373' // neutral-500 — tick labels
const LINE = '#1d4ed8' // blue-700 — the profile curve
const MLE = '#dc2626' // red-600 — the MLE marker
const THRESH = '#a16207' // yellow-700 — the CI loglik threshold
const BAND = '#10b981' // emerald-500 — faint CI wash
const MONO = 'var(--font-mono)'

/**
 * A profile-likelihood curve: optimized log-likelihood (y) against the profiled
 * value (x). The MLE is the peak (red rule); the 95% CI is the band where the
 * curve stays within `ci_drop` of the max, drawn as a faint wash bracketed by
 * the threshold rule. An open bound (grid didn't bracket that side) fades to the
 * plot edge. Self-measuring so it draws on first paint (incl. headless capture).
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
    const pts = data.points
      .filter((p) => p.loglik > -1e99)
      .map((p) => ({ value: p.coords[0]!, loglik: p.loglik }))
    const mleValue = data.mle_coords[0]!
    const param = data.params[0]!
    const xs = pts.map((p) => p.value)
    const xLo = Math.min(...xs)
    const xHi = Math.max(...xs)
    // Pad the x-domain so an edge point (often the MLE) isn't clipped by the frame.
    const pad = (xHi - xLo || 1) * 0.04
    const thresh = data.mle_loglik - data.ci_drop
    // CI band edges; an open side (null) runs to the last data point, reading as
    // "bounded by the grid, open beyond".
    const bandLo = data.ci_lo ?? xLo
    const bandHi = data.ci_hi ?? xHi

    const marks: Plot.Markish[] = [
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
      // The MLE — vertical marker + a dot on the curve.
      Plot.ruleX([mleValue], { stroke: MLE, strokeWidth: 0.75 }),
      Plot.line(pts, {
        x: 'value',
        y: 'loglik',
        stroke: LINE,
        strokeWidth: 1.25,
        curve: 'monotone-x',
      }),
      Plot.dot(pts, {
        x: 'value',
        y: 'loglik',
        fill: LINE,
        r: 2.5,
        stroke: 'white',
        strokeWidth: 0.5,
      }),
      Plot.dot([{ value: mleValue, loglik: data.mle_loglik }], {
        x: 'value',
        y: 'loglik',
        fill: MLE,
        r: 4,
        stroke: 'white',
        strokeWidth: 0.75,
      }),
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
