import { useEffect, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { MleRestart, MleSearch } from '@/api/client'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { fmtValue } from '@/lib/format'
import { ink, token } from '@/lib/theme'

const LEAD = ink('mark-red')
const LINE = token('ink')
const LIVE = ink('mark-blue-deep')
const WARN = token('warn')
const AXIS = token('muted')
const FRAME = token('line')
const MONO = 'var(--font-mono)'

/** Gap ticks on the symmetric-log axis: 0, then one per decade below. */
const GAP_TICKS = [0, -0.1, -1, -10, -100, -1000, -1e4, -1e5]

/**
 * How each optimiser search climbs: its best log-likelihood so far against its
 * evaluation count, from camdl's live `opt_trace.tsv`. The y axis is the gap to
 * the best value any search has found, on a symmetric-log scale, so the climb
 * from a poor start and the last fraction of a unit near the top are both
 * readable. A line still rising at its last evaluation was cut off by its
 * budget; a line that flattened converged in practice, whatever status NLopt
 * gave it. Blue lines are searches still running; red is the leading search; a
 * hollow amber end marks a search that stopped at `max_evals`. A search whose
 * first evaluation was not finite starts at an open diamond, where it first
 * found a finite value; one with none yet has no line and is named below.
 */
export function StartsProgress({
  searches,
  restarts,
  live,
}: {
  searches: MleSearch[]
  restarts: MleRestart[]
  /** The run is still going (a fresh heartbeat). */
  live: boolean
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

  const finite = searches.filter((s) => s.best_loglik != null)
  const none = searches.filter((s) => s.best_loglik == null)

  useEffect(() => {
    const el = ref.current
    if (!el || width <= 0 || finite.length === 0) return
    const top = Math.max(...finite.map((s) => s.best_loglik!))
    const lead = finite.find((s) => s.best_loglik === top)!
    const maxeval = new Set(restarts.filter((r) => r.hit_maxeval).map((r) => r.chain))

    type Pt = { chain: string; eval: number; gap: number; best: number }
    const lineOf = (s: MleSearch): Pt[] =>
      s.trace
        .filter((p): p is [number, number] => p[1] != null)
        .map(([e, b]) => ({ chain: `c${s.chain}`, eval: e, gap: b - top, best: b }))
    const strokeOf = (s: MleSearch) => (s === lead ? LEAD : !s.finished && live ? LIVE : LINE)
    const ends = finite.map((s) => {
      const pts = lineOf(s)
      return { ...pts[pts.length - 1]!, s }
    })
    const starts = finite
      .filter((s) => s.started_nonfinite)
      .map((s) => lineOf(s)[0]!)
      .filter(Boolean)
    const tip = (d: Pt) =>
      `${d.chain} · eval ${d.eval.toLocaleString()}\nbest logL ${fmtValue(d.best)}` +
      (d.gap < 0 ? ` (${fmtValue(-d.gap)} below the top)` : ' (the top)')
    const minGap = Math.min(...finite.flatMap((s) => lineOf(s).map((p) => p.gap)))

    const node = Plot.plot({
      width,
      height: 240,
      marginLeft: 52,
      marginRight: 30,
      marginTop: 10,
      marginBottom: 34,
      style: { background: 'transparent', color: AXIS, fontSize: '10px', fontFamily: MONO },
      x: { label: 'evaluations →', labelAnchor: 'center', tickSize: 3, ticks: 6 },
      y: {
        type: 'symlog',
        constant: 0.1,
        label: '↑ best logL − top',
        labelAnchor: 'center',
        domain: [Math.min(minGap, -0.1), 0.05],
        ticks: GAP_TICKS.filter((t) => t >= Math.min(minGap, -0.1)),
        tickFormat: (d: number) => (d === 0 ? '0' : fmtValue(d)),
        tickSize: 3,
      },
      marks: [
        Plot.ruleY([0], { stroke: FRAME }),
        ...finite
          .filter((s) => s !== lead)
          .map((s) =>
            Plot.line(lineOf(s), {
              x: 'eval',
              y: 'gap',
              stroke: strokeOf(s),
              strokeOpacity: 0.75,
              strokeWidth: 1.25,
              curve: 'step-after',
            }),
          ),
        Plot.line(lineOf(lead), { x: 'eval', y: 'gap', stroke: LEAD, strokeWidth: 2, curve: 'step-after' }),
        Plot.dot(starts, { x: 'eval', y: 'gap', symbol: 'diamond', r: 4, fill: 'none', stroke: AXIS, title: tip }),
        Plot.dot(ends, {
          x: 'eval',
          y: 'gap',
          r: 3.5,
          fill: (d: { s: MleSearch }) =>
            maxeval.has(d.s.chain) ? token('surface') : strokeOf(d.s),
          stroke: (d: { s: MleSearch }) => (maxeval.has(d.s.chain) ? WARN : strokeOf(d.s)),
          strokeWidth: 1.5,
          title: tip,
        }),
        Plot.text(ends, {
          x: 'eval',
          y: 'gap',
          text: 'chain',
          dx: 6,
          textAnchor: 'start',
          fill: AXIS,
        }),
        Plot.frame({ stroke: FRAME, strokeWidth: 0.5 }),
      ],
    })
    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [finite, restarts, live, width])

  if (searches.length === 0) return null
  return (
    <div>
      {finite.length > 0 ? (
        <div className="group/fig relative">
          <div ref={ref} className="w-full min-w-0" role="img" aria-label="best log-likelihood by evaluation, per search" />
          <PlotDownloadButton targetRef={ref} name="search-progress" />
        </div>
      ) : (
        <p className="py-2 font-mono text-[11px] text-faint">
          no search has a finite log-likelihood yet
        </p>
      )}
      {none.length > 0 && finite.length > 0 && (
        <p className="mt-1 font-mono text-[10px] text-warn">
          no finite log-likelihood yet: {none.map((s) => `c${s.chain} (${s.evals} evals)`).join(', ')}
        </p>
      )}
    </div>
  )
}
