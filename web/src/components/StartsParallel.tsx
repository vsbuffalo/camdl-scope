import { useEffect, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { MleParam, MleRestart } from '@/api/client'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { fmtTick, fmtValue } from '@/lib/format'
import { ink, token } from '@/lib/theme'

const BEST = ink('mark-red')
const LINE = token('ink')
const AXIS = token('muted')
const RULE = token('line-strong')
const MONO = 'var(--font-mono)'

/**
 * Where each start's optimum sits, parameter by parameter: one vertical axis
 * per parameter running from its lower bound (bottom) to its upper bound (top),
 * and one line per start through its value on every axis. Starts that agree on
 * a parameter pinch to one point on that axis; disagreement is a vertical
 * spread, read as a share of the search range. Lines that stay parallel between
 * two axes are starts trading one parameter against another along a ridge of
 * equal likelihood.
 *
 * A line's darkness is its log-likelihood: the best start is red, starts close
 * to it in log-likelihood are dark, and starts far below it fade, so a few
 * starts stuck in a poor basin do not hide the ones that matter. A parameter
 * without bounds is scaled to the range of the starts' values instead.
 */
export function StartsParallel({
  starts,
  params,
}: {
  /** Finished, non-failed starts. */
  starts: MleRestart[]
  params: MleParam[]
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
    if (!el || width <= 0 || starts.length === 0) return

    const bestLL = Math.max(...starts.map((s) => s.loglik))
    const best = starts.find((s) => s.loglik === bestLL)!
    // Each axis's range: the parameter's bounds, else the starts' own spread.
    const ranges = params.map((p): [number, number, boolean] => {
      if (p.bounds && p.bounds[1] > p.bounds[0]) return [p.bounds[0], p.bounds[1], true]
      const vs = starts.map((s) => s.values?.[p.name]).filter((v): v is number => v != null)
      const lo = Math.min(...vs)
      const hi = Math.max(...vs)
      return hi > lo ? [lo, hi, false] : [lo - 1, hi + 1, false]
    })
    // Darkness from the log-likelihood drop: full within 0.1 of the best,
    // fading on a log scale to faint at a drop of 100.
    const opacity = (drop: number) =>
      drop <= 0.1 ? 0.9 : Math.max(0.12, 0.9 - (0.78 * Math.log10(drop / 0.1)) / 3)

    type Pt = { start: string; param: string; y: number; v: number; drop: number; best: boolean }
    const rows: Pt[] = []
    // Worst first, so better lines draw on top.
    for (const s of [...starts].sort((a, b) => a.loglik - b.loglik)) {
      params.forEach((p, i) => {
        const v = s.values?.[p.name]
        if (v == null) return
        const [lo, hi] = ranges[i]!
        rows.push({
          start: `c${s.chain}`,
          param: p.name,
          y: (v - lo) / (hi - lo),
          v,
          drop: bestLL - s.loglik,
          best: s === best,
        })
      })
    }
    const others = rows.filter((r) => !r.best)
    const bestRows = rows.filter((r) => r.best)
    const names = params.map((p) => p.name)
    const tip = (r: Pt) =>
      `${r.start} · ${r.param} = ${fmtValue(r.v)}\n` +
      (r.best ? 'best start' : `logL ${fmtValue(r.drop)} below the best`)
    const ends = params.flatMap((p, i) => {
      const [lo, hi, bounded] = ranges[i]!
      return [
        { param: p.name, y: 0, label: bounded ? fmtTick(lo) : `${fmtTick(lo)} (min)` },
        { param: p.name, y: 1, label: bounded ? fmtTick(hi) : `${fmtTick(hi)} (max)` },
      ]
    })

    const node = Plot.plot({
      width,
      height: 240,
      marginTop: 22,
      marginBottom: 22,
      marginLeft: 28,
      marginRight: 28,
      style: { background: 'transparent', color: AXIS, fontSize: '10px', fontFamily: MONO },
      x: { type: 'point', domain: names, axis: 'top', label: null, tickSize: 0, padding: 0.3 },
      y: { domain: [0, 1], axis: null },
      opacity: { type: 'identity' },
      marks: [
        Plot.ruleX(names, { stroke: RULE, strokeWidth: 1 }),
        Plot.line(others, {
          x: 'param',
          y: 'y',
          z: 'start',
          stroke: LINE,
          strokeOpacity: (r: Pt) => opacity(r.drop),
          strokeWidth: 1.25,
        }),
        Plot.dot(others, {
          x: 'param',
          y: 'y',
          r: 2.5,
          fill: LINE,
          fillOpacity: (r: Pt) => opacity(r.drop),
          title: tip,
        }),
        Plot.line(bestRows, { x: 'param', y: 'y', stroke: BEST, strokeWidth: 2 }),
        Plot.dot(bestRows, { x: 'param', y: 'y', r: 3.5, fill: BEST, title: tip }),
        Plot.text(ends, {
          x: 'param',
          y: 'y',
          text: 'label',
          dx: 8,
          textAnchor: 'start',
          fill: AXIS,
        }),
      ],
    })
    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [starts, params, width])

  return (
    <div className="group/fig relative">
      <div ref={ref} className="w-full min-w-0" role="img" aria-label="each start's optimum per parameter" />
      <PlotDownloadButton targetRef={ref} name="starts-parameters" />
    </div>
  )
}
