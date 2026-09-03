/**
 * Hover readout for a ribbon panel — the crosshair and tip shared by the
 * Predictive check and the Quantities trajectories.
 *
 * Why a *band* readout rather than a nearest-point one. At any x these panels
 * show five numbers, not one; Plot's default `tip` would name the single
 * nearest datum and hide the other four, which is the opposite of the question
 * being asked ("what does the model actually predict here?"). So the pointer
 * snaps in x only, and the tip reports every quantile drawn at that time —
 * plus, where there are observations, the observed value and whether it fell
 * outside the 90% band. That last line is what makes an outlier readable: the
 * eye can see a point sits above the ribbon, but not by how much.
 *
 * Shared for the same reason `BandLayers` is: the *substrate* is genuinely
 * common — a time, a colour, and q05…q95 — while each tab composes its own
 * panel around it. This exports marks, not a component, so neither caller has
 * to give up control of its own `Plot.plot` call.
 *
 * Cost is one extra mark per panel. The pointer transform does an x-lookup on
 * pointermove within the hovered plot only, so it does not scale with the
 * number of panels on screen, and a few hundred points per series is far below
 * anything measurable — which is why this is always on rather than a toggle.
 */
import * as Plot from '@observablehq/plot'

/** One time step of a banded series. Quantiles are optional so a partially
 *  written artifact (or a censored scalar) degrades to the layers it has. */
export interface BandPoint {
  time: number
  q05?: number | null
  q25?: number | null
  q50?: number | null
  q75?: number | null
  q95?: number | null
}

/** One overlaid arm: a scenario in Quantities, an arm in Predictive. */
export interface BandSeries {
  label: string
  color: string
  points: BandPoint[]
}

/** An observed value to report beside the prediction (Predictive only). */
export interface ObservedPoint {
  time: number
  value?: number | null
}

/** Which band layers are currently drawn. The tip quotes only what is on
 *  screen — quoting a hidden q05 would report a number the panel does not
 *  show, and the y-axis is not even scaled to it. */
export interface DrawnLayers {
  median: boolean
  p50: boolean
  p90: boolean
}

export interface BandTipOptions {
  series: BandSeries[]
  observed?: ObservedPoint[]
  /** The panel's x accessor — numeric model time, or a Date under a calendar. */
  xOf: (d: { time: number }) => Date | number
  /** How the hovered time is spelled in the tip's first line. */
  fmtX: (time: number) => string
  /** How a value is spelled. */
  fmtY: (value: number) => string
  layers: DrawnLayers
  /** Crosshair colour; defaults to a hairline grey. */
  rule?: string
  fontFamily?: string
}

const num = (v: number | null | undefined): number | null =>
  v == null || !Number.isFinite(v) ? null : v

/** Pad a label so the tip's columns line up in a monospace face. */
const pad = (s: string, w: number) => s.padEnd(w, ' ')

/**
 * One row per distinct time across every series and the observations, carrying
 * the readout for that time. `y` positions the tip near the data rather than
 * at the axis floor.
 */
function rows(o: BandTipOptions) {
  const times = new Set<number>()
  for (const s of o.series) for (const p of s.points) times.add(p.time)
  for (const ob of o.observed ?? []) times.add(ob.time)

  const byTime = o.series.map(
    (s) => [s, new Map(s.points.map((p) => [p.time, p]))] as const,
  )
  const obsByTime = new Map((o.observed ?? []).map((ob) => [ob.time, ob]))
  const solo = o.series.length === 1
  // One label column across every arm AND the "observed" row, so the values
  // line up down the tip in a monospace face.
  const labelW = Math.max(
    9,
    ...o.series.map((s) => s.label.length + 2),
  )

  return [...times]
    .sort((a, b) => a - b)
    .map((time) => {
      const lines: string[] = [o.fmtX(time)]
      let anchor: number | null = null

      const ob = obsByTime.get(time)
      const obsV = num(ob?.value)
      if (obsV != null) {
        anchor = obsV
        lines.push(`${pad('observed', labelW)}${o.fmtY(obsV)}`)
      }

      /** Where the observation sits relative to one arm's 90% band. Reported
       *  per arm rather than once: with several arms overlaid, a bare "outside
       *  90%" does not say WHICH band was missed, and they can disagree. */
      const flag = (p: BandPoint): string => {
        if (obsV == null || !o.layers.p90) return ''
        const lo = num(p.q05)
        const hi = num(p.q95)
        if (lo == null || hi == null) return ''
        if (obsV > hi) return '  above 90%'
        if (obsV < lo) return '  below 90%'
        return ''
      }

      for (const [s, index] of byTime) {
        const p = index.get(time)
        if (!p) continue
        const q05 = num(p.q05)
        const q25 = num(p.q25)
        const q50 = num(p.q50)
        const q75 = num(p.q75)
        const q95 = num(p.q95)
        if (anchor == null) anchor = q50 ?? q95 ?? q05
        if (solo) {
          // One arm: room for the full ladder, outermost first so it reads
          // down the panel the way the ribbon is stacked. The band edge the
          // observation missed carries the flag, so the comparison is right
          // next to the two numbers it is between.
          const out = flag(p)
          if (o.layers.p90 && q95 != null)
            lines.push(`${pad('q95', labelW)}${o.fmtY(q95)}${out === '  above 90%' ? out : ''}`)
          if (o.layers.p50 && q75 != null) lines.push(`${pad('q75', labelW)}${o.fmtY(q75)}`)
          if (o.layers.median && q50 != null)
            lines.push(`${pad('median', labelW)}${o.fmtY(q50)}`)
          if (o.layers.p50 && q25 != null) lines.push(`${pad('q25', labelW)}${o.fmtY(q25)}`)
          if (o.layers.p90 && q05 != null)
            lines.push(`${pad('q05', labelW)}${o.fmtY(q05)}${out === '  below 90%' ? out : ''}`)
        } else {
          // Several arms: one line each, or the tip outgrows the panel.
          const mid = o.layers.median && q50 != null ? o.fmtY(q50) : '—'
          const span =
            o.layers.p90 && q05 != null && q95 != null
              ? `  [${o.fmtY(q05)}–${o.fmtY(q95)}]`
              : o.layers.p50 && q25 != null && q75 != null
                ? `  [${o.fmtY(q25)}–${o.fmtY(q75)}]`
                : ''
          lines.push(`${pad(s.label, labelW)}${mid}${span}${flag(p)}`)
        }
      }
      return { time, y: anchor ?? 0, title: lines.join('\n') }
    })
    .filter((r) => r.title.includes('\n')) // a bare timestamp is not a readout
}

/**
 * Crosshair + tip marks for a ribbon panel. Push these LAST so the tip paints
 * over the ribbons; an empty array when there is nothing to report.
 */
export function bandTipMarks(o: BandTipOptions): Plot.Markish[] {
  const data = rows(o)
  if (data.length === 0) return []
  const x = (d: { time: number }) => o.xOf(d)
  return [
    Plot.ruleX(data, Plot.pointerX({ x, stroke: o.rule ?? '#a3a3a3', strokeWidth: 0.75 })),
    Plot.tip(
      data,
      Plot.pointerX({
        x,
        y: 'y',
        title: 'title',
        fontFamily: o.fontFamily ?? 'var(--font-mono)',
        fontSize: 10,
        lineHeight: 1.2,
      }),
    ),
  ]
}
