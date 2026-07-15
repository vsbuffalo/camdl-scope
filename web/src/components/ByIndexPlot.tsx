import * as Plot from '@observablehq/plot'
import { Figure } from '@/components/Figure'
import { fmtTick } from '@/lib/format'
import type { ByIndexProfile } from '@/lib/byindex'

const AXIS = '#737373'
const MONO = 'var(--font-mono)'

/**
 * A by-index profile: the chosen index dimension on x (in schema order), the
 * value marginalised over time by mean, drawn as a line + filled dot per facet
 * level. Where an `obs` overlay exists (predictive) it's a hollow ring; a
 * derived quantity has none. Fully general — no dimension or value name is
 * hard-coded.
 */
export function ByIndexPlot({
  profile,
  xLabel,
  yLabel,
}: {
  profile: ByIndexProfile
  xLabel: string
  yLabel: string
}) {
  return (
    <Figure
      name="by-index"
      aria="by-index profile"
      deps={[profile, xLabel, yLabel]}
      render={(el, width) => {
        const { xLevels, facets } = profile
        if (xLevels.length === 0) {
          el.replaceChildren()
          return
        }
        const predPts = facets.flatMap((f) =>
          f.points.filter((p) => p.pred != null).map((p) => ({ xi: p.xi, y: p.pred as number, color: f.color })),
        )
        const obsPts = facets.flatMap((f) =>
          f.points.filter((p) => p.obs != null).map((p) => ({ xi: p.xi, y: p.obs as number, color: f.color })),
        )
        const lines = facets.map((f) => ({
          color: f.color,
          pts: f.points.filter((p) => p.pred != null).map((p) => ({ xi: p.xi, y: p.pred as number })),
        }))
        let lo = Infinity
        let hi = -Infinity
        for (const p of predPts) {
          lo = Math.min(lo, p.y)
          hi = Math.max(hi, p.y)
        }
        for (const p of obsPts) {
          lo = Math.min(lo, p.y)
          hi = Math.max(hi, p.y)
        }
        if (!Number.isFinite(lo)) {
          lo = 0
          hi = 1
        }
        const pad = (hi - lo) * 0.08 || 1
        const node = Plot.plot({
          width,
          height: Math.min(Math.round(width * 0.55), 320),
          marginTop: 10,
          marginBottom: 40,
          marginLeft: 52,
          marginRight: 12,
          style: { background: 'transparent', color: AXIS, fontSize: '10px', fontFamily: MONO },
          x: {
            domain: [-0.4, xLevels.length - 0.6],
            label: `${xLabel} →`,
            ticks: xLevels.map((_, i) => i),
            tickFormat: (i: number) => xLevels[Math.round(i)] ?? '',
            tickSize: 0,
          },
          y: {
            label: `↑ ${yLabel}`,
            labelAnchor: 'center',
            domain: [lo - pad, hi + pad],
            grid: true,
            tickFormat: (d: number) => fmtTick(d),
          },
          marks: [
            ...lines.map((l) =>
              Plot.line(l.pts, { x: 'xi', y: 'y', stroke: l.color, strokeWidth: 1.5, strokeOpacity: 0.85 }),
            ),
            Plot.dot(predPts, { x: 'xi', y: 'y', fill: 'color', r: 3.5, stroke: 'white', strokeWidth: 0.5 }),
            Plot.dot(obsPts, { x: 'xi', y: 'y', stroke: 'color', fill: 'white', r: 3.5, strokeWidth: 1.5 }),
          ],
        })
        el.replaceChildren(node)
      }}
    />
  )
}

/** Swatches mapping each facet level to its colour (the by-index legend). */
export function LevelLegend({ levelColor }: { levelColor: Map<string, string> }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-neutral-400">
      {[...levelColor].map(([lvl, color]) => (
        <span key={lvl} className="flex items-center gap-1">
          <span
            className="inline-block size-2 rounded-full"
            style={{ background: color }}
          />
          {lvl || '∅'}
        </span>
      ))}
    </div>
  )
}
