import { useEffect, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { ProfileResponse } from '@/api/client'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { cn } from '@/lib/utils'

const AXIS = '#737373' // neutral-500 — tick labels
const MLE = '#dc2626' // red-600 — the MLE cell outline
const SELECTED = '#2563eb' // blue-600 — the clicked cell outline
const FAILED = '#e5e5e5' // neutral-200 — a failed/infeasible cell (sentinel loglik)
const MONO = 'var(--font-mono)'

/** The bits of an Observable Plot band scale we hit-test clicks against. */
type BandScale = { apply: (v: number) => number; bandwidth: number }
// Δlog-likelihood floor for the colour ramp: cells worse than this pin to the
// darkest, so the ramp spends its contrast on the near-MLE structure (the peak
// and its confidence region) instead of being flattened by far, very-negative
// cells. ~4× the 2D 95% drop (≈3.0).
const DL_FLOOR = 12
// camdl writes a huge-negative sentinel (≈ −1e100) for a grid cell whose
// optimization failed / was infeasible. Those aren't real likelihoods, so we
// draw them as a distinct "failed" tile rather than the ramp's darkest genuine
// value (which would read as a real low-likelihood region).
const SENTINEL = -1e99

/** Grid-axis label: enough significant figures to keep log-spaced values
 *  distinct (fmtTick collapses e.g. 0.005 / 0.0073 / 0.01 all to "0.01"). */
const fmtGrid = (v: number): string => Number(v.toPrecision(3)).toString()

/**
 * A 2D profile likelihood as a grid-mesh surface: one filled cell per evaluated
 * `(param_x, param_y)` grid point, coloured by its optimized log-likelihood
 * relative to the current best (Δlogℓ ≤ 0, brightest at the peak). Missing cells
 * are blank, so a *running* profile shows the mesh filling in as camdl lands each
 * point. The MLE cell is outlined red; cells inside the 95% joint region
 * (Δlogℓ ≥ −`ci_drop`, ½·χ²₂ ≈ 3.0) are outlined white. Categorical (band) axes —
 * the grid values are discrete and log-spaced, so equal-width tiles read as a
 * clean matrix rather than a distorted scatter. Self-measuring so it draws on
 * first paint (incl. headless capture).
 */
type ColorMode = 'delta' | 'absolute'

export function ProfileSurface({ data }: { data: ProfileResponse }) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  // Colour by Δ-from-peak (near-peak structure) or absolute logL (compare the
  // surface against the log-likelihood of *other* fits/models).
  const [mode, setMode] = useState<ColorMode>('delta')
  // The clicked cell (by grid coords) — its nuisance MLE + logL show in a readout
  // below. Kept as coords (not the point object) so it survives a live poll.
  const [selected, setSelected] = useState<{ gx: number; gy: number } | null>(null)

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

    const [nameX, nameY] = data.params
    const maxLL = data.mle_loglik
    const isAbs = mode === 'absolute'
    const all = data.points
      .filter((p) => p.coords.length >= 2)
      .map((p) => ({
        gx: p.coords[0]!,
        gy: p.coords[1]!,
        ll: p.loglik,
        dl: p.loglik - maxLL, // ≤ 0
      }))
    if (all.length === 0) {
      el.replaceChildren()
      return
    }
    // Real (colour-ramped) cells vs failed/infeasible sentinels (drawn muted).
    const cells = all.filter((c) => c.ll > SENTINEL)
    const failed = all.filter((c) => c.ll <= SENTINEL)

    // Discrete, ascending grid axes (band domains keyed on the numeric values) —
    // spanning every evaluated cell so the mesh is complete, failed ones included.
    const xs = [...new Set(all.map((c) => c.gx))].sort((a, b) => a - b)
    const ys = [...new Set(all.map((c) => c.gy))].sort((a, b) => a - b)
    const inRegion = (c: { dl: number }) => c.dl >= -data.ci_drop
    const mleCell = { gx: data.mle_coords[0]!, gy: data.mle_coords[1]! }

    // Square-ish tiles: size to the width, then set height from the row count.
    const cell = Math.max(14, Math.min(46, Math.floor((width - 74) / xs.length)))
    const height = ys.length * cell + 54

    const node = Plot.plot({
      width,
      height,
      marginTop: 6,
      marginBottom: 44,
      marginLeft: 66,
      marginRight: 14,
      style: {
        background: 'transparent',
        color: AXIS,
        fontSize: '9px',
        fontFamily: MONO,
      },
      x: {
        type: 'band',
        domain: xs,
        padding: 0, // cells abut; a hairline `inset` keeps them legible
        label: `${nameX} →`,
        tickFormat: fmtGrid,
        tickRotate: -45,
        tickSize: 2,
      },
      y: {
        type: 'band',
        domain: ys,
        padding: 0,
        reverse: true, // smallest at the bottom (conventional up-is-more)
        label: `↑ ${nameY}`,
        tickFormat: fmtGrid,
        tickSize: 2,
      },
      color: {
        type: 'linear',
        // Same DL_FLOOR-wide window either way (keeps near-peak contrast); the
        // absolute view just labels the ramp in real logL so you can read a
        // cell's value off the legend and set it beside another fit's.
        domain: isAbs ? [maxLL - DL_FLOOR, maxLL] : [-DL_FLOOR, 0],
        clamp: true,
        scheme: 'viridis',
        legend: true,
        label: isAbs ? 'log-likelihood' : 'Δ log-likelihood (from peak)',
      },
      marks: [
        // Failed/infeasible cells: a muted tile so they read as "tried, no fit"
        // rather than a genuine low-likelihood basin.
        Plot.cell(failed, {
          x: 'gx',
          y: 'gy',
          fill: FAILED,
          inset: 0.5,
          title: (c: { gx: number; gy: number }) =>
            `${nameX}=${fmtGrid(c.gx)}, ${nameY}=${fmtGrid(c.gy)}\nno feasible fit (sentinel)`,
        }),
        Plot.cell(cells, {
          x: 'gx',
          y: 'gy',
          fill: isAbs ? 'll' : 'dl',
          inset: 0.5,
          title: (c: { gx: number; gy: number; ll: number; dl: number }) =>
            `${nameX}=${fmtGrid(c.gx)}, ${nameY}=${fmtGrid(c.gy)}\n` +
            `logL=${c.ll.toFixed(2)}  (Δ ${c.dl.toFixed(2)})`,
        }),
        // The 95% joint region, outlined as it forms.
        Plot.cell(cells.filter(inRegion), {
          x: 'gx',
          y: 'gy',
          fill: 'none',
          stroke: '#ffffff',
          strokeOpacity: 0.7,
          strokeWidth: 1,
          inset: 0.5,
        }),
        // The MLE cell.
        Plot.cell([mleCell], {
          x: 'gx',
          y: 'gy',
          fill: 'none',
          stroke: MLE,
          strokeWidth: 2,
          inset: 0.5,
        }),
        // The clicked cell (drawn last, on top).
        ...(selected
          ? [
              Plot.cell([selected], {
                x: 'gx',
                y: 'gy',
                fill: 'none',
                stroke: SELECTED,
                strokeWidth: 2.5,
                inset: 0.5,
              }),
            ]
          : []),
      ],
    })

    // Click-to-select: hit-test the click against the band scales → grid coords.
    const sx = node.scale('x') as unknown as BandScale | undefined
    const sy = node.scale('y') as unknown as BandScale | undefined
    // With a colour legend, `node` is a <figure> whose first <svg> is the legend
    // ramp — pick the plot svg (the one bearing the cell rects), not that.
    const svg = ([...node.querySelectorAll('svg')] as SVGSVGElement[]).sort(
      (a, b) => b.querySelectorAll('rect').length - a.querySelectorAll('rect').length,
    )[0]
    const onClick = (event: MouseEvent) => {
      if (!svg || !sx || !sy) return
      const r = svg.getBoundingClientRect()
      const px = ((event.clientX - r.left) * width) / r.width
      const py = ((event.clientY - r.top) * height) / r.height
      const gx = xs.find((d) => px >= sx.apply(d) && px < sx.apply(d) + sx.bandwidth)
      const gy = ys.find((d) => py >= sy.apply(d) && py < sy.apply(d) + sy.bandwidth)
      setSelected(gx !== undefined && gy !== undefined ? { gx, gy } : null)
    }
    if (svg) {
      svg.style.cursor = 'pointer'
      svg.addEventListener('click', onClick)
    }

    el.replaceChildren(node)
    return () => {
      svg?.removeEventListener('click', onClick)
      node.remove()
    }
  }, [data, width, mode, selected])

  return (
    <div className="group/fig relative">
      <div className="mb-2 flex items-center gap-2">
        <span className="font-mono text-[10px] uppercase tracking-wide text-neutral-400">
          colour
        </span>
        <ColorToggle value={mode} onChange={setMode} />
      </div>
      <div
        ref={ref}
        className="w-full min-w-0 overflow-x-auto"
        role="img"
        aria-label={`profile likelihood surface of ${data.params.join(' × ')}`}
      />
      <PlotDownloadButton
        targetRef={ref}
        name={`profile-${data.params.join('-')}`}
      />
      <CellReadout data={data} selected={selected} mode={mode} />
    </div>
  )
}

/** The clicked cell's detail: its coords, logL (in the selected scaling), and the
 *  conditional MLE of the OTHER (nuisance) params optimized there. */
function CellReadout({
  data,
  selected,
  mode,
}: {
  data: ProfileResponse
  selected: { gx: number; gy: number } | null
  mode: ColorMode
}) {
  if (!selected) {
    return (
      <p className="mt-2 font-mono text-[10px] text-neutral-400">
        click a cell for its log-likelihood and the nuisance-param MLE there
      </p>
    )
  }
  const [nameX, nameY] = data.params
  const pt = data.points.find(
    (p) => p.coords[0] === selected.gx && p.coords[1] === selected.gy,
  )
  const failed = !pt || pt.loglik <= -1e99
  const dl = pt ? pt.loglik - data.mle_loglik : 0
  const isAbs = mode === 'absolute'
  const nuisance = pt ? Object.entries(pt.nuisance) : []

  return (
    <div className="mt-2 rounded-sm border border-neutral-200 bg-neutral-50 px-3 py-2 font-mono text-[11px] text-neutral-700">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span>
          <span className="text-neutral-400">cell </span>
          {nameX}={fmtGrid(selected.gx)}, {nameY}={fmtGrid(selected.gy)}
        </span>
        {failed ? (
          <span className="text-neutral-400">no feasible fit (sentinel)</span>
        ) : isAbs ? (
          <span>
            logL <span className="font-semibold text-neutral-900">{pt!.loglik.toFixed(2)}</span>{' '}
            <span className="text-neutral-400">(Δ {dl.toFixed(2)})</span>
          </span>
        ) : (
          <span>
            Δ <span className="font-semibold text-neutral-900">{dl.toFixed(2)}</span>{' '}
            <span className="text-neutral-400">(logL {pt!.loglik.toFixed(2)})</span>
          </span>
        )}
      </div>
      {!failed && nuisance.length > 0 && (
        <div className="mt-1 flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <span className="text-[10px] uppercase tracking-wide text-neutral-400">
            nuisance MLE
          </span>
          {nuisance.map(([k, v]) => (
            <span key={k}>
              {k} = <span className="tabular-nums text-neutral-900">{fmtGrid(v)}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

/** Segmented control: colour the surface by Δ-from-peak or absolute logL. */
function ColorToggle({
  value,
  onChange,
}: {
  value: ColorMode
  onChange: (m: ColorMode) => void
}) {
  const opts: { v: ColorMode; label: string }[] = [
    { v: 'delta', label: 'Δ from peak' },
    { v: 'absolute', label: 'absolute logL' },
  ]
  return (
    <div className="inline-flex border border-neutral-200">
      {opts.map((o, i) => (
        <button
          key={o.v}
          type="button"
          onClick={() => onChange(o.v)}
          aria-pressed={value === o.v}
          className={cn(
            'px-2 py-0.5 font-mono text-[11px] transition-colors',
            i > 0 && 'border-l border-neutral-200',
            value === o.v
              ? 'bg-neutral-900 text-white'
              : 'text-neutral-500 hover:text-neutral-800',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}
