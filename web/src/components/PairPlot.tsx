import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import * as Plot from '@observablehq/plot'
import type { DrawsResponse, PosteriorResponse } from '@/api/client'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { fmtTick } from '@/lib/format'
import { chainColor } from '@/lib/colors'
import { cn } from '@/lib/utils'

// Cell sizing: square cells clamp between these so few params fill the width
// while many params shrink toward MIN and the grid scrolls.
const MIN_CELL = 72
const MAX_CELL = 200
const LABEL_W = 26 // px gutter for the left row labels (rotated vertical, so narrow)
const Y_AXIS_W = 42 // px strip down the left holding each row's value y-axis ticks
const X_AXIS_H = 24 // px strip along the bottom holding each column's value x-axis ticks

const FRAME = '#e5e5e5' // neutral-200 — hairline cell border
const POST_BAR = '#525252' // neutral-600 — posterior marginal (primary, darker)
const PRIOR_FILL = '#d4d4d4' // neutral-300 — prior marginal (secondary, light)
const PRIOR_STROKE = '#a3a3a3' // neutral-400 — prior step outline
const MEDIAN = '#171717' // neutral-900
const SYMBOL = '#525252' // neutral-600 — in-cell symbol

type Domain = [number, number] | undefined

type CellSpec =
  | {
      kind: 'scatter'
      x: number[]
      y: number[]
      chain: number[]
      xDomain: Domain
      yDomain: Domain
    }
  | {
      kind: 'diag'
      values: number[]
      priorDensity: { x: number[]; y: number[] } | null
      median: number | null
      symbol: string
      domain: Domain
    }

function extent(xs: number[]): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const x of xs) {
    if (x < lo) lo = x
    if (x > hi) hi = x
  }
  return [lo, hi]
}

/** Evenly-spaced bin edges over a domain — shared by prior + posterior so the
 *  two histograms bin identically and their heights are directly comparable. */
function makeThresholds([d0, d1]: [number, number], nbins: number): number[] {
  const step = (d1 - d0) / nbins
  return Array.from({ length: nbins + 1 }, (_, i) => d0 + step * i)
}

/** One corner-plot cell: a scatter (lower) or a marginal histogram (diagonal). */
function PairCell({
  spec,
  chainDomain,
  size,
}: {
  spec: CellSpec
  chainDomain: number[]
  size: number
}) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el || size <= 0) return

    const fontSize = Math.round(Math.max(9, Math.min(13, size / 14)))
    const base = {
      width: size,
      height: size,
      margin: 2,
      style: { background: 'transparent' as const },
      x: { axis: null },
      y: { axis: null },
    }

    let node: ReturnType<typeof Plot.plot>
    if (spec.kind === 'scatter') {
      node = Plot.plot({
        ...base,
        x: { axis: null, domain: spec.xDomain },
        y: { axis: null, domain: spec.yDomain },
        color: {
          type: 'categorical',
          domain: chainDomain,
          range: chainDomain.map(chainColor),
        },
        marks: [
          Plot.dot(spec.chain, {
            x: (_d: number, i: number) => spec.x[i],
            y: (_d: number, i: number) => spec.y[i],
            fill: (d: number) => d,
            r: 1.1,
            fillOpacity: 0.25,
            stroke: null,
          }),
          Plot.frame({ stroke: FRAME, strokeWidth: 0.5 }),
        ],
      })
    } else {
      const domain = spec.domain
      // More room with bigger cells → a few more bins; clamp so it stays crisp.
      const nbins = Math.max(14, Math.min(28, Math.round(size / 9)))
      const marks: Plot.Markish[] = []

      if (domain) {
        const thresholds = makeThresholds(domain, nbins)
        const binwidth = (domain[1] - domain[0]) / nbins
        // Prior first (behind): the smooth ANALYTIC density, scaled into the
        // posterior's `proportion` units (density × bin width) so heights are
        // directly comparable. A binned histogram of prior samples clipped to
        // this window reads as noise; the analytic curve is exact and smooth.
        const pc = spec.priorDensity
        if (pc && pc.x.length > 1) {
          const pts = pc.x.map((x, i) => ({ x, y: (pc.y[i] ?? 0) * binwidth }))
          marks.push(
            Plot.areaY(pts, { x: 'x', y: 'y', fill: PRIOR_FILL, fillOpacity: 0.55 }),
            Plot.line(pts, {
              x: 'x',
              y: 'y',
              stroke: PRIOR_STROKE,
              strokeWidth: 0.75,
              strokeOpacity: 0.85,
            }),
          )
        }
        // Posterior on top: darker solid bars. Same bins + `proportion` reducer
        // ⇒ equal bin width ⇒ proportion ∝ density, height-comparable to prior.
        marks.push(
          Plot.rectY(
            spec.values,
            Plot.binX<Plot.RectYOptions>(
              { y: 'proportion' },
              {
                x: (d: number) => d,
                thresholds,
                fill: POST_BAR,
                fillOpacity: 0.82,
                insetLeft: 0.5,
                insetRight: 0.5,
              },
            ),
          ),
        )
      }

      if (spec.median != null) {
        marks.push(Plot.ruleX([spec.median], { stroke: MEDIAN, strokeWidth: 1 }))
      }
      marks.push(
        Plot.text([spec.symbol], {
          frameAnchor: 'top-left',
          dx: 3,
          dy: 2,
          fill: SYMBOL,
          fontSize,
          fontWeight: 600,
        }),
        Plot.frame({ stroke: FRAME, strokeWidth: 0.5 }),
      )
      node = Plot.plot({
        ...base,
        x: { axis: null, domain },
        y: { axis: null },
        marks,
      })
    }

    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [spec, chainDomain, size])

  return <div style={{ width: size, height: size }} ref={ref} />
}

/**
 * A dedicated outer-edge axis strip: a single Observable Plot containing ONLY
 * an axis (no data marks), self-drawn into a ref like the cells. Living in its
 * own grid track keeps the data cells full-bleed and pixel-aligned — the strip
 * borrows the cells' `margin:2` on the shared axis so its ticks sit directly
 * under (x) or beside (y) the column/row's data.
 */
function AxisStrip({
  kind,
  domain,
  w,
  h,
}: {
  kind: 'x' | 'y'
  domain: [number, number] | undefined
  w: number
  h: number
}) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el || w <= 0 || h <= 0 || !domain) return

    const style = {
      background: 'transparent' as const,
      color: '#737373',
      fontSize: '9px',
      fontFamily: 'var(--font-mono)',
    }

    const node =
      kind === 'x'
        ? Plot.plot({
            width: w,
            height: h,
            marginLeft: 2,
            marginRight: 2,
            marginTop: 0,
            marginBottom: h - 4,
            style,
            x: {
              domain,
              ticks: 3,
              tickFormat: (d: number) => fmtTick(d),
              tickSize: 2,
              tickPadding: 3,
              label: null,
            },
            y: { axis: null },
            marks: [],
          })
        : Plot.plot({
            width: w,
            height: h,
            marginTop: 2,
            marginBottom: 2,
            marginRight: 0,
            marginLeft: w - 4,
            style,
            y: {
              domain,
              ticks: 4,
              tickFormat: (d: number) => fmtTick(d),
              tickSize: 2,
              tickPadding: 3,
              label: null,
            },
            x: { axis: null },
            marks: [],
          })

    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [kind, domain, w, h])

  return <div style={{ width: w, height: h }} ref={ref} />
}

/** x-limit mode: `posterior` fits each axis to the draws; `prior` widens it to
 *  the prior's breadth so the posterior reads as a spike inside the prior. */
export type PriorXlimMode = 'posterior' | 'prior'

/** Built-in blurbs for the objective aux columns — they carry no model `#'`
 *  docs, and what they mean (especially on a PGAS fit) is exactly the thing
 *  worth being reminded of. Model params use their own docs instead. */
const OBJECTIVE_DOCS: Record<string, string> = {
  log_posterior:
    'The sampler’s objective: log prior + log-likelihood. On a chain-binomial (PGAS) fit this is the COMPLETE-DATA log posterior — dominated by the latent-path mass, not the data fit; pair parameters against obs_ll for the data-fit signal.',
  log_likelihood: 'The model log-likelihood p(y | θ) (marginal — MH/ODE fits).',
  obs_ll:
    'Data-fit term of the complete-data likelihood: log p(y | path, θ). On a PGAS fit, this — not log_posterior — is the target that shows which parameter values fit the data.',
  transition_ll:
    'Latent-path term of the complete-data likelihood: log p(path | θ). Together with obs_ll it composes log_complete_data_ll.',
  log_complete_data_ll:
    'The complete-data log-likelihood log p(y, path | θ) — path-dominated; see obs_ll for the data-fit part.',
}

interface PairPlotProps {
  draws: DrawsResponse
  posterior?: PosteriorResponse
  /** Ordered, already-filtered visible params (the selection, in estimated order). */
  params: string[]
  priorXlimMode?: PriorXlimMode
}

/**
 * Corner plot over the SELECTED parameters: lower-triangle bivariate scatter
 * (colored by chain, doubling as a mixing check), diagonal marginals with a
 * prior overlay, upper triangle left blank by convention. Correlations are NOT
 * computed here — the stats stay in Python; this view shows the joint geometry
 * (ridges, identifiability, chain separation) the marginals alone can't. Cells
 * are sized to fill the container width and scroll only when they can't fit.
 */
export function PairPlot({
  draws,
  posterior,
  params,
  priorXlimMode = 'posterior',
}: PairPlotProps) {
  const n = params.length

  const containerRef = useRef<HTMLDivElement>(null)
  // The full-width figure (legend + grid) is the PNG capture target — ref this,
  // not the scroll container, so a wide grid isn't clipped to the viewport.
  const figRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(0)

  // Measure the scroll container's visible width once on mount (seeded
  // synchronously so cells draw on first paint, incl. headless capture) and on
  // resize. One observer here drives every cell's size — cells don't self-measure.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const measure = () => {
      const w = Math.round(el.getBoundingClientRect().width)
      if (w > 0) setContainerWidth(w)
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const cell = useMemo(() => {
    if (n < 1 || containerWidth <= 0) return MIN_CELL
    const avail = containerWidth - LABEL_W - Y_AXIS_W
    return Math.max(MIN_CELL, Math.min(MAX_CELL, Math.floor(avail / n)))
  }, [containerWidth, n])

  const chainDomain = useMemo(
    () => [...new Set(draws.chain)].sort((a, b) => a - b),
    [draws.chain],
  )

  const meta = useMemo(() => {
    const byName = new Map(
      (posterior?.params ?? []).map((p) => [p.name, p] as const),
    )
    return params.map((name) => ({
      name,
      symbol: byName.get(name)?.symbol ?? name,
      median: byName.get(name)?.q50 ?? null,
      description: byName.get(name)?.description ?? OBJECTIVE_DOCS[name] ?? null,
      reference: byName.get(name)?.reference ?? null,
    }))
  }, [params, posterior])

  // Doc lookup: clicking a row label opens this variable's description strip
  // above the grid (click again, click another, or ✕ to change/close).
  const [infoParam, setInfoParam] = useState<string | null>(null)
  const info = infoParam ? meta.find((m) => m.name === infoParam) : undefined

  const anyPrior = useMemo(
    () => params.some((p) => (draws.prior_density?.[p]?.x.length ?? 0) > 1),
    [params, draws.prior_density],
  )

  // One domain per dimension, reused for that param's column-x, row-y, axis
  // strips, and diagonal — so every cell touching a param shares an axis. In
  // `prior` mode the domain widens to the prior curve's extent (its breadth),
  // so the posterior collapses to a spike inside the broad prior.
  const domainOf = useMemo(() => {
    const m = new Map<string, Domain>()
    for (const p of params) {
      const post = draws.draws[p] ?? []
      const [lo, hi] = extent(post)
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) {
        m.set(p, undefined)
        continue
      }
      let d0 = lo
      let d1 = hi
      const pc = draws.prior_density?.[p]
      if (priorXlimMode === 'prior' && pc && pc.x.length > 1) {
        d0 = Math.min(d0, pc.x[0]!)
        d1 = Math.max(d1, pc.x[pc.x.length - 1]!)
      }
      const pad = (d1 - d0) * 0.04 || Math.abs(d0) * 0.04 || 1
      m.set(p, [d0 - pad, d1 + pad])
    }
    return m
  }, [params, draws, priorXlimMode])

  const labelFs = Math.round(Math.max(11, Math.min(15, cell / 13)))

  return (
    // ALWAYS mount this outer measuring block — even when n<2 — so the ref is
    // attached and the width is measured before the grid renders. (With cached
    // data the grid mounts only after the selection populates; an early return
    // here would let the measure effect run with no element and never re-fire,
    // stranding the grid at MIN_CELL on tab re-entry.) Measure the OUTER block,
    // not the inner overflow-x-auto box (which shrink-wraps to the grid).
    <div ref={containerRef} className="group/fig relative">
      {n < 2 ? (
        <p className="text-sm text-neutral-500">
          The pair plot needs at least two selected parameters — open{' '}
          <span className="font-mono">⚙ params</span> to choose more.
        </p>
      ) : (
        <>
        {info && (
          <div className="mb-2 flex items-baseline gap-2 border border-neutral-200 bg-neutral-50 px-3 py-2">
            <span className="shrink-0 font-mono text-[12px] font-semibold text-neutral-900">
              {info.symbol}
            </span>
            {info.symbol !== info.name && (
              <span className="shrink-0 font-mono text-[10px] text-neutral-400">
                {info.name}
              </span>
            )}
            <span className="min-w-0 text-[12px] leading-snug text-neutral-600">
              {info.description ?? (
                <span className="italic text-neutral-400">
                  no description — add a{' '}
                  <span className="font-mono not-italic">{"#'"}</span> doc
                  comment above this parameter in the model
                </span>
              )}
              {info.reference && (
                <span className="ml-1.5 text-[11px] italic text-neutral-400">
                  {info.reference}
                </span>
              )}
            </span>
            <button
              type="button"
              onClick={() => setInfoParam(null)}
              aria-label="close description"
              className="ml-auto shrink-0 font-mono text-[11px] text-neutral-400 transition-colors hover:text-neutral-700"
            >
              ✕
            </button>
          </div>
        )}
        <div className="overflow-x-auto">
          <div ref={figRef} className="w-max bg-white">
          {anyPrior && <Legend />}
          <div
            className="grid gap-px"
            style={{
              width: 'max-content',
              gridTemplateColumns: `${LABEL_W}px ${Y_AXIS_W}px repeat(${n}, ${cell}px)`,
              gridTemplateRows: `repeat(${n}, ${cell}px) ${X_AXIS_H}px auto`,
            }}
          >
          {params.map((rowName, r) => (
            // One matrix row: left symbol gutter, value y-axis strip, then N cells.
            <Row key={rowName}>
              <button
                type="button"
                onClick={() =>
                  setInfoParam((cur) => (cur === rowName ? null : rowName))
                }
                aria-pressed={infoParam === rowName}
                className={cn(
                  'flex cursor-pointer items-center justify-center font-medium transition-colors',
                  infoParam === rowName
                    ? 'bg-neutral-100 text-neutral-900'
                    : 'text-neutral-500 hover:text-neutral-900',
                )}
                title={`${meta[r]!.symbol} — click for description`}
              >
                {/* rotate(-90deg) reads bottom-to-top (upright glyphs) — the
                    conventional y-axis direction; `sideways-lr` isn't in Chrome. */}
                <span
                  className="whitespace-nowrap"
                  style={{ fontSize: labelFs, transform: 'rotate(-90deg)' }}
                >
                  {meta[r]!.symbol}
                </span>
              </button>
              {/* y-axis value ticks for this row's data. Row 0 is a marginal
                  density (no meaningful value y-axis) → blank. */}
              {r === 0 ? (
                <div />
              ) : (
                <AxisStrip
                  kind="y"
                  domain={domainOf.get(params[r]!)}
                  w={Y_AXIS_W}
                  h={cell}
                />
              )}
              {params.map((colName, c) => {
                if (c > r) {
                  // Upper triangle — blank by convention.
                  return <div key={colName} />
                }
                if (c === r) {
                  return (
                    <PairCell
                      key={colName}
                      size={cell}
                      chainDomain={chainDomain}
                      spec={{
                        kind: 'diag',
                        values: draws.draws[rowName] ?? [],
                        priorDensity: draws.prior_density?.[rowName] ?? null,
                        median: meta[r]!.median,
                        symbol: meta[r]!.symbol,
                        domain: domainOf.get(rowName),
                      }}
                    />
                  )
                }
                return (
                  <PairCell
                    key={colName}
                    size={cell}
                    chainDomain={chainDomain}
                    spec={{
                      kind: 'scatter',
                      x: draws.draws[colName] ?? [],
                      y: draws.draws[rowName] ?? [],
                      chain: draws.chain,
                      xDomain: domainOf.get(colName),
                      yDomain: domainOf.get(rowName),
                    }}
                  />
                )
              })}
            </Row>
          ))}

          {/* Value x-axis strip: empty label + y-axis corners, then ticks
              under each column (aligned to that column's data x-domain). */}
          <div />
          <div />
          {params.map((colName, c) => (
            <AxisStrip
              key={colName}
              kind="x"
              domain={domainOf.get(params[c]!)}
              w={cell}
              h={X_AXIS_H}
            />
          ))}

          {/* Bottom edge: empty corners, then one symbol under each column. */}
          <div />
          <div />
          {params.map((colName, c) => (
            <div
              key={colName}
              className="flex items-start justify-center overflow-hidden pt-1 font-medium text-neutral-500"
              style={{ fontSize: labelFs }}
            >
              <span className="max-w-full truncate" title={meta[c]!.symbol}>
                {meta[c]!.symbol}
              </span>
            </div>
          ))}
          </div>
          </div>
        </div>
        <PlotDownloadButton targetRef={figRef} name="pairplot" />
        </>
      )}
    </div>
  )
}

/** Minimal mono legend distinguishing the two diagonal histograms. */
function Legend() {
  return (
    <div className="mb-2 flex items-center gap-3 font-mono text-[10px] text-neutral-400">
      <span className="flex items-center gap-1.5">
        <span
          className="inline-block h-2 w-3 rounded-[1px]"
          style={{ background: POST_BAR, opacity: 0.82 }}
        />
        posterior
      </span>
      <span className="text-neutral-300">·</span>
      <span className="flex items-center gap-1.5">
        <span
          className="inline-block h-2 w-3 rounded-[1px] border"
          style={{ background: PRIOR_FILL, borderColor: PRIOR_STROKE, opacity: 0.7 }}
        />
        prior
      </span>
    </div>
  )
}

/**
 * `display: contents` so each matrix row's children participate directly in the
 * parent grid (auto-flow places them left-to-right) while still grouping under
 * a stable React key.
 */
function Row({ children }: { children: ReactNode }) {
  return <div style={{ display: 'contents' }}>{children}</div>
}
