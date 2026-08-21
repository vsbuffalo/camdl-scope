import { useEffect, useMemo, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type {
  QuantityBandPoint,
  QuantityInfo,
  QuantityScalarRow,
} from '@/api/client'
import { useQuantityScalars, useQuantitySeries, useRun } from '@/api/queries'
import { ForestSkeleton, MutedNotice, NoPosteriorNotice } from '@/components/States'
import { BandLayerChecks, type BandLayer } from '@/components/BandLayers'
import { ScenarioChecks } from '@/components/ScenarioChecks'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { Segmented } from '@/components/Segmented'
import { ByIndexPlot, LevelLegend } from '@/components/ByIndexPlot'
import { Card } from '@/components/ui/card'
import { fmtTick, fmtValue } from '@/lib/format'
import { buildByIndexProfile, type IndexRecord } from '@/lib/byindex'
import { dayToDate, fmtModelDate, isTimePointUnit } from '@/lib/calendar'
import { logYOptions } from '@/lib/plot-scale'
import { loadJson, saveJson } from '@/lib/persist'
import { buildScenarioColors, referenceScenario, SCENARIO_REFERENCE } from '@/lib/scenario'
import { cn } from '@/lib/utils'

const AXIS = '#737373'
const PANEL_HEIGHT = 190
const MONO = 'var(--font-mono)'

function sourceTag(source: string): string | null {
  if (source === 'observations') return 'obs'
  if (source === 'derived') return 'derived'
  return null
}

function stratumLabel(stratum: Record<string, string>): string {
  return Object.entries(stratum)
    .map(([k, v]) => `${k}=${v}`)
    .join(' · ')
}

/** A small scenario swatch + name — the shared legend/chip ink. */
function ScenarioChip({ scenario, color }: { scenario: string; color: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block size-2 shrink-0 rounded-[1px]"
        style={{ background: color }}
        aria-hidden
      />
      <span className="text-neutral-600">{scenario}</span>
    </span>
  )
}

// ── Series quantities — scenario-overlaid banded ribbons ────────────────────

interface ScenarioSeries {
  scenario: string
  color: string
  points: QuantityBandPoint[]
}

/** One stratum's banded trajectory, overlaid by scenario. Which layers draw is
 *  the caller's choice (median / 50% / 90%); with several arms overlaid the
 *  fills lighten so five ribbons stay legible rather than mixing to mud. */
function BandPanel({
  title,
  series,
  toDate,
  logY,
  hiddenLayers,
}: {
  title: string
  series: ScenarioSeries[]
  toDate: ((t: number) => Date) | null
  /** Log y-axis. A quantity that goes non-positive (a difference, a log-ratio)
   *  has no viable log domain and stays linear — see lib/plot-scale. */
  logY: boolean
  hiddenLayers: ReadonlySet<BandLayer>
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  const solo = series.length === 1

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
    // Numeric time → Date when the fit carries a calendar, so the axis reads as
    // real dates; otherwise the raw numeric model-time index.
    const xOf = (d: { time: number }) => (toDate ? toDate(d.time) : d.time)
    const marks: Plot.Markish[] = []
    // Values of the marks drawn, for the log domain (see lib/plot-scale).
    const drawn: number[] = []
    const showP90 = !hiddenLayers.has('p90')
    const showP50 = !hiddenLayers.has('p50')
    const showMedian = !hiddenLayers.has('median')
    for (const s of series) {
      const pts = [...s.points].sort((a, b) => a.time - b.time)
      for (const p of pts) {
        if (showP90) drawn.push(p.q05 ?? NaN, p.q95 ?? NaN)
        if (showP50) drawn.push(p.q25 ?? NaN, p.q75 ?? NaN)
        if (showMedian) drawn.push(p.q50 ?? NaN)
      }
      if (showP90) {
        marks.push(
          Plot.areaY(pts, {
            x: xOf,
            y1: 'q05',
            y2: 'q95',
            fill: s.color,
            fillOpacity: solo ? 0.16 : 0.1,
          }),
        )
      }
      if (showP50) {
        marks.push(
          Plot.areaY(pts, {
            x: xOf,
            y1: 'q25',
            y2: 'q75',
            fill: s.color,
            fillOpacity: solo ? 0.22 : 0.16,
          }),
        )
      }
      if (showMedian) {
        marks.push(Plot.line(pts, { x: xOf, y: 'q50', stroke: s.color, strokeWidth: 1.3 }))
      }
    }
    marks.push(Plot.ruleY([0], { stroke: '#e5e5e5', strokeWidth: 0.5 }))

    const node = Plot.plot({
      width,
      height: PANEL_HEIGHT,
      marginTop: 8,
      // Two-line date ticks (day over month) need the extra bottom room.
      marginBottom: toDate ? 34 : 22,
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
        ...logYOptions(logY, drawn),
      },
      marks,
    })
    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [series, width, solo, toDate, logY, hiddenLayers])

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
        name={`quantity-${title.replace(/[^\w.-]+/g, '-')}`}
      />
    </div>
  )
}

function SeriesQuantity({
  runId,
  q,
  colorOf,
  activeScenarios,
  toDate,
  dimLevels,
  logY,
  hiddenLayers,
}: {
  runId: string
  q: QuantityInfo
  colorOf: (scenario: string) => string
  activeScenarios: string[]
  toDate: ((t: number) => Date) | null
  dimLevels: Map<string, string[]>
  logY: boolean
  hiddenLayers: ReadonlySet<BandLayer>
}) {
  const { data, isPending, isError } = useQuantitySeries(runId, q.name)
  const tag = sourceTag(q.source)

  // Inner view: the per-stratum ribbons over time, or the by-index profile
  // (marginalise time by mean, one index dim on x). Only offer by-index when the
  // quantity actually has an index dimension.
  const [view, setView] = useState<'series' | 'byindex'>('series')
  const [byIndexX, setByIndexX] = useState<string | null>(null)
  const [byIndexFacet, setByIndexFacet] = useState<string | null>(null)
  const indexDims = q.index_dims
  const activeByX =
    byIndexX && indexDims.includes(byIndexX) ? byIndexX : (indexDims[0] ?? null)
  const activeByFacet =
    byIndexFacet && byIndexFacet !== activeByX && indexDims.includes(byIndexFacet)
      ? byIndexFacet
      : null

  // Points of the selected scenarios (mirrors the ribbon panels' filter).
  const shownPoints = useMemo(() => {
    if (!data) return []
    const keep = new Set(activeScenarios.length ? activeScenarios : ['as_fitted'])
    return data.points.filter((p) => keep.has(p.scenario))
  }, [data, activeScenarios])

  // One panel per stratum cell; within a panel, one ribbon per (selected) scenario.
  const panels = useMemo(() => {
    const byStratum = new Map<
      string,
      { stratum: Record<string, string>; byScenario: Map<string, QuantityBandPoint[]> }
    >()
    for (const p of shownPoints) {
      const key = JSON.stringify(p.stratum)
      let g = byStratum.get(key)
      if (!g) {
        g = { stratum: p.stratum, byScenario: new Map() }
        byStratum.set(key, g)
      }
      const arr = g.byScenario.get(p.scenario)
      if (arr) arr.push(p)
      else g.byScenario.set(p.scenario, [p])
    }
    const order = activeScenarios.length ? activeScenarios : ['as_fitted']
    return [...byStratum.values()].map((g) => ({
      stratum: g.stratum,
      series: order
        .filter((s) => g.byScenario.has(s))
        .map((s) => ({ scenario: s, color: colorOf(s), points: g.byScenario.get(s)! })),
    }))
  }, [shownPoints, colorOf, activeScenarios])

  // By-index profile — a derived quantity has no observed overlay, so obs=null.
  const byIndex = useMemo(() => {
    if (!activeByX || shownPoints.length === 0) return null
    const records: IndexRecord[] = shownPoints.map((p) => ({
      stratum: p.stratum,
      pred: p.q50,
      obs: null,
    }))
    return buildByIndexProfile(records, activeByX, activeByFacet, dimLevels)
  }, [shownPoints, activeByX, activeByFacet, dimLevels])

  const hasSymbol = Boolean(q.symbol && q.symbol !== q.name)
  const canByIndex = indexDims.length > 0
  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 px-3 py-2">
        <span className="text-sm font-semibold text-neutral-900">
          {q.symbol ?? q.name}
        </span>
        {hasSymbol && (
          <span className="font-mono text-[11px] font-medium text-neutral-400">
            {q.name}
          </span>
        )}
        {tag && (
          <span className="font-mono text-[10px] uppercase tracking-wide text-neutral-400">
            {tag}
          </span>
        )}
        {q.description ? (
          <span className="text-xs font-medium text-neutral-500" title={q.description}>
            {q.description}
          </span>
        ) : (
          <span className="font-mono text-[10px] text-neutral-400">banded over draws</span>
        )}
        {q.reference && (
          <span className="ml-auto hidden shrink-0 text-[11px] italic text-neutral-400 sm:inline">
            {q.reference}
          </span>
        )}
      </div>
      {isPending && <ForestSkeleton rows={1} />}
      {isError && (
        <MutedNotice
          bordered={false}
          title="Couldn't load this quantity"
          detail="The backend returned an error reading its TSV."
        />
      )}
      {data && canByIndex && (
        <div className="flex gap-4 border-t border-neutral-100 px-3 pt-2">
          {(
            [
              ['series', 'Time series'],
              ['byindex', 'By index'],
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
      )}
      {data &&
        (view === 'series' || !canByIndex ? (
          panels.map((p) => (
            <BandPanel
              key={JSON.stringify(p.stratum)}
              title={stratumLabel(p.stratum) || 'all'}
              series={p.series}
              toDate={toDate}
              logY={logY}
              hiddenLayers={hiddenLayers}
            />
          ))
        ) : byIndex && byIndex.facets.some((f) => f.points.some((p) => p.pred != null)) ? (
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
                  levelColor={new Map(byIndex.facets.map((f) => [f.level, f.color]))}
                />
              )}
              <span className="font-mono text-[10px] text-neutral-400">
                line = median · marginalised over time
              </span>
            </div>
            <ByIndexPlot
              profile={byIndex}
              xLabel={activeByX ?? ''}
              yLabel={`mean ${q.symbol ?? q.name}`}
            />
          </>
        ) : (
          <MutedNotice
            bordered={false}
            title="Nothing to profile"
            detail="This quantity has no index dimension with values to profile."
          />
        ))}
    </Card>
  )
}

// ── Scalar quantities — one censoring-aware table, faceted by scenario ──────

export function ScalarBand({
  q50,
  q05,
  q95,
  unit,
  toDate,
}: {
  q50: number | null | undefined
  q05: number | null | undefined
  q95: number | null | undefined
  /** The quantity's declared unit from the manifest. An anchored time point
   *  (``unit: "time"``) renders as calendar dates when the fit has a calendar;
   *  any other non-null unit (a duration like "days", a rate) is appended as a
   *  suffix. camdl currently emits ``unit: null`` — this lights up when the
   *  manifest starts carrying units. */
  unit?: string | null
  toDate?: ((t: number) => Date) | null
}) {
  if (q50 == null) {
    return <span className="text-neutral-400">— censored —</span>
  }
  if (unit != null && isTimePointUnit(unit) && toDate) {
    return (
      <span>
        <span className="font-semibold text-neutral-900">
          {fmtModelDate(toDate, q50)}
        </span>{' '}
        <span className="text-neutral-500">
          [{q05 != null ? fmtModelDate(toDate, q05) : '—'},{' '}
          {q95 != null ? fmtModelDate(toDate, q95) : '—'}]
        </span>
      </span>
    )
  }
  return (
    <span>
      <span className="font-semibold text-neutral-900">{fmtValue(q50)}</span>{' '}
      <span className="text-neutral-500">
        [{fmtValue(q05)}, {fmtValue(q95)}]
      </span>
      {unit != null && !isTimePointUnit(unit) && (
        <span className="ml-1 text-[10px] text-neutral-400">{unit}</span>
      )}
    </span>
  )
}

function ScalarTable({
  runId,
  showScenario,
  colorOf,
  infoOf,
  activeScenarios,
  toDate,
}: {
  runId: string
  showScenario: boolean
  colorOf: (scenario: string) => string
  infoOf: Map<string, QuantityInfo>
  activeScenarios: string[]
  toDate: ((t: number) => Date) | null
}) {
  const { data, isPending, isError } = useQuantityScalars(runId)
  const hasStrata = useMemo(
    () => (data?.rows ?? []).some((r) => Object.keys(r.stratum).length > 0),
    [data],
  )
  const [open, setOpenState] = useState(() =>
    loadJson('quantities:scalars-open', true),
  )
  const setOpen = (v: boolean) => {
    setOpenState(v)
    saveJson('quantities:scalars-open', v)
  }

  // Group the selected scenarios' rows by quantity (the name shows once per
  // group), preserving order.
  const groups = useMemo(() => {
    const keep = new Set(activeScenarios)
    const m = new Map<string, QuantityScalarRow[]>()
    for (const r of data?.rows ?? []) {
      if (showScenario && !keep.has(r.scenario)) continue
      const arr = m.get(r.name)
      if (arr) arr.push(r)
      else m.set(r.name, [r])
    }
    return [...m.entries()]
  }, [data, activeScenarios, showScenario])

  if (isPending) return <ForestSkeleton rows={2} />
  if (isError)
    return (
      <MutedNotice
        bordered={false}
        title="Couldn't load the scalar quantities"
        detail="The backend returned an error reading the quantities TSVs."
      />
    )
  if (!data || data.rows.length === 0) return null

  return (
    <Card className="overflow-hidden">
      {/* Collapsible: this table runs to ~650px on a real fit, which pushes the
          series figures below the fold and separates them from the controls
          that change them. Collapsed state persists — a reader working on the
          trajectories should not have to re-collapse it every visit. */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-baseline gap-2 px-3 py-2 text-left transition-colors hover:bg-neutral-50"
      >
        <span className="text-[10px] text-neutral-400">{open ? '▾' : '▸'}</span>
        <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
          scalar quantities
        </span>
        <span className="font-mono text-[10px] tabular-nums text-neutral-400">
          ({groups.length})
        </span>
        {!open && (
          <span className="truncate font-mono text-[10px] text-neutral-400">
            {groups.map(([name]) => name).join(' · ')}
          </span>
        )}
      </button>
      <div className={cn('overflow-x-auto border-t border-neutral-200', !open && 'hidden')}>
        <table className="w-full border-collapse font-mono text-[12px] tabular-nums">
          <thead>
            <tr className="border-b border-neutral-200 text-[10px] uppercase tracking-wide text-neutral-400">
              <th className="px-3 py-2 text-left font-medium">Quantity</th>
              {showScenario && <th className="px-2 py-2 text-left font-medium">scenario</th>}
              <th className="px-2 py-2 text-left font-medium">reduce</th>
              {hasStrata && <th className="px-2 py-2 text-left font-medium">stratum</th>}
              <th className="px-3 py-2 text-right font-medium">median [90%]</th>
              <th className="px-3 py-2 text-right font-medium">censored</th>
            </tr>
          </thead>
          <tbody>
            {groups.map(([name, rows]) =>
              rows.map((r, i) => {
                const censored = r.p_censored == null ? null : Math.round(r.p_censored * 100)
                const first = i === 0
                return (
                  <tr
                    key={`${name}-${r.scenario}-${JSON.stringify(r.stratum)}`}
                    className={cn(first && 'border-t border-neutral-100')}
                  >
                    <td className="px-3 py-1.5 text-left align-top">
                      {first &&
                        (() => {
                          const info = infoOf.get(name)
                          const hasSymbol = Boolean(info?.symbol && info.symbol !== name)
                          return (
                            <>
                              <div className="flex items-baseline gap-1.5">
                                <span className="font-semibold text-neutral-900">
                                  {info?.symbol ?? name}
                                </span>
                                {hasSymbol && (
                                  <span className="font-mono text-[10px] text-neutral-400">
                                    {name}
                                  </span>
                                )}
                                {sourceTag(r.source) && (
                                  <span className="font-mono text-[10px] uppercase tracking-wide text-neutral-400">
                                    {sourceTag(r.source)}
                                  </span>
                                )}
                              </div>
                              {info?.description && (
                                <div
                                  className="mt-0.5 max-w-[20rem] truncate text-[11px] font-normal text-neutral-500"
                                  title={info.description}
                                >
                                  {info.description}
                                </div>
                              )}
                            </>
                          )
                        })()}
                    </td>
                    {showScenario && (
                      <td className="px-2 py-1.5 text-left">
                        <ScenarioChip scenario={r.scenario} color={colorOf(r.scenario)} />
                      </td>
                    )}
                    <td className="px-2 py-1.5 text-left text-neutral-500">
                      {first ? (r.reduce ?? '—') : ''}
                    </td>
                    {hasStrata && (
                      <td className="px-2 py-1.5 text-left text-neutral-600">
                        {stratumLabel(r.stratum) || '—'}
                      </td>
                    )}
                    <td className="px-3 py-1.5 text-right">
                      <ScalarBand q50={r.q50} q05={r.q05} q95={r.q95} unit={infoOf.get(r.name)?.unit} toDate={toDate} />
                    </td>
                    <td
                      className={cn(
                        'px-3 py-1.5 text-right',
                        censored && censored > 0 ? 'text-amber-600' : 'text-neutral-400',
                      )}
                    >
                      {censored == null ? '—' : `${censored}%`}
                    </td>
                  </tr>
                )
              }),
            )}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

// ── Tab ─────────────────────────────────────────────────────────────────────

/**
 * Generated quantities (`camdl fit predict`'s `quantities/` sidecar): named,
 * non-scored reductions of what the model produces, banded over draws and — for
 * a scenario-aware predict — overlaid by scenario. The manifest's `shape` drives
 * the layout: `scalar` quantities collapse into one table (a row per scenario),
 * `series` quantities each get a scenario-overlaid ribbon.
 */
export function QuantitiesTab({ runId }: { runId: string }) {
  const run = useRun(runId)
  const quantities = useMemo(
    () => run.data?.available_quantities ?? [],
    [run.data],
  )
  const scenarios = useMemo(
    () => run.data?.quantity_scenarios ?? [],
    [run.data],
  )
  const series = quantities.filter((q) => q.shape === 'series')
  const scalars = quantities.filter((q) => q.shape === 'scalar')

  const colorMap = useMemo(() => buildScenarioColors(scenarios), [scenarios])
  const colorOf = (s: string) => colorMap.get(s) ?? SCENARIO_REFERENCE
  const showScenario = scenarios.length > 1
  const infoOf = useMemo(
    () => new Map(quantities.map((q) => [q.name, q])),
    [quantities],
  )
  // Fit-level calendar → dates on every quantity time axis; canonical level
  // ordering for the by-index profile (age/village don't sort lexically).
  const toDate = useMemo(() => dayToDate(run.data?.calendar), [run.data?.calendar])
  const dimLevels = useMemo(() => {
    const m = new Map<string, string[]>()
    for (const d of run.data?.dimensions ?? []) m.set(d.name, d.levels)
    return m
  }, [run.data])

  // Scenario overlay filter (null = all) — isolate one arm so a stepped/wide
  // series (e.g. SIA campaigns in Reff) reads cleanly instead of as overlay mush.
  const [selectedScenarios, setSelectedScenarios] = useState<readonly string[] | null>(
    null,
  )
  // The reference arm (fitted / as_fitted / baseline) is pinned always-on;
  // the selection governs only the scenario overlays (see PredictiveTab).
  const reference = useMemo(() => referenceScenario(scenarios), [scenarios])
  const overlayOptions = useMemo(
    () => scenarios.filter((s) => s !== reference),
    [scenarios, reference],
  )
  const activeScenarios = useMemo(() => {
    // Default: with a pinned reference, open on the clean posterior predictive
    // (overlays deselected — add arms deliberately); without one, default to
    // all arms so a scenario-only sidecar doesn't open empty.
    const set = new Set(selectedScenarios ?? (reference ? [] : scenarios))
    return scenarios.filter((s) => s === reference || set.has(s))
  }, [selectedScenarios, scenarios, reference])
  const toggleScenario = (s: string) =>
    setSelectedScenarios(
      activeScenarios.includes(s)
        ? activeScenarios.filter((x) => x !== s)
        : [...activeScenarios, s],
    )

  // Log y for every series panel on the tab — a display preference like the
  // Predictive tab's, persisted for the same reason (the tab unmounts on every
  // tab switch, and this is a setting, not per-visit state). Kept tab-wide
  // rather than per-quantity: reading several quantities on one scale is the
  // point, and a quantity with no positive values just stays linear.
  const [logY, setLogYState] = useState(() =>
    loadJson('quantities:log-y', false),
  )
  const setLogY = (v: boolean) => {
    setLogYState(v)
    saveJson('quantities:log-y', v)
  }
  // Which band layers draw, shared across every quantity panel (the same
  // median / 50% / 90% vocabulary as the Predictive check). Persisted like the
  // other display preferences; stored as an array since a Set is not JSON.
  const [hiddenLayers, setHiddenLayersState] = useState<ReadonlySet<BandLayer>>(
    () => new Set(loadJson<BandLayer[]>('quantities:hidden-layers', [])),
  )
  const toggleLayer = (layer: BandLayer) => {
    const next = new Set(hiddenLayers)
    if (next.has(layer)) next.delete(layer)
    else next.add(layer)
    setHiddenLayersState(next)
    saveJson('quantities:hidden-layers', [...next])
  }
  // The figures section collapses like the scalar table — same treatment for
  // the two halves of one quantity list, which differ only in shape.
  const [seriesOpen, setSeriesOpenState] = useState(() =>
    loadJson('quantities:series-open', true),
  )
  const setSeriesOpen = (v: boolean) => {
    setSeriesOpenState(v)
    saveJson('quantities:series-open', v)
  }

  if (run.isPending) {
    return (
      <Card className="overflow-hidden">
        <ForestSkeleton rows={2} />
      </Card>
    )
  }

  if (quantities.length === 0) {
    return (
      <NoPosteriorNotice
        status={run.data?.status}
        whenDone={{
          title: 'No generated quantities',
          detail: (
            <>
              This fit has no <span className="font-mono">quantities</span> sidecar.
              Add a <span className="font-mono">quantities {'{}'}</span> block to the
              model and run <span className="font-mono">camdl fit predict</span>.
            </>
          ),
        }}
      />
    )
  }

  return (
    <div className="space-y-4">
      {/* SCENARIO stays at the top: it is the one control whose scope spans
          both sections (the scalar table AND the figures). Controls that only
          change the figures live with the figures — see the series header. */}
      {showScenario && (
        <div className="px-1">
          <ScenarioChecks
            options={overlayOptions}
            selected={activeScenarios.filter((s) => s !== reference)}
            colorOf={colorOf}
            onToggle={toggleScenario}
            onSetAll={setSelectedScenarios}
            pinned={reference}
          />
        </div>
      )}
      {scalars.length > 0 && (
        <ScalarTable
          runId={runId}
          showScenario={showScenario}
          colorOf={colorOf}
          infoOf={infoOf}
          activeScenarios={activeScenarios}
          toDate={toDate}
        />
      )}
      {series.length > 0 && (
        <div>
          {/* The figures' own control strip, directly above them and sticky, so
              toggling a scale never scrolls the figure it changes off screen —
              with several series quantities the last one is far from the top of
              the tab. Sticks *below* the app header (--app-header-h) and under
              its z-index, so the two stack rather than overlap. */}
          <div
            className="sticky z-30 flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-neutral-200 bg-white px-1 py-2"
            style={{ top: 'var(--app-header-h)' }}
          >
            <button
              type="button"
              onClick={() => setSeriesOpen(!seriesOpen)}
              aria-expanded={seriesOpen}
              className="flex items-baseline gap-1 text-[10px] font-medium uppercase tracking-wider text-neutral-400 transition-colors hover:text-neutral-600"
            >
              <span>{seriesOpen ? '▾' : '▸'}</span>
              <span>series quantities</span>
              <span className="font-mono tabular-nums normal-case">
                ({series.length})
              </span>
            </button>
            {seriesOpen && (
              <>
                <BandLayerChecks hidden={hiddenLayers} onToggle={toggleLayer} />
                <label className="flex cursor-pointer items-center gap-1.5 font-mono text-xs">
                  <input
                    type="checkbox"
                    checked={logY}
                    onChange={() => setLogY(!logY)}
                    className="size-3 accent-neutral-800"
                  />
                  <span className={logY ? 'text-neutral-900' : 'text-neutral-500'}>
                    log y
                  </span>
                </label>
              </>
            )}
          </div>
          {seriesOpen && (
            <div className="space-y-4 pt-4">
              {series.map((q) => (
                <SeriesQuantity
                  key={q.name}
                  runId={runId}
                  q={q}
                  colorOf={colorOf}
                  activeScenarios={activeScenarios}
                  toDate={toDate}
                  dimLevels={dimLevels}
                  logY={logY}
                  hiddenLayers={hiddenLayers}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
