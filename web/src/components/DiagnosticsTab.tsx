import { useEffect, useRef, useState, type ReactNode } from 'react'
import * as Plot from '@observablehq/plot'
import type {
  ChainMixing,
  DiagnosticsResponse,
  ParamDiagnostic,
  SamplerPanel,
} from '@/api/client'
import { useDiagnostics } from '@/api/queries'
import { ChainSelect } from '@/components/ChainSelect'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { includedChains, type ChainControls } from '@/lib/chains'
import { WarmupControl } from '@/components/WarmupControl'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { fmtEss, fmtRhat, fmtTick, fmtValue } from '@/lib/format'
import { cn } from '@/lib/utils'

const DEFAULT_WARMUP_PCT = 50

// Convergence signal thresholds — the same P&L coding the Posterior forest uses,
// so a number reads identically across tabs.
const RHAT_HIGH = 1.1 // > this reads red (elevated, suspect)
const RHAT_OK = 1.05 // <= this reads muted green (healthy)
const ESS_LOW = 100 // < this reads amber (thin effective sample)

// Mixing-bar palette: a bar within the healthy band is quiet neutral; one
// outside it lights up red. The band itself is a faint emerald wash.
const BAR_IN = '#525252' // neutral-600 — within the healthy band
const BAR_OUT = '#dc2626' // red-600 — outside the band
const BAND_FILL = '#10b981' // emerald-500 — faint healthy-zone wash
const BAND_RULE = '#34d399' // emerald-400 — band edges
const FRAME = '#e5e5e5' // neutral-200 — hairline frame
const AXIS = '#737373' // neutral-500 — tick labels
const MONO = 'var(--font-mono)'
const ROW_H = 24 // px per chain bar

/** Two-sided color for R̂: healthy green / neutral / elevated red. */
function rhatClass(rhat: number | null | undefined): string {
  if (rhat == null) return 'text-neutral-400'
  if (rhat > RHAT_HIGH) return 'text-red-600 font-medium'
  if (rhat <= RHAT_OK) return 'text-emerald-600'
  return 'text-neutral-500'
}

/** ESS color: amber when thin, muted neutral when healthy. */
function essClass(ess: number | null | undefined): string {
  if (ess == null) return 'text-neutral-400'
  if (ess < ESS_LOW) return 'text-amber-600'
  return 'text-neutral-400'
}

/** Compact readout: ≥100 or 0 as integers, else 3 sig-figs (e.g. 0.290). */
function fmtEff(v: number | null | undefined): string {
  if (v == null) return '—'
  if (v === 0) return '0'
  if (v >= 100 || v < 0.001) return v.toPrecision(3)
  return v >= 1 ? v.toFixed(2) : v.toPrecision(2)
}

/** Finding-line color keyed off camdl's severity. */
function severityClass(severity: string): string {
  if (severity === 'error') return 'text-red-600'
  if (severity === 'warn') return 'text-amber-600'
  return 'text-emerald-600'
}

/** Small uppercase tracked section header, matching the other tabs' strips. */
function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="border-b border-neutral-100 px-3 py-2 text-[10px] font-medium uppercase tracking-wider text-neutral-400">
      {children}
    </div>
  )
}

type MixRow = { chain: number; value: number; out: boolean }

/**
 * Per-chain mixing metric as a compact horizontal bar chart — one bar per chain,
 * the healthy `band` drawn as a faint emerald wash bracketed by two reference
 * rules. A bar outside the band reads red, inside it reads neutral. Self-measuring
 * via a width-seeding ref so it draws on first paint (incl. headless capture).
 */
function MixingChart({ mixing }: { mixing: ChainMixing }) {
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

    const band = mixing.band
    // Label by the chain id the API sends, not by array position: camdl's
    // chains are 1-based and a mixing series may cover a subset (a chain still
    // warming up contributes no value), so position names the wrong chain.
    const rows: MixRow[] = mixing.values.map((value, i) => ({
      chain: mixing.chains?.[i] ?? i + 1,
      value,
      out: band ? value < band[0] || value > band[1] : false,
    }))
    const height = rows.length * ROW_H + 24

    const marks: Plot.Markish[] = []
    if (band) {
      // Faint wash spanning the full frame height between the band edges, then
      // dashed rules at lo/hi so the healthy zone is legible even behind a bar.
      marks.push(
        Plot.rectX([band], {
          x1: (d: [number, number]) => d[0],
          x2: (d: [number, number]) => d[1],
          fill: BAND_FILL,
          fillOpacity: 0.08,
        }),
        Plot.ruleX(band, {
          stroke: BAND_RULE,
          strokeWidth: 0.75,
          strokeDasharray: '2,2',
        }),
      )
    }
    marks.push(
      Plot.barX(rows, {
        y: 'chain',
        x: 'value',
        fill: (d: MixRow) => (d.out ? BAR_OUT : BAR_IN),
        insetTop: 2,
        insetBottom: 2,
      }),
      Plot.ruleX([0], { stroke: FRAME, strokeWidth: 0.5 }),
      Plot.frame({ stroke: FRAME, strokeWidth: 0.5 }),
    )

    const node = Plot.plot({
      width,
      height,
      marginTop: 4,
      marginBottom: 18,
      marginLeft: 30,
      marginRight: 10,
      style: {
        background: 'transparent',
        color: AXIS,
        fontSize: '9px',
        fontFamily: MONO,
      },
      x: {
        label: null,
        ticks: 5,
        tickSize: 2,
        tickPadding: 3,
        tickFormat: (d: number) => fmtTick(d),
      },
      y: {
        label: null,
        domain: rows.map((r) => r.chain),
        tickSize: 0,
        tickPadding: 4,
        tickFormat: (d: number) => `c${d}`,
      },
      marks,
    })

    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [mixing, width])

  return (
    <div className="group/fig relative">
      <div
        ref={ref}
        className="w-full min-w-0"
        role="img"
        aria-label={`per-chain ${mixing.label}`}
      />
      <PlotDownloadButton targetRef={ref} name="chain-mixing" />
    </div>
  )
}

/**
 * A method-specific sampler panel as a chain × metric table.
 *
 * One component for every sampler: the backend decides which panels a run
 * supports and what each column means, so a new sampler diagnostic arrives as
 * data — a column with a label, a note and an optional healthy band — rather
 * than as a new component here. A value outside its band reads red; a counter
 * that should be zero (divergences) reads red on any nonzero.
 */
function SamplerPanelTable({ panel }: { panel: SamplerPanel }) {
  const flagged = (v: number | null, col: SamplerPanel['columns'][number]) => {
    if (v == null) return false
    if (col.key === 'n_divergent') return v > 0
    return col.band ? v < col.band[0] || v > col.band[1] : false
  }
  return (
    <>
      <SectionLabel>{panel.title}</SectionLabel>
      {panel.note && (
        <p className="px-3 pt-1 text-[11px] leading-snug text-neutral-500">
          {panel.note}
        </p>
      )}
      <div className="scroll-x-visible overflow-x-auto px-3 py-2">
        <table className="w-full border-collapse font-mono text-[11px] tabular-nums">
          <thead>
            <tr className="border-b border-neutral-200 text-[9px] uppercase tracking-wider text-neutral-400">
              <th className="px-2 py-1.5 text-left font-medium">chain</th>
              {panel.columns.map((c) => (
                <th
                  key={c.key}
                  className="px-2 py-1.5 text-right font-medium"
                  title={c.note ?? undefined}
                >
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {panel.rows.map((chain, r) => (
              <tr key={chain} className="border-t border-neutral-100">
                <td className="px-2 py-1 text-left text-neutral-500">c{chain}</td>
                {panel.columns.map((c, i) => {
                  const v = panel.values[r]?.[i] ?? null
                  return (
                    <td
                      key={c.key}
                      className={cn(
                        'px-2 py-1 text-right',
                        flagged(v, c)
                          ? 'font-medium text-red-600'
                          : 'text-neutral-700',
                      )}
                    >
                      {v == null ? '—' : fmtValue(v)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* The columns worth a sentence explain themselves under the table —
          a tooltip is not readable on a phone and not discoverable anywhere. */}
      {panel.columns.some((c) => c.note) && (
        <ul className="space-y-0.5 px-3 pb-2 text-[10px] leading-snug text-neutral-400">
          {panel.columns
            .filter((c) => c.note)
            .map((c) => (
              <li key={c.key}>
                <span className="text-neutral-600">{c.label}</span> — {c.note}
              </li>
            ))}
        </ul>
      )}
    </>
  )
}

/** The verdict header + the findings list (or the all-clear line). */
function Verdict({ data }: { data: DiagnosticsResponse }) {
  const isCamdl = data.source === 'camdl'
  return (
    <div className="border-b border-neutral-100 px-3 py-2.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="font-mono text-[11px] text-neutral-500">
          {isCamdl
            ? `camdl · ${data.stage ?? '—'} verdict`
            : 'live arviz estimate — no stage summary yet'}
        </span>
        <span
          className="font-mono text-[10px] uppercase tracking-wider text-neutral-400"
          title="objective the diagnostics summarise"
        >
          {data.logpost_label}
        </span>
      </div>

      {data.n_chains_warming > 0 && (
        <p className="mt-1 font-mono text-[11px] text-amber-600">
          {data.n_chains_warming} chain{data.n_chains_warming > 1 ? 's' : ''} still
          warming up (no post-warm-up draws) — R̂/ESS computed over the{' '}
          {data.n_chains} with draws.
        </p>
      )}

      <div className="mt-1.5 space-y-1">
        {data.findings.length === 0 ? (
          <p className="text-xs text-emerald-600">no findings — within thresholds</p>
        ) : (
          data.findings.map((f, i) => (
            <p key={i} className={cn('text-xs', severityClass(f.severity))}>
              <span className="mr-1 font-mono text-[10px] text-neutral-400">
                [{f.kind}]
              </span>
              {f.headline}
            </p>
          ))
        )}
      </div>
    </div>
  )
}

/** One run-level efficiency readout: label + value + hover explanation. */
function Metric({
  label,
  value,
  title,
}: {
  label: string
  value: number | null | undefined
  title: string
}) {
  return (
    <span className="flex items-baseline gap-1.5" title={title}>
      <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-400">
        {label}
      </span>
      <span className="font-mono text-xs tabular-nums text-neutral-800">
        {fmtEff(value)}
      </span>
    </span>
  )
}

/**
 * Run-level, thinning-invariant efficiency — the numbers `camdl fit summary`
 * reports, method-agnostic (PGAS / PMMH / mh-ode / nuts). ESS/iter is min-param
 * ESS ÷ raw sampling iterations (`n_samples × thin`): the algorithm-quality
 * metric to compare samplers with, invariant to thinning. ESS/sec is min-param
 * ESS ÷ wall-clock: the runtime metric (hardware-dependent). Shown only when the
 * authoritative summary carries them (a finished stage on a recent camdl).
 */
function EfficiencyStrip({ data }: { data: DiagnosticsResponse }) {
  if (data.ess_per_iter == null && data.ess_per_sec == null) return null
  return (
    <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1 border-b border-neutral-100 px-3 py-2">
      <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
        efficiency
      </span>
      {data.ess_per_iter != null && (
        <Metric
          label="ESS/iter"
          value={data.ess_per_iter}
          title="effective samples per raw sampling iteration (n_samples × thin) — thinning-invariant algorithm quality; compare samplers with this"
        />
      )}
      {data.ess_per_sec != null && (
        <Metric
          label="ESS/sec"
          value={data.ess_per_sec}
          title="effective samples per second of wall-clock — runtime metric (hardware-dependent)"
        />
      )}
    </div>
  )
}

/** Dense, mono, financial-register table — one row per parameter. */
function ParamTable({
  params,
  perChainCols,
}: {
  params: ParamDiagnostic[]
  /** Per-chain ESS columns: `at` is the position in `ess_per_chain`, `id` the
   *  chain it belongs to (camdl's 1-based name) — they are not the same. */
  perChainCols: { at: number; id: number }[]
}) {
  return (
    // The per-chain columns make this wide; let it scroll on a phone rather than
    // squeezing the numerics.
    <div className="overflow-x-auto">
      <table className="w-full min-w-max border-collapse font-mono text-[11px] tabular-nums">
        <thead>
          <tr className="border-b border-neutral-200 text-[9px] uppercase tracking-wider text-neutral-400">
            <th className="px-2 py-1.5 text-left font-medium">parameter</th>
            <th className="px-2 py-1.5 text-right font-medium">R&#x0302;</th>
            <th className="px-2 py-1.5 text-right font-medium">ess</th>
            <th className="px-2 py-1.5 text-right font-medium">tail-ess</th>
            <th className="px-2 py-1.5 text-right font-medium">mcse</th>
            <th className="px-2 py-1.5 text-right font-medium">sep</th>
            {perChainCols.map((c) => (
              <th key={c.at} className="px-2 py-1.5 text-right font-medium">
                c{c.id}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {params.map((p) => {
            const hasSymbol = Boolean(p.symbol && p.symbol !== p.name)
            return (
              <tr
                key={p.name}
                className="border-b border-neutral-100 last:border-b-0"
              >
                <td className="whitespace-nowrap px-2 py-1.5 text-left">
                  {hasSymbol && (
                    <span className="font-semibold text-neutral-900">
                      {p.symbol}{' '}
                    </span>
                  )}
                  <span className={hasSymbol ? 'text-neutral-400' : 'text-neutral-900'}>
                    {p.name}
                  </span>
                </td>
                <td className={cn('px-2 py-1.5 text-right', rhatClass(p.rhat))}>
                  {fmtRhat(p.rhat)}
                </td>
                <td className={cn('px-2 py-1.5 text-right', essClass(p.ess_bulk))}>
                  {fmtEss(p.ess_bulk)}
                </td>
                <td className={cn('px-2 py-1.5 text-right', essClass(p.ess_tail))}>
                  {fmtEss(p.ess_tail)}
                </td>
                <td className="px-2 py-1.5 text-right text-neutral-500">
                  {fmtValue(p.mcse)}
                </td>
                <td className="px-2 py-1.5 text-right text-neutral-500">
                  {fmtValue(p.sep)}
                </td>
                {perChainCols.map((c) => {
                  const v = p.ess_per_chain[c.at]
                  return (
                    <td
                      key={c.at}
                      className={cn(
                        'px-2 py-1.5 text-right',
                        v != null && v < ESS_LOW
                          ? 'text-red-600'
                          : 'text-neutral-400',
                      )}
                    >
                      {v != null ? fmtEss(v) : '—'}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/**
 * Convergence diagnostics for one run: camdl's verdict (or the watcher's live
 * arviz estimate), per-chain mixing, the per-parameter R̂/ESS table, and the
 * PMMH MAP. Owns the warm-up cutoff — diagnostics recompute as it moves.
 */
export function DiagnosticsTab({
  runId,
  chainIds,
  excludedChains,
  onToggleChain,
  onResetChains,
}: { runId: string } & ChainControls) {
  const [warmupPct, setWarmupPct] = useState(DEFAULT_WARMUP_PCT)
  const chains = includedChains({ chainIds, excludedChains, onToggleChain, onResetChains })
  const { data, isPending, isError, isPlaceholderData } = useDiagnostics(
    runId,
    warmupPct,
    chains,
  )

  const hasContent = Boolean(data && data.params.length > 0 && data.n_tail > 0)
  // The per-chain ESS columns only appear when camdl shipped a per-chain
  // breakdown; the widest param's array sets the column count.
  const perChainN = data
    ? data.params.reduce((m, p) => Math.max(m, p.ess_per_chain.length), 0)
    : 0
  // Column position → chain id. The ids come from the API (camdl names chains
  // 1-based), so these headers read the same as the chain selector's; the
  // fallback keeps a column labelled if an older response omits the ids.
  const perChainCols = Array.from({ length: perChainN }, (_, i) => ({
    at: i,
    id: data?.chain_ids?.[i] ?? i + 1,
  }))

  return (
    <div className="max-w-4xl">
      <Card
        className={cn(
          'overflow-hidden transition-opacity',
          isPlaceholderData && 'opacity-60',
        )}
      >
        <WarmupControl
          value={warmupPct}
          onChange={setWarmupPct}
          cutoff={data?.warmup_cutoff ?? null}
          nTail={data?.n_tail ?? null}
        />

        {chainIds.length > 1 && (
          <ChainSelect
            chainIds={chainIds}
            excluded={excludedChains}
            onToggle={onToggleChain}
            onReset={onResetChains}
          />
        )}

        {isPending && <ForestSkeleton />}

        {isError && (
          <MutedNotice
            bordered={false}
            title="Couldn't load diagnostics"
            detail="The backend returned an error for this run. It may still be warming up."
          />
        )}

        {data && !hasContent && (
          <MutedNotice
            bordered={false}
            title="No draws yet"
            detail="This run hasn't produced post-warmup draws to diagnose. Check back once it has sampled past the cutoff."
          />
        )}

        {data && hasContent && (
          <>
            <Verdict data={data} />
            <EfficiencyStrip data={data} />

            {data.mixing && (
              <>
                <SectionLabel>
                  per-chain mixing · {data.mixing.label}
                </SectionLabel>
                <div className="px-3 py-2">
                  <MixingChart mixing={data.mixing} />
                  {data.mixing.band && (
                    <p className="mt-1 font-mono text-[10px] text-neutral-400">
                      healthy band [{fmtValue(data.mixing.band[0])},{' '}
                      {fmtValue(data.mixing.band[1])}]
                    </p>
                  )}
                </div>
              </>
            )}

            {data.sampler_panels?.map((panel) => (
              <SamplerPanelTable key={panel.id} panel={panel} />
            ))}

            <SectionLabel>per-parameter convergence</SectionLabel>
            <ParamTable params={data.params} perChainCols={perChainCols} />

            {data.map_loglik != null && (
              <div className="border-t border-neutral-100 px-3 py-2">
                <span className="font-mono text-[11px] text-neutral-600">
                  MAP log-lik {fmtValue(data.map_loglik)}
                  {data.map_chain != null ? ` · chain ${data.map_chain}` : ''}
                </span>
              </div>
            )}
          </>
        )}
      </Card>
    </div>
  )
}
