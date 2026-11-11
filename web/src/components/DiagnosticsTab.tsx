import { useEffect, useRef, useState, type ReactNode } from 'react'
import * as Plot from '@observablehq/plot'
import type {
  ChainMixing,
  DiagnosticsResponse,
  LeafRowModel,
  ParamDiagnostic,
  SamplerPanel,
} from '@/api/client'
import { useDiagnostics } from '@/api/queries'
import { ChainSelect } from '@/components/ChainSelect'
import { ChainLoglikTable } from '@/components/ChainLoglikTable'
import { PlotDownloadButton } from '@/components/PlotDownloadButton'
import { includedChains, type ChainControls } from '@/lib/chains'
import { WarmupControl } from '@/components/WarmupControl'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { TargetDecomposition } from '@/components/TargetDecomposition'
import { CrossChainMatrix } from '@/components/CrossChainMatrix'
import { ParticleHealth, fmtFraction } from '@/components/ParticleHealth'
import { fmtEss, fmtRhat, fmtTick, fmtValue } from '@/lib/format'
import { statusChipClasses, statusTextClass } from '@/lib/convergence'
import { cn } from '@/lib/utils'


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

/** R̂ colour is camdl's (findings and verdict, via the API's `rhat_status`);
 *  ESS carries no status because camdl assigns none per parameter. Both
 *  numbers are shown regardless. */
function rhatClass(p: ParamDiagnostic): string {
  if (p.rhat == null && p.rhat_text == null) return 'text-neutral-400'
  return statusTextClass(p.rhat_status)
}

function essClass(ess: number | null | undefined): string {
  return ess == null ? 'text-neutral-300' : 'text-neutral-400'
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

/**
 * A section that opens on demand. Closed by default: these answer a follow-up
 * question ("what KIND of disagreement is this?") that only arises once the
 * verdict above has raised one, and left open they push the per-parameter
 * table off the first screen for every fit, including the healthy ones.
 */
function CollapsibleSection({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border-b border-neutral-100 last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-neutral-400 transition-colors hover:bg-neutral-50 hover:text-neutral-600"
      >
        <span className="text-neutral-300">{open ? '▾' : '▸'}</span>
        {label}
      </button>
      {open && children}
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
                      {/* Proportions (acceptance, renewal) read as percentages;
                          counts and step sizes keep their native form. The
                          backend declares which is which — see FRACTION_COLUMNS. */}
                      {v == null
                        ? '—'
                        : c.is_fraction
                          ? fmtFraction(v)
                          : fmtValue(v)}
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

/** Every method leaf of the problem folder, whichever is open: completed,
 *  running and failed, each with camdl's verdict for it. Shown only when the
 *  folder holds more than the open leaf, since one leaf is the verdict above. */
function LeafTable({ leaves }: { leaves: LeafRowModel[] }) {
  if (leaves.length <= 1) return null
  return (
    <table className="mt-2 w-full border-collapse font-mono text-[11px]">
      <thead>
        <tr className="text-left text-neutral-400">
          <th className="py-0.5 pr-3 font-normal">method</th>
          <th className="py-0.5 pr-3 font-normal">leaf</th>
          <th className="py-0.5 pr-3 font-normal">status</th>
          <th className="py-0.5 font-normal">camdl&apos;s verdict</th>
        </tr>
      </thead>
      <tbody>
        {leaves.map((l) => (
          <tr key={l.leaf} className={l.is_open ? 'bg-neutral-50' : undefined}>
            <td className="py-0.5 pr-3 text-neutral-700">{l.method}</td>
            <td className="py-0.5 pr-3 text-neutral-400">
              {l.leaf}
              {l.is_open ? ' (open)' : ''}
            </td>
            <td className="py-0.5 pr-3 text-neutral-500">{l.status}</td>
            <td className={statusTextClass(l.verdict.status)}>{l.verdict.text}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** The verdict header + the findings list (or the all-clear line). */
function Verdict({ data }: { data: DiagnosticsResponse }) {
  const isCamdl = data.source === 'camdl'
  // camdl can give a verdict while giving no R̂ (a frozen chain makes it
  // undefined), so the numbers below are then the live estimate under camdl's
  // verdict. A leaf with no report (running, failed, not recorded) has no
  // verdict to speak of.
  const hasCamdlVerdict = !['running', 'failed', 'not_recorded', 'unreadable'].includes(
    data.verdict.kind,
  )
  return (
    <div className="border-b border-neutral-100 px-3 py-2.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="font-mono text-[11px] text-neutral-500">
          {isCamdl
            ? `camdl · ${data.stage ?? '—'} verdict`
            : hasCamdlVerdict
              ? 'camdl verdict; the numbers below are a live arviz estimate (camdl reported no R̂)'
              : 'live arviz estimate — camdl has not assessed this run'}
        </span>
        <span
          className="font-mono text-[10px] uppercase tracking-wider text-neutral-400"
          title="objective the diagnostics summarise"
        >
          {data.logpost_label}
        </span>
      </div>

      {/* Dead chains get a banner, not a line: a run can report "done" with
          half its chains never having sampled, and the posterior you are
          reading is then built from the survivors only — a selected sample,
          since the chains that died are exactly those whose starting points
          the model could not evaluate. */}
      {data.n_chains_dead > 0 && (
        <div className="mt-1.5 rounded-sm border border-red-300 bg-red-50 px-2 py-1.5">
          <p className="font-mono text-[11px] font-medium text-red-700">
            {data.n_chains_dead} of{' '}
            {data.n_chains + data.n_chains_dead} chains produced no draws
            {data.dead_chain_ids.length > 0 && (
              <> — c{data.dead_chain_ids.join(', c')}</>
            )}
          </p>
          <p className="mt-0.5 text-[11px] leading-snug text-red-700/80">
            Everything below is computed from the {data.n_chains} surviving
            chain{data.n_chains === 1 ? '' : 's'}. Chains skipped at
            initialisation are not a random subset — see the findings for the
            reason camdl gave.
          </p>
        </div>
      )}

      {data.n_chains_warming > 0 && (
        <p className="mt-1 font-mono text-[11px] text-amber-600">
          {data.n_chains_warming} chain{data.n_chains_warming > 1 ? 's' : ''} still
          warming up (no post-warm-up draws) — R̂/ESS computed over the{' '}
          {data.n_chains} with draws.
        </p>
      )}

      {/* camdl's verdict, as persisted: the sentence, never a grade of ours. */}
      <p
        className={cn(
          'mt-1.5 inline-block rounded-sm border px-1.5 py-0.5 font-mono text-[11px] font-medium',
          statusChipClasses(data.verdict.status),
        )}
      >
        {data.verdict.text}
      </p>

      <LeafTable leaves={data.leaves} />

      <div className="mt-1.5 space-y-1">
        {data.findings.length === 0 ? (
          <p className="text-xs text-neutral-500">
            camdl has reported no findings for this run yet
          </p>
        ) : (
          data.findings.map((f, i) => (
            <p key={i} className={cn('text-xs', severityClass(f.severity))}>
              <span className="mr-1 font-mono text-[10px] text-neutral-400">
                [{f.kind}]
              </span>
              {f.headline}
              {/* camdl's own sentence for each chain it raised this finding
                  for, one line per chain. */}
              {f.chain_messages.length > 0 && (
                <span className="mt-0.5 block space-y-0.5 pl-3 text-[11px] leading-snug">
                  {f.chain_messages.map((m) => (
                    <span key={m.chain} className="block">
                      <span className="font-mono text-neutral-500">c{m.chain}</span>{' '}
                      {m.message}
                    </span>
                  ))}
                </span>
              )}
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
  const unreportable = data.ess_unreportable ?? []
  if (data.ess_per_iter == null && data.ess_per_sec == null && unreportable.length === 0)
    return null
  return (
    <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1 border-b border-neutral-100 px-3 py-2">
      <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
        efficiency
      </span>
      {/* A withheld efficiency is a finding, not a gap: the minimum is bounded
          by a parameter whose chains disagree, so any number here would be the
          minimum over the converged parameters only — which improves as the fit
          gets worse. Say which parameters, the way camdl's fit summary does. */}
      {unreportable.length > 0 && (
        <span className="font-mono text-[11px] text-amber-600">
          not reportable — no ESS for{' '}
          <span className="font-medium">{unreportable.slice(0, 3).join(', ')}</span>
          {unreportable.length > 3 && ` (+${unreportable.length - 3} more)`}{' '}
          {/* Only an older camdl left ESS blank because chains disagreed; the
              current one never withholds bulk-ESS for a high R̂ (camdl
              4e418b17), so a blank there is a refusal, not that diagnosis. */}
          {data.stats_vintage === 'classic' ? '(chains disagree)' : '(camdl reported none)'}
        </span>
      )}
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

/** Column heads by what the numbers are. camdl changed the statistics behind
 *  `rhat` / `ess` without changing the keys, and a stored fit keeps the older
 *  ones, so the head says which a reader is looking at. */
const VEHTARI = 'Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021), Bayesian Analysis 16(2):667–718'
const HEAD = {
  rank_bulk: {
    rhat: 'R̂ rank',
    rhatTitle: `max of rank-normalized and folded split-R̂ — ${VEHTARI}`,
    ess: 'bulk-ess',
    essTitle: `bulk effective sample size — ${VEHTARI}`,
  },
  classic: {
    rhat: 'R̂ classic',
    rhatTitle: 'Gelman–Rubin R̂, as written by camdl before 2026-08-22 (no rank normalization)',
    ess: 'ess Σchain',
    essTitle: 'sum of per-chain ESS, as written by camdl before 2026-08-22; left blank when chains disagreed',
  },
} as const
function vintageKey(v: string | null): keyof typeof HEAD {
  return v === 'classic' ? 'classic' : 'rank_bulk'
}

/** Dense, mono, financial-register table — one row per parameter. */
function ParamTable({
  params,
  perChainCols,
  vintage,
  frozen,
}: {
  params: ParamDiagnostic[]
  /** Chains camdl recorded as taking no step after burn-in. */
  frozen: Set<number>
  /** Which R̂ / ESS the numbers are: camdl's current rank-normalized R̂ and
   *  bulk-ESS, its older classic R̂ and summed ESS (same JSON keys), or `null`
   *  for the live arviz estimate, which is rank-normalized / bulk. */
  vintage: string | null
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
            <th className="px-2 py-1.5 text-right font-medium" title={HEAD[vintageKey(vintage)].rhatTitle}>
              {HEAD[vintageKey(vintage)].rhat}
            </th>
            <th className="px-2 py-1.5 text-right font-medium" title={HEAD[vintageKey(vintage)].essTitle}>
              {HEAD[vintageKey(vintage)].ess}
            </th>
            <th className="px-2 py-1.5 text-right font-medium">tail-ess</th>
            <th className="px-2 py-1.5 text-right font-medium">mcse</th>
            <th className="px-2 py-1.5 text-right font-medium">sep</th>
            {perChainCols.map((c) => (
              <th
                key={c.at}
                className={cn(
                  'px-2 py-1.5 text-right font-medium',
                  frozen.has(c.id) && 'text-red-600',
                )}
                title={frozen.has(c.id) ? 'camdl: this chain took no step after burn-in' : undefined}
              >
                c{c.id}
                {frozen.has(c.id) && ' frozen'}
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
                <td
                  className={cn('px-2 py-1.5 text-right', rhatClass(p))}
                  title={p.rhat == null && p.rhat_refused ? `camdl reported no R̂: ${p.rhat_refused.replace(/_/g, ' ')}` : undefined}
                >
                  {p.rhat == null && p.rhat_refused ? (
                    <span className="text-neutral-400">{p.rhat_refused.replace(/_/g, ' ')}</span>
                  ) : (
                    (p.rhat_text ?? fmtRhat(p.rhat))
                  )}
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
                      className="px-2 py-1.5 text-right text-neutral-400"
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
  onSetExcludedChains,
  deadChainIds,
  frozenChainIds,
  warmupPct,
  onWarmupPct,
  warmupDeclaredPct,
}: { runId: string } & ChainControls) {
  const chains = includedChains({ chainIds, excludedChains })
  const { data, isPending, isError, isPlaceholderData } = useDiagnostics(
    runId,
    warmupPct,
    chains,
  )

  // R̂/ESS need a post-warm-up tail in EVERY chain; a running job with chains
  // that started staggered has none until the last one catches up.
  const hasContent = Boolean(data && data.params.length > 0 && data.n_tail > 0)
  // What the sampler reports about itself, straight off the trace's diagnostic
  // columns — available from the first sweep, so it is shown independently.
  const hasLive = Boolean(data?.mixing || data?.sampler_panels?.length)
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
          declaredPct={warmupDeclaredPct}
          onChange={onWarmupPct}
          cutoff={data?.warmup_cutoff ?? null}
          nTail={data?.n_tail ?? null}
        />

        {chainIds.length > 1 && (
          <ChainSelect
            chainIds={chainIds}
            excluded={excludedChains}
            onToggle={onToggleChain}
            onReset={onResetChains}
            onSetExcluded={onSetExcludedChains}
            deadIds={deadChainIds}
            frozenIds={frozenChainIds}
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

        {/* The verdict leads when there is one to give. */}
        {data && hasContent && (
          <>
            <Verdict data={data} />
            <EfficiencyStrip data={data} />
          </>
        )}

        {data && !hasContent && (
          <MutedNotice
            bordered={false}
            title={hasLive ? 'No convergence diagnostics yet' : 'No draws yet'}
            detail={
              hasLive
                ? "R̂ and ESS need a post-warm-up tail in every chain, and this run doesn't have one yet — chains start staggered, so an early sweep leaves one short. What the sampler reports about itself is below, live from the first sweep."
                : "This run hasn't produced post-warmup draws to diagnose. Check back once it has sampled past the cutoff."
            }
          />
        )}

        {/* Live sampler telemetry — renewal / acceptance and the per-chain
            counters. Deliberately OUTSIDE the convergence gate: these are read
            straight off the trace's diagnostic columns and exist from the
            first sweep, whereas R̂/ESS need a post-warm-up tail in every chain.
            Gating them together hid the renewal rate for exactly the window a
            running job is being watched in. */}
        {data?.mixing && (
          <>
            <SectionLabel>per-chain mixing · {data.mixing.label}</SectionLabel>
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

        {data?.sampler_panels?.map((panel) => (
          <SamplerPanelTable key={panel.id} panel={panel} />
        ))}

        {data && (
          <>
            <SectionLabel>per-chain log-likelihood · camdl&apos;s comparison</SectionLabel>
            <ChainLoglikTable
              record={data.chain_logliks}
              error={data.chain_logliks_error}
              frozenChains={data.frozen_chains}
            />
          </>
        )}

        {data?.particle_health && (
          <>
            <SectionLabel>particle health</SectionLabel>
            <ParticleHealth health={data.particle_health} />
          </>
        )}

        {data?.decomposition && (
          <CollapsibleSection label="what separates the chains">
            <TargetDecomposition dec={data.decomposition} />
          </CollapsibleSection>
        )}

        {data?.cross_chain && (
          <CollapsibleSection label="cross-chain path compatibility">
            <CrossChainMatrix compat={data.cross_chain} />
          </CollapsibleSection>
        )}

        {data && hasContent && (
          <>
            <SectionLabel>per-parameter convergence</SectionLabel>
            <ParamTable
              params={data.params}
              perChainCols={perChainCols}
              vintage={data.stats_vintage ?? null}
              frozen={new Set(data.frozen_chains ?? [])}
            />

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
