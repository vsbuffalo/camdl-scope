import type { ReactNode } from 'react'
import type { CompareResponse, CompareRow } from '@/api/client'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const f2 = (x: number | null | undefined) => (x == null ? '—' : x.toFixed(2))
const f3 = (x: number | null | undefined) => (x == null ? '—' : x.toFixed(3))
const fPit = (x: number | null | undefined) => (x == null ? '—' : x.toFixed(2))

/** Δelpd with an explicit sign — the P&L convention (gain vs baseline). */
function fDelta(x: number | null | undefined): string {
  if (x == null) return '—'
  return (x >= 0 ? '+' : '') + x.toFixed(2)
}

/** E_T (e-value): compact decimal in the interesting band, else scientific. */
function fE(x: number | null | undefined): string {
  if (x == null) return '—'
  if (x === 0) return '0'
  return x >= 1000 || x < 0.001 ? x.toExponential(2) : x.toFixed(3)
}

/** Δelpd sign → P&L color: beats baseline green, worse red, baseline neutral. */
function deltaClass(d: number | null | undefined): string {
  if (d == null) return 'text-faint'
  if (d > 0) return 'text-good'
  if (d < 0) return 'text-danger'
  return 'text-muted'
}

const ALIAS_CHIP =
  'inline-flex shrink-0 items-center border border-line-strong px-1 font-mono text-[10px] font-medium leading-tight text-body-dim'

/**
 * The prequential comparison — the financial-terminal read of out-of-sample
 * predictive accuracy. elpd is absolute (higher = better); Δelpd is paired
 * against the baseline and P&L-colored, bolded when the gap clears 2·se(Δ); E_T
 * is the e-value / Bayes factor; evidence is the Jeffreys tier; PIT₉₀ flags
 * overconfidence. Rows arrive best-first.
 *
 * Wide table on desktop; one card per model on mobile (the 9-column table can't
 * fit a phone, and the long fit labels are aliased to M-tickers throughout).
 */
export function CompareTable({
  data,
  aliasOf,
}: {
  data: CompareResponse
  aliasOf: Map<string, string>
}) {
  return (
    <div className="space-y-2">
      {!data.commensurable && (
        <div className="border border-warn-line bg-warn-surface px-3 py-2 text-[12px] leading-relaxed text-warn-deep">
          Horizons differ across models (T_score mismatch) — Δ columns are not
          commensurable and are suppressed. The absolute elpd / CRPS / PIT are
          still each model's own out-of-sample scores.
        </div>
      )}

      {/* Desktop: the dense table. */}
      <Card className="hidden overflow-hidden sm:block">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-right font-mono text-[12px] tabular-nums">
            <thead>
              <tr className="border-b border-line text-[10px] uppercase tracking-wide text-faint">
                <th className="px-2 py-2 text-left font-medium">—</th>
                <th className="px-3 py-2 text-left font-medium">Model</th>
                <th className="px-2 py-2 font-medium">T</th>
                <th className="px-2 py-2 font-medium">elpd</th>
                <th className="px-2 py-2 font-medium">Δelpd</th>
                <th className="px-2 py-2 font-medium">E_T</th>
                <th className="px-2 py-2 font-medium">se(Δ)</th>
                <th className="px-3 py-2 text-left font-medium">evidence</th>
                <th className="px-2 py-2 font-medium">CRPS</th>
                <th className="px-2 py-2 font-medium">PIT₉₀</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => (
                <Row key={r.run_id} r={r} alias={aliasOf.get(r.run_id) ?? ''} />
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Mobile: one card per model — everything visible, no horizontal scroll. */}
      <div className="space-y-2 sm:hidden">
        {data.rows.map((r) => (
          <CompareCard key={r.run_id} r={r} alias={aliasOf.get(r.run_id) ?? ''} />
        ))}
      </div>
    </div>
  )
}

function Row({ r, alias }: { r: CompareRow; alias: string }) {
  return (
    <tr
      className={cn(
        'border-b border-hairline last:border-b-0',
        r.is_baseline && 'bg-sunken',
      )}
    >
      <td className="px-2 py-2 text-left">
        <span className={ALIAS_CHIP}>{alias}</span>
      </td>
      <td className="px-3 py-2 text-left">
        <div className="flex items-center gap-2">
          <span className="max-w-[20rem] truncate text-ink-dim">
            {r.label}
          </span>
          {r.is_baseline && (
            <span className="shrink-0 text-[10px] uppercase tracking-wide text-faint">
              baseline
            </span>
          )}
        </div>
      </td>
      <td className="px-2 py-2 text-muted">{r.t_score}</td>
      <td className="px-2 py-2 text-ink-dim">{f2(r.elpd)}</td>
      <td
        className={cn(
          'px-2 py-2',
          deltaClass(r.delta_elpd),
          r.gap_is_real && 'font-semibold',
        )}
      >
        {fDelta(r.delta_elpd)}
      </td>
      <td className="px-2 py-2 text-muted">{fE(r.e_t)}</td>
      <td className="px-2 py-2 text-faint">{f2(r.se_delta_elpd)}</td>
      <td className="px-3 py-2 text-left text-body-dim">
        {r.evidence_label ?? '—'}
      </td>
      <td className="px-2 py-2 text-body-dim">{f3(r.mean_crps)}</td>
      <td
        className={cn(
          'whitespace-nowrap px-2 py-2',
          r.overconfident ? 'text-warn' : 'text-muted',
        )}
      >
        {fPit(r.pit_cov90)}
        {r.overconfident && ' ⚠'}
      </td>
    </tr>
  )
}

function Metric({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-[10px] uppercase tracking-wide text-faint">
        {label}
      </span>
      <span>{children}</span>
    </div>
  )
}

function CompareCard({ r, alias }: { r: CompareRow; alias: string }) {
  return (
    <div
      className={cn(
        'border border-line',
        r.is_baseline && 'bg-sunken',
      )}
    >
      <div className="flex items-center gap-2 border-b border-hairline px-3 py-2">
        <span className={ALIAS_CHIP}>{alias}</span>
        <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink-dim">
          {r.label}
        </span>
        {r.is_baseline && (
          <span className="shrink-0 text-[10px] uppercase tracking-wide text-faint">
            baseline
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 px-3 py-2.5 font-mono text-[12px] tabular-nums">
        <Metric label="elpd">
          <span className="text-ink-dim">{f2(r.elpd)}</span>
        </Metric>
        <Metric label="Δelpd">
          <span
            className={cn(deltaClass(r.delta_elpd), r.gap_is_real && 'font-semibold')}
          >
            {fDelta(r.delta_elpd)}
          </span>
        </Metric>
        <Metric label="E_T">
          <span className="text-muted">{fE(r.e_t)}</span>
        </Metric>
        <Metric label="se(Δ)">
          <span className="text-faint">{f2(r.se_delta_elpd)}</span>
        </Metric>
        <Metric label="evidence">
          <span className="text-body-dim">{r.evidence_label ?? '—'}</span>
        </Metric>
        <Metric label="T">
          <span className="text-muted">{r.t_score}</span>
        </Metric>
        <Metric label="CRPS">
          <span className="text-body-dim">{f3(r.mean_crps)}</span>
        </Metric>
        <Metric label="PIT₉₀">
          <span className={r.overconfident ? 'text-warn' : 'text-muted'}>
            {fPit(r.pit_cov90)}
            {r.overconfident && ' ⚠'}
          </span>
        </Metric>
      </div>
    </div>
  )
}
