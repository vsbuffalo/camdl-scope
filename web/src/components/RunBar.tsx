import type { RunSummary } from '@/api/client'
import { RunSelect } from '@/components/RunSelect'
import { StatusBadge } from '@/components/StatusBadge'

interface RunBarProps {
  runs: RunSummary[]
  value: string | undefined
  onChange: (runId: string) => void
  loading: boolean
  error: boolean
}

function RunArea({ runs, value, onChange, loading, error }: RunBarProps) {
  if (error) {
    return (
      <span className="font-mono text-[11px] text-neutral-400">
        backend offline
      </span>
    )
  }
  if (loading) {
    return (
      <div className="h-8 w-full animate-pulse rounded-none bg-neutral-100 sm:w-[22rem]" />
    )
  }
  if (runs.length === 0) {
    return <span className="font-mono text-[11px] text-neutral-400">no runs</span>
  }
  return <RunSelect runs={runs} value={value} onChange={onChange} />
}

/**
 * Dense monospace identity line — the "ticker" — reading the selected run's
 * identity and shape: `run_id · ALGO/BACKEND · N CHAINS · M PARAMS` plus a
 * status light. Flows and wraps on narrow screens so it never overflows.
 */
function Ticker({ run }: { run: RunSummary }) {
  const sep = <span className="text-neutral-300">·</span>
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 py-1.5 font-mono text-[11px] text-neutral-500">
      <span className="text-neutral-700">{run.run_id}</span>
      {sep}
      <span className="uppercase">
        {run.algorithm}/{run.backend}
      </span>
      {sep}
      {run.fit_kind === 'mle' ? (
        <span className="uppercase">MLE</span>
      ) : (
        <span className="tabular-nums">{run.n_chains} CHAINS</span>
      )}
      {sep}
      <span className="tabular-nums">{run.n_params} PARAMS</span>
    </div>
  )
}

/**
 * The run's status line — the single status indicator, sitting above its
 * progress bar. Leads with the {@link StatusBadge} (swatch + status) for every
 * run, so it's not duplicated in the ticker. For a live run it adds the
 * heartbeat phase (burn-in vs sampling, when known), a sweep counter, and a
 * completion bar — driven by camdl's `progress.json` when present, else the
 * trace's `max_iter / target_sweeps`. A clean failure appends its reason.
 */
function ProgressBlurb({ run }: { run: RunSummary }) {
  const p = run.progress
  const failed = run.status === 'failed' || p?.state === 'failed'
  const live = run.status === 'running' || run.status === 'warming'

  // Only a real heartbeat phase (burn-in / sampling) — never fall back to the
  // status, which would just echo the badge.
  const phase = live && p?.phase ? p.phase.replace(/_/g, '-') : null

  // Sweep-position fallback when there's no heartbeat pct: last stored sweep
  // against the configured total (clamped — max_iter is a raw sweep index).
  const sweepPct =
    run.max_iter != null && run.target_sweeps
      ? Math.min(100, Math.round((100 * run.max_iter) / run.target_sweeps))
      : null
  const pct = live ? (p?.pct ?? sweepPct) : null

  const counter =
    live && p?.step != null && p?.total != null
      ? `${p.step.toLocaleString()}/${p.total.toLocaleString()}`
      : live && run.max_iter != null && run.target_sweeps
        ? `${run.max_iter.toLocaleString()}/${run.target_sweeps.toLocaleString()}`
        : null

  return (
    <div className="py-1">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 font-mono text-[11px] text-neutral-500">
        <StatusBadge status={run.status} />
        {phase && (
          <>
            <span className="text-neutral-300">·</span>
            <span className="uppercase tracking-wide">{phase}</span>
          </>
        )}
        {counter && (
          <>
            <span className="text-neutral-300">·</span>
            <span className="tabular-nums">{counter}</span>
          </>
        )}
        {pct != null && (
          <span className="tabular-nums text-neutral-400">{pct}%</span>
        )}
        {failed && p?.reason && (
          <>
            <span className="text-neutral-300">·</span>
            <span className="min-w-0 truncate text-red-500">{p.reason}</span>
          </>
        )}
      </div>
      {pct != null && (
        <div className="mt-1 h-1 w-full max-w-[22rem] overflow-hidden bg-neutral-100">
          <div
            className="h-full bg-blue-700 transition-[width] duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
    </div>
  )
}

/**
 * The Explore workspace's selector bar: the run dropdown over its identity
 * ticker (and, for a live or failed run, a progress blurb). Lives in the
 * content column and is left-aligned with the panels below it.
 */
export function RunBar(props: RunBarProps) {
  const { runs, value } = props
  const selected = runs.find((r) => r.run_id === value)

  return (
    <div className="mb-4 border-b border-neutral-200 pb-1">
      <div className="flex min-w-0 items-center py-1">
        <RunArea {...props} />
      </div>
      {selected && <Ticker run={selected} />}
      {selected && <ProgressBlurb run={selected} />}
    </div>
  )
}
