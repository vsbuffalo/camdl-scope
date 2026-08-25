import { Fragment } from 'react'
import type { RunSummary } from '@/api/client'
import { useRun } from '@/api/queries'
import { CopyButton } from '@/components/CopyButton'
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
 * Sampler config keys the ticker already reports in another form — the
 * algorithm/backend pair and the chain count. Repeating them would only
 * lengthen the line.
 */
const CONFIG_ALREADY_SHOWN = new Set(['algorithm', 'backend', 'chains'])

/**
 * Digit-group an integer knob (`1200` → `1,200`). Non-integers are left as
 * written: an optimizer tolerance of `1e-6` would group to `0`.
 */
function formatConfigValue(v: string | number): string {
  if (typeof v !== 'number') return v
  return Number.isInteger(v) ? v.toLocaleString() : String(v)
}

/**
 * Dense monospace identity line — the "ticker" — reading the selected run's
 * identity and shape: `run_id · ALGO/BACKEND · N CHAINS · M PARAMS` plus a
 * status light. Flows and wraps on narrow screens so it never overflows.
 *
 * The sampler's static configuration follows, whatever camdl recorded for the
 * algorithm — a PGAS filter's particle count decides whether its trajectories
 * degenerate, so reading a fit means knowing it.
 */
function Ticker({ run }: { run: RunSummary }) {
  const sep = <span className="text-neutral-300">·</span>
  // Every tab already loads this run's detail, so this shares that cache entry.
  const config = useRun(run.run_id).data?.algorithm_config ?? {}
  const knobs = Object.entries(config).filter(
    ([key]) => !CONFIG_ALREADY_SHOWN.has(key),
  )
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 py-1.5 font-mono text-[11px] text-neutral-500">
      <span className="inline-flex items-center gap-1">
        <span className="text-neutral-700">{run.run_id}</span>
        <CopyButton text={run.run_id} label="run id" />
      </span>
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
      {knobs.map(([key, value]) => (
        <Fragment key={key}>
          {sep}
          <span className="uppercase tabular-nums">
            {formatConfigValue(value)} {key.replace(/_/g, ' ')}
          </span>
        </Fragment>
      ))}
    </div>
  )
}

/**
 * The run's status line, sitting above its progress bar. It carries information
 * beyond the badge only while a run is *live* (the heartbeat phase, a sweep
 * counter, a completion bar) or has *failed* / *stalled* (worth flagging). On a
 * clean `done` finish it would just echo the status already shown in the run
 * dropdown, so it's omitted — one status light, not two.
 */
function ProgressBlurb({ run }: { run: RunSummary }) {
  // Clean terminal state: the dropdown already shows `done`; don't repeat it.
  if (run.status === 'done') return null

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
