import type { CompareCandidate, RunSummary } from '@/api/client'
import { StatusBadge } from '@/components/StatusBadge'
import { cn } from '@/lib/utils'

/**
 * The Compare workspace's selector: only runs that carry a `prequential.json`
 * are comparable, so the checklist shows those, and counts the rest as
 * not-comparable rather than hiding why they're absent. A candidate is either a
 * listed run or a standalone particle-filter scoring run (camdl writes one
 * `[method]` per config, so scoring is a run folder of its own, which the run
 * list omits); the latter shows no status badge. Selection drives the
 * `camdl compare` call.
 */
export function PrequentialPicker({
  candidates,
  others,
  runs,
  selected,
  onToggle,
  aliasOf,
}: {
  candidates: CompareCandidate[]
  /** Listed runs without a `prequential.json`. */
  others: number
  /** The run list, for each listed candidate's status. */
  runs: RunSummary[]
  selected: Set<string>
  onToggle: (runId: string) => void
  /** Alias (M1, M2, …) per selected run — shown as a ticker on its row. */
  aliasOf: Map<string, string>
}) {
  const comparable = candidates
  const statusOf = new Map(runs.map((r) => [r.run_id, r.status]))

  return (
    <div className="border border-line">
      <div className="flex items-center justify-between border-b border-hairline px-3 py-2">
        <span className="font-mono text-[11px] uppercase tracking-wide text-muted">
          Models to compare
        </span>
        <span className="font-mono text-[11px] tabular-nums text-faint">
          {selected.size} selected
        </span>
      </div>

      {comparable.length === 0 ? (
        <p className="px-3 py-4 text-sm leading-relaxed text-faint">
          No runs have a <code className="text-muted">prequential.json</code>{' '}
          yet. Score a pfilter stage (
          <code className="text-muted">camdl pfilter --save-prequential</code>
          , or a <code className="text-muted">fit run</code> with a pfilter
          stage) to enable model comparison.
        </p>
      ) : (
        <div className="max-h-64 overflow-y-auto">
          {comparable.map((run) => {
            const on = selected.has(run.run_id)
            return (
              <button
                key={run.run_id}
                type="button"
                onClick={() => onToggle(run.run_id)}
                aria-pressed={on}
                className={cn(
                  'flex w-full items-center gap-2.5 border-b border-hairline px-3 py-1.5 text-left last:border-b-0 hover:bg-sunken',
                  on && 'bg-accent-surface/40',
                )}
              >
                <span
                  className={cn(
                    'flex size-3.5 shrink-0 items-center justify-center border text-[9px] font-bold leading-none text-on-inverse',
                    on ? 'border-accent bg-accent' : 'border-line-strong',
                  )}
                  aria-hidden
                >
                  {on ? '✓' : ''}
                </span>
                {statusOf.has(run.run_id) ? (
                  <StatusBadge status={statusOf.get(run.run_id)!} />
                ) : (
                  <span className="font-mono text-[10px] text-faint">scored</span>
                )}
                <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink-dim">
                  {run.label}
                </span>
                {on && (
                  <span className="shrink-0 border border-line-strong px-1 font-mono text-[10px] font-medium leading-tight text-body-dim">
                    {aliasOf.get(run.run_id)}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      )}

      {others > 0 && (
        <div className="border-t border-hairline px-3 py-1.5 font-mono text-[10px] text-faint">
          {others} run{others === 1 ? '' : 's'} without a prequential.json — not
          comparable
        </div>
      )}
    </div>
  )
}
