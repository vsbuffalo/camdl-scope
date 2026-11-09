import type { RunSummary } from '@/api/client'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from '@/components/ui/select'
import { StatusBadge } from '@/components/StatusBadge'

interface RunSelectProps {
  runs: RunSummary[]
  value: string | undefined
  onChange: (runId: string) => void
}

/**
 * The fit dropdown: one row per problem folder. A folder holding several runs
 * (one per fit seed) is one fit to choose; which of its runs is open is the
 * run bar's seed picker. The row stands for the open run when it is in that
 * folder, else for the folder's default.
 */
export function RunSelect({ runs, value, onChange }: RunSelectProps) {
  const selected = runs.find((r) => r.run_id === value)
  const counts = new Map<string, number>()
  for (const r of runs) counts.set(r.folder, (counts.get(r.folder) ?? 0) + 1)
  const rows = runs.filter((r) =>
    selected && r.folder === selected.folder ? r === selected : r.folder_default,
  )
  const nRuns = (r: RunSummary) => counts.get(r.folder) ?? 1

  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger
        className="min-w-0 sm:w-[22rem]"
        aria-label="Select a run"
      >
        {selected ? (
          <span className="flex min-w-0 items-center gap-2">
            <StatusBadge status={selected.status} />
            <span className="truncate font-mono text-[13px] text-neutral-900">
              {selected.label}
            </span>
          </span>
        ) : (
          <span className="text-neutral-400">Select a run…</span>
        )}
      </SelectTrigger>
      <SelectContent>
        {rows.map((run) => (
          <SelectItem key={run.run_id} value={run.run_id}>
            <span className="flex items-center gap-2">
              <StatusBadge status={run.status} />
              <span className="truncate font-mono text-[13px]">{run.label}</span>
              {nRuns(run) > 1 && (
                <span className="font-mono text-[11px] text-neutral-400">
                  · {nRuns(run)} runs
                </span>
              )}
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
