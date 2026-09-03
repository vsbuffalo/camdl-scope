import type { RunSummary } from '@/api/client'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from '@/components/ui/select'
import { StatusBadge } from '@/components/StatusBadge'
import { runRows, selectedTitle } from '@/lib/run-groups'

interface RunSelectProps {
  runs: RunSummary[]
  value: string | undefined
  onChange: (runId: string) => void
}

/** Left padding per nesting level: folder → configuration → seed. */
const INDENT = ['pl-2.5', 'pl-6', 'pl-10']

/**
 * The fit dropdown: one row per sampler configuration (camdl's method
 * directory), with replicate seeds nested under their configuration and the
 * configurations of one problem under a folder header. See `run-groups.ts`
 * for why the grouping follows camdl's levels. The closed dropdown names the
 * open seed when its configuration has several.
 */
export function RunSelect({ runs, value, onChange }: RunSelectProps) {
  const selected = runs.find((r) => r.run_id === value)
  const rows = runRows(runs)

  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger
        className="min-w-0 sm:w-[28rem]"
        aria-label="Select a run"
      >
        {selected ? (
          <span className="flex min-w-0 items-center gap-2">
            <StatusBadge status={selected.status} />
            <span className="truncate font-mono text-[13px] text-ink">
              {selectedTitle(selected, runs)}
            </span>
          </span>
        ) : (
          <span className="text-faint">Select a run…</span>
        )}
      </SelectTrigger>
      <SelectContent>
        {rows.map((row) => {
          if (row.kind === 'folder') {
            return (
              <div
                key={row.key}
                className="border-b border-hairline bg-sunken py-1 pl-2.5 pr-8 font-mono text-[11px] text-muted"
              >
                {row.title}
              </div>
            )
          }
          if (row.kind === 'config') {
            return (
              <div
                key={row.key}
                className={`border-b border-hairline py-1 pr-8 font-mono text-[12px] text-body-dim ${INDENT[row.indent]}`}
              >
                {row.title}
                <span className="ml-1.5 text-[11px] text-faint">
                  · {row.nSeeds} seeds
                </span>
              </div>
            )
          }
          return (
            <SelectItem key={row.key} value={row.run.run_id} className={INDENT[row.indent]}>
              <span className="flex items-center gap-2">
                <StatusBadge status={row.run.status} />
                <span className="truncate font-mono text-[13px]">{row.text}</span>
                {row.isDefault && (
                  <span className="font-mono text-[11px] text-faint">default</span>
                )}
              </span>
            </SelectItem>
          )
        })}
      </SelectContent>
    </Select>
  )
}
