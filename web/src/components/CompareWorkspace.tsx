import { useState } from 'react'
import type { CompareResponse } from '@/api/client'
import { useCompare, useRuns } from '@/api/queries'
import { PrequentialPicker } from '@/components/PrequentialPicker'
import { CompareTable } from '@/components/CompareTable'
import { DeltaElpdPlot } from '@/components/DeltaElpdPlot'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

/** Flat baseline picker: `auto` (camdl's argmax-elpd) or a specific model by
 *  its alias — the reference every Δelpd is measured against. */
function BaselineSelect({
  ids,
  aliasOf,
  value,
  onChange,
}: {
  ids: string[]
  aliasOf: Map<string, string>
  value: string | undefined
  onChange: (id: string | undefined) => void
}) {
  const opts: { id: string | undefined; label: string }[] = [
    { id: undefined, label: 'auto (best elpd)' },
    ...ids.map((id) => ({ id, label: aliasOf.get(id) ?? id })),
  ]
  return (
    <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
        Baseline
      </span>
      <div className="flex flex-wrap items-center gap-3">
        {opts.map((o) => (
          <button
            key={o.id ?? 'auto'}
            type="button"
            onClick={() => onChange(o.id)}
            className={cn(
              '-mb-px border-b-2 border-transparent pb-0.5 font-mono text-xs transition-colors',
              o.id === value
                ? 'border-neutral-900 text-neutral-900'
                : 'text-neutral-500 hover:text-neutral-800',
            )}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  )
}

/**
 * Compare models by prequential (out-of-sample) predictive accuracy. Owns its
 * own multi-selection — independent of the Explore workspace — gated to runs
 * that carry a `prequential.json`. The actual scoring is the authoritative
 * `camdl compare` (shelled out server-side); this renders its verdict: the
 * comparison table (cards on mobile), the Δelpd error-bar plot, and the caveats.
 *
 * Each selected model gets a short alias (M1, M2, … by selection order) — a
 * ticker that stands in for the long fit label in the dense table and the plot.
 */
export function CompareWorkspace() {
  const { data, isError } = useRuns()
  const runs = data ?? []
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const selectedIds = runs
    .filter((r) => selected.has(r.run_id))
    .map((r) => r.run_id)

  // Stable alias per selected model, assigned in selection order.
  const aliasOf = new Map(selectedIds.map((id, i) => [id, `M${i + 1}`]))

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  // Baseline: `undefined` = auto (camdl's argmax elpd); ignored if it's no
  // longer among the selected runs.
  const [baseline, setBaseline] = useState<string>()
  const effectiveBaseline =
    baseline && selectedIds.includes(baseline) ? baseline : undefined
  const cmp = useCompare(selectedIds, { baseline: effectiveBaseline })

  return (
    <div>
      <PrequentialPicker
        runs={runs}
        selected={selected}
        onToggle={toggle}
        aliasOf={aliasOf}
      />

      <div className="mt-4">
        {selectedIds.length >= 2 && !isError && (
          <BaselineSelect
            ids={selectedIds}
            aliasOf={aliasOf}
            value={effectiveBaseline}
            onChange={setBaseline}
          />
        )}
        {isError ? (
          <MutedNotice
            title="Backend not reachable"
            detail="Couldn't reach /api. Is camdl-scope running and serving this store?"
          />
        ) : selectedIds.length < 2 ? (
          <MutedNotice
            title="Select at least two models"
            detail="Pick two or more runs with prequential scores to compare their out-of-sample predictive accuracy (elpd, CRPS, calibration)."
          />
        ) : cmp.isError ? (
          <MutedNotice
            title="Comparison failed"
            detail={
              (cmp.error as Error)?.message ??
              'camdl compare did not return a result.'
            }
          />
        ) : !cmp.data ? (
          <Card className="overflow-hidden">
            <ForestSkeleton />
          </Card>
        ) : (
          <CompareResult data={cmp.data} aliasOf={aliasOf} />
        )}
      </div>
    </div>
  )
}

function CompareResult({
  data,
  aliasOf,
}: {
  data: CompareResponse
  aliasOf: Map<string, string>
}) {
  const notes = data.notes ?? []
  const missing = data.missing_prequential ?? []
  const labelOf = new Map(data.rows.map((r) => [r.run_id, r.label]))

  return (
    <div className="space-y-4">
      <CompareTable data={data} aliasOf={aliasOf} />

      {data.commensurable && data.rows.length > 0 && (
        <Card className="overflow-hidden">
          <div className="px-3 pb-1 pt-3 font-mono text-[10px] uppercase tracking-wide text-neutral-400">
            Δelpd ± 2·se vs baseline
          </div>
          <div className="px-2 pb-2">
            <DeltaElpdPlot rows={data.rows} aliasOf={aliasOf} />
          </div>
          {/* Alias legend — decodes the plot's M-tickers on any screen. */}
          <div className="flex flex-wrap gap-x-3 gap-y-1 border-t border-neutral-100 px-3 py-2 font-mono text-[10px] text-neutral-500">
            {[...aliasOf.entries()].map(([id, alias]) => (
              <span key={id} className="flex items-center gap-1">
                <span className="font-semibold text-neutral-700">{alias}</span>
                <span className="text-neutral-300">=</span>
                <span className="truncate">{labelOf.get(id) ?? id}</span>
              </span>
            ))}
          </div>
        </Card>
      )}

      {(notes.length > 0 || missing.length > 0) && (
        <div className="space-y-1 px-1 text-[11px] leading-relaxed text-neutral-500">
          {notes.map((n, i) => (
            <p key={i} className="flex gap-1.5">
              <span className="shrink-0 text-neutral-400">ⓘ</span>
              <span>{n}</span>
            </p>
          ))}
          {missing.length > 0 && (
            <p className="text-neutral-400">
              Dropped (no prequential.json): {missing.join(', ')}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
