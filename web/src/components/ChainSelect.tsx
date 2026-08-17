import { useState } from 'react'
import { cn } from '@/lib/utils'
import { chainColor } from '@/lib/colors'
import { DEMO } from '@/lib/demo'

interface ChainSelectProps {
  /** All chain ids in the run, ascending. */
  chainIds: number[]
  /** Chains the user has dropped (empty = all included). */
  excluded: Set<number>
  /** Toggle one chain's inclusion. */
  onToggle: (id: number) => void
  /** Restore all chains. */
  onReset: () => void
}

/**
 * A `chains (N/total)` toggle — sibling to the Pair tab's `⚙ params` control —
 * that opens an inline panel of per-chain checkboxes, coloured to match the
 * trace grid and pair scatter. Unchecking a chain drops it from this run's
 * pair plot, traces, AND diagnostics (R̂/ESS recompute on the retained chains),
 * so a stuck chain can be excluded once and stays excluded across the three
 * tabs. The last remaining chain can't be unchecked — the view never blanks.
 *
 * The swatch is keyed by chain **id** (`chainColor`), the same mapping the pair
 * scatter and trace grid use, so the colour you see misbehaving in a plot is
 * the colour of its checkbox here — and it doesn't shift as chains are dropped.
 * Hidden by the caller when a run has fewer than two chains (nothing to
 * exclude).
 */
export function ChainSelect({
  chainIds,
  excluded,
  onToggle,
  onReset,
}: ChainSelectProps) {
  const [open, setOpen] = useState(false)
  // Static demo: chain filtering refetches with a `chains=` param a static host
  // can't serve, so the control is hidden (all chains shown).
  if (DEMO) return null
  const includedCount = chainIds.length - excluded.size
  const anyExcluded = excluded.size > 0

  return (
    <div className="border-b border-neutral-100 px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex items-center gap-1.5 rounded-sm border border-neutral-200 px-2 py-1 font-mono text-[11px] text-neutral-600 transition-colors hover:bg-neutral-50"
        >
          <span className="text-neutral-400">{open ? '▾' : '▸'}</span>
          <span>chains</span>
          <span className="tabular-nums text-neutral-400">
            ({includedCount}/{chainIds.length})
          </span>
        </button>

        {anyExcluded && (
          <span className="font-mono text-[10px] text-amber-600">
            {excluded.size} excluded — recomputed on the rest
          </span>
        )}

        {open && anyExcluded && (
          <button
            type="button"
            onClick={onReset}
            className="font-mono text-[10px] text-neutral-400 underline-offset-2 hover:text-neutral-700 hover:underline"
          >
            reset
          </button>
        )}
      </div>

      {open && (
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1.5 rounded-sm border border-neutral-200 bg-[#fafafa] p-2.5">
          {chainIds.map((c) => {
            const included = !excluded.has(c)
            // Never let the user uncheck the final included chain.
            const lockedOn = included && includedCount === 1
            return (
              <label
                key={c}
                className={cn(
                  'flex select-none items-center gap-1.5 font-mono text-[11px]',
                  lockedOn
                    ? 'cursor-not-allowed text-neutral-400'
                    : 'cursor-pointer text-neutral-700',
                )}
                title={lockedOn ? 'at least one chain must stay selected' : undefined}
              >
                <input
                  type="checkbox"
                  checked={included}
                  disabled={lockedOn}
                  onChange={() => onToggle(c)}
                  className="h-3 w-3 accent-neutral-700"
                />
                <span
                  className="inline-block h-2 w-3 rounded-[1px]"
                  style={{
                    background: chainColor(c),
                    opacity: included ? 1 : 0.3,
                  }}
                />
                <span className={included ? '' : 'line-through opacity-60'}>
                  c{c}
                </span>
              </label>
            )
          })}
        </div>
      )}
    </div>
  )
}
