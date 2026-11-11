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
  /** Replace the exclusion set outright (the all / none shortcuts). */
  onSetExcluded?: (next: Set<number>) => void
  /** Chains that produced no draws and never will — camdl skips a chain whose
   *  initial log-posterior is non-finite and finishes the run anyway. Shown
   *  greyed and struck through: they are not a choice the reader has, and a
   *  finished fit can be missing half its chains while reporting `done`. */
  deadIds?: number[]
  /** Chains camdl recorded as taking no step after burn-in: every retained draw
   *  is one parameter vector. Tagged, not disabled: they are in the pooled
   *  numbers until the reader drops them. */
  frozenIds?: number[]
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
  onSetExcluded,
  deadIds,
  frozenIds,
}: ChainSelectProps) {
  const [open, setOpen] = useState(false)
  // Static demo: chain filtering refetches with a `chains=` param a static host
  // can't serve, so the control is hidden (all chains shown).
  if (DEMO) return null
  const includedCount = chainIds.length - excluded.size
  const anyExcluded = excluded.size > 0
  const dead = new Set(deadIds ?? [])
  const frozen = new Set(frozenIds ?? [])
  // The chains a reader can actually choose between. A dead chain contributes
  // nothing whether it is "included" or not, so `all` simply clears the
  // exclusions rather than excluding the dead ones — doing the latter made the
  // count fall from 24/24 to 13/24 on a click labelled "all", which reads as a
  // bug even though both numbers were true.
  const live = chainIds.filter((c) => !dead.has(c))
  // "none" cannot mean zero: the backend treats an empty include-list as "all",
  // so a literal none would silently show everything. It leaves the first live
  // chain, which is the "solo one, then add" workflow this exists for.
  const soloFirst = () => {
    const keep = live[0] ?? chainIds[0]
    if (keep == null || !onSetExcluded) return
    onSetExcluded(new Set(chainIds.filter((c) => c !== keep)))
  }

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

        {dead.size > 0 && (
          <span className="font-mono text-[10px] text-neutral-400">
            {dead.size} dead (no draws)
          </span>
        )}

        {frozen.size > 0 && (
          <span
            className="font-mono text-[10px] text-red-600"
            title="camdl's record: every retained draw of these chains is one parameter vector"
          >
            {frozen.size} frozen (took no step after burn-in)
          </span>
        )}

        {anyExcluded && (
          <span className="font-mono text-[10px] text-amber-600">
            {excluded.size} excluded — recomputed on the rest
          </span>
        )}

        {open && onSetExcluded && (
          <div className="flex items-center gap-2 font-mono text-[10px] text-neutral-400">
            <button
              type="button"
              onClick={onReset}
              className="underline-offset-2 transition-colors hover:text-neutral-700 hover:underline"
            >
              all
            </button>
            <span className="text-neutral-300">·</span>
            <button
              type="button"
              onClick={soloFirst}
              title="leaves one chain — an empty selection is read as 'all'"
              className="underline-offset-2 transition-colors hover:text-neutral-700 hover:underline"
            >
              none
            </button>
          </div>
        )}
      </div>

      {open && (
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1.5 rounded-sm border border-neutral-200 bg-[#fafafa] p-2.5">
          {chainIds.map((c) => {
            const included = !excluded.has(c)
            const isDead = dead.has(c)
            // Never let the user uncheck the final included chain.
            const lockedOn = included && includedCount === 1
            return (
              <label
                key={c}
                className={cn(
                  'flex select-none items-center gap-1.5 font-mono text-[11px]',
                  isDead
                    ? 'cursor-not-allowed text-neutral-300'
                    : lockedOn
                      ? 'cursor-not-allowed text-neutral-400'
                      : 'cursor-pointer text-neutral-700',
                )}
                title={
                  isDead
                    ? 'no draws — camdl skipped this chain at initialisation'
                    : frozen.has(c)
                      ? 'camdl: this chain took no step after burn-in; its draws are one parameter vector'
                      : lockedOn
                      ? 'at least one chain must stay selected'
                      : undefined
                }
              >
                <input
                  type="checkbox"
                  checked={included && !isDead}
                  disabled={lockedOn || isDead}
                  onChange={() => onToggle(c)}
                  className="h-3 w-3 accent-neutral-700"
                />
                <span
                  className="inline-block h-2 w-3 rounded-[1px]"
                  style={{
                    background: chainColor(c),
                    opacity: isDead ? 0.15 : included ? 1 : 0.3,
                  }}
                />
                <span
                  className={
                    isDead
                      ? 'line-through opacity-70'
                      : included
                        ? ''
                        : 'line-through opacity-60'
                  }
                >
                  c{c}
                </span>
                {frozen.has(c) && !isDead && (
                  <span className="text-[10px] text-red-600">frozen</span>
                )}
              </label>
            )
          })}
        </div>
      )}
    </div>
  )
}
