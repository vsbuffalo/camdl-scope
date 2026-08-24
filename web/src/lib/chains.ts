/**
 * Run-level chain-exclusion state, owned by the Explore workspace and threaded
 * into the Posterior / Pair / Traces / Diagnostics tabs so a dropped chain stays
 * dropped across all of them (a stuck chain is stuck everywhere — unlike
 * warm-up, which each tab owns as a per-view lens).
 */
/** Discard the first half of each chain unless told otherwise — the
 *  conventional default, and now stated once rather than per tab. */
export const DEFAULT_WARMUP_PCT = 50

export interface ChainControls {
  chainIds: number[]
  excludedChains: Set<number>
  onToggleChain: (id: number) => void
  onResetChains: () => void
  /** The run's warm-up percentage, shared by every tab that summarises draws —
   *  a property of the fit, not of the view looking at it. */
  warmupPct: number
  onWarmupPct: (pct: number) => void
}

/** The include-list a query hook wants: ascending kept ids, or `null` for all.
 *  Takes only the two members it reads, so a caller assembling the object
 *  inline is not forced to supply the whole control surface. */
export function includedChains(
  c: Pick<ChainControls, 'chainIds' | 'excludedChains'>,
): number[] | null {
  if (c.excludedChains.size === 0) return null
  return c.chainIds.filter((id) => !c.excludedChains.has(id))
}
