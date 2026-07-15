/**
 * Run-level chain-exclusion state, owned by the Explore workspace and threaded
 * into the Posterior / Pair / Traces / Diagnostics tabs so a dropped chain stays
 * dropped across all of them (a stuck chain is stuck everywhere — unlike
 * warm-up, which each tab owns as a per-view lens).
 */
export interface ChainControls {
  chainIds: number[]
  excludedChains: Set<number>
  onToggleChain: (id: number) => void
  onResetChains: () => void
}

/** The include-list a query hook wants: ascending kept ids, or `null` for all. */
export function includedChains(c: ChainControls): number[] | null {
  if (c.excludedChains.size === 0) return null
  return c.chainIds.filter((id) => !c.excludedChains.has(id))
}
