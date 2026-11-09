/**
 * Run-level chain-exclusion state, owned by the Explore workspace and threaded
 * into the Posterior / Pair / Traces / Diagnostics tabs so a dropped chain stays
 * dropped across all of them (a stuck chain is stuck everywhere — unlike
 * warm-up, which each tab owns as a per-view lens).
 */
/**
 * A warm-up choice: the share of each chain to discard, or `null` for the
 * warm-up the run declares (its fit.toml `burn_in`). `null` is the default,
 * because camdl's own R̂, ESS and verdict are computed at that cutoff, and a
 * view showing them beside numbers from another cutoff contradicts itself.
 * The server falls back to half of each chain for a run that declares none.
 */
export type WarmupSetting = number | null

export interface ChainControls {
  chainIds: number[]
  excludedChains: Set<number>
  onToggleChain: (id: number) => void
  onResetChains: () => void
  /** Replace the exclusion set outright — the all / none shortcuts. */
  onSetExcludedChains: (next: Set<number>) => void
  /** Chains that produced no draws and never will. A property of the run, so
   *  it travels with the rest of the chain state rather than being refetched
   *  per tab. */
  deadChainIds: number[]
  /** The run's warm-up percentage, shared by every tab that summarises draws —
   *  a property of the fit, not of the view looking at it. */
  warmupPct: WarmupSetting
  onWarmupPct: (pct: WarmupSetting) => void
  /** Where the declared warm-up sits on the slider, `null` when undeclared. */
  warmupDeclaredPct: number | null
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
