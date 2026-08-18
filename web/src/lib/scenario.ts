/**
 * Shared scenario ink — keeps the scenario overlay colors consistent across the
 * Quantities table/ribbons and the Predictive ribbons. `baseline` / `as_fitted`
 * are the dark *reference* arm; intervention scenarios get distinct hues.
 */
export const SCENARIO_REFERENCE = '#171717' // neutral-900

/** The names that mark a sidecar's *reference* arm, in priority order:
 *  camdl's `fitted` sentinel (fit predict's identity patch), the watcher's
 *  `as_fitted` normalization of scenario-less sidecars, and `baseline`
 *  (simulate's identity sentinel / the conventional no-change arm). The
 *  reference is the posterior predictive itself — scenarios overlay it. */
const REFERENCE_NAMES = ['fitted', 'as_fitted', 'baseline'] as const

/** The reference arm present in this scenario set, or null (e.g. a predict
 *  run with an explicit --scenario list that omitted the fitted sentinel). */
export function referenceScenario(scenarios: readonly string[]): string | null {
  for (const name of REFERENCE_NAMES) if (scenarios.includes(name)) return name
  return null
}

const PALETTE = [
  '#1d4ed8', '#b45309', '#047857', '#be123c',
  '#6d28d9', '#0e7490', '#a16207', '#9f1239',
] as const

/** Stable scenario→color, assigned in the given order. */
export function buildScenarioColors(scenarios: string[]): Map<string, string> {
  const m = new Map<string, string>()
  const ref = referenceScenario(scenarios)
  let i = 0
  for (const s of scenarios) {
    if (s === ref) m.set(s, SCENARIO_REFERENCE)
    else m.set(s, PALETTE[i++ % PALETTE.length]!)
  }
  return m
}
