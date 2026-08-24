/**
 * Tiny localStorage-backed persistence for UI settings that should survive a
 * page reload (the selected run, active tab, per-run chain exclusions). Kept
 * deliberately small: JSON in/out under a namespaced key, and every access is
 * guarded so a disabled/full store (private mode, quota) degrades to in-memory
 * state rather than throwing. This is view state, not data — never load-bearing.
 */
const PREFIX = 'camdl-watch:'

export function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    return raw == null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}

export function saveJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value))
  } catch {
    // localStorage unavailable or over quota — settings just won't persist.
  }
}

/** Storage key for a run's excluded-chain set (per-run memory). */
export function excludedChainsKey(runId: string): string {
  return `excluded:${runId}`
}

/** Storage key for a run's warm-up percentage (per-run memory).
 *
 *  Per RUN, not per tab: how much of a chain is burn-in is a property of the
 *  sampling, so a cutoff chosen on the traces has to be the same cutoff the
 *  forest and the diagnostics summarise — otherwise two tabs report different
 *  numbers for one fit and neither says which draws it used. */
export function warmupKey(runId: string): string {
  return `warmup:${runId}`
}
