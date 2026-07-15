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
