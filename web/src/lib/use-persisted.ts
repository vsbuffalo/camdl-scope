import { useCallback, useEffect, useState } from 'react'
import { loadJson, saveJson } from '@/lib/persist'

/**
 * `useState` that survives the component leaving the screen.
 *
 * Every tab unmounts when you switch away from it, so a display preference
 * held in plain `useState` — which layers are drawn, whether the axis is log,
 * how the residuals are laid out — silently resets on the way back. That reads
 * as the app forgetting what you told it, and it is why this exists as one
 * helper rather than a `loadJson`/`saveJson` pair hand-written per toggle:
 * scattered pairs are how half the toggles ended up persisted and half not.
 *
 * Persist a preference about *how things are drawn*. Do NOT persist a
 * selection that names something artifact-specific — a scenario arm, a
 * parameter, an index dimension — since restoring it against a different fit
 * either silently selects nothing or selects the wrong thing. Those belong to
 * the run, and are keyed per run where they are worth keeping at all.
 */
export function usePersisted<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => loadJson(key, initial))
  const set = useCallback(
    (next: T) => {
      saveJson(key, next)
      setValue(next)
    },
    [key],
  )
  return [value, set]
}

/**
 * The same, for a `Set` of string keys — the shape every "which layers are
 * hidden" toggle uses. Stored as an array, since a `Set` is not JSON.
 */
export function usePersistedSet<T extends string>(
  key: string,
  initial: readonly T[] = [],
): [ReadonlySet<T>, (v: ReadonlySet<T>) => void] {
  const [value, setValue] = useState<ReadonlySet<T>>(
    () => new Set(loadJson<T[]>(key, [...initial])),
  )
  const set = useCallback(
    (next: ReadonlySet<T>) => {
      saveJson(key, [...next])
      setValue(next)
    },
    [key],
  )
  return [value, set]
}

/**
 * A SELECTION that names artifact-specific things — parameters, scenario arms,
 * objectives — remembered per run.
 *
 * The third category, and the one that was missing. A selection cannot be
 * persisted globally: restoring "show r_eff, tau, gamma" against a different
 * model selects nothing, or worse, a same-named parameter that means something
 * else. But scoped to the run that produced it, and filtered on load against
 * what that run actually has, it is safe — and it is what a reader expects,
 * since choosing which parameters to look at is work they did.
 *
 * Returns `null` until something has been stored, so the caller can apply a
 * data-derived default (a recommended selection that only arrives with the
 * run's metadata) rather than being handed an empty set that looks deliberate.
 */
export function usePersistedRunSelection(
  runId: string,
  kind: string,
): [ReadonlySet<string> | null, (v: ReadonlySet<string>) => void] {
  const key = `${kind}:${runId}`
  const [value, setValue] = useState<ReadonlySet<string> | null>(() => {
    const stored = loadJson<string[] | null>(key, null)
    return stored ? new Set(stored) : null
  })
  // A run switch must re-read rather than carry the previous run's choice.
  useEffect(() => {
    const stored = loadJson<string[] | null>(`${kind}:${runId}`, null)
    setValue(stored ? new Set(stored) : null)
  }, [kind, runId])
  const set = useCallback(
    (next: ReadonlySet<string>) => {
      saveJson(key, [...next])
      setValue(next)
    },
    [key],
  )
  return [value, set]
}

/** Keep only the members that still exist, so a stored selection made against
 *  an earlier version of a run cannot resurrect names it no longer has. */
export function stillValid(
  selection: ReadonlySet<string> | null,
  available: readonly string[],
): Set<string> | null {
  if (selection == null) return null
  const have = new Set(available)
  const kept = [...selection].filter((s) => have.has(s))
  return kept.length > 0 ? new Set(kept) : null
}

/** Toggle one member of a persisted set — the common case for a checkbox row. */
export function toggleInSet<T extends string>(
  current: ReadonlySet<T>,
  member: T,
): ReadonlySet<T> {
  const next = new Set(current)
  if (next.has(member)) next.delete(member)
  else next.add(member)
  return next
}
