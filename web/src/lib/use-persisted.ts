import { useCallback, useState } from 'react'
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
