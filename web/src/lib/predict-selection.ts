import { useCallback, useSyncExternalStore } from 'react'
import { loadJson, saveJson } from '@/lib/persist'

/**
 * Which `camdl fit predict` output of a run the viewer reads, remembered per run.
 *
 * camdl files predict output per run and per option set (gh#896):
 * `predict/<run_id12>/` for the default options, `predict/<run_id12>-<opts8>/`
 * for any other `--horizon`, `--stream`, `--n-draws`, …. One choice drives every
 * tab that reads predict output (Predictive, Quantities, the Posterior tab's
 * scalar quantities), so it lives here rather than in one tab's state.
 *
 * `null` means "the run's default". A stored key is only a request: the run's
 * own list of predicts decides whether it still exists (see `useRun`), so a
 * predict deleted since it was chosen falls back to the default instead of
 * failing the run.
 */
const listeners = new Set<() => void>()

function storageKey(runId: string): string {
  return `predict:${runId}`
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb)
  return () => listeners.delete(cb)
}

export function usePredictChoice(
  runId: string | undefined,
): [string | null, (key: string | null) => void] {
  const value = useSyncExternalStore(subscribe, () =>
    runId ? loadJson<string | null>(storageKey(runId), null) : null,
  )
  const set = useCallback(
    (key: string | null) => {
      if (!runId) return
      saveJson(storageKey(runId), key)
      listeners.forEach((l) => l())
    },
    [runId],
  )
  return [value, set]
}
