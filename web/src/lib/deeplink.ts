/**
 * Deep links: a URL that opens one exact fit.
 *
 * `?run=<token>&tab=<tab>` — so a fit can be handed over in a message, a ticket
 * or an agent's output and land the reader on the same run, rather than on
 * whatever the app would otherwise have opened.
 *
 * The token is deliberately forgiving, because whoever writes the link usually
 * has something *other* than the run id to hand. camdl prints a leaf path
 * (`results/fits/fit_x-1344469c/pgas-79d6825a/seed_1-…`); `run.json` carries the
 * full 64-hex fit hash; the viewer's ticker copies the 8-hex run id. All three
 * resolve, along with a stem when the store holds only one fit of it.
 *
 * What it will not do is guess. A token matching several runs resolves to
 * *nothing* and says so, because the failure that matters here is silently
 * opening a different fit than the link named — the reader would have no way to
 * tell, and would read the wrong numbers.
 */

/** Shortest hash prefix accepted. Four hex digits over a few hundred runs is
 *  already unlikely to collide, and a collision refuses rather than guesses. */
const HASH_MIN = 4

export interface DeepLink {
  /** Run token exactly as written in the URL (unresolved). */
  run?: string
  /** Inner tab to open, e.g. `diagnostics`. Ignored if the run has no such tab. */
  tab?: string
}

/** How a run token resolved against the store. */
export type RunResolution =
  | { kind: 'ok'; runId: string }
  | { kind: 'none' }
  | { kind: 'ambiguous'; matches: string[] }

/** The content-hash suffix of a run id: `fit_x-1344469c` → `1344469c`. */
function hashOf(runId: string): string {
  const cut = runId.lastIndexOf('-')
  return cut < 0 ? '' : runId.slice(cut + 1)
}

/** Resolve one already-split token (no path separators). */
function matchToken(token: string, runIds: string[]): RunResolution {
  const t = token.toLowerCase()

  const exact = runIds.find((id) => id.toLowerCase() === t)
  if (exact) return { kind: 'ok', runId: exact }

  // A hash, from either end: the run directory carries the first 8 hex digits,
  // while camdl's own output and `run.json` carry all 64. Either may be the
  // prefix of the other, so accept both directions.
  if (new RegExp(`^[0-9a-f]{${HASH_MIN},}$`).test(t)) {
    const byHash = runIds.filter((id) => {
      const h = hashOf(id).toLowerCase()
      return h.length >= HASH_MIN && (h.startsWith(t) || t.startsWith(h))
    })
    if (byHash.length === 1) return { kind: 'ok', runId: byHash[0]! }
    if (byHash.length > 1) return { kind: 'ambiguous', matches: byHash }
  }

  // A config stem, which is unique only when the store holds one fit of it.
  const byPrefix = runIds.filter((id) => id.toLowerCase().startsWith(t))
  if (byPrefix.length === 1) return { kind: 'ok', runId: byPrefix[0]! }
  if (byPrefix.length > 1) return { kind: 'ambiguous', matches: byPrefix }

  return { kind: 'none' }
}

/**
 * Resolve a run token against the run ids in the store.
 *
 * A token containing `/` is treated as a path and each segment is tried in
 * order, so a leaf path pasted straight out of camdl's stdout works: the
 * segment level (`fit_x-1344469c`) is reached before the method and seed levels,
 * and those never match anything the segment did not.
 */
export function resolveRunToken(token: string, runIds: string[]): RunResolution {
  const raw = token.trim()
  if (!raw) return { kind: 'none' }
  let ambiguous: RunResolution | null = null
  for (const seg of raw.split('/').filter(Boolean)) {
    const got = matchToken(seg, runIds)
    if (got.kind === 'ok') return got
    // Remember an ambiguity, but keep looking: a later, more specific segment
    // may still resolve cleanly.
    if (got.kind === 'ambiguous' && ambiguous === null) ambiguous = got
  }
  return ambiguous ?? { kind: 'none' }
}

/** The deep link the current URL carries, if any. */
export function readDeepLink(): DeepLink {
  if (typeof window === 'undefined') return {}
  const p = new URLSearchParams(window.location.search)
  return {
    run: p.get('run') ?? undefined,
    tab: p.get('tab') ?? undefined,
  }
}

/**
 * Reflect what is on screen back into the address bar, so the URL is always a
 * link to the current view and can just be copied.
 *
 * `replaceState`, not `push`: clicking through six tabs should not put six
 * entries in the history and make Back feel broken.
 */
export function writeDeepLink(next: Partial<DeepLink>): void {
  if (typeof window === 'undefined') return
  const url = new URL(window.location.href)
  for (const [key, value] of Object.entries(next)) {
    if (value) url.searchParams.set(key, value)
    else url.searchParams.delete(key)
  }
  window.history.replaceState(null, '', url)
}

/** An absolute, shareable link to one run (optionally on a given tab). */
export function runLink(runId: string, tab?: string): string {
  if (typeof window === 'undefined') return ''
  const url = new URL(window.location.origin + window.location.pathname)
  url.searchParams.set('run', runId)
  if (tab) url.searchParams.set('tab', tab)
  return url.toString()
}
