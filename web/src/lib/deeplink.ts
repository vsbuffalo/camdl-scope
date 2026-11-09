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
 * resolve, along with a stem when the store holds only one fit of it. A problem
 * folder holding several runs (one per fit seed) opens its default run unless
 * the token names one: a leaf path, a seed run's id, camdl's run_id, or its
 * label.
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

/** What resolution needs to know about each run in the store. A problem folder
 *  holds every run of one problem — one per fit seed — and each is an entry. */
export interface RunRef {
  run_id: string
  /** The problem folder, e.g. `fit_x-1344469c`. */
  folder: string
  /** Whether a bare folder name opens this run. */
  folder_default: boolean
  /** The run's directory under the folder, e.g. `pmmh-83841c1b/seed_6-48b478e8`. */
  leaf: string
  /** camdl's 64-hex run_id, from the run's run.json. */
  leaf_run_id?: string | null
  /** The run's label (camdl's own when it has one). */
  label: string
}

/** The content-hash suffix of a folder name: `fit_x-1344469c` → `1344469c`. */
function hashOf(folder: string): string {
  const cut = folder.lastIndexOf('-')
  return cut < 0 ? '' : folder.slice(cut + 1)
}

function result(ids: string[]): RunResolution {
  const unique = [...new Set(ids)]
  if (unique.length === 1) return { kind: 'ok', runId: unique[0]! }
  if (unique.length > 1) return { kind: 'ambiguous', matches: unique }
  return { kind: 'none' }
}

/** The run a folder name opens. */
function defaultOf(folder: string, runs: RunRef[]): string | undefined {
  return runs.find((r) => r.folder === folder && r.folder_default)?.run_id
}

/** Resolve one already-split token (no path separators). */
function matchToken(token: string, runs: RunRef[]): RunResolution {
  const t = token.toLowerCase().replace(/^@/, '')

  const exact = runs.find((r) => r.run_id.toLowerCase() === t)
  if (exact) return { kind: 'ok', runId: exact.run_id }
  const folder = runs.find((r) => r.folder.toLowerCase() === t)
  if (folder) return { kind: 'ok', runId: defaultOf(folder.folder, runs)! }

  // camdl's `@label` names one run.
  const byLabel = runs.filter((r) => r.label.toLowerCase() === t).map((r) => r.run_id)
  if (byLabel.length) return result(byLabel)

  // A hash: camdl's run_id names one run, and a folder's 8-hex suffix names
  // the problem. Either may be the prefix of the other, so both directions.
  if (new RegExp(`^[0-9a-f]{${HASH_MIN},}$`).test(t)) {
    const byRun = runs
      .filter((r) => {
        const h = (r.leaf_run_id ?? '').toLowerCase()
        return h.length >= HASH_MIN && (h.startsWith(t) || t.startsWith(h))
      })
      .map((r) => r.run_id)
    const byFolder = [
      ...new Set(
        runs
          .filter((r) => {
            const h = hashOf(r.folder).toLowerCase()
            return h.length >= HASH_MIN && (h.startsWith(t) || t.startsWith(h))
          })
          .map((r) => r.folder),
      ),
    ].map((f) => defaultOf(f, runs)!)
    const got = result([...byRun, ...byFolder])
    if (got.kind !== 'none') return got
  }

  // A config stem, which is unique only when the store holds one fit of it.
  const folders = [
    ...new Set(runs.filter((r) => r.folder.toLowerCase().startsWith(t)).map((r) => r.folder)),
  ]
  return result(folders.map((f) => defaultOf(f, runs)!))
}

/**
 * Resolve a run token against the runs in the store.
 *
 * A token containing `/` is treated as a path, so a leaf path pasted straight
 * out of camdl's stdout (`results/fits/fit_x-…/pmmh-…/seed_6-…`) opens that
 * seed's run, and a path that stops at the folder opens the folder's default.
 * Otherwise each segment is tried in order.
 */
export function resolveRunToken(token: string, runs: RunRef[]): RunResolution {
  const raw = token.trim()
  if (!raw) return { kind: 'none' }
  const segs = raw.split('/').filter(Boolean)
  if (segs.length > 1) {
    const path = `/${segs.join('/')}/`
    const byLeaf = runs.filter((r) => path.includes(`/${r.folder}/${r.leaf}/`))
    if (byLeaf.length) return result(byLeaf.map((r) => r.run_id))
  }
  let ambiguous: RunResolution | null = null
  for (const seg of segs) {
    const got = matchToken(seg, runs)
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
