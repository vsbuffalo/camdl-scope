/**
 * How the fit dropdown lists the store's runs.
 *
 * camdl files a fit as `<folder>/<method dir>/<seed dir>/`, and each level is
 * hashed from its own slice of the inputs: the folder from the model, data and
 * fit problem, the method directory from the sampler settings (starts
 * included), the seed directory from the seed. So runs sharing a method
 * directory are replicate seeds of one sampler configuration, and runs in
 * different method directories were sampled differently — a warm-started
 * re-run, a longer chain, another algorithm — even when they share a folder.
 *
 * The list follows that: one row per method directory, seeds indented under it
 * only when it holds several, and a folder header only when the folder holds
 * more than one method directory, so configurations of one problem sit
 * together.
 */

/** What grouping needs to know about a run. */
export interface GroupRun {
  run_id: string
  /** camdl's label when the run has one, else the derived one. */
  label: string
  folder: string
  method: string
  method_default: boolean
  seed: number | null
  leaf: string
  /** camdl's own label, when the run has one. */
  user_label: string | null
  /** `<config stem> · <algorithm>/<backend>`. */
  config_label: string
}

export type Row<R extends GroupRun> =
  /** A folder holding several sampler configurations. Not selectable. */
  | { kind: 'folder'; key: string; title: string }
  /** A sampler configuration with several seeds. Not selectable. */
  | { kind: 'config'; key: string; title: string; indent: number; nSeeds: number }
  /** One run. */
  | { kind: 'run'; key: string; run: R; text: string; indent: number; isDefault: boolean }

interface Config<R extends GroupRun> {
  method: string
  runs: R[]
}

interface Folder<R extends GroupRun> {
  folder: string
  configs: Config<R>[]
}

function bySeed(a: GroupRun, b: GroupRun): number {
  return (a.seed ?? 0) - (b.seed ?? 0) || (a.leaf < b.leaf ? -1 : a.leaf > b.leaf ? 1 : 0)
}

/** Text order with digit runs compared as numbers: `tau_3` < `tau_3.5` < `tau_10`. */
function natural(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true })
}

/** A folder's config stem: `m1_profile-9d94fe18` → `m1_profile`. */
function stemOf(folder: string): string {
  const cut = folder.lastIndexOf('-')
  return cut < 0 ? folder : folder.slice(0, cut)
}

/** Folders by config stem, then by their runs' names, so the points of one
 *  `fit run --sweep` (a folder per value, the hash in no useful order) read in
 *  value order; method directories by name within a folder. */
function nest<R extends GroupRun>(runs: R[]): Folder<R>[] {
  const folders = new Map<string, Map<string, R[]>>()
  for (const r of runs) {
    const methods = folders.get(r.folder) ?? new Map<string, R[]>()
    folders.set(r.folder, methods)
    methods.set(r.method, [...(methods.get(r.method) ?? []), r])
  }
  const nested = [...folders].map(([folder, methods]) => ({
    folder,
    configs: [...methods]
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([method, rs]) => ({ method, runs: [...rs].sort(bySeed) })),
  }))
  const nameOf = (f: Folder<R>) =>
    f.configs.map((c) => configTitle(c, f.configs.length > 1)).sort(natural)[0] ?? ''
  return nested.sort(
    (a, b) => natural(stemOf(a.folder), stemOf(b.folder)) || natural(nameOf(a), nameOf(b)),
  )
}

/** A configuration's name among its folder's: camdl's label when one run
 *  carries it, else the method directory — or, alone in its folder, the
 *  run's label. */
function configTitle(c: Config<GroupRun>, shared: boolean): string {
  if (c.runs.length === 1) {
    const r = c.runs[0]!
    return r.user_label ?? (shared ? c.method : r.label)
  }
  return shared ? c.method : c.runs[0]!.config_label
}

function seedText(r: GroupRun): string {
  const seed = r.seed != null ? `seed ${r.seed}` : r.leaf
  return r.user_label ? `${seed} · ${r.user_label}` : seed
}

/** The dropdown's rows, in display order. */
export function runRows<R extends GroupRun>(runs: R[]): Row<R>[] {
  const rows: Row<R>[] = []
  for (const f of nest(runs)) {
    const shared = f.configs.length > 1
    const base = shared ? 1 : 0
    if (shared) rows.push({ kind: 'folder', key: `f:${f.folder}`, title: f.folder })
    for (const c of f.configs) {
      if (c.runs.length === 1) {
        const r = c.runs[0]!
        rows.push({
          kind: 'run', key: r.run_id, run: r, text: configTitle(c, shared),
          indent: base, isDefault: false,
        })
        continue
      }
      rows.push({
        kind: 'config', key: `c:${f.folder}/${c.method}`, title: configTitle(c, shared),
        indent: base, nSeeds: c.runs.length,
      })
      for (const r of c.runs) {
        rows.push({
          kind: 'run', key: r.run_id, run: r, text: seedText(r),
          indent: base + 1, isDefault: r.method_default,
        })
      }
    }
  }
  return rows
}

/** What the closed dropdown says for the open run: enough to tell it from
 *  every other run without opening the list, including which seed it is.
 *  Kept short — the ticker beneath already carries the folder and algorithm,
 *  and a long title truncates the seed, the one part that changes. */
export function selectedTitle(run: GroupRun, runs: GroupRun[]): string {
  const inFolder = runs.filter((r) => r.folder === run.folder)
  const shared = new Set(inFolder.map((r) => r.method)).size > 1
  const siblings = inFolder.filter((r) => r.method === run.method).sort(bySeed)
  const name =
    run.user_label ?? (shared ? run.method : run.label)
  if (siblings.length < 2) return name
  const seed = run.seed != null ? `seed ${run.seed}` : run.leaf
  return `${name} · ${seed} · ${siblings.length} seeds`
}
