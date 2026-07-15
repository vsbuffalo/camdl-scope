/**
 * The "by index" profile shared by the Predictive and Quantities tabs: take a
 * value that varies over (time, index dims…), put one chosen index dim on x,
 * marginalise time and every other dim by mean, and split into one line per
 * level of an optional facet dim. Fully general over a model's index tensor —
 * no dimension or value name is hard-coded.
 */

// Categorical palette for index-level colouring (village / age / …); cycles
// when a dimension has more levels than colours.
export const LEVEL_PALETTE = [
  '#2563eb', '#16a34a', '#dc2626', '#9333ea', '#ea580c',
  '#0891b2', '#ca8a04', '#db2777', '#4f46e5', '#65a30d',
]
export const NEUTRAL = '#525252' // neutral-600 — the single line when unfaceted

/** Levels of `dim` restricted to those `present`, in the schema's canonical
 *  order (age/village labels don't sort lexicographically); any present level
 *  not in the schema is appended in sorted order. */
export function orderedLevels(
  dim: string,
  present: string[],
  dimLevels: Map<string, string[]>,
): string[] {
  const seen = new Set(present)
  const canonical = dimLevels.get(dim) ?? []
  const ordered = canonical.filter((l) => seen.has(l))
  const extras = [...seen].filter((l) => !canonical.includes(l)).sort()
  return [...ordered, ...extras]
}

/** One per-(stratum, time) sample fed to the aggregator. `pred` is the modelled
 *  value (a predictive median or a quantity median); `obs` is the observed
 *  overlay where one exists (predictive) or null (a derived quantity). */
export type IndexRecord = {
  stratum: Record<string, string>
  pred: number | null
  obs: number | null
}

export type ByIndexPoint = { x: string; xi: number; pred: number | null; obs: number | null }
export type ByIndexFacet = { level: string; color: string; points: ByIndexPoint[] }
export type ByIndexProfile = { xLevels: string[]; facets: ByIndexFacet[] }

/** Marginalise `records` over time (and every dim except `xDim`/`facetDim`) by
 *  mean, giving the mean pred/obs per (x-level, facet-level). x levels come out
 *  in schema order; facets are coloured from `LEVEL_PALETTE` (or `NEUTRAL` when
 *  unfaceted). */
export function buildByIndexProfile(
  records: readonly IndexRecord[],
  xDim: string,
  facetDim: string | null,
  dimLevels: Map<string, string[]>,
): ByIndexProfile {
  const acc = new Map<
    string,
    { x: string; f: string; predSum: number; predN: number; obsSum: number; obsN: number }
  >()
  for (const r of records) {
    const xL = r.stratum[xDim] ?? ''
    const fL = facetDim ? (r.stratum[facetDim] ?? '') : ''
    const key = JSON.stringify([xL, fL])
    let e = acc.get(key)
    if (!e) {
      e = { x: xL, f: fL, predSum: 0, predN: 0, obsSum: 0, obsN: 0 }
      acc.set(key, e)
    }
    if (r.pred != null && Number.isFinite(r.pred)) {
      e.predSum += r.pred
      e.predN += 1
    }
    if (r.obs != null && Number.isFinite(r.obs)) {
      e.obsSum += r.obs
      e.obsN += 1
    }
  }
  const cells = [...acc.values()]
  const order = orderedLevels(xDim, cells.map((c) => c.x), dimLevels)
  const xIndex = new Map(order.map((l, i) => [l, i]))
  const facetLevels = orderedLevels(facetDim ?? '', cells.map((c) => c.f), dimLevels)
  const facets = facetLevels.map((level, i) => ({
    level,
    color: facetDim ? (LEVEL_PALETTE[i % LEVEL_PALETTE.length] ?? NEUTRAL) : NEUTRAL,
    points: cells
      .filter((c) => c.f === level)
      .map((c) => ({
        x: c.x,
        xi: xIndex.get(c.x) ?? 0,
        pred: c.predN ? c.predSum / c.predN : null,
        obs: c.obsN ? c.obsSum / c.obsN : null,
      }))
      .sort((a, b) => a.xi - b.xi),
  }))
  return { xLevels: order, facets }
}
