/**
 * Log y-axis for count-like series.
 *
 * Epidemic streams are counts, so zero is a legal value — an early `q05` sits
 * on it for most of an outbreak — and a naive log scale sends those points to
 * −∞, breaking the band paths rather than just hiding a point. So the domain
 * spans the *positive* values actually drawn and the scale clamps: a zero pins
 * to the axis floor, which is the convention epidemic curves use (an
 * alternative, symlog, is the only way to place zero truthfully, at the cost
 * of a kink in the scale).
 *
 * Callers pass the values of the marks they are drawing — not of all the data —
 * so hiding a wild outer band rescales the axis to what remains, exactly as the
 * linear auto-domain would.
 */

/** Padded [lo, hi] over the positive values, or ``null`` when a log axis is
 *  not viable (nothing positive to show). */
export function logYDomain(values: Iterable<number>): [number, number] | null {
  let lo = Infinity
  let hi = -Infinity
  for (const v of values) {
    if (!Number.isFinite(v)) continue
    if (v > 0 && v < lo) lo = v
    if (v > hi) hi = v
  }
  if (!Number.isFinite(lo) || hi <= 0) return null
  return [lo * 0.8, hi * 1.1]
}

/**
 * Observable Plot y-scale options for a log axis over ``values``, or an empty
 * object when `log` is off or a log axis is not viable — spread into the scale
 * so the linear default stands:
 *
 * ```ts
 * y: { label: null, grid: true, ...logYOptions(logY, drawn) }
 * ```
 */
export function logYOptions(
  enabled: boolean,
  values: Iterable<number>,
): { type: 'log'; domain: [number, number]; clamp: true } | Record<string, never> {
  if (!enabled) return {}
  const domain = logYDomain(values)
  return domain ? { type: 'log' as const, domain, clamp: true } : {}
}
