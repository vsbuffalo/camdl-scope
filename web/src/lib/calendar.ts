import type { components } from '@/api/types'

/** The fit's time-axis calendar (fit-level, on RunDetail), forwarded from the
 *  CAS artifact sidecar. */
export type Calendar = components['schemas']['Calendar']

const DAY_MS = 86_400_000

/**
 * A converter from a numeric stream ``time`` to a Date, using the fit's
 * calendar (``origin + time × days_per_unit`` days). Returns null when there's
 * no usable calendar — a relative-time fit — so callers keep the numeric axis.
 *
 * Parsed as UTC so day-granularity dates don't drift across timezones; format
 * ticks with UTC too (`Plot`'s default is fine at the year/month granularity
 * these axes land on).
 */
export function dayToDate(
  cal: Calendar | null | undefined,
): ((t: number) => Date) | null {
  if (!cal?.origin) return null
  const originMs = Date.parse(`${cal.origin}T00:00:00Z`)
  if (Number.isNaN(originMs)) return null
  const scale = (cal.days_per_unit || 1) * DAY_MS
  return (t: number) => new Date(originMs + t * scale)
}
