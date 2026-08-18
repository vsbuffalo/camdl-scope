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

/**
 * Is a quantity's declared ``unit`` an ANCHORED time point — a model-time value
 * that should render as a calendar date (`time_of_max(I)` → 2026-08-12)?
 * Duration units ("days", "weeks") are deliberately NOT time points: a
 * generation interval is a length, not a location, and must never be anchored
 * to the calendar origin. camdl's manifest is expected to mark points with
 * ``unit: "time"`` (spec §quantities: "a time (a date in an anchored model)").
 */
export function isTimePointUnit(unit: string | null | undefined): boolean {
  return unit === 'time' || unit === 'date'
}

/** ISO day string (UTC) for a model-time value under the fit's calendar. */
export function fmtModelDate(toDate: (t: number) => Date, t: number): string {
  return toDate(t).toISOString().slice(0, 10)
}
