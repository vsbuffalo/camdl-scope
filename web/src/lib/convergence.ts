/**
 * How camdl's verdict is drawn.
 *
 * The verdict is computed once, by camdl, and persisted with the run; the API
 * hands it over as a `kind`, a `status` and the words to show (`VerdictModel`).
 * Nothing here compares a number against a bar. `status` is camdl's category
 * (converged, not converged, a finding's severity) as a colour class, and a run
 * camdl has not assessed — still running, a method with no chains to compare, a
 * leaf written before camdl persisted its verdict — is `neutral`, never green.
 *
 * R̂ and ESS stay on screen as numbers; they are not graded here.
 */

export type VerdictStatus = 'ok' | 'warn' | 'error' | 'neutral'

export function asStatus(s: string | null | undefined): VerdictStatus {
  return s === 'ok' || s === 'warn' || s === 'error' ? s : 'neutral'
}

/** Text colour for a number camdl has (or has not) flagged. */
export function statusTextClass(s: string | null | undefined): string {
  switch (asStatus(s)) {
    case 'error':
      return 'text-red-600 font-medium'
    case 'warn':
      return 'text-amber-600 font-medium'
    case 'ok':
      return 'text-emerald-600'
    case 'neutral':
      return 'text-neutral-500'
  }
}

/** Tailwind classes for a verdict chip: text, background, border. */
export function statusChipClasses(s: string | null | undefined): string {
  switch (asStatus(s)) {
    case 'error':
      return 'bg-red-50 text-red-700 border-red-200'
    case 'warn':
      return 'bg-amber-50 text-amber-700 border-amber-200'
    case 'ok':
      return 'bg-emerald-50 text-emerald-700 border-emerald-200'
    case 'neutral':
      return 'bg-neutral-50 text-neutral-500 border-neutral-200'
  }
}
