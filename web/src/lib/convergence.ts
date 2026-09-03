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
      return 'text-danger font-medium'
    case 'warn':
      return 'text-warn font-medium'
    case 'ok':
      return 'text-good'
    case 'neutral':
      return 'text-muted'
  }
}

/** Tailwind classes for a verdict chip: text, background, border. */
export function statusChipClasses(s: string | null | undefined): string {
  switch (asStatus(s)) {
    case 'error':
      return 'bg-danger-surface text-danger-strong border-danger-line'
    case 'warn':
      return 'bg-warn-surface text-warn-strong border-warn-line'
    case 'ok':
      return 'bg-good-surface text-good-strong border-good-line'
    case 'neutral':
      return 'bg-sunken text-muted border-line'
  }
}
