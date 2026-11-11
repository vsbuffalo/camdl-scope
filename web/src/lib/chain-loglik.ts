/**
 * How camdl's per-chain log-likelihood comparison is read out.
 *
 * camdl compares each chain's mean log-likelihood with the pooled mean of the
 * others, holds Z = |diff| / se against a cutoff, and persists the result, flag
 * included. Nothing here recomputes any of it: this maps the persisted record to
 * the words in each cell, and a reason camdl gave for not assessing a chain to a
 * sentence.
 */
import type { components } from '@/api/types'

export type ChainLoglikRow = components['schemas']['ChainLoglikRow']

/** camdl's reason codes for a chain it did not assess, in words. An unknown
 *  code is shown as written (underscores to spaces), never dropped. */
const REASONS: Record<string, string> = {
  non_finite_mean: 'mean is not finite',
  too_few_effective_draws: 'too few effective draws',
  no_standard_error: 'no standard error',
  too_few_chains: 'fewer than two chains to compare',
}

export function reasonText(row: Pick<ChainLoglikRow, 'reason' | 'ess'>): string {
  const base = row.reason
    ? (REASONS[row.reason] ?? row.reason.replace(/_/g, ' '))
    : 'no reason recorded'
  return row.ess != null ? `${base} (ESS ${row.ess})` : base
}

export interface ChainLoglikCells {
  mean: string
  diff: string
  z: string
  cutoff: string
  /** `flagged` is camdl's own verdict on the chain; `not assessed` carries its
   *  reason in `note`. */
  flag: 'flagged' | 'not flagged' | 'not assessed'
  note: string | null
}

export function chainLoglikCells(row: ChainLoglikRow): ChainLoglikCells {
  const dash = '—'
  if (row.status === 'assessed') {
    return {
      mean: row.mean ?? dash,
      diff: row.diff ?? dash,
      z: row.z ?? dash,
      cutoff: row.cutoff ?? dash,
      flag: row.flagged ? 'flagged' : 'not flagged',
      note: null,
    }
  }
  return {
    mean: row.mean ?? dash,
    diff: dash,
    z: dash,
    cutoff: dash,
    flag: 'not assessed',
    note: reasonText(row),
  }
}
