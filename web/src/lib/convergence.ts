/**
 * Convergence grading for a posterior sample.
 *
 * Upstream (`camdl fit predict`) reports NUMBERS, not a verdict: each
 * predictive row carries `rhat_max` / `ess_min` from the producing stage's
 * Gelman–Rubin summary, or blank when the stage reported none (a single-chain
 * stage — `NotAssessed` upstream). The judgment is the consumer's, so the
 * thresholds live here, in one place, shared with the Diagnostics tab.
 *
 * The grade is deliberately NOT binary. Three failure modes read differently
 * and want different words:
 *
 *  - `ok` — R̂ within tolerance and ESS adequate. The usual case; the badge
 *    stays quiet so it doesn't cry wolf.
 *  - `thin` — R̂ fine, but ESS below {@link ESS_LOW}. The chains agree on
 *    *where* the posterior is; there just aren't many independent draws, so
 *    the band's tails are noisy. Estimates are usable, extreme quantiles are
 *    not. This is a precision problem, not a validity one.
 *  - `marginal` — R̂ between {@link RHAT_OK} and {@link RHAT_HIGH}. Chains are
 *    close but not interchangeable; usually more sampling fixes it.
 *  - `unconverged` — R̂ above {@link RHAT_HIGH}. The chains are exploring
 *    different regions; a pooled predictive mixes incompatible posteriors and
 *    means nothing until this is resolved. Loud by design.
 *  - `unknown` — no summary was reported. Absence of a diagnostic is not
 *    evidence of health, so it reads neutral-but-noted, never green.
 *
 * Thresholds follow the standard convention rather than anything camdl-specific:
 * R̂ ≤ 1.01 is the modern recommendation (Vehtari et al. 2021, "Rank-normalized
 * R̂", Bayesian Analysis 16(2), §3), with 1.1 the older, looser bound still in
 * wide use; the watcher grades at 1.05 / 1.1 to match its own Diagnostics tab
 * and the Posterior forest colouring. ESS ≥ 100 per the same paper's guidance
 * that ESS should comfortably exceed 100 for stable quantile estimates.
 */

export const RHAT_OK = 1.05 // ≤ this reads healthy
export const RHAT_HIGH = 1.1 // > this reads unconverged
export const ESS_LOW = 100 // < this reads thin

export type ConvergenceGrade =
  | 'ok'
  | 'thin'
  | 'marginal'
  | 'unconverged'
  | 'unknown'

export type Convergence = {
  grade: ConvergenceGrade
  rhat: number | null
  ess: number | null
  /** Short label for the badge chip. */
  label: string
  /** One sentence: what this means for reading the plot below. */
  detail: string
}

/** Grade a sample from its worst-case R̂ and ESS. Nulls mean "not reported". */
export function gradeConvergence(
  rhat: number | null | undefined,
  ess: number | null | undefined,
): Convergence {
  const r = rhat ?? null
  const e = ess ?? null
  const base = { rhat: r, ess: e }

  if (r == null && e == null) {
    return {
      ...base,
      grade: 'unknown',
      label: 'convergence not assessed',
      detail:
        'The producing stage reported no R̂ / ESS summary (e.g. a single-chain stage), so this predictive carries no convergence evidence either way.',
    }
  }
  if (r != null && r > RHAT_HIGH) {
    return {
      ...base,
      grade: 'unconverged',
      label: 'not converged',
      detail: `R̂ = ${fmt(r)} (> ${RHAT_HIGH}) — the chains are exploring different regions of parameter space, so this predictive pools incompatible posteriors. Read nothing quantitative from it until the fit is resolved.`,
    }
  }
  if (r != null && r > RHAT_OK) {
    return {
      ...base,
      grade: 'marginal',
      label: 'marginal',
      detail: `R̂ = ${fmt(r)} (> ${RHAT_OK}) — the chains nearly agree but are not yet interchangeable. Usually more sampling; treat interval widths as provisional.`,
    }
  }
  if (e != null && e < ESS_LOW) {
    return {
      ...base,
      grade: 'thin',
      label: 'thin sample',
      detail: `ESS = ${fmt(e)} (< ${ESS_LOW}) — the chains agree, but there are few effectively independent draws, so the band's tails (q05 / q95) are noisy even though the median is usable.`,
    }
  }
  return {
    ...base,
    grade: 'ok',
    label: 'converged',
    detail: `R̂ ${r != null ? `= ${fmt(r)} ` : ''}within tolerance${e != null ? `, ESS = ${fmt(e)}` : ''} — the chains agree and the sample is adequate.`,
  }
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return '—'
  if (v >= 1000) return Math.round(v).toLocaleString()
  if (v >= 100) return Math.round(v).toString()
  return v.toFixed(v < 10 ? 3 : 1).replace(/\.?0+$/, '')
}

/** Tailwind classes for a grade's chip: text, background, border. */
export function gradeClasses(grade: ConvergenceGrade): string {
  switch (grade) {
    case 'unconverged':
      return 'bg-red-50 text-red-700 border-red-200'
    case 'marginal':
      return 'bg-amber-50 text-amber-700 border-amber-200'
    case 'thin':
      return 'bg-amber-50 text-amber-700 border-amber-200'
    case 'unknown':
      return 'bg-neutral-50 text-neutral-500 border-neutral-200'
    case 'ok':
      return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  }
}

/** Whether a grade warrants the full explanatory line, not just the chip. */
export function isNoteworthy(grade: ConvergenceGrade): boolean {
  return grade !== 'ok'
}
