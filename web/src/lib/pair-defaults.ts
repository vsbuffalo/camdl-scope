/**
 * The objective the pair grid pairs parameters against before the reader picks.
 *
 * On a fit with a complete-data split (the trace writes `obs_ll`),
 * `log_posterior` is dominated by the latent path's density and rises with any
 * parameter that shrinks the implied epidemic, such as an ascertainment share,
 * whatever the data say (camdl-scope#8: misread twice, on `rho` and on
 * `s_asc`). `obs_ll` is the data-fit term, so it comes first; a fit without
 * the split pairs against `log_posterior`.
 */
export function defaultPairObjectives(objectives: readonly string[]): string[] {
  if (objectives.includes('obs_ll')) return ['obs_ll']
  if (objectives.includes('log_posterior')) return ['log_posterior']
  return []
}
