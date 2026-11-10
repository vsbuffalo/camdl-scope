import type { ParamPosterior } from '@/api/client'
import { Description } from '@/components/Description'
import { MarginalDensity } from '@/components/MarginalDensity'
import { fmtEss, fmtRhat, fmtValue } from '@/lib/format'
import { statusTextClass } from '@/lib/convergence'

const DENSITY_HEIGHT = 66

/** R̂ colour is camdl's: the status the API read from its findings and
 *  verdict (warn / error where camdl flagged this parameter, else neutral). The
 *  number is shown either way; no bar is applied to it here. */
function rhatClass(param: ParamPosterior): string {
  if (param.rhat == null && param.rhat_text == null) return 'text-neutral-400'
  return statusTextClass(param.rhat_status)
}

/** A non-finite R̂ is shown as camdl spells it (inf, NaN), not as a blank. */
function rhatText(param: ParamPosterior): string {
  return param.rhat_text ?? fmtRhat(param.rhat)
}

/**
 * One parameter as a dense, two-line table row (no per-row card): a label line
 * (symbol · name · description · citation) over a data line (marginal density,
 * median [90%], R̂/ESS). Many of these stack under hairline dividers in a single
 * bubble — a Tufte-style table, not a gallery of boxes.
 *
 * The density draws on its OWN labelled x-axis: params live on incomparable
 * supports, so the per-row axis is what gives each marginal its meaning.
 */
export function ForestRow({
  param,
  draws,
}: {
  param: ParamPosterior
  /** Row-aligned post-warmup draws for this param; `[]` until loaded. */
  draws: number[]
}) {
  const hasSymbol = Boolean(param.symbol && param.symbol !== param.name)

  return (
    <div className="px-4 py-2.5">
      {/* label line — symbol, name, citation. The doc comment gets its own
          line below: it is prose, and clipping it to the tail of this row hid
          the reasoning it exists to carry. */}
      <div className="flex min-w-0 items-baseline gap-2">
        <span className="shrink-0 text-base font-semibold leading-none text-neutral-900">
          {param.symbol ?? param.name}
        </span>
        {hasSymbol && (
          <span className="shrink-0 font-mono text-[11px] font-medium text-neutral-400">
            {param.name}
          </span>
        )}
        {param.reference && (
          <span className="ml-auto hidden shrink-0 text-[11px] italic text-neutral-400 sm:inline">
            {param.reference}
          </span>
        )}
      </div>
      {param.description && (
        <Description
          text={param.description}
          className="text-xs font-medium text-neutral-500"
        />
      )}

      {/* data line — on a phone the density takes the FULL width with the stats
          stacked below it; on >=sm it sits left of the right-aligned stat columns
          (otherwise the w-36 + w-28 columns starve the histogram to ~80px). */}
      <div className="mt-2 flex flex-col gap-1.5 sm:flex-row sm:items-center sm:gap-3">
        <div className="w-full min-w-0 sm:max-w-[340px]">
          {draws.length > 0 ? (
            <MarginalDensity
              values={draws}
              q05={param.q05}
              q25={param.q25}
              q50={param.q50}
              q75={param.q75}
              q95={param.q95}
              bounds={param.bounds}
            />
          ) : (
            // Hold the row's height so loading draws don't jolt the layout.
            <div
              className="w-full animate-pulse rounded bg-neutral-100"
              style={{ height: DENSITY_HEIGHT }}
            />
          )}
        </div>
        <div className="flex items-baseline justify-between gap-4 sm:ml-auto sm:items-center sm:justify-end">
          <div className="shrink-0 whitespace-nowrap text-right font-mono tabular-nums sm:w-36">
            <div className="text-sm font-semibold leading-tight text-neutral-900">
              {fmtValue(param.q50)}
            </div>
            <div className="text-[11px] font-medium leading-tight text-neutral-500">
              [{fmtValue(param.q05)}, {fmtValue(param.q95)}]
            </div>
          </div>
          <div className="shrink-0 whitespace-nowrap text-right font-mono text-[11px] font-medium leading-tight tabular-nums sm:w-28">
            <div className={rhatClass(param)}>
              R&#x0302; {rhatText(param)}
            </div>
            <div className="text-neutral-400">ESS {fmtEss(param.ess)}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
