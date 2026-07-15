import { useMle } from '@/api/queries'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { fmtValue } from '@/lib/format'

/**
 * The MLE point-estimate view: the optimized θ̂ per parameter, doc-labelled like
 * the posterior forest but showing a *point* — no marginal, since an
 * optimization fit has no distribution. The "converged spread" column is the
 * range of that coordinate across the restarts that converged: tight ⇒
 * well-determined, wide ⇒ sloppy/multimodal (a cheap identifiability read).
 * A bound-pinned estimate (θ̂ at an edge) is flagged.
 */
export function EstimateTab({ runId }: { runId: string }) {
  const { data, isPending, isError } = useMle(runId)

  return (
    <div className="max-w-4xl">
      <Card className="overflow-hidden">
        {isPending && <ForestSkeleton rows={4} />}

        {isError && (
          <MutedNotice
            bordered={false}
            title="Couldn't load the estimate"
            detail="The backend returned an error for this MLE fit."
          />
        )}

        {data && (
          <>
            <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 border-b border-neutral-100 px-3 py-2.5">
              <span className="font-mono text-[11px] text-neutral-500">
                MLE point estimate ·{' '}
                <span className="text-neutral-800">
                  logL {fmtValue(data.loglik)}
                </span>
              </span>
              <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-400">
                {data.algorithm} / {data.backend} ·{' '}
                <span
                  className={
                    data.n_converged < data.n_restarts ? 'text-amber-600' : ''
                  }
                >
                  {data.n_converged}/{data.n_restarts} restarts converged
                </span>
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-max border-collapse font-mono text-[11px] tabular-nums">
                <thead>
                  <tr className="border-b border-neutral-200 text-[9px] uppercase tracking-wider text-neutral-400">
                    <th className="px-3 py-1.5 text-left font-medium">
                      parameter
                    </th>
                    <th className="px-3 py-1.5 text-right font-medium">
                      MLE θ&#x0302;
                    </th>
                    <th className="px-3 py-1.5 text-right font-medium">
                      converged spread
                    </th>
                    <th className="px-3 py-1.5 text-right font-medium">bounds</th>
                  </tr>
                </thead>
                <tbody>
                  {data.params.map((p) => {
                    const hasSymbol = Boolean(p.symbol && p.symbol !== p.name)
                    const atBound =
                      p.bounds != null &&
                      p.value != null &&
                      (Math.abs(p.value - p.bounds[0]) <=
                        1e-6 * (Math.abs(p.bounds[0]) + 1) ||
                        Math.abs(p.value - p.bounds[1]) <=
                          1e-6 * (Math.abs(p.bounds[1]) + 1))
                    return (
                      <tr
                        key={p.name}
                        className="border-b border-neutral-100 last:border-b-0"
                      >
                        <td className="whitespace-nowrap px-3 py-1.5 text-left">
                          {hasSymbol && (
                            <span className="font-semibold text-neutral-900">
                              {p.symbol}{' '}
                            </span>
                          )}
                          <span
                            className={
                              hasSymbol ? 'text-neutral-400' : 'text-neutral-900'
                            }
                            title={p.description ?? undefined}
                          >
                            {p.name}
                          </span>
                        </td>
                        <td className="px-3 py-1.5 text-right font-semibold text-neutral-900">
                          {fmtValue(p.value)}
                          {atBound && (
                            <span
                              className="ml-1 text-amber-600"
                              title="at a parameter bound"
                            >
                              ⚠
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-1.5 text-right text-neutral-500">
                          {p.restart_lo != null && p.restart_hi != null
                            ? `[${fmtValue(p.restart_lo)}, ${fmtValue(p.restart_hi)}]`
                            : '—'}
                        </td>
                        <td className="px-3 py-1.5 text-right text-neutral-400">
                          {p.bounds != null
                            ? `[${fmtValue(p.bounds[0])}, ${fmtValue(p.bounds[1])}]`
                            : '—'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {data.n_converged < data.n_restarts && (
              <p className="border-t border-neutral-100 px-3 py-2 font-mono text-[10px] text-amber-600">
                {data.n_restarts - data.n_converged} of {data.n_restarts}{' '}
                restarts failed to converge — see the Restarts tab. A point
                estimate from an unreliable optimization should be read with
                care.
              </p>
            )}
          </>
        )}
      </Card>
    </div>
  )
}
