import { useMle } from '@/api/queries'
import { MleRunNotes } from '@/components/MleRunNotes'
import { startsFinished } from '@/lib/mle'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { fmtValue } from '@/lib/format'

/**
 * The MLE point-estimate view: the optimized θ̂ per parameter, doc-labelled like
 * the posterior forest but showing a *point* — no marginal, since an
 * optimization fit has no distribution. The "converged spread" column is the
 * range of that coordinate across the starts that converged: tight ⇒
 * well-determined, wide ⇒ sloppy/multimodal (a cheap identifiability read).
 * A bound-pinned estimate (θ̂ at an edge) is flagged.
 *
 * While the optimizer runs, the point shown is the best finished start's, and
 * the header says so: it is provisional until camdl writes the estimate.
 */
export function EstimateTab({
  runId,
  status,
}: {
  runId: string
  status?: string
}) {
  const { data, isPending, isError } = useMle(runId, status)

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

        {data && data.provisional && data.n_restarts === 0 && (
          <>
            <MleRunNotes data={data} status={status} />
            <MutedNotice
              bordered={false}
              title="No estimate yet"
              detail="The best finished start's point appears here as soon as the first start finishes."
            />
          </>
        )}

        {data && !(data.provisional && data.n_restarts === 0) && (
          <>
            <MleRunNotes data={data} status={status} />
            <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 border-b border-hairline px-3 py-2.5">
              <span className="font-mono text-[11px] text-muted">
                {data.provisional
                  ? `Best of ${data.n_restarts} finished ${data.n_restarts === 1 ? 'start' : 'starts'} so far`
                  : 'MLE point estimate'}{' '}
                ·{' '}
                <span className="text-ink-dim">
                  logL {fmtValue(data.loglik)}
                </span>
              </span>
              <span className="font-mono text-[10px] uppercase tracking-wider text-faint">
                {data.algorithm} / {data.backend} ·{' '}
                <span
                  className={
                    data.n_converged - data.n_maxeval < data.n_restarts
                      ? 'text-warn'
                      : ''
                  }
                >
                  {data.provisional
                    ? startsFinished(data)
                    : `${data.n_converged - data.n_maxeval}/${data.n_restarts} starts converged`}
                </span>
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-max border-collapse font-mono text-[11px] tabular-nums">
                <thead>
                  <tr className="border-b border-line text-[9px] uppercase tracking-wider text-faint">
                    <th className="px-3 py-1.5 text-left font-medium">
                      parameter
                    </th>
                    <th className="px-3 py-1.5 text-right font-medium">
                      {data.provisional ? 'best so far' : <>MLE θ&#x0302;</>}
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
                        className="border-b border-hairline last:border-b-0"
                      >
                        <td className="whitespace-nowrap px-3 py-1.5 text-left">
                          {hasSymbol && (
                            <span className="font-semibold text-ink">
                              {p.symbol}{' '}
                            </span>
                          )}
                          <span
                            className={
                              hasSymbol ? 'text-faint' : 'text-ink'
                            }
                            title={p.description ?? undefined}
                          >
                            {p.name}
                          </span>
                        </td>
                        <td
                          className={
                            data.provisional
                              ? 'px-3 py-1.5 text-right text-body'
                              : 'px-3 py-1.5 text-right font-semibold text-ink'
                          }
                        >
                          {fmtValue(p.value)}
                          {atBound && (
                            <span
                              className="ml-1 text-warn"
                              title="at a parameter bound"
                            >
                              ⚠
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-1.5 text-right text-muted">
                          {p.restart_lo != null && p.restart_hi != null
                            ? `[${fmtValue(p.restart_lo)}, ${fmtValue(p.restart_hi)}]`
                            : '—'}
                        </td>
                        <td className="px-3 py-1.5 text-right text-faint">
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
              <p className="border-t border-hairline px-3 py-2 font-mono text-[10px] text-warn">
                {data.n_restarts - data.n_converged} of {data.n_restarts}{' '}
                starts failed — see the Starts tab. A point
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
