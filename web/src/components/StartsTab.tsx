import { useEffect, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { MleRestart } from '@/api/client'
import { useMle } from '@/api/queries'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { ChainLoglikTable } from '@/components/ChainLoglikTable'
import { MleRunNotes } from '@/components/MleRunNotes'
import { StartsParallel } from '@/components/StartsParallel'
import { startsFinished } from '@/lib/mle'
import { fmtValue } from '@/lib/format'
import { cn } from '@/lib/utils'
import { ink, token } from '@/lib/theme'

const FAILED = -1e99
const BEST = ink('mark-red') // red-600 — the best start
const DOT = token('body-dim') // neutral-600 — a converged start
const AXIS = token('muted')
const MONO = 'var(--font-mono)'

/**
 * Multi-start diagnostics — the MLE analogue of R̂/ESS. The optimizer runs
 * independent local searches from several starting points (camdl's `chains`,
 * placed by its `starts` rule) and reports the best. It is only as trustworthy
 * as its starts: many converging to the same log-likelihood is confidence the
 * mode is real; a wide scatter (or mostly failures) means the reported θ̂ may be
 * one lucky — or unlucky — local optimum. Shows the converged starts on a
 * log-likelihood axis (best in red) and the full table of starts.
 *
 * While the optimizer runs, the table fills in as starts finish (camdl writes
 * them in completion order; they are shown best first).
 */
export function StartsTab({
  runId,
  status,
}: {
  runId: string
  status?: string
}) {
  const { data, isPending, isError } = useMle(runId, status)
  const converged = (data?.restarts ?? []).filter((r) => r.loglik > FAILED)
  const best = converged.length ? Math.max(...converged.map((r) => r.loglik)) : 0

  return (
    <div className="max-w-4xl">
      <Card className="overflow-hidden">
        {isPending && <ForestSkeleton rows={4} />}
        {isError && (
          <MutedNotice
            bordered={false}
            title="Couldn't load the starts"
            detail="The backend returned an error for this MLE fit."
          />
        )}

        {data && (
          <>
            <MleRunNotes data={data} status={status} />
            <div className="border-b border-hairline px-3 py-2.5 font-mono text-[11px] text-muted">
              {data.provisional
                ? startsFinished(data)
                : `${data.n_restarts} starts`}{' '}
              ·{' '}
              {/* A start that ran out of evaluations did not fail, but it did
                  not converge either: count it apart. */}
              <span
                className={
                  data.n_converged - data.n_maxeval < data.n_restarts
                    ? 'text-warn'
                    : 'text-good'
                }
              >
                {data.n_converged - data.n_maxeval} converged
              </span>
              {data.n_converged < data.n_restarts && (
                <span className="text-faint">
                  {' '}
                  · {data.n_restarts - data.n_converged} failed
                </span>
              )}
              {data.n_maxeval > 0 && (
                <span className="text-warn">
                  {' '}
                  · {data.n_maxeval} at max_evals
                </span>
              )}
              {converged.length > 1 && (
                <span className="text-faint">
                  {' '}
                  · best logL{' '}
                  <span className="text-ink-dim">{fmtValue(best)}</span>,
                  spread{' '}
                  {fmtValue(best - Math.min(...converged.map((r) => r.loglik)))}
                </span>
              )}
            </div>

            {converged.length > 0 ? (
              <div className="px-3 py-3">
                <StartStrip starts={converged} best={best} />
                <p className="mt-1 font-mono text-[10px] text-faint">
                  each dot is where a start ended (failed starts are left out);
                  the red one is{' '}
                  {data.provisional ? 'the best so far' : 'the reported θ̂'}.
                  Scattered ⇒ multiple local optima. Tightly clustered ⇒ the
                  starts reach the same height, not necessarily the same point:
                  the plot below shows whether their parameters agree.
                </p>
                {converged.length > 1 && (
                  <div className="mt-3">
                    <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-faint">
                      do the starts agree on the parameters?
                    </div>
                    <StartsParallel starts={converged} params={data.params} />
                    <p className="mt-1 font-mono text-[10px] text-faint">
                      one axis per parameter, from its lower bound (bottom) to
                      its upper bound (top); one line per start through its
                      optimum. Lines pinching to a point ⇒ the starts agree on
                      that parameter; spread ⇒ they don&apos;t. Red is the best
                      start; the further a start&apos;s log-likelihood falls
                      below it, the fainter its line.
                    </p>
                  </div>
                )}
              </div>
            ) : data.n_restarts === 0 ? (
              <MutedNotice
                bordered={false}
                title="No start has finished yet"
                detail="Each start's optimum is added here as it finishes."
              />
            ) : (
              <MutedNotice
                bordered={false}
                title="No start converged"
                detail="Every start hit the failure sentinel — the optimizer never found a feasible optimum. The reported estimate, if any, is not trustworthy."
              />
            )}

            <div className="overflow-x-auto border-t border-hairline">
              <table className="w-full min-w-max border-collapse font-mono text-[11px] tabular-nums">
                <thead>
                  <tr className="border-b border-line text-[9px] uppercase tracking-wider text-faint">
                    <th className="px-3 py-1.5 text-left font-medium">start</th>
                    <th className="px-3 py-1.5 text-right font-medium">logL</th>
                    <th className="px-3 py-1.5 text-left font-medium">status</th>
                    <th className="px-3 py-1.5 text-right font-medium">evals</th>
                    {data.params.map((p) => (
                      <th
                        key={p.name}
                        className="px-3 py-1.5 text-right font-medium normal-case"
                      >
                        {p.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.restarts.map((r, i) => {
                    const failed = r.loglik <= FAILED
                    // Rows arrive best first; starts that converged to the same
                    // optimum tie, and only one of them is the point shown.
                    const isBest = i === 0 && !failed
                    return (
                      <tr
                        key={r.chain}
                        className="border-b border-hairline last:border-b-0"
                      >
                        <td className="px-3 py-1.5 text-left text-muted">
                          c{r.chain}
                          {isBest && (
                            <span
                              className="ml-1 text-danger"
                              title={data.provisional ? 'best so far' : 'reported θ̂'}
                            >
                              ★
                            </span>
                          )}
                        </td>
                        <td
                          className={cn(
                            'px-3 py-1.5 text-right',
                            failed
                              ? 'text-faint'
                              : isBest
                                ? 'font-semibold text-ink'
                                : 'text-body',
                          )}
                        >
                          {failed ? 'failed' : fmtValue(r.loglik)}
                        </td>
                        <td
                          className={cn(
                            'px-3 py-1.5 text-left',
                            failed || r.hit_maxeval ? 'text-warn' : 'text-muted',
                          )}
                          title={
                            r.hit_maxeval
                              ? 'stopped because its evaluation budget ran out, not because it converged'
                              : undefined
                          }
                        >
                          {r.status}
                          {r.hit_maxeval && ' ⚠'}
                        </td>
                        <td className="px-3 py-1.5 text-right text-faint">
                          {r.n_evals.toLocaleString()}
                        </td>
                        {data.params.map((p) => (
                          <td
                            key={p.name}
                            className="px-3 py-1.5 text-right text-body"
                          >
                            {fmtValue(r.values?.[p.name] ?? null)}
                          </td>
                        ))}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* camdl writes this comparison with the finished fit, so a
                provisional fit has none yet — not an older leaf's absence.
                An nl-* optimizer never writes one (camdl sets
                chain_loglik_agreement to None: a start ends at one point, so
                there is no per-start mean to compare); only IF2 does. */}
            {!data.provisional && !data.algorithm.startsWith('nl-') && (
              <div className="border-t border-hairline">
                <div className="px-3 py-2 text-[10px] font-medium uppercase tracking-wider text-faint">
                  per-start log-likelihood · camdl&apos;s comparison
                </div>
                <ChainLoglikTable
                  record={data.chain_logliks}
                  error={data.chain_logliks_error}
                />
              </div>
            )}
          </>
        )}
      </Card>
    </div>
  )
}

/** A one-row strip of converged starts' logliks — self-measuring so it draws on
 *  first paint (incl. headless capture). */
function StartStrip({
  starts,
  best,
}: {
  starts: MleRestart[]
  best: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    setWidth(Math.round(el.getBoundingClientRect().width))
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0
      if (w > 0) setWidth(Math.round(w))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const el = ref.current
    if (!el || width <= 0) return
    const pts = starts.map((r) => ({ loglik: r.loglik, best: r.loglik === best }))
    const node = Plot.plot({
      width,
      height: 78,
      marginTop: 6,
      marginBottom: 30,
      marginLeft: 10,
      marginRight: 10,
      style: { background: 'transparent', color: AXIS, fontSize: '10px', fontFamily: MONO },
      x: {
        label: 'log-likelihood at optimum →',
        labelAnchor: 'center',
        ticks: 5,
        tickSize: 3,
        tickFormat: (d: number) => fmtValue(d),
      },
      y: { axis: null, domain: [-1, 1] },
      marks: [
        Plot.ruleY([0], { stroke: token('line'), strokeWidth: 0.5 }),
        Plot.dot(pts, {
          x: 'loglik',
          y: () => 0,
          r: (d: { best: boolean }) => (d.best ? 5 : 3.5),
          fill: (d: { best: boolean }) => (d.best ? BEST : DOT),
          fillOpacity: 0.85,
          stroke: token('surface'),
          strokeWidth: 0.75,
        }),
      ],
    })
    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [starts, best, width])

  return <div ref={ref} className="w-full min-w-0" role="img" aria-label="log-likelihood of each start" />
}
