import { useEffect, useRef, useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { MleRestart } from '@/api/client'
import { useMle } from '@/api/queries'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { fmtValue } from '@/lib/format'
import { cn } from '@/lib/utils'

const FAILED = -1e99
const BEST = '#dc2626' // red-600 — the best restart
const DOT = '#525252' // neutral-600 — a converged restart
const AXIS = '#737373'
const MONO = 'var(--font-mono)'

/**
 * Multi-start diagnostics — the MLE analogue of R̂/ESS. An optimizer is only as
 * trustworthy as its restarts: many converging to the same log-likelihood is
 * confidence the mode is real; a wide scatter (or mostly failures) means the
 * reported θ̂ may be one lucky — or unlucky — local optimum. Shows the converged
 * restarts on a log-likelihood axis (best in red) and the full restart table.
 */
export function RestartsTab({ runId }: { runId: string }) {
  const { data, isPending, isError } = useMle(runId)
  const converged = (data?.restarts ?? []).filter((r) => r.loglik > FAILED)
  const best = converged.length ? Math.max(...converged.map((r) => r.loglik)) : 0

  return (
    <div className="max-w-4xl">
      <Card className="overflow-hidden">
        {isPending && <ForestSkeleton rows={4} />}
        {isError && (
          <MutedNotice
            bordered={false}
            title="Couldn't load the restarts"
            detail="The backend returned an error for this MLE fit."
          />
        )}

        {data && (
          <>
            <div className="border-b border-neutral-100 px-3 py-2.5 font-mono text-[11px] text-neutral-500">
              {data.n_restarts} restarts ·{' '}
              <span
                className={
                  data.n_converged < data.n_restarts
                    ? 'text-amber-600'
                    : 'text-emerald-600'
                }
              >
                {data.n_converged} converged
              </span>
              {data.n_converged < data.n_restarts && (
                <span className="text-neutral-400">
                  {' '}
                  · {data.n_restarts - data.n_converged} failed
                </span>
              )}
              {converged.length > 1 && (
                <span className="text-neutral-400">
                  {' '}
                  · best logL{' '}
                  <span className="text-neutral-800">{fmtValue(best)}</span>,
                  spread{' '}
                  {fmtValue(best - Math.min(...converged.map((r) => r.loglik)))}
                </span>
              )}
            </div>

            {converged.length > 0 ? (
              <div className="px-3 py-3">
                <RestartStrip restarts={converged} best={best} />
                <p className="mt-1 font-mono text-[10px] text-neutral-400">
                  each dot is a converged restart's optimum; the red one is the
                  reported θ̂. Tightly clustered ⇒ the mode is reliable; scattered
                  ⇒ multiple local optima.
                </p>
              </div>
            ) : (
              <MutedNotice
                bordered={false}
                title="No restart converged"
                detail="Every restart hit the failure sentinel — the optimizer never found a feasible optimum. The reported estimate, if any, is not trustworthy."
              />
            )}

            <div className="overflow-x-auto border-t border-neutral-100">
              <table className="w-full min-w-max border-collapse font-mono text-[11px] tabular-nums">
                <thead>
                  <tr className="border-b border-neutral-200 text-[9px] uppercase tracking-wider text-neutral-400">
                    <th className="px-3 py-1.5 text-left font-medium">restart</th>
                    <th className="px-3 py-1.5 text-right font-medium">logL</th>
                    <th className="px-3 py-1.5 text-left font-medium">status</th>
                    <th className="px-3 py-1.5 text-right font-medium">evals</th>
                  </tr>
                </thead>
                <tbody>
                  {data.restarts.map((r) => {
                    const failed = r.loglik <= FAILED
                    const isBest = !failed && r.loglik === best
                    return (
                      <tr
                        key={r.chain}
                        className="border-b border-neutral-100 last:border-b-0"
                      >
                        <td className="px-3 py-1.5 text-left text-neutral-500">
                          c{r.chain}
                          {isBest && (
                            <span className="ml-1 text-red-600" title="reported θ̂">
                              ★
                            </span>
                          )}
                        </td>
                        <td
                          className={cn(
                            'px-3 py-1.5 text-right',
                            failed
                              ? 'text-neutral-400'
                              : isBest
                                ? 'font-semibold text-neutral-900'
                                : 'text-neutral-700',
                          )}
                        >
                          {failed ? 'failed' : fmtValue(r.loglik)}
                        </td>
                        <td
                          className={cn(
                            'px-3 py-1.5 text-left',
                            failed ? 'text-amber-600' : 'text-neutral-500',
                          )}
                        >
                          {r.status}
                        </td>
                        <td className="px-3 py-1.5 text-right text-neutral-400">
                          {r.n_evals.toLocaleString()}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Card>
    </div>
  )
}

/** A one-row strip of converged restart logliks — self-measuring so it draws on
 *  first paint (incl. headless capture). */
function RestartStrip({
  restarts,
  best,
}: {
  restarts: MleRestart[]
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
    const pts = restarts.map((r) => ({ loglik: r.loglik, best: r.loglik === best }))
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
        Plot.ruleY([0], { stroke: '#e5e5e5', strokeWidth: 0.5 }),
        Plot.dot(pts, {
          x: 'loglik',
          y: () => 0,
          r: (d: { best: boolean }) => (d.best ? 5 : 3.5),
          fill: (d: { best: boolean }) => (d.best ? BEST : DOT),
          fillOpacity: 0.85,
          stroke: 'white',
          strokeWidth: 0.75,
        }),
      ],
    })
    el.replaceChildren(node)
    return () => {
      node.remove()
    }
  }, [restarts, best, width])

  return <div ref={ref} className="w-full min-w-0" role="img" aria-label="restart log-likelihoods" />
}
