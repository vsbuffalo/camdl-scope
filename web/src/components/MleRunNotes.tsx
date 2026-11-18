import type { MleResponse } from '@/api/client'
import { isLive, startsFinished } from '@/lib/mle'

/**
 * What the reader must know before trusting the numbers below it: that the
 * estimate is provisional (the run is going, or stopped before writing one),
 * and that every finished start ran out of evaluations. Renders nothing for a
 * finished fit whose starts converged.
 */
export function MleRunNotes({
  data,
  status,
}: {
  data: MleResponse
  status: string | undefined
}) {
  const live = isLive(status)
  const lines: { tone: string; text: string }[] = []
  if (data.provisional) {
    lines.push(
      live
        ? {
            tone: 'text-notice',
            text:
              data.n_restarts === 0 && data.loglik == null
                ? 'Optimizer running — no start has finished yet.'
                : data.provisional_from_running
                  ? `Optimizer running — ${data.n_restarts === 0 ? 'no start has finished yet' : startsFinished(data)}. The estimate is the best point a search still running has reached; it can only rise, and it is not the fit's answer, which camdl writes when every start is in.`
                  : `Optimizer running — ${startsFinished(data)}. The estimate is the best of the finished starts so far, not the fit's answer; camdl writes that when every start is in.`,
          }
        : {
            tone: 'text-warn',
            text: `The run stopped before writing its estimate (${startsFinished(data)}). What is shown is the best of the starts that finished, not a reported optimum.`,
          },
    )
  }
  if (data.all_maxeval) {
    lines.push({
      tone: 'text-warn',
      text: `Every ${data.provisional ? 'finished ' : ''}start stopped at its max_evals budget, so the ${data.provisional ? 'best-so-far point' : 'reported optimum'} is where the best start's budget ran out, not a converged optimum.`,
    })
  }
  if (!lines.length) return null
  return (
    <div className="border-b border-hairline px-3 py-2 font-mono text-[10px] leading-relaxed">
      {lines.map((l) => (
        <p key={l.text} className={l.tone}>
          {l.text}
        </p>
      ))}
    </div>
  )
}
