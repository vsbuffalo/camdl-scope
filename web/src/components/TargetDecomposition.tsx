/**
 * Which term of the sampled target separates the chains.
 *
 * A PGAS fit targets `log p(θ, X | y) = obs_ll + transition_ll +
 * initial_state_ll + log prior`. R̂ says the chains disagree; this says what
 * about. The bars carry each term's share of the between-chain variance in the
 * target — `Var(T) = Σ Cov(Xᵢ, T)`, exact because the terms sum to the total —
 * with the range in nats beside it as the readable magnitude.
 *
 * The shares, not the ranges, are the bars. `max−min` of a sum is not the sum
 * of `max−min`: on one real fit the four ranges total 159.3 nats against a
 * total range of 131.0, so plotting ranges as if they partitioned the whole
 * would over-explain it. A share can also be negative, which a range cannot
 * express — and a negative prior share is the sharpest available answer to "is
 * a prior overruling the data?": it means the prior is pulling the chains
 * together.
 *
 * The headline states the proportion rather than the winner. Across this
 * project's store the trajectory term ranks first in 51 of 61 fits, so the
 * ranking alone is nearly constant and carries little; the split, and the
 * minority where observation leads, are the content.
 */
import type { DiagnosticsResponse } from '@/api/client'
import { cn } from '@/lib/utils'

type Decomposition = NonNullable<DiagnosticsResponse['decomposition']>
type Term = Decomposition['terms'][number]

/** Colour by role, not by rank, so a term keeps its hue as fits change. */
const TERM_COLOR: Record<string, string> = {
  obs_ll: 'bg-red-600',
  transition_ll: 'bg-blue-700',
  initial_state_ll: 'bg-emerald-700',
  prior: 'bg-amber-500',
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`

/** How negative an observation share must be before it is worth alarming over.
 *  See `obsOpposes` for the measurement behind the number. */
const MATERIAL_OPPOSING = -0.05

function fmtNats(x: number): string {
  if (!Number.isFinite(x)) return '—'
  return Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(1)
}

/**
 * Where zero sits, and how wide a share draws, for a set of shares.
 *
 * The axis spans [min(0, smallest), max(0, largest)], so an all-positive set —
 * the common case — starts at the left edge and uses the full width, while a
 * set containing a negative share reserves exactly the room that share needs.
 * A fixed centre line wasted half the panel on every fit whose terms all point
 * the same way.
 */
function axisOf(shares: number[]) {
  const lo = Math.min(0, ...shares)
  const hi = Math.max(0, ...shares)
  const span = hi - lo || 1
  return { zero: (-lo / span) * 100, span }
}

/** One signed bar, anchored at the axis zero. */
function ShareBar({
  share,
  axis,
  color,
  height,
}: {
  share: number
  axis: { zero: number; span: number }
  color: string
  height: string
}) {
  const w = (Math.abs(share) / axis.span) * 100
  return (
    <div className={cn('relative min-w-0 flex-1', height)}>
      {axis.zero > 0.5 && (
        <div
          className="absolute inset-y-0 w-px bg-neutral-200"
          style={{ left: `${axis.zero}%` }}
        />
      )}
      <div
        className={cn('absolute inset-y-0 rounded-[2px]', color)}
        style={
          share < 0
            ? { right: `${100 - axis.zero}%`, width: `${w}%` }
            : { left: `${axis.zero}%`, width: `${w}%` }
        }
      />
    </div>
  )
}

function TermBar({
  term,
  axis,
}: {
  term: Term
  axis: { zero: number; span: number }
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-28 shrink-0 truncate font-mono text-[11px] text-neutral-700 sm:w-44">
        {term.label}
      </span>
      <ShareBar
        share={term.share}
        axis={axis}
        color={TERM_COLOR[term.key] ?? 'bg-neutral-500'}
        height="h-3"
      />
      <span className="w-14 shrink-0 text-right font-mono text-[11px] tabular-nums text-neutral-900 sm:w-16">
        {pct(term.share)}
      </span>
      {/* Secondary to the share; dropped on a phone so the bar keeps width. */}
      <span className="hidden w-20 shrink-0 text-right font-mono text-[10px] tabular-nums text-neutral-500 sm:block">
        {fmtNats(term.range_nats)} nats
      </span>
    </div>
  )
}

export function TargetDecomposition({ dec }: { dec: Decomposition }) {
  if (!dec.available) {
    return (
      <p className="px-3 py-2 font-mono text-[10px] leading-relaxed text-neutral-500">
        No target decomposition — {dec.reason}
      </p>
    )
  }

  const terms = dec.terms
  const axis = axisOf(terms.map((t) => t.share))
  const streamAxis = axisOf(dec.streams.map((s) => s.share))
  const obs = terms.find((t) => t.key === 'obs_ll')
  const traj = terms.find((t) => t.key === 'transition_ll')
  const dataLeads = obs && traj && obs.share > traj.share
  // Anti-correlated with the total, and by enough to mean something. Firing on
  // any negative share was wrong: over this project's store the observation
  // share has median +1.7% with a 5th percentile of −1.7%, and a standard
  // deviation near 64% on fits with under six chains — so a −2% is squarely
  // inside the noise and 32 of 141 fits are negative at all. The threshold sits
  // above that bulk; the genuinely anti-correlated fits run −20% to −90%.
  const obsOpposes = Boolean(obs && obs.share <= MATERIAL_OPPOSING)
  // Shares are covariances over CHAIN MEANS, so the sample size is the chain
  // count. Four of them is a noisy estimate and the reader should know.
  const fewChains = dec.chain_ids.length < 6
  const residualBroken = dec.residual_max > dec.residual_tol

  return (
    <div className="flex flex-col gap-2 px-3 py-2">
      {/* The proportion, in words, before the chart. A reader who takes only
          one line away should take this one. A negative share cannot be
          phrased as "explains −15% of the spread" — nothing explains a
          negative fraction — so that case gets its own sentence. */}
      <p className="font-mono text-[11px] leading-relaxed text-neutral-700">
        Across {dec.chain_ids.length} chains the target spans{' '}
        <span className="text-neutral-900">{fmtNats(dec.total_range)} nats</span>.
        {traj && (
          <>
            {' '}
            The latent path accounts for{' '}
            <span className="text-neutral-900">{pct(traj.share)}</span> of that
            spread
          </>
        )}
        {obs && obs.share >= 0 && (
          <>
            ; the data accounts for{' '}
            <span className="text-neutral-900">{pct(obs.share)}</span>
          </>
        )}
        {obs && obs.share < 0 && (
          <>
            . The data term runs{' '}
            <span className="text-neutral-900">against</span> it at{' '}
            <span className="text-neutral-900">{pct(obs.share)}</span>
            {obs.share > MATERIAL_OPPOSING && (
              <span className="text-neutral-500">
                {' '}
                — small enough to be sampling noise
              </span>
            )}
          </>
        )}
        .
      </p>

      {obsOpposes && (
        <p className="border-l-2 border-red-600 bg-red-50 px-2 py-1 font-mono text-[10px] leading-relaxed text-red-800">
          The observation term moves <em>against</em> the target: the chains the
          sampler ranks highest are the ones fitting the data worst. This is the
          complete-data target being gamed — a chain buys more trajectory
          density than it loses in data fit, and comes out ahead. The ranking
          the sampler is optimising is not the ranking the data supports, so
          treat between-chain comparisons of the log posterior with suspicion.
        </p>
      )}

      {dataLeads && (
        <p className="border-l-2 border-red-600 bg-red-50 px-2 py-1 font-mono text-[10px] leading-relaxed text-red-800">
          The observation term outranks the trajectory term. The chains disagree
          about something the data can see — unusual, and worth following into
          the per-stream breakdown below.
        </p>
      )}

      <div className="flex flex-col gap-1.5">
        {terms.map((t) => (
          <TermBar key={t.key} term={t} axis={axis} />
        ))}
      </div>

      <p className="font-mono text-[10px] leading-relaxed text-neutral-500">
        Share of the between-chain variance in the sampled target, signed —
        shares sum to 100%, so one above it means another is negative. A
        negative share means the term is anti-correlated with the total: chains
        scoring higher overall do <em>worse</em> on it. For the prior that reads
        as pulling the chains together; for the observation term it means the
        sampler prefers chains that fit the data less well. The nats column is
        each term&apos;s own range across chain means and is <em>not</em>
        additive.
      </p>

      {fewChains && (
        <p className="font-mono text-[10px] leading-relaxed text-neutral-500">
          Estimated from {dec.chain_ids.length} chain means, so the shares carry
          real sampling error — read the sign and the rough magnitude, not the
          decimal.
        </p>
      )}

      {dec.initial_state_derived && (
        <p className="font-mono text-[10px] text-neutral-500">
          This run predates the <code>initial_state_ll</code> column; the
          initial-state term was recovered by subtraction, so it carries no
          independent check.
        </p>
      )}

      {residualBroken && (
        <p className="font-mono text-[10px] text-amber-700">
          The terms do not reconcile — largest residual{' '}
          {dec.residual_max.toExponential(1)} nats against a tolerance of{' '}
          {dec.residual_tol}. camdl is writing a term this view does not know
          about, and it is being absorbed into the implied prior. Treat the
          prior row as unreliable until that is resolved.
        </p>
      )}

      {dec.streams.length > 0 && (
        <div className="mt-1 border-t border-neutral-100 pt-2">
          <div className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-neutral-400">
            observation term, by stream
          </div>
          <div className="flex flex-col gap-1">
            {dec.streams.map((s) => (
              <div key={s.stream} className="flex items-center gap-3">
                <span className="w-28 shrink-0 truncate font-mono text-[11px] text-neutral-600 sm:w-44">
                  {s.stream}
                </span>
                <ShareBar
                  share={s.share}
                  axis={streamAxis}
                  color="bg-red-600 opacity-70"
                  height="h-2"
                />
                <span className="w-14 shrink-0 text-right font-mono text-[11px] tabular-nums text-neutral-900 sm:w-16">
                  {pct(s.share)}
                </span>
                <span className="hidden w-20 shrink-0 text-right font-mono text-[10px] tabular-nums text-neutral-500 sm:block">
                  {fmtNats(s.range_nats)} nats
                </span>
              </div>
            ))}
          </div>
          <p className="mt-1 font-mono text-[10px] leading-relaxed text-neutral-500">
            Share of the chain disagreement <em>within</em> the observation
            term. Sums to 100% over streams, so a share above it means another
            stream&apos;s is negative.
          </p>
        </div>
      )}

      {/* Per-chain detail: the bars name the term, this names the chain — the
          handoff into the chain selector. */}
      <details className="mt-1 border-t border-neutral-100 pt-2">
        <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-wider text-neutral-400 hover:text-neutral-700">
          per-chain term means
        </summary>
        <div className="mt-1.5 overflow-x-auto">
          <table className="w-full border-collapse font-mono text-[10px]">
            <thead>
              <tr className="text-neutral-500">
                <th className="px-2 py-1 text-left font-medium">chain</th>
                {terms.map((t) => (
                  <th key={t.key} className="px-2 py-1 text-right font-medium">
                    {t.label.split(' ')[0]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dec.chain_ids.map((cid, i) => (
                <tr key={cid} className="border-t border-neutral-100">
                  <td className="px-2 py-0.5 text-neutral-700">c{cid}</td>
                  {terms.map((t) => {
                    const v = t.chain_means[i]
                    const mean =
                      t.chain_means.reduce((a, b) => a + b, 0) /
                      t.chain_means.length
                    const dev = v == null ? 0 : v - mean
                    // Deviation from the cross-chain mean, so an outlier chain
                    // is visible without mentally subtracting a 5-digit number.
                    return (
                      <td
                        key={t.key}
                        className={cn(
                          'px-2 py-0.5 text-right tabular-nums',
                          Math.abs(dev) > 2 * (t.sd_nats || 1)
                            ? 'text-red-700'
                            : 'text-neutral-600',
                        )}
                      >
                        {dev >= 0 ? '+' : ''}
                        {fmtNats(dev)}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 px-2 font-mono text-[10px] text-neutral-500">
            Deviation from the cross-chain mean, in nats. Red is more than 2
            standard deviations from it — the chain to try excluding.
          </p>
        </div>
      </details>
    </div>
  )
}
