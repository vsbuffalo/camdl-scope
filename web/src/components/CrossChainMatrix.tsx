/**
 * Cross-chain path compatibility — is each chain welded to its own mode?
 *
 * `M[i][j] = log p(x_j | θ_i)`: chain i's parameters scoring chain j's latent
 * path. R̂ says the chains disagree; this says whether they are stuck on their
 * own (θ, x) pairs — a strong diagonal — or exchangeable in the augmented
 * space, where the separation lives in the marginal posterior instead and the
 * off-diagonals come close to the diagonal.
 *
 * Three rendering decisions, each because the obvious alternative misleads:
 *
 * 1. **Cells show `M[i][j] − M[i][i]`, not the raw density.** Rows are not
 *    comparable: a different θ puts a row on a different scale, so the largest
 *    raw number identifies the chain with the loosest process noise, not the
 *    most compatible pairing. Centred, a cell reads directly as "how many nats
 *    worse chain j's path is under chain i's parameters", and the diagonal is
 *    exactly zero.
 * 2. **Diverging at zero, with the number printed.** A positive entry —
 *    another chain's path being *more* probable under my parameters than my
 *    own — means that chain never found its own mode. It is the alarming case
 *    and a sequential ramp would bury it at the pale end. k is 8–16, so the
 *    numbers fit in the cells and the colour is a second encoding rather than
 *    the only one.
 * 3. **Rows and columns follow the clustering, not the chain id.** Blocks of
 *    mutually compatible chains are the multimodality signal: two blocks is two
 *    modes, k singletons is k stuck chains. Chain-id order scatters that.
 */
import type { DiagnosticsResponse } from '@/api/client'
import { cn } from '@/lib/utils'

type Compat = NonNullable<DiagnosticsResponse['cross_chain']>

/** Diverging blue↔red through a neutral grey, so zero reads as "nothing" and
 *  the two directions read as opposite. `t` in [-1, 1]. */
function cellColor(t: number): string {
  const a = Math.min(1, Math.abs(t))
  if (t >= 0) {
    // Positive: another chain's path scores BETTER. The alarming direction.
    return `color-mix(in oklab, #e34948 ${Math.round(a * 88)}%, #f0efec)`
  }
  return `color-mix(in oklab, #2a78d6 ${Math.round(a * 72)}%, #f0efec)`
}

function fmtCell(v: number): string {
  const a = Math.abs(v)
  if (a >= 1000) return v.toFixed(0)
  if (a >= 100) return v.toFixed(0)
  if (a >= 10) return v.toFixed(1)
  return v.toFixed(2)
}

export function CrossChainMatrix({ compat }: { compat: Compat }) {
  const { chains, centred, order } = compat
  const k = chains.length

  // Scale on the off-diagonal magnitudes only. The diagonal is zero by
  // construction and would not contribute, but a single huge outlier row can
  // still flatten everything else, so the ramp is set by the 90th percentile
  // rather than the max — the extreme cell stays saturated and the structure
  // among the rest stays visible.
  const offs: number[] = []
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const v = centred[i]?.[j]
      if (i !== j && typeof v === 'number') offs.push(Math.abs(v))
    }
  }
  offs.sort((a, b) => a - b)
  const scale = offs.length ? offs[Math.floor(offs.length * 0.9)] || 1 : 1

  return (
    <div className="flex flex-col gap-2 px-3 py-2">
      <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1">
        <span className="font-mono text-[11px] text-neutral-700">
          {k} chains scored
        </span>
        {compat.diagonal_dominance != null && (
          <span className="font-mono text-[11px] text-neutral-600">
            diagonal dominance{' '}
            <span className="text-neutral-900 tabular-nums">
              {compat.diagonal_dominance.toFixed(2)}
            </span>
          </span>
        )}
        {compat.asymmetry != null && (
          <span className="font-mono text-[11px] text-neutral-600">
            asymmetry{' '}
            <span className="text-neutral-900 tabular-nums">
              {compat.asymmetry.toFixed(2)}
            </span>
          </span>
        )}
        {compat.terms && (
          <span className="font-mono text-[10px] text-neutral-400">
            {compat.terms}
          </span>
        )}
      </div>

      {compat.has_positive_offdiagonal && (
        <p className="border-l-2 border-red-600 bg-red-50 px-2 py-1 font-mono text-[10px] leading-relaxed text-red-800">
          At least one chain scores another chain&apos;s path higher than its
          own. That chain has not found a mode of its own — it is sitting
          somewhere its own parameters do not prefer.
        </p>
      )}

      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-0 font-mono text-[10px] tabular-nums">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-white px-1.5 py-1 text-right font-medium text-neutral-400">
                θ ↓ / path →
              </th>
              {order.map((j) => (
                <th
                  key={j}
                  className="px-1.5 py-1 text-center font-medium text-neutral-500"
                >
                  c{chains[j]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {order.map((i) => (
              <tr key={i}>
                <th className="sticky left-0 z-10 bg-white px-1.5 py-1 text-right font-medium text-neutral-500">
                  c{chains[i]}
                </th>
                {order.map((j) => {
                  const v = centred[i]?.[j]
                  if (typeof v !== 'number') {
                    return (
                      <td
                        key={j}
                        title="non-finite"
                        className="border border-white bg-neutral-50 px-1.5 py-1 text-center text-neutral-300"
                      >
                        —
                      </td>
                    )
                  }
                  const diag = i === j
                  return (
                    <td
                      key={j}
                      title={`log p(x_c${chains[j]} | θ_c${chains[i]}) − log p(x_c${chains[i]} | θ_c${chains[i]}) = ${v.toFixed(2)} nats`}
                      className={cn(
                        'border border-white px-1.5 py-1 text-center',
                        diag
                          ? 'font-semibold text-neutral-500'
                          : 'text-neutral-900',
                      )}
                      style={
                        diag
                          ? { background: '#f0efec' }
                          : { background: cellColor(v / scale) }
                      }
                    >
                      {diag ? '0' : fmtCell(v)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="font-mono text-[10px] leading-relaxed text-neutral-500">
        Cell (row θ, column path) is{' '}
        <span className="text-neutral-700">
          log p(x_col | θ_row) − log p(x_row | θ_row)
        </span>{' '}
        in nats — how much worse another chain&apos;s path is under this
        row&apos;s parameters. Zero on the diagonal by construction. Blue is
        worse, <span className="text-red-700">red is better than the row&apos;s
        own path</span>. Rows and columns follow a clustering of the symmetrised
        matrix, so mutually-compatible chains sit together: a block is a mode,
        and all-singletons means every chain is stuck somewhere different.
      </p>

      {compat.n_nonfinite > 0 && (
        <p className="font-mono text-[10px] text-amber-700">
          {compat.n_nonfinite} non-finite{' '}
          {compat.n_nonfinite === 1 ? 'cell' : 'cells'}, drawn blank — a path
          with zero probability under those parameters.
        </p>
      )}

      <p className="font-mono text-[10px] leading-relaxed text-neutral-500">
        Chains refused at seeding are absent from this matrix, so {k} is what
        was scored, not what was launched
        {chains.length > 1 &&
          chains[chains.length - 1]! - chains[0]! + 1 > k &&
          ` — ids run c${chains[0]} to c${chains[chains.length - 1]}, so ${chains[chains.length - 1]! - chains[0]! + 1 - k} did not start`}
        .{compat.draw && ` Scored at: ${compat.draw}.`}
      </p>
    </div>
  )
}
