import type { ChainLoglikModel } from '@/api/client'
import { chainLoglikCells } from '@/lib/chain-loglik'
import { cn } from '@/lib/utils'

/**
 * camdl's per-chain log-likelihood comparison, as persisted with the run: each
 * chain's mean against the pooled mean of the others (`diff`, in nats, chain
 * minus the others), Z = |diff| / se, the cutoff camdl held Z against, and
 * camdl's own flag. The flag is read, never recomputed here. A leaf that does
 * not carry the record says so rather than showing an empty table.
 */
export function ChainLoglikTable({
  record,
  error,
  frozenChains,
}: {
  record: ChainLoglikModel | null | undefined
  error?: string | null
  /** Chains camdl recorded as taking no step after burn-in, marked in the
   *  chain column. */
  frozenChains?: number[] | null
}) {
  if (error) {
    return <p className="px-3 py-2 font-mono text-[11px] text-amber-600">{error}</p>
  }
  if (!record) {
    return (
      <p className="px-3 py-2 font-mono text-[11px] text-neutral-400">
        not recorded — this leaf was written before camdl persisted the per-chain
        log-likelihood comparison
      </p>
    )
  }
  const frozen = new Set(frozenChains ?? [])
  return (
    <>
      <p className="px-3 pt-2 font-mono text-[10px] leading-snug text-neutral-400">
        each chain's mean <span className="text-neutral-600">{record.column}</span> against
        the mean of the others
        {record.family_alpha != null && <> · family-wise α {record.family_alpha}</>}
        {record.min_ess != null && <> · chains with fewer than {record.min_ess} effective draws are not assessed</>}
      </p>
      {record.column_absent && (
        <p className="px-3 pt-1 font-mono text-[11px] text-amber-600">
          camdl read the chains' traces and none carries {record.column}
        </p>
      )}
      <div className="overflow-x-auto">
        <table className="w-full min-w-max border-collapse font-mono text-[11px] tabular-nums">
          <thead>
            <tr className="border-b border-neutral-200 text-[9px] uppercase tracking-wider text-neutral-400">
              <th className="px-3 py-1.5 text-left font-medium">chain</th>
              <th className="px-3 py-1.5 text-right font-medium">mean</th>
              <th className="px-3 py-1.5 text-right font-medium" title="this chain's mean minus the others', in nats">
                Δ others
              </th>
              <th className="px-3 py-1.5 text-right font-medium" title="|Δ| over its standard error">
                Z
              </th>
              <th className="px-3 py-1.5 text-right font-medium" title="the bar camdl held Z against">
                cutoff
              </th>
              <th className="px-3 py-1.5 text-left font-medium">camdl</th>
            </tr>
          </thead>
          <tbody>
            {record.chains.map((r) => {
              const c = chainLoglikCells(r)
              return (
                <tr key={r.chain} className="border-b border-neutral-100 last:border-b-0">
                  <td className="px-3 py-1.5 text-left text-neutral-500">
                    c{r.chain}
                    {frozen.has(r.chain) && (
                      <span
                        className="ml-1.5 text-red-600"
                        title="camdl recorded this chain as taking no step after burn-in"
                      >
                        frozen
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-1.5 text-right text-neutral-700">{c.mean}</td>
                  <td className="px-3 py-1.5 text-right text-neutral-700">{c.diff}</td>
                  <td className="px-3 py-1.5 text-right text-neutral-700">{c.z}</td>
                  <td className="px-3 py-1.5 text-right text-neutral-400">{c.cutoff}</td>
                  <td
                    className={cn(
                      'px-3 py-1.5 text-left',
                      c.flag === 'flagged' && 'font-medium text-amber-600',
                      c.flag === 'not flagged' && 'text-neutral-500',
                      c.flag === 'not assessed' && 'text-neutral-400',
                    )}
                  >
                    {c.flag}
                    {c.note && <span className="text-neutral-400"> · {c.note}</span>}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </>
  )
}
