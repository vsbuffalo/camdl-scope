/**
 * Particle-filter health, shaped by the sampler that ran.
 *
 * PGAS and PMMH are not two configurations of one panel: they store different
 * things and fail differently, so each gets the evidence it actually has.
 *
 * PGAS keeps one conditioned path, and its failure is that path freezing while
 * θ keeps moving — sampling that looks like sampling and is not. The scalar
 * renewal rate hides where that happens, so the profile across the trajectory
 * leads: camdl bins renewal into tenths of the substeps, which is the
 * update-rate-against-time diagnostic Lindsten, Jordan & Schön (2014) and
 * Chopin & Singh (2015) recommend in place of a rule for the particle count.
 * The ancestor-sampling counters say *why* a profile is flat, and the two
 * causes have opposite fixes — a degenerate filter wants more particles, while
 * a schedule with no ancestry draws will not improve with any number.
 *
 * PMMH stores no path, so none of that exists. It sticks instead: an
 * over-estimated log-likelihood is accepted and nothing beats it. A healthy
 * average acceptance can hide one long stall, so the longest run of rejections
 * is reported beside the rate.
 */
import type { DiagnosticsResponse } from '@/api/client'
import { cn } from '@/lib/utils'

type Health = NonNullable<DiagnosticsResponse['particle_health']>

/** A proportion as a percentage — how a rate is read out loud. Two decimals
 *  below 1% so a near-frozen 0.15% does not round to 0%, one above. */
export function fmtFraction(v: number): string {
  const pctv = v * 100
  if (pctv > 0 && pctv < 1) return `${pctv.toFixed(2)}%`
  return `${pctv.toFixed(1)}%`
}

function fmt(v: number, unit: string): string {
  // Only proportions become percentages. A percentage of a count (ancestry
  // draws per sweep, particles, sweeps stalled) would be meaningless.
  if (unit === 'fraction') return fmtFraction(v)
  if (unit === 'sweeps') return Math.round(v).toLocaleString()
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString()
  if (Math.abs(v) >= 10) return v.toFixed(1)
  return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
}

/**
 * Renewal against position along the trajectory. Bars rather than a line: the
 * ten bins are a partition of the trajectory, not samples of a continuum, and
 * a line would invite reading a slope between bin centres.
 */
function RenewalProfile({ profile }: { profile: number[] }) {
  const max = Math.max(...profile, 0.05)
  const first = profile[0] ?? 0
  const last = profile[profile.length - 1] ?? 0
  // A profile that climbs steeply is the interesting shape: the early path is
  // pinned while the late path moves freely.
  const skewed = last > 0.3 && first < 0.25 * last
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex h-16 items-end gap-1">
        {profile.map((v, i) => (
          <div
            key={i}
            className="group relative flex-1"
            title={`bin ${i + 1} of ${profile.length}: renewal ${fmtFraction(v)}`}
          >
            <div
              className={cn(
                'w-full rounded-t-[2px]',
                v < 0.05 ? 'bg-orange-500' : 'bg-blue-700',
              )}
              style={{ height: `${Math.max(2, (v / max) * 60)}px` }}
            />
          </div>
        ))}
      </div>
      <div className="flex justify-between font-mono text-[10px] text-neutral-400">
        <span>start of trajectory</span>
        <span>end</span>
      </div>
      {skewed && (
        <p className="font-mono text-[10px] leading-relaxed text-amber-700">
          Renewal climbs from {fmtFraction(first)} to {fmtFraction(last)} across the
          trajectory: the early path is effectively frozen while the late path
          moves freely. Draws are being reused for the start of the epidemic,
          so anything the early trajectory determines — seeding time, initial
          size — is far less explored than its marginal suggests.
        </p>
      )}
    </div>
  )
}

export function ParticleHealth({ health }: { health: Health }) {
  if (!health.available) {
    return (
      <p className="px-3 py-2 font-mono text-[10px] leading-relaxed text-neutral-500">
        No particle diagnostics — {health.reason}
      </p>
    )
  }
  const chainRenewal = Object.entries(health.renewal_by_chain ?? {})
  const stalls = Object.entries(health.longest_stall_by_chain ?? {})

  return (
    <div className="flex flex-col gap-3 px-3 py-2">
      <div className="font-mono text-[10px] uppercase tracking-wider text-neutral-400">
        {health.method || 'sampler'}
      </div>

      <div className="flex flex-col gap-1.5">
        {/* Stacked on a phone, four columns from `sm` up. The fixed label /
            value / unit columns total ~366px, which is wider than a 390px
            viewport's content box — as one row they squeezed the note to zero
            width, and a zero-width paragraph is 500px tall. `sm:contents`
            dissolves the mobile wrapper so its children rejoin the outer row on
            a wide screen. */}
        {health.stats.map((s) => (
          <div
            key={s.key}
            className="flex flex-col gap-0.5 border-b border-neutral-50 pb-1.5 last:border-b-0 sm:flex-row sm:items-baseline sm:gap-3 sm:border-b-0 sm:pb-0"
          >
            <div className="flex items-baseline gap-2 sm:contents">
              <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-neutral-700 sm:w-48 sm:flex-none sm:shrink-0">
                {s.label}
              </span>
              <span
                className={cn(
                  'shrink-0 text-right font-mono text-[12px] tabular-nums sm:w-24',
                  s.severity === 'warn' ? 'text-orange-700' : 'text-neutral-900',
                )}
              >
                {s.value == null ? '—' : fmt(s.value, s.unit)}
              </span>
              <span className="shrink-0 font-mono text-[10px] text-neutral-400 sm:w-16">
                {s.unit === 'fraction' ? '' : s.unit}
              </span>
            </div>
            <span className="font-mono text-[10px] leading-relaxed text-neutral-500 sm:min-w-0 sm:flex-1">
              {s.note}
              {s.band && (
                <span className="text-neutral-400">
                  {' '}
                  Conventional range{' '}
                  {s.unit === 'fraction'
                    ? `${fmtFraction(s.band[0])}–${fmtFraction(s.band[1])}`
                    : `[${s.band[0]}, ${s.band[1]}]`}
                  .
                </span>
              )}
            </span>
          </div>
        ))}
      </div>

      {health.renewal_profile.length > 0 && (
        <div className="border-t border-neutral-100 pt-2">
          <div className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-neutral-400">
            renewal along the trajectory
          </div>
          <RenewalProfile profile={health.renewal_profile} />
          <p className="mt-1.5 font-mono text-[10px] leading-relaxed text-neutral-500">
            Fraction of the conditioned path replaced per sweep, in tenths of
            the substeps. This is the shape to read rather than the scalar: a
            flat-but-low profile is a filter that cannot renew anywhere, while a
            profile low at one end is a trajectory pinned there. Orange marks a
            bin under 5%.
          </p>
        </div>
      )}

      {chainRenewal.length > 1 && (
        <div className="border-t border-neutral-100 pt-2">
          <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-neutral-400">
            renewal by chain
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {chainRenewal
              .sort((a, b) => Number(a[0]) - Number(b[0]))
              .map(([cid, v]) => (
                <span key={cid} className="font-mono text-[10px]">
                  <span className="text-neutral-400">c{cid}</span>{' '}
                  <span
                    className={cn(
                      'tabular-nums',
                      v < 0.01 ? 'text-orange-700' : 'text-neutral-700',
                    )}
                  >
                    {fmtFraction(v)}
                  </span>
                </span>
              ))}
          </div>
          <p className="mt-1 font-mono text-[10px] text-neutral-500">
            One frozen chain among healthy ones is invisible in the pooled mean.
          </p>
        </div>
      )}

      {stalls.length > 0 && (
        <div className="border-t border-neutral-100 pt-2">
          <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-neutral-400">
            longest stall by chain
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {stalls
              .sort((a, b) => b[1] - a[1])
              .map(([cid, v]) => (
                <span key={cid} className="font-mono text-[10px]">
                  <span className="text-neutral-400">c{cid}</span>{' '}
                  <span className="tabular-nums text-neutral-700">
                    {v.toLocaleString()}
                  </span>
                </span>
              ))}
          </div>
          <p className="mt-1 font-mono text-[10px] text-neutral-500">
            Consecutive rejected proposals. A long stall is estimator variance,
            not a tuning problem — the fix is more particles, not more sweeps.
          </p>
        </div>
      )}
    </div>
  )
}
