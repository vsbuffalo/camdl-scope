import { useState } from 'react'
import type { ProfileResponse, ProfileSummary } from '@/api/client'
import { useProfile, useProfiles } from '@/api/queries'
import { ProfilePlot } from '@/components/ProfilePlot'
import { ProfileSurface } from '@/components/ProfileSurface'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { StatusBadge } from '@/components/StatusBadge'
import { Card } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from '@/components/ui/select'
import { fmtValue } from '@/lib/format'
import { loadJson, saveJson } from '@/lib/persist'
import { readDeepLink, writeDeepLink } from '@/lib/deeplink'
import { cn } from '@/lib/utils'

/**
 * The Profile workspace: a selector over the `profiles/` CAS tree, then the
 * chosen profile likelihood — a 1D curve (loglik vs the profiled value, its MLE
 * and 95% CI) or, for a two-parameter profile, a 2D likelihood surface on the
 * grid mesh. Read-only, and polls so a running profile fills in live. Empty
 * until `camdl profile` has written a profile for the store.
 */
export function ProfileWorkspace() {
  const { data, isPending, isError } = useProfiles()
  const list = data ?? []
  // Persist the chosen profile — two profiles of the same model share a label
  // (`· g` vs `· g × Cscale`) and the default flips by mtime, so without this a
  // reload can bounce you to the other one.
  const [selectedId, setSelectedId] = useState<string | undefined>(
    () => readDeepLink().profile ?? loadJson<string | undefined>('profile:base', undefined),
  )
  const selected = list.find((p) => p.base_id === selectedId) ?? list[0]
  const selectProfile = (id: string) => {
    saveJson('profile:base', id)
    writeDeepLink({ profile: id })
    setSelectedId(id)
  }

  if (isPending) {
    return (
      <div className="max-w-4xl">
        <Card className="overflow-hidden">
          <ForestSkeleton rows={4} />
        </Card>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="max-w-4xl">
        <MutedNotice
          title="Backend not reachable"
          detail="Couldn't reach /api/profiles. Is camdl-scope running and serving this store?"
        />
      </div>
    )
  }

  if (list.length === 0) {
    return (
      <div className="max-w-4xl">
        <MutedNotice
          title="No profiles yet"
          detail="Nothing under the store's profiles/ tree. Run `camdl profile` to sweep a parameter's likelihood, and it'll appear here."
        />
      </div>
    )
  }

  return (
    <div className="max-w-4xl">
      <ProfileSelect
        profiles={list}
        value={selected?.base_id}
        onChange={selectProfile}
      />
      {selected && <ProfileCard baseId={selected.base_id} />}
    </div>
  )
}

/**
 * The profile picker — a dropdown over the `profiles/` bases, mirroring the
 * Explore run selector so "pick the thing to view" reads the same across
 * workspaces. Each item is one `(model · param)` profile; the trigger echoes
 * the selection. Fixed-height regardless of how many profiles exist.
 */
function ProfileSelect({
  profiles,
  value,
  onChange,
}: {
  profiles: ProfileSummary[]
  value: string | undefined
  onChange: (id: string) => void
}) {
  const selected = profiles.find((p) => p.base_id === value)

  return (
    <div className="mb-4 border-b border-line pb-1">
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger
          className="min-w-0 sm:w-[22rem]"
          aria-label="Select a profile"
        >
          {selected ? (
            <span className="truncate font-mono text-[13px] text-ink">
              {selected.label}
              <span className="ml-1.5 text-faint">
                · {selected.params.join(' × ')}
              </span>
              <ProfileListStatus p={selected} />
            </span>
          ) : (
            <span className="text-faint">Select a profile…</span>
          )}
        </SelectTrigger>
        <SelectContent>
          {profiles.map((p) => (
            <SelectItem key={p.base_id} value={p.base_id}>
              <span className="truncate font-mono text-[13px]">
                {p.label}
                <span className="ml-1.5 text-faint">
                  · {p.params.join(' × ')}
                </span>
                <ProfileListStatus p={p} />
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

/** The identity strip + MLE/CI readout + the profile-likelihood plot. */
function ProfileCard({ baseId }: { baseId: string }) {
  const { data, isPending, isError, isPlaceholderData } = useProfile(baseId)

  return (
    <Card
      className={cn(
        'overflow-hidden transition-opacity',
        isPlaceholderData && 'opacity-60',
      )}
    >
      {isPending && <ForestSkeleton rows={4} />}

      {isError && (
        <MutedNotice
          bordered={false}
          title="Couldn't load the profile"
          detail="The backend returned an error for this profile."
        />
      )}

      {data &&
        (() => {
          const is2d = data.params.length >= 2
          // camdl flags a failed/infeasible cell with a huge-negative sentinel;
          // those aren't real optima, so gate the readout/curve on the cells
          // that actually fit (a just-started profile can be all sentinels).
          const realPoints = data.points.filter((p) => p.loglik > -1e99)
          const hasValid = realPoints.length > 0 && data.mle_loglik != null
          const pending = data.pending ?? []
          return (
            <>
              <div className="border-b border-hairline px-3 py-2.5">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <span className="font-mono text-[11px] text-muted">
                    {data.label} · profile of{' '}
                    <span className="text-ink-dim">
                      {data.params.join(' × ')}
                    </span>
                  </span>
                  <span className="font-mono text-[10px] uppercase tracking-wider text-faint">
                    {data.method} · {data.loglik_type}
                  </span>
                </div>

                <dl className="mt-1.5 flex flex-wrap items-baseline gap-x-5 gap-y-1">
                  {!hasValid ? (
                    <Stat
                      label="status"
                      value="no valid optimum yet"
                      hint={
                        realPoints.length === 0 && data.points.length > 0
                          ? 'every cell so far failed to fit'
                          : 'no grid value has finished yet'
                      }
                    />
                  ) : (
                    <>
                      <Stat
                        label={`${(data.n_estimate_points ?? 1) > 0 ? 'MLE' : 'best so far'} ${data.params.join(', ')}`}
                        value={data.mle_coords.map((v) => fmtValue(v)).join(', ')}
                      />
                      <Stat label="logL" value={fmtValue(data.mle_loglik)} />
                      {is2d ? (
                        <Stat
                          label={`${Math.round(data.ci_level * 100)}% region`}
                          value={`Δlogℓ ≥ −${data.ci_drop.toFixed(2)}`}
                        />
                      ) : (
                        (data.n_estimate_points ?? 2) < 2 ? (
                          <Stat
                            label={`${Math.round(data.ci_level * 100)}% CI`}
                            value="—"
                            hint="needs two complete grid values"
                          />
                        ) : (
                          <Stat
                            label={`${Math.round(data.ci_level * 100)}% CI`}
                            value={`[${fmtValue(data.ci_lo)}, ${fmtValue(data.ci_hi)}]`}
                            hint={
                              data.ci_lo == null || data.ci_hi == null
                                ? 'open — grid edge'
                                : undefined
                            }
                          />
                        )
                      )}
                    </>
                  )}
                </dl>
                <ProfileRunLine data={data} />
              </div>

              {is2d ? (
                data.points.length < 1 && pending.length < 1 ? (
                  <MutedNotice
                    bordered={false}
                    title="Surface filling in"
                    detail="No grid cells have landed yet — they'll appear here as `camdl profile` evaluates them."
                  />
                ) : (
                  <div className="px-3 py-3">
                    <ProfileSurface data={data} />
                    <p className="mt-1.5 font-mono text-[10px] text-faint">
                      {data.points.length + pending.length} grid cells ·
                      brighter = higher likelihood, grey = no feasible fit · red
                      outline = MLE, thin outline = 95% region (Δlogℓ ≥ −
                      {data.ci_drop.toFixed(2)}), both from complete cells ·
                      faded, dashed = partial (some starts still running) ·
                      empty, dashed = no start finished yet
                    </p>
                  </div>
                )
              ) : realPoints.length < 1 || realPoints.length + pending.length < 2 ? (
                <MutedNotice
                  bordered={false}
                  title={
                    hasValid ? 'Not enough grid points' : 'No valid points yet'
                  }
                  detail={
                    hasValid
                      ? 'This profile has fewer than two fitted points — nothing to curve yet.'
                      : `Every evaluated point so far failed to fit (${data.points.length} tried). The curve appears once points optimize successfully.`
                  }
                />
              ) : (
                <div className="px-3 py-3">
                  <ProfilePlot data={data} />
                  <p className="mt-1 font-mono text-[10px] text-faint">
                    {realPoints.length} fitted grid point
                    {realPoints.length === 1 ? '' : 's'}
                    {data.points.length > realPoints.length &&
                      ` (${data.points.length - realPoints.length} failed)`}
                    {pending.length > 0 && `, ${pending.length} not started`}{' '}
                    · CI is the {data.ci_drop.toFixed(2)} log-likelihood drop from
                    the peak, over complete values · hollow = partial (a lower
                    bound) · open triangle = no start finished yet
                  </p>
                  <GridTable data={data} />
                </div>
              )}
            </>
          )
        })()}
    </Card>
  )
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint?: string
}) {
  return (
    <div className="flex items-baseline gap-1.5" title={hint}>
      <dt className="font-mono text-[10px] uppercase tracking-wider text-faint">
        {label}
      </dt>
      <dd className="font-mono text-xs tabular-nums text-ink-dim">
        {value}
        {hint && <span className="ml-1 text-warn">{hint}</span>}
      </dd>
    </div>
  )
}

/** A profile's status in the selector, when it is not simply finished. */
function ProfileListStatus({ p }: { p: ProfileSummary }) {
  if (p.status === 'done' || p.status === 'unknown' || !p.status) return null
  return (
    <span className="ml-2 inline-flex items-baseline gap-1.5">
      <StatusBadge status={p.status} />
      {p.n_starts_expected != null && (
        <span className="text-[11px] tabular-nums text-faint">
          {p.n_starts_finished}/{p.n_starts_expected}
        </span>
      )}
    </span>
  )
}

/** Seconds as a short age: `4 s`, `3 min`, `2 h`. */
function age(seconds: number): string {
  if (seconds < 90) return `${Math.max(0, Math.round(seconds))} s`
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`
  return `${Math.round(seconds / 3600)} h`
}

/**
 * Whether the profile is running, from camdl's invocation heartbeat, how many
 * of its starts have finished, and whether the estimate above is provisional.
 * Nothing for a profile camdl kept no invocation record of.
 */
function ProfileRunLine({ data }: { data: ProfileResponse }) {
  if (data.status === 'unknown' || !data.status) return null
  const beat =
    data.heartbeat_at != null ? age(Date.now() / 1000 - data.heartbeat_at) : null
  return (
    <div className="mt-1.5 flex flex-wrap items-baseline gap-x-2 gap-y-0.5 font-mono text-[11px] text-muted">
      <StatusBadge status={data.status} />
      {data.n_starts_expected != null && (
        <>
          <span className="text-ghost">·</span>
          <span className="tabular-nums">
            {data.n_starts_finished} of {data.n_starts_expected} starts finished
          </span>
        </>
      )}
      {beat && (data.status === 'running' || data.status === 'stalled') && (
        <>
          <span className="text-ghost">·</span>
          <span className={data.status === 'stalled' ? 'text-warn' : 'text-faint'}>
            heartbeat {beat} ago
            {data.status === 'stalled' && ' — stopped partway'}
          </span>
        </>
      )}
      {data.provisional && (
        <>
          <span className="text-ghost">·</span>
          <span className="text-notice">
            provisional: MLE and interval from complete grid values only
          </span>
        </>
      )}
    </div>
  )
}

/** Every grid value of a 1D profile: its logL, how many of its starts have
 *  finished, and their spread. Pending values are listed with 0 starts. */
function GridTable({ data }: { data: ProfileResponse }) {
  const param = data.params[0]!
  const rows = [
    ...data.points.map((p) => ({
      v: p.coords[0]!,
      ll: p.loglik > -1e99 ? p.loglik : null,
      failed: p.loglik <= -1e99,
      n: p.n_starts,
      of: p.n_expected ?? null,
      complete: p.complete ?? true,
      spread: p.loglik_spread ?? null,
    })),
    ...(data.pending ?? []).map((p) => ({
      v: p.coords[0]!,
      ll: null,
      failed: false,
      n: 0,
      of: p.n_expected,
      complete: false,
      spread: null,
    })),
  ].sort((a, b) => a.v - b.v)
  return (
    <div className="mt-3 overflow-x-auto border-t border-hairline">
      <table className="w-full min-w-max border-collapse font-mono text-[11px] tabular-nums">
        <thead>
          <tr className="border-b border-line text-[9px] uppercase tracking-wider text-faint">
            <th className="px-3 py-1.5 text-left font-medium normal-case">{param}</th>
            <th className="px-3 py-1.5 text-right font-medium">logL</th>
            <th className="px-3 py-1.5 text-right font-medium">starts</th>
            <th className="px-3 py-1.5 text-right font-medium">spread</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.v} className="border-b border-hairline last:border-b-0">
              <td className="px-3 py-1 text-left text-body">{fmtValue(r.v)}</td>
              <td className={cn('px-3 py-1 text-right', r.complete ? 'text-ink' : 'text-faint')}>
                {r.failed ? 'failed' : r.ll == null ? '—' : fmtValue(r.ll)}
                {!r.complete && r.ll != null && (
                  <span className="ml-1 text-faint" title="partial: a lower bound">≥</span>
                )}
              </td>
              <td className={cn('px-3 py-1 text-right', r.complete ? 'text-muted' : 'text-notice')}>
                {r.of != null ? `${r.n} of ${r.of}` : r.n}
              </td>
              <td className="px-3 py-1 text-right text-faint">
                {r.spread != null ? fmtValue(r.spread) : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
