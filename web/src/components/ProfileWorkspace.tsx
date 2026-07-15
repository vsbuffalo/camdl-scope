import { useState } from 'react'
import type { ProfileSummary } from '@/api/client'
import { useProfile, useProfiles } from '@/api/queries'
import { ProfilePlot } from '@/components/ProfilePlot'
import { ProfileSurface } from '@/components/ProfileSurface'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from '@/components/ui/select'
import { fmtValue } from '@/lib/format'
import { loadJson, saveJson } from '@/lib/persist'
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
  const [selectedId, setSelectedId] = useState<string | undefined>(() =>
    loadJson<string | undefined>('profile:base', undefined),
  )
  const selected = list.find((p) => p.base_id === selectedId) ?? list[0]
  const selectProfile = (id: string) => {
    saveJson('profile:base', id)
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
          detail="Couldn't reach /api/profiles. Is camdl-watch running and serving this store?"
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
    <div className="mb-4 border-b border-neutral-200 pb-1">
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger
          className="min-w-0 sm:w-[22rem]"
          aria-label="Select a profile"
        >
          {selected ? (
            <span className="truncate font-mono text-[13px] text-neutral-900">
              {selected.label}
              <span className="ml-1.5 text-neutral-400">
                · {selected.params.join(' × ')}
              </span>
            </span>
          ) : (
            <span className="text-neutral-400">Select a profile…</span>
          )}
        </SelectTrigger>
        <SelectContent>
          {profiles.map((p) => (
            <SelectItem key={p.base_id} value={p.base_id}>
              <span className="truncate font-mono text-[13px]">
                {p.label}
                <span className="ml-1.5 text-neutral-400">
                  · {p.params.join(' × ')}
                </span>
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
          const hasValid = realPoints.length > 0
          return (
            <>
              <div className="border-b border-neutral-100 px-3 py-2.5">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <span className="font-mono text-[11px] text-neutral-500">
                    {data.label} · profile of{' '}
                    <span className="text-neutral-800">
                      {data.params.join(' × ')}
                    </span>
                  </span>
                  <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-400">
                    {data.method} · {data.loglik_type}
                  </span>
                </div>

                <dl className="mt-1.5 flex flex-wrap items-baseline gap-x-5 gap-y-1">
                  {!hasValid ? (
                    <Stat
                      label="status"
                      value="no valid optimum yet"
                      hint="every cell so far failed to fit"
                    />
                  ) : (
                    <>
                      <Stat
                        label={`MLE ${data.params.join(', ')}`}
                        value={data.mle_coords.map((v) => fmtValue(v)).join(', ')}
                      />
                      <Stat label="logL" value={fmtValue(data.mle_loglik)} />
                      {is2d ? (
                        <Stat
                          label={`${Math.round(data.ci_level * 100)}% region`}
                          value={`Δlogℓ ≥ −${data.ci_drop.toFixed(2)}`}
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
                      )}
                    </>
                  )}
                </dl>
              </div>

              {is2d ? (
                data.points.length < 1 ? (
                  <MutedNotice
                    bordered={false}
                    title="Surface filling in"
                    detail="No grid cells have landed yet — they'll appear here as `camdl profile` evaluates them."
                  />
                ) : (
                  <div className="px-3 py-3">
                    <ProfileSurface data={data} />
                    <p className="mt-1.5 font-mono text-[10px] text-neutral-400">
                      {data.points.length} grid cells · brighter = higher
                      likelihood, grey = no feasible fit · red outline = MLE,
                      white = 95% region (Δlogℓ ≥ −{data.ci_drop.toFixed(2)}) ·
                      updates live as cells land
                    </p>
                  </div>
                )
              ) : realPoints.length < 2 ? (
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
                  <p className="mt-1 font-mono text-[10px] text-neutral-400">
                    {realPoints.length} fitted grid point
                    {realPoints.length === 1 ? '' : 's'}
                    {data.points.length > realPoints.length &&
                      ` (${data.points.length - realPoints.length} failed)`}{' '}
                    · CI is the {data.ci_drop.toFixed(2)} log-likelihood drop from
                    the peak
                  </p>
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
      <dt className="font-mono text-[10px] uppercase tracking-wider text-neutral-400">
        {label}
      </dt>
      <dd className="font-mono text-xs tabular-nums text-neutral-800">
        {value}
        {hint && <span className="ml-1 text-amber-600">{hint}</span>}
      </dd>
    </div>
  )
}
