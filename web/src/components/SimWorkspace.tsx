import { useState } from 'react'
import * as Plot from '@observablehq/plot'
import type { SimSeriesResponse, SimSummary } from '@/api/client'
import { useSims, useSimSeries } from '@/api/queries'
import { Figure } from '@/components/Figure'
import { Segmented } from '@/components/Segmented'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { fmtTick } from '@/lib/format'
import { dayToDate } from '@/lib/calendar'
import { LEVEL_PALETTE } from '@/lib/byindex'
import { loadJson, saveJson } from '@/lib/persist'
import { cn } from '@/lib/utils'

const AXIS = '#737373'
const MONO = 'var(--font-mono)'
const ACCENT = '#2563eb' // band + single-member ink

/** A member's line colour: a single member is the accent; a small sweep cycles
 *  the categorical palette (matching the legend). */
function memberColor(i: number, n: number): string {
  return n === 1 ? ACCENT : (LEVEL_PALETTE[i % LEVEL_PALETTE.length] ?? ACCENT)
}

/**
 * The Sims workspace: a selector over the `sims/` CAS tree, then the chosen
 * forward simulation's compartment trajectories across its parameter *sweep* —
 * a small sweep overlays its members, a large one collapses to a quantile band
 * (with a few sample members to toggle on). Read-only; empty until
 * `camdl simulate` has written a sim for the store.
 */
export function SimWorkspace() {
  const { data, isPending, isError } = useSims()
  const list = data ?? []
  const [selectedId, setSelectedId] = useState<string | undefined>(() =>
    loadJson<string | undefined>('sim:id', undefined),
  )
  const selected = list.find((s) => s.sim_id === selectedId) ?? list[0]
  const select = (id: string) => {
    saveJson('sim:id', id)
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
          detail="Couldn't reach /api/sims. Is camdl-watch running and serving this store?"
        />
      </div>
    )
  }
  if (list.length === 0) {
    return (
      <div className="max-w-4xl">
        <MutedNotice
          title="No simulations yet"
          detail="Nothing under the store's sims/ tree. Run `camdl simulate` and it'll appear here."
        />
      </div>
    )
  }

  return (
    <div className="max-w-4xl">
      <SimSelect sims={list} value={selected?.sim_id} onChange={select} />
      {selected && <SimCard key={selected.sim_id} sim={selected} />}
    </div>
  )
}

/** The sim picker — a dropdown over `sims/`, mirroring the Explore/Profile
 *  selectors; each item is one model with its sweep size. */
function SimSelect({
  sims,
  value,
  onChange,
}: {
  sims: SimSummary[]
  value: string | undefined
  onChange: (id: string) => void
}) {
  const selected = sims.find((s) => s.sim_id === value)
  const tag = (s: SimSummary) =>
    `· ${s.n_members} member${s.n_members === 1 ? '' : 's'}${s.n_members > 1 ? ' · sweep' : ''}`
  return (
    <div className="mb-4 border-b border-neutral-200 pb-1">
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="min-w-0 sm:w-[24rem]" aria-label="Select a sim">
          {selected ? (
            <span className="truncate font-mono text-[13px] text-neutral-900">
              {selected.model}
              <span className="ml-1.5 text-neutral-400">{tag(selected)}</span>
            </span>
          ) : (
            <span className="text-neutral-400">Select a sim…</span>
          )}
        </SelectTrigger>
        <SelectContent>
          {sims.map((s) => (
            <SelectItem key={s.sim_id} value={s.sim_id}>
              <span className="truncate font-mono text-[13px]">
                {s.model}
                <span className="ml-1.5 text-neutral-400">{tag(s)}</span>
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

/** Identity strip + compartment picker + the adaptive trajectory plot. */
function SimCard({ sim }: { sim: SimSummary }) {
  const [state, setState] = useState<string | undefined>(undefined)
  // Committed zoom window (null = full); `drag` tracks the thumbs live so the
  // readout follows without refetching until release.
  const [win, setWin] = useState<[number, number] | null>(null)
  const [drag, setDrag] = useState<[number, number] | null>(null)
  const { data, isPending, isError, isPlaceholderData } = useSimSeries(
    sim.sim_id,
    state,
    win ?? undefined,
  )
  const [showMembers, setShowMembers] = useState(false)

  const bounds: [number, number] = data ? [data.t_min, data.t_max] : [0, 1]
  const toDate = data ? dayToDate(data.calendar) : null
  const sliderVal = drag ?? win ?? bounds
  const fmtBound = (t: number) =>
    toDate ? String(toDate(t).getUTCFullYear()) : String(Math.round(t))

  return (
    <Card
      className={cn(
        'overflow-hidden transition-opacity',
        isPlaceholderData && 'opacity-60',
      )}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 border-b border-neutral-100 px-3 py-2.5">
        <span className="text-sm font-semibold text-neutral-900">{sim.model}</span>
        <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-400">
          forward sim · {sim.n_members} member{sim.n_members === 1 ? '' : 's'}
        </span>
      </div>

      {isPending && <ForestSkeleton rows={3} />}
      {isError && (
        <MutedNotice
          bordered={false}
          title="Couldn't load the simulation"
          detail="The backend returned an error reading this sim's trajectories."
        />
      )}

      {data && data.states.length > 0 && (
        <>
          <div className="flex flex-col gap-2 border-b border-neutral-100 px-3 py-2">
            <Segmented
              label="Compartment"
              options={data.states}
              value={data.state}
              onChange={setState}
            />
            {data.mode === 'band' && (
              <label className="flex cursor-pointer items-center gap-1.5 font-mono text-xs text-neutral-700">
                <input
                  type="checkbox"
                  checked={showMembers}
                  onChange={() => setShowMembers((v) => !v)}
                  className="size-3 accent-neutral-800"
                />
                <span>
                  show {data.members.length} sample member
                  {data.members.length === 1 ? '' : 's'}
                </span>
              </label>
            )}
            {data.mode === 'members' && data.members.length > 1 && (
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-neutral-500">
                {data.members.map((m, i) => (
                  <span key={m.member + i} className="flex items-center gap-1">
                    <span
                      className="inline-block h-[3px] w-3 rounded-full"
                      style={{ background: memberColor(i, data.members.length) }}
                    />
                    {m.member}
                  </span>
                ))}
              </div>
            )}
            <span className="font-mono text-[10px] text-neutral-400">
              {data.mode === 'band'
                ? `${data.n_members} sweep members → quantile band (5–95% · 25–75%) + median`
                : `${data.n_members} sweep member${data.n_members === 1 ? '' : 's'} overlaid · summed over strata`}
            </span>
            {bounds[1] > bounds[0] && (
              <div className="mt-0.5 flex flex-col gap-1">
                <div className="flex items-baseline justify-between font-mono text-[10px] text-neutral-400">
                  <span className="uppercase tracking-wider">window</span>
                  <span className="text-neutral-600">
                    {fmtBound(sliderVal[0])} – {fmtBound(sliderVal[1])}
                    {win && (
                      <button
                        type="button"
                        onClick={() => {
                          setWin(null)
                          setDrag(null)
                        }}
                        className="ml-2 text-neutral-400 underline-offset-2 hover:text-neutral-700 hover:underline"
                      >
                        reset
                      </button>
                    )}
                  </span>
                </div>
                <Slider
                  value={sliderVal}
                  min={bounds[0]}
                  max={bounds[1]}
                  step={Math.max(1, Math.round((bounds[1] - bounds[0]) / 500))}
                  onValueChange={(v) => setDrag([v[0] ?? bounds[0], v[1] ?? bounds[1]])}
                  onValueCommit={(v) => {
                    const lo = v[0] ?? bounds[0]
                    const hi = v[1] ?? bounds[1]
                    setWin(lo <= bounds[0] && hi >= bounds[1] ? null : [lo, hi])
                    setDrag(null)
                  }}
                />
              </div>
            )}
          </div>
          <SimPlot data={data} showMembers={showMembers} />
        </>
      )}

      {data && data.states.length === 0 && (
        <MutedNotice
          bordered={false}
          title="No state trajectories"
          detail="This sim's trajectory has no state columns to plot."
        />
      )}
    </Card>
  )
}

/**
 * One compartment's trajectories over time. Small sweep: a coloured line per
 * member. Large sweep: the quantile band + median, with sample members faint
 * behind when toggled on. x is numeric model-time (days) — a sim carries no
 * calendar yet.
 */
function SimPlot({
  data,
  showMembers,
}: {
  data: SimSeriesResponse
  showMembers: boolean
}) {
  return (
    <Figure
      name={`sim-${data.model}-${data.state}`}
      aria="sim trajectories"
      deps={[data, showMembers]}
      render={(el, width) => {
        // Date the axis when the sim's model declared a calendar; else numeric
        // model-time (days). `xt` maps a raw time to what the scale plots.
        const toDate = dayToDate(data.calendar)
        const xt = (t: number): number | Date => (toDate ? toDate(t) : t)
        const marks: Plot.Markish[] = []
        if (data.mode === 'band') {
          const bandX = (d: { time: number }) => xt(d.time)
          marks.push(
            Plot.areaY(data.band, { x: bandX, y1: 'q05', y2: 'q95', fill: ACCENT, fillOpacity: 0.12 }),
            Plot.areaY(data.band, { x: bandX, y1: 'q25', y2: 'q75', fill: ACCENT, fillOpacity: 0.2 }),
          )
          if (showMembers) {
            for (const m of data.members) {
              const pts = m.time.map((t, i) => ({ t: xt(t), v: m.value[i]! }))
              marks.push(
                Plot.line(pts, { x: 't', y: 'v', stroke: ACCENT, strokeWidth: 0.5, strokeOpacity: 0.4 }),
              )
            }
          }
          marks.push(Plot.line(data.band, { x: bandX, y: 'q50', stroke: ACCENT, strokeWidth: 1.6 }))
        } else {
          data.members.forEach((m, i) => {
            const pts = m.time.map((t, j) => ({ t: xt(t), v: m.value[j]! }))
            marks.push(
              Plot.line(pts, {
                x: 't', y: 'v',
                stroke: memberColor(i, data.members.length),
                strokeWidth: 1.3, strokeOpacity: 0.85,
              }),
            )
          })
        }
        marks.push(Plot.ruleY([0], { stroke: '#e5e5e5', strokeWidth: 0.5 }))

        const node = Plot.plot({
          width,
          height: Math.min(Math.round(width * 0.5), 320),
          marginTop: 10,
          marginBottom: 30,
          marginLeft: 54,
          marginRight: 12,
          style: { background: 'transparent', color: AXIS, fontSize: '10px', fontFamily: MONO },
          x: toDate
            ? { label: 'date →', labelAnchor: 'center', ticks: 6 }
            : {
                label: 'model-time (days) →',
                labelAnchor: 'center',
                ticks: 6,
                tickFormat: (d: number) => fmtTick(d),
              },
          y: {
            label: `↑ ${data.state}`,
            labelAnchor: 'center',
            ticks: 5,
            tickFormat: (d: number) => fmtTick(d),
            grid: true,
          },
          marks,
        })
        el.replaceChildren(node)
      }}
    />
  )
}
