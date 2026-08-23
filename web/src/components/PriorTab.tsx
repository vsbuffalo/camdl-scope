import * as Plot from '@observablehq/plot'
import type { PriorPosteriorRow } from '@/api/client'
import { usePriorPosterior, usePriorPredictive, useRun } from '@/api/queries'
import { Figure } from '@/components/Figure'
import { Segmented } from '@/components/Segmented'
import { dayToDate } from '@/lib/calendar'
import { fmtTick } from '@/lib/format'
import { logYOptions } from '@/lib/plot-scale'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { WarmupControl } from '@/components/WarmupControl'
import { ChainSelect } from '@/components/ChainSelect'
import { includedChains, type ChainControls } from '@/lib/chains'
import { Card } from '@/components/ui/card'
import { fmtValue } from '@/lib/format'
import { loadJson, saveJson } from '@/lib/persist'
import { cn } from '@/lib/utils'
import { useState } from 'react'

// Contraction below this means the data barely moved the parameter: the
// posterior is close to the prior restated, so any "estimate" is mostly an
// assumption. Above the high mark the likelihood is clearly in charge.
const CONTRACTION_WEAK = 0.1
const CONTRACTION_STRONG = 0.7
// |z| beyond this is prior/data tension worth reading: the posterior sits more
// than two prior SDs from where the prior was centred.
const Z_TENSION = 2
// Any appreciable posterior mass against a declared bound means the constraint
// is doing work the data should be doing.
const BOUND_PRESSURE = 0.01

/** Contraction colour: red when the data said nothing (or the posterior is
 *  wider than the prior), green when it clearly informed the parameter. */
function contractionClass(c: number | null | undefined): string {
  if (c == null) return 'text-neutral-400'
  if (c < CONTRACTION_WEAK) return 'text-red-600 font-medium'
  if (c >= CONTRACTION_STRONG) return 'text-emerald-600'
  return 'text-neutral-700'
}

function zClass(z: number | null | undefined): string {
  if (z == null) return 'text-neutral-400'
  return Math.abs(z) >= Z_TENSION ? 'text-amber-600 font-medium' : 'text-neutral-700'
}

/** How to read a row, in one phrase — the interpretation the numbers support,
 *  so a reader does not have to hold the thresholds in their head. */
function reading(r: PriorPosteriorRow): { text: string; tone: string } {
  if (r.contraction == null) return { text: 'no prior scale', tone: 'text-neutral-400' }
  if (r.contraction < 0)
    return { text: 'posterior wider than prior', tone: 'text-red-600' }
  if (r.contraction < CONTRACTION_WEAK)
    return { text: 'prior-dominated', tone: 'text-red-600' }
  if ((r.bound_pressure ?? 0) >= BOUND_PRESSURE)
    return { text: 'against its bound', tone: 'text-amber-600' }
  if (r.z != null && Math.abs(r.z) >= Z_TENSION)
    return { text: 'moved off the prior', tone: 'text-amber-600' }
  if (r.contraction >= CONTRACTION_STRONG)
    return { text: 'data-informed', tone: 'text-emerald-600' }
  return { text: 'partly informed', tone: 'text-neutral-500' }
}

/**
 * The prior → posterior half of a Bayesian workflow: what the data actually did
 * to each parameter.
 *
 * Deliberately a table, not a plot. The question here — "which parameters is
 * this fit actually estimating, and which are echoing my priors back at me?" —
 * is answered by three numbers per parameter, and reading eleven of them side
 * by side is faster than eleven overlaid densities. The Posterior tab already
 * draws the prior against the marginal where the shapes matter.
 */
/**
 * The prior predictive check: prior draws pushed through the observation model,
 * banded, with the data overlaid. The gate question is COVERAGE, not sharpness
 * — a prior band that cannot reach the observed series describes a world the
 * data rules out, and no amount of sampling fixes that.
 *
 * camdl writes no prior predictive of its own (camdl#711); this renders one
 * generated into the run directory, so the section is absent until it is.
 */
function PriorPredictivePanel({
  runId,
  streams,
  toDate,
}: {
  runId: string
  streams: string[]
  toDate: ((t: number) => Date) | null
}) {
  const [stream, setStream] = useState<string>()
  const active = stream && streams.includes(stream) ? stream : streams[0]
  const { data } = usePriorPredictive(runId, active)
  const [logY, setLogYState] = useState(() => loadJson('prior:pp-log-y', false))
  const setLogY = (v: boolean) => {
    setLogYState(v)
    saveJson('prior:pp-log-y', v)
  }

  return (
    <>
      <div className="flex flex-col gap-2 border-t border-neutral-200 px-3 py-2">
        {streams.length > 1 && (
          <Segmented
            label="Stream"
            options={streams}
            value={active ?? ''}
            onChange={setStream}
          />
        )}
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
            Show
          </span>
          <label className="flex cursor-pointer items-center gap-1.5 font-mono text-xs">
            <input
              type="checkbox"
              checked={logY}
              onChange={() => setLogY(!logY)}
              className="size-3 accent-neutral-800"
            />
            <span className={logY ? 'text-neutral-900' : 'text-neutral-500'}>
              log y
            </span>
          </label>
        </div>
      </div>
      <Figure
        name={`prior-predictive-${active ?? ''}`}
        aria={`prior predictive ${active ?? ''}`}
        deps={[data, logY, toDate]}
        render={(el, width) => {
          const pred = [...(data?.predictive ?? [])].sort((a, b) => a.time - b.time)
          const obs = (data?.observed ?? []).filter(
            (o) => o.value != null && Number.isFinite(o.value),
          )
          if (pred.length === 0) {
            el.replaceChildren()
            return
          }
          const xOf = (d: { time: number }) => (toDate ? toDate(d.time) : d.time)
          const drawn: number[] = []
          for (const p of pred) drawn.push(p.q05, p.q25, p.q50, p.q75, p.q95)
          for (const o of obs) drawn.push(o.value as number)
          const node = Plot.plot({
            width,
            height: 260,
            marginTop: 10,
            marginBottom: toDate ? 34 : 24,
            marginLeft: 52,
            marginRight: 12,
            style: {
              background: 'transparent',
              color: '#737373',
              fontSize: '10px',
              fontFamily: 'var(--font-mono)',
            },
            x: { label: null, tickSize: 2, tickPadding: 4, ticks: 6 },
            y: {
              label: null,
              tickSize: 2,
              tickPadding: 4,
              ticks: 5,
              tickFormat: (d: number) => fmtTick(d),
              grid: true,
              ...logYOptions(logY, drawn),
            },
            marks: [
              Plot.areaY(pred, {
                x: xOf,
                y1: 'q05',
                y2: 'q95',
                fill: PRIOR_INK,
                fillOpacity: 0.12,
              }),
              Plot.areaY(pred, {
                x: xOf,
                y1: 'q25',
                y2: 'q75',
                fill: PRIOR_INK,
                fillOpacity: 0.22,
              }),
              Plot.line(pred, { x: xOf, y: 'q50', stroke: PRIOR_INK, strokeWidth: 1.3 }),
              Plot.dot(obs, {
                x: xOf,
                y: 'value',
                fill: '#171717',
                r: 2.5,
                stroke: 'white',
                strokeWidth: 0.5,
              }),
              Plot.ruleY([0], { stroke: '#e5e5e5', strokeWidth: 0.5 }),
            ],
          })
          el.replaceChildren(node)
        }}
      />
      <p className="px-3 pb-2 text-[10px] leading-snug text-neutral-400">
        Band = prior predictive (50% / 90%), ● = observed. The check is whether
        the band <em>covers</em> the data: a prior that cannot produce what was
        seen will not be rescued by the likelihood, and one that covers
        everything imaginable has not constrained anything.
      </p>
    </>
  )
}

/** Prior ink — distinct from the posterior arms so the two checks never read as
 *  the same object. */
const PRIOR_INK = '#7c3aed'

export function PriorTab({
  runId,
  chainIds,
  excludedChains,
  onToggleChain,
  onResetChains,
}: { runId: string } & ChainControls) {
  const [warmupPct, setWarmupPct] = useState<number>(() =>
    loadJson('prior:warmup', 50),
  )
  const onWarmup = (v: number) => {
    saveJson('prior:warmup', v)
    setWarmupPct(v)
  }
  const chains = includedChains({
    chainIds,
    excludedChains,
    onToggleChain,
    onResetChains,
  })
  const { data, isPending, isError } = usePriorPosterior(runId, warmupPct, chains)
  const run = useRun(runId)
  const priorStreams = run.data?.available_prior_streams ?? []
  const toDate = dayToDate(run.data?.calendar)

  return (
    <div className="max-w-4xl">
      <Card className="overflow-hidden">
        <WarmupControl
          value={warmupPct}
          onChange={onWarmup}
          cutoff={data?.warmup_cutoff ?? null}
          nTail={data?.n_tail ?? null}
        />
        {chainIds.length > 1 && (
          <div className="border-t border-neutral-100 px-3 py-2">
            <ChainSelect
              chainIds={chainIds}
              excluded={excludedChains}
              onToggle={onToggleChain}
              onReset={onResetChains}
            />
          </div>
        )}

        {priorStreams.length > 0 && (
          <PriorPredictivePanel
            runId={runId}
            streams={priorStreams}
            toDate={toDate}
          />
        )}

        {isPending && <ForestSkeleton rows={3} />}
        {isError && (
          <MutedNotice
            bordered={false}
            title="Couldn't load the prior comparison"
            detail="The backend returned an error for this run."
          />
        )}

        {data && data.rows.length === 0 && (
          <MutedNotice
            bordered={false}
            title="No draws yet"
            detail="This run hasn't produced post-warm-up draws to compare against its priors."
          />
        )}

        {data && data.rows.length > 0 && (
          <>
            <div className="scroll-x-visible overflow-x-auto">
              <table className="w-full border-collapse font-mono text-[11px] tabular-nums">
                <thead>
                  <tr className="border-b border-neutral-200 text-[9px] uppercase tracking-wider text-neutral-400">
                    <th className="px-2 py-1.5 text-left font-medium">parameter</th>
                    <th className="px-2 py-1.5 text-left font-medium">prior</th>
                    <th className="px-2 py-1.5 text-right font-medium">prior sd</th>
                    <th className="px-2 py-1.5 text-right font-medium">post sd</th>
                    <th className="px-2 py-1.5 text-right font-medium">contraction</th>
                    <th className="px-2 py-1.5 text-right font-medium">z</th>
                    <th className="px-2 py-1.5 text-left font-medium">reading</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => {
                    const read = reading(r)
                    return (
                      <tr key={r.param} className="border-t border-neutral-100">
                        <td className="px-2 py-1.5 text-left">
                          <span className="font-medium text-neutral-900">
                            {r.symbol ?? r.param}
                          </span>
                          {r.symbol && r.symbol !== r.param && (
                            <span className="ml-1.5 text-[10px] text-neutral-400">
                              {r.param}
                            </span>
                          )}
                        </td>
                        <td className="px-2 py-1.5 text-left text-[10px] text-neutral-500">
                          {r.prior_label ?? '—'}
                        </td>
                        <td className="px-2 py-1.5 text-right text-neutral-500">
                          {r.prior_sd == null ? '—' : fmtValue(r.prior_sd)}
                        </td>
                        <td className="px-2 py-1.5 text-right text-neutral-700">
                          {r.post_sd == null ? '—' : fmtValue(r.post_sd)}
                        </td>
                        <td
                          className={cn(
                            'px-2 py-1.5 text-right',
                            contractionClass(r.contraction),
                          )}
                        >
                          {r.contraction == null ? '—' : r.contraction.toFixed(2)}
                        </td>
                        <td className={cn('px-2 py-1.5 text-right', zClass(r.z))}>
                          {r.z == null ? '—' : r.z.toFixed(2)}
                        </td>
                        <td className={cn('px-2 py-1.5 text-left text-[10px]', read.tone)}>
                          {read.text}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <ul className="space-y-0.5 border-t border-neutral-100 px-3 py-2 text-[10px] leading-snug text-neutral-400">
              <li>
                <span className="text-neutral-600">contraction</span> — 1 −
                (post sd / prior sd)². Near 1 the likelihood determines the
                parameter; near 0 the posterior is the prior restated, and the
                "estimate" is an assumption. Negative means the posterior came
                out wider than the prior, which is a modelling problem, not a
                result.
              </li>
              <li>
                <span className="text-neutral-600">z</span> — how far the
                posterior mean sits from the prior mean, in prior standard
                deviations. Read it with contraction: large |z| with strong
                contraction is the data overruling the prior; large |z| with
                weak contraction is prior/data conflict with neither winning.
              </li>
              <li>
                A parameter flagged{' '}
                <span className="text-amber-600">against its bound</span> has
                posterior mass piled against a declared limit — the constraint
                is doing the work, so its interval is not an estimate.
              </li>
            </ul>
          </>
        )}
      </Card>
    </div>
  )
}
