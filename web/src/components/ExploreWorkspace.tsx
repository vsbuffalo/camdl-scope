import { useEffect, useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { RunSummary } from '@/api/client'
import { useRun, useRuns } from '@/api/queries'
import { RunBar } from '@/components/RunBar'
import { PosteriorTab } from '@/components/PosteriorTab'
import { PriorTab } from '@/components/PriorTab'
import { EstimateTab } from '@/components/EstimateTab'
import { RestartsTab } from '@/components/RestartsTab'
import { PairTab } from '@/components/PairTab'
import { PredictiveTab } from '@/components/PredictiveTab'
import { QuantitiesTab } from '@/components/QuantitiesTab'
import { TracesTab } from '@/components/TracesTab'
import { DiagnosticsTab } from '@/components/DiagnosticsTab'
import { SourceTab } from '@/components/SourceTab'
import { ForestSkeleton, MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { excludedChainsKey, loadJson, saveJson, warmupKey } from '@/lib/persist'
import { DEFAULT_WARMUP_PCT } from '@/lib/chains'

/** The single-fit tab grid — the inner navigation level of the Explore
 *  workspace. Its tab set depends on the fit *kind*: a posterior (sampling) fit
 *  gets the forest/traces/diagnostics view; an MLE ('scout') fit gets the
 *  point-estimate + restarts view. Both reuse the Source (and Quantities) tabs. */
function ResultsTabs({ run }: { run: RunSummary }) {
  return run.fit_kind === 'mle' ? (
    <MleTabs run={run} />
  ) : (
    <PosteriorTabs run={run} />
  )
}

/** MLE fit view: the point estimate and the multi-start restart diagnostic. */
function MleTabs({ run }: { run: RunSummary }) {
  const detail = useRun(run.run_id)
  const hasQuantities = (detail.data?.available_quantities?.length ?? 0) > 0

  const tabs = [
    { value: 'estimate', label: 'Estimate' },
    { value: 'restarts', label: 'Restarts' },
    ...(hasQuantities ? [{ value: 'quantities', label: 'Quantities' }] : []),
    { value: 'source', label: 'Model' },
  ]
  const [tab, setTab] = useState<string>(() =>
    loadJson('explore:mle-tab', 'estimate'),
  )
  const activeTab = tabs.some((t) => t.value === tab) ? tab : 'estimate'
  const onTab = (v: string) => {
    saveJson('explore:mle-tab', v)
    setTab(v)
  }

  return (
    <Tabs value={activeTab} onValueChange={onTab} className="w-full">
      <TabsList>
        {tabs.map((t) => (
          <TabsTrigger key={t.value} value={t.value}>
            {t.label}
          </TabsTrigger>
        ))}
      </TabsList>
      <TabsContent value="estimate">
        <EstimateTab runId={run.run_id} />
      </TabsContent>
      <TabsContent value="restarts">
        <RestartsTab runId={run.run_id} />
      </TabsContent>
      {hasQuantities && (
        <TabsContent value="quantities">
          <QuantitiesTab runId={run.run_id} />
        </TabsContent>
      )}
      <TabsContent value="source">
        <SourceTab runId={run.run_id} />
      </TabsContent>
    </Tabs>
  )
}

/** Posterior (sampling) fit view: the forest, pair, traces, diagnostics tabs. */
function PosteriorTabs({ run }: { run: RunSummary }) {
  const detail = useRun(run.run_id)
  const hasQuantities =
    (detail.data?.available_quantities?.length ?? 0) > 0

  // Chain exclusion and warm-up are both properties of the RUN — a stuck chain
  // is stuck everywhere, and how much of a chain is burn-in is a fact about the
  // sampling, not about the view looking at it. Both are shared across the
  // Posterior / Prior / Pair / Traces / Diagnostics tabs and persisted per run,
  // so a cutoff chosen while reading the traces is the cutoff the forest and
  // the diagnostics then summarise. Held per tab, they disagreed silently: two
  // tabs reporting different R̂ for one fit, neither saying which draws it used.
  // Persisted per run and reloaded on a run switch, so a page reload (or coming
  // back to a run later) keeps the chains you dropped. `excluded` empty = all in.
  //
  // Use the run's REAL chain ids (camdl numbers from 1) — synthesizing 0..n-1
  // would mislabel the checkboxes and, worse, silently drop the top chain, since
  // its id could never appear in the include-list sent to the backend.
  const chainIds = useMemo(
    () => [...run.chain_ids].sort((a, b) => a - b),
    [run.chain_ids],
  )
  const [excludedChains, setExcludedChains] = useState<Set<number>>(
    () => new Set(loadJson(excludedChainsKey(run.run_id), [] as number[])),
  )
  useEffect(() => {
    setExcludedChains(new Set(loadJson(excludedChainsKey(run.run_id), [] as number[])))
  }, [run.run_id])
  const commitExcluded = (next: Set<number>) => {
    saveJson(excludedChainsKey(run.run_id), [...next])
    return next
  }
  const onToggleChain = (id: number) =>
    setExcludedChains((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return commitExcluded(next)
    })
  const onResetChains = () => setExcludedChains(commitExcluded(new Set()))

  const [warmupPct, setWarmupPct] = useState<number>(() =>
    loadJson(warmupKey(run.run_id), DEFAULT_WARMUP_PCT),
  )
  useEffect(() => {
    setWarmupPct(loadJson(warmupKey(run.run_id), DEFAULT_WARMUP_PCT))
  }, [run.run_id])
  const onWarmupPct = (pct: number) => {
    saveJson(warmupKey(run.run_id), pct)
    setWarmupPct(pct)
  }

  const chainProps = {
    chainIds,
    excludedChains,
    onToggleChain,
    onResetChains,
    warmupPct,
    onWarmupPct,
  }

  const tabs = [
    { value: 'posterior', label: 'Posterior' },
    { value: 'prior', label: 'Prior' },
    { value: 'pair', label: 'Pair' },
    { value: 'predictive', label: 'Predictive' },
    ...(hasQuantities ? [{ value: 'quantities', label: 'Quantities' }] : []),
    { value: 'traces', label: 'Traces' },
    { value: 'diagnostics', label: 'Diagnostics' },
    { value: 'source', label: 'Model' },
  ]

  // Persist the active tab across reloads, but fall back to Posterior if the
  // stored tab isn't available for this run (e.g. Quantities on a run without a
  // sidecar) so we never land on an empty panel.
  const [tab, setTab] = useState<string>(() => loadJson('explore:tab', 'posterior'))
  const activeTab = tabs.some((t) => t.value === tab) ? tab : 'posterior'
  const onTab = (v: string) => {
    saveJson('explore:tab', v)
    setTab(v)
  }

  return (
    <Tabs value={activeTab} onValueChange={onTab} className="w-full">
      <TabsList>
        {tabs.map((tab) => (
          <TabsTrigger key={tab.value} value={tab.value}>
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>

      <TabsContent value="posterior">
        <PosteriorTab runId={run.run_id} {...chainProps} />
      </TabsContent>
      <TabsContent value="prior">
        <PriorTab runId={run.run_id} {...chainProps} />
      </TabsContent>
      <TabsContent value="pair">
        <PairTab runId={run.run_id} {...chainProps} />
      </TabsContent>
      <TabsContent value="predictive">
        <PredictiveTab runId={run.run_id} />
      </TabsContent>
      {hasQuantities && (
        <TabsContent value="quantities">
          <QuantitiesTab runId={run.run_id} />
        </TabsContent>
      )}
      <TabsContent value="traces">
        <TracesTab runId={run.run_id} {...chainProps} />
      </TabsContent>
      <TabsContent value="diagnostics">
        <DiagnosticsTab runId={run.run_id} {...chainProps} />
      </TabsContent>
      <TabsContent value="source">
        <SourceTab runId={run.run_id} />
      </TabsContent>
    </Tabs>
  )
}

/**
 * Explore one fit: a run selector + its identity ticker over the six-tab
 * single-fit viewer. Owns run discovery and selection — the Compare workspace
 * owns its own (multi-)selection independently.
 */
export function ExploreWorkspace() {
  const { data, isPending, isError } = useRuns()
  const runs = data ?? []
  // Restore the last-viewed run across reloads; fall back to the newest if it's
  // gone.
  const [selectedId, setSelectedId] = useState<string | undefined>(() =>
    loadJson<string | undefined>('explore:run', undefined),
  )
  const selected =
    runs.find((r) => r.run_id === selectedId) ?? runs[0] ?? undefined
  // Pin the *effective* run (the default counts): the live store reorders as
  // fits land, so without pinning a reload could drop you on a different run —
  // and its chain exclusions wouldn't apply. A fresh visitor still opens on the
  // newest; once a run is in view it stays put until they pick another.
  const selectedRunId = selected?.run_id
  useEffect(() => {
    if (selectedRunId) saveJson('explore:run', selectedRunId)
  }, [selectedRunId])

  // Live monitoring: while the open run is still sampling, refresh its data (and
  // the run list, for the progress blurb) on a short interval so the tabs track
  // the fit instead of freezing at load. Finished runs don't poll.
  const queryClient = useQueryClient()
  const liveId =
    selected && (selected.status === 'running' || selected.status === 'warming')
      ? selected.run_id
      : undefined
  useEffect(() => {
    if (!liveId) return
    const t = setInterval(() => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      queryClient.invalidateQueries({
        predicate: (q) =>
          Array.isArray(q.queryKey) && q.queryKey.includes(liveId),
      })
    }, 5000)
    return () => clearInterval(t)
  }, [liveId, queryClient])

  return (
    <div>
      <RunBar
        runs={runs}
        value={selected?.run_id}
        onChange={setSelectedId}
        loading={isPending}
        error={isError}
      />

      {isError ? (
        <MutedNotice
          title="Backend not reachable"
          detail="Couldn't reach /api. Is camdl-watch running and serving this store?"
        />
      ) : isPending ? (
        <Card className="overflow-hidden">
          <ForestSkeleton />
        </Card>
      ) : runs.length === 0 ? (
        <MutedNotice
          title="No runs found"
          detail="This store has no discoverable fits yet."
        />
      ) : selected ? (
        <ResultsTabs run={selected} />
      ) : null}
    </div>
  )
}
