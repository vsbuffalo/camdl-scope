/**
 * Store workspace: what is on disk, and what it is costing.
 *
 * Scoped to the store *family* — the parent holding `fits`/`sims`/`ensembles`
 * — rather than to the runs the app can open, because that is where the bytes
 * are. On the ebola project the watcher's own `fits` subtree is 4.7 GB of a
 * 36 GB tree.
 *
 * Two sizes are reported everywhere and the gap between them is a finding, not
 * a rounding error: `bytes` is apparent size, `alloc_bytes` is what the blocks
 * cost. A tree of five million ~1.9 kB sim files pays a 4 kB block each, so the
 * two differ by ~2.8x there and by almost nothing on a fits tree. Showing only
 * one would either contradict `du` or hide the slack.
 */
import { useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { StoreKindRow, StoreRunRow } from '@/api/client'
import { ApiError, rescanStore } from '@/api/client'
import { useStore } from '@/api/queries'
import { qk } from '@/api/queries'
import { MutedNotice } from '@/components/States'
import { Card } from '@/components/ui/card'
import { StatusBadge } from '@/components/StatusBadge'
import { cn } from '@/lib/utils'

const GIB = 1024 ** 3
const MIB = 1024 ** 2

/** Bytes at a human scale. Binary units, because that is what `du` reports and
 *  a reader will cross-check against `du`. */
function fmtBytes(b: number): string {
  if (b >= GIB) return `${(b / GIB).toFixed(b / GIB >= 10 ? 0 : 2)} GB`
  if (b >= MIB) return `${(b / MIB).toFixed(b / MIB >= 10 ? 0 : 1)} MB`
  if (b >= 1024) return `${(b / 1024).toFixed(0)} kB`
  return `${b} B`
}

const fmtInt = (n: number) => n.toLocaleString()

function fmtAgo(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 90) return `${Math.round(s)}s ago`
  if (s < 5400) return `${Math.round(s / 60)}m ago`
  if (s < 172800) return `${Math.round(s / 3600)}h ago`
  return `${Math.round(s / 86400)}d ago`
}

function fmtDur(s: number): string {
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`
  if (s < 90) return `${s.toFixed(1)} s`
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

/** Uppercase tracked section header, matching the other workspaces' strips. */
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="border-t border-neutral-100 bg-neutral-50/60 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-neutral-500">
      {children}
    </div>
  )
}

/** One headline number. */
function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string
  value: string
  sub?: string
  tone?: 'warn'
}) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5 px-3 py-2">
      <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-neutral-500">
        {label}
      </span>
      <span
        className={cn(
          'font-mono text-lg leading-tight tabular-nums',
          tone === 'warn' ? 'text-orange-700' : 'text-neutral-900',
        )}
      >
        {value}
      </span>
      {sub && (
        <span className="font-mono text-[10px] leading-tight text-neutral-500">
          {sub}
        </span>
      )}
    </div>
  )
}

/**
 * A horizontal proportion bar. `alloc` draws behind `bytes` so the slack — the
 * disk paid but not filled — is visible as the lighter remainder rather than
 * being a number you have to subtract yourself.
 */
function SizeBar({
  bytes,
  alloc,
  max,
  color = 'bg-blue-800',
}: {
  bytes: number
  alloc: number
  max: number
  color?: string
}) {
  const pa = max > 0 ? Math.max(0, Math.min(100, (alloc / max) * 100)) : 0
  const pb = max > 0 ? Math.max(0, Math.min(100, (bytes / max) * 100)) : 0
  return (
    <div className="relative h-2.5 w-full overflow-hidden rounded-[2px] bg-neutral-100">
      <div
        className={cn('absolute inset-y-0 left-0 rounded-[2px] opacity-25', color)}
        style={{ width: `${pa}%` }}
      />
      <div
        className={cn('absolute inset-y-0 left-0 rounded-[2px]', color)}
        style={{ width: `${pb}%` }}
      />
    </div>
  )
}

/** The scan strip: where it looked, when, how long it took, and a refresh. */
function ScanBar({
  root,
  scannedAt,
  duration,
  state,
  files,
  dirs,
  onRescan,
  busy,
}: {
  root: string | null
  scannedAt: number | null
  duration: number | null
  state: string
  files: number
  dirs: number
  onRescan: () => void
  busy: boolean
}) {
  const scanning = state === 'scanning'
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-neutral-100 px-3 py-2">
      <span className="truncate font-mono text-[11px] text-neutral-700">
        {root ?? '—'}
      </span>
      <span className="font-mono text-[10px] text-neutral-500">
        {scanning
          ? `scanning · ${fmtInt(files)} files · ${fmtInt(dirs)} dirs`
          : scannedAt != null
            ? `scanned ${fmtAgo(scannedAt)}${duration != null ? ` · took ${fmtDur(duration)}` : ''}`
            : 'not scanned'}
      </span>
      <button
        type="button"
        onClick={onRescan}
        disabled={busy || scanning}
        className="ml-auto rounded border border-neutral-200 px-2 py-0.5 font-mono text-[10px] text-neutral-700 hover:bg-neutral-50 disabled:opacity-50"
      >
        {scanning ? 'scanning…' : 'rescan'}
      </button>
    </div>
  )
}

const RUN_PAGE = 25

/** Shown when /api/store 404s: the fix is a restart, not a rescan. */
const STALE_BACKEND =
  'The page was reloaded and picked up this tab, but the running camdl-scope process was started before the /api/store route existed — the SPA is static files read per request, while routes load once at startup. Restart camdl-scope and this tab will work.'

export function StoreWorkspace() {
  const { data, isPending, isError, error } = useStore()
  const qc = useQueryClient()
  const [busy, setBusy] = useState(false)
  const [rescanError, setRescanError] = useState<string | null>(null)
  const [showAllRuns, setShowAllRuns] = useState(false)

  const snap = data?.snapshot ?? null
  const scan = data?.scan
  // A 404 here is not a broken store — it is a server started before this route
  // existed. The SPA is served from disk and updates on reload, so the frontend
  // can show a tab the running Python process knows nothing about.
  const staleBackend =
    (error instanceof ApiError && error.status === 404) ||
    rescanError === STALE_BACKEND

  const onRescan = async () => {
    setBusy(true)
    setRescanError(null)
    try {
      await rescanStore()
      await qc.invalidateQueries({ queryKey: qk.store })
    } catch (e) {
      // An unhandled rejection here surfaced as a bare "error" with nothing to
      // act on; the 404 case in particular has an exact fix.
      setRescanError(
        e instanceof ApiError && e.status === 404
          ? STALE_BACKEND
          : e instanceof Error
            ? e.message
            : 'rescan failed',
      )
    } finally {
      setBusy(false)
    }
  }

  const maxChild = useMemo(
    () => Math.max(1, ...(snap?.children ?? []).map((c) => c.alloc_bytes)),
    [snap],
  )
  const maxKind = useMemo(
    () => Math.max(1, ...(snap?.kinds ?? []).map((k) => k.alloc_bytes)),
    [snap],
  )
  const runs: StoreRunRow[] = snap?.runs ?? []
  const maxRun = useMemo(
    () => Math.max(1, ...runs.map((r) => r.alloc_bytes)),
    [runs],
  )
  const shownRuns = showAllRuns ? runs : runs.slice(0, RUN_PAGE)

  // Status tally over the runs the watcher can open, ordered worst-first so a
  // store full of stalled fits reads as such at a glance.
  const statuses = useMemo(() => {
    const counts = snap?.status_counts ?? {}
    const order = ['running', 'warming', 'stalled', 'failed', 'done']
    return order
      .filter((s) => counts[s])
      .map((s) => ({ status: s, n: counts[s]! }))
      .concat(
        Object.entries(counts)
          .filter(([s]) => !order.includes(s))
          .map(([status, n]) => ({ status, n })),
      )
  }, [snap])

  const slack = snap ? snap.total_alloc - snap.total_bytes : 0

  return (
    <div className="max-w-5xl">
      <Card className="overflow-hidden">
        <ScanBar
          root={data?.root ?? null}
          scannedAt={snap?.scanned_at ?? null}
          duration={snap?.duration_s ?? null}
          state={scan?.state ?? 'idle'}
          files={scan?.files ?? 0}
          dirs={scan?.dirs ?? 0}
          onRescan={onRescan}
          busy={busy}
        />

        {(isError || rescanError) && (
          <MutedNotice
            bordered={false}
            title={
              staleBackend
                ? 'This camdl-scope process predates the store view'
                : "Couldn't reach the store API"
            }
            detail={
              staleBackend
                ? STALE_BACKEND
                : (rescanError ?? 'The backend returned an error for /api/store.')
            }
          />
        )}

        {!snap && !isError && (
          <MutedNotice
            bordered={false}
            title={scan?.state === 'scanning' ? 'Walking the store' : 'No scan yet'}
            detail={
              scan?.state === 'scanning'
                ? `A full results/ tree is millions of files and takes minutes. ${fmtInt(scan.files)} files and ${fmtInt(scan.dirs)} directories so far, ${fmtBytes(scan.total_bytes)} of data occupying ${fmtBytes(scan.total_alloc)}.`
                : isPending
                  ? 'Loading.'
                  : 'Hit rescan to walk the store.'
            }
          />
        )}

        {snap && (
          <>
            <div className="grid grid-cols-2 divide-x divide-neutral-100 border-b border-neutral-100 sm:grid-cols-4">
              <Stat
                label="on disk"
                value={fmtBytes(snap.total_alloc)}
                sub={`${fmtBytes(snap.total_bytes)} of data`}
              />
              <Stat
                label="slack"
                value={fmtBytes(slack)}
                sub={
                  snap.total_bytes > 0
                    ? `${(snap.total_alloc / snap.total_bytes).toFixed(2)}× apparent`
                    : undefined
                }
                tone={slack > snap.total_bytes ? 'warn' : undefined}
              />
              <Stat
                label="files"
                value={fmtInt(snap.total_files)}
                sub={`${fmtInt(snap.total_dirs)} directories`}
              />
              <Stat
                label="runs"
                value={fmtInt(runs.length)}
                sub={
                  snap.orphan_dirs
                    ? `${snap.orphan_dirs} not openable`
                    : 'all openable'
                }
                tone={snap.orphan_dirs ? 'warn' : undefined}
              />
            </div>

            {statuses.length > 0 && (
              <>
                <SectionLabel>run status</SectionLabel>
                <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-3 py-2">
                  {statuses.map(({ status, n }) => (
                    <span key={status} className="flex items-center gap-2">
                      <span className="font-mono text-[14px] tabular-nums text-neutral-900">
                        {n}
                      </span>
                      <StatusBadge status={status} />
                    </span>
                  ))}
                  {snap.orphan_dirs > 0 && (
                    <span className="flex items-center gap-1.5">
                      <span className="size-2 rounded-[1px] bg-neutral-300" />
                      <span className="font-mono text-[12px] tabular-nums text-neutral-900">
                        {snap.orphan_dirs}
                      </span>
                      <span className="font-mono text-[10px] text-neutral-500">
                        not a readable run
                      </span>
                    </span>
                  )}
                </div>
              </>
            )}

            <SectionLabel>by run kind</SectionLabel>
            <div className="flex flex-col gap-1.5 px-3 py-2">
              {snap.children.map((c) => (
                <div key={c.name} className="flex items-center gap-3">
                  <span
                    className={cn(
                      'w-28 shrink-0 truncate font-mono text-[11px]',
                      c.is_store ? 'text-neutral-900' : 'text-neutral-600',
                    )}
                    title={c.is_store ? 'the store the watcher indexes' : undefined}
                  >
                    {c.name}
                    {c.is_store && <span className="text-blue-800"> ·</span>}
                  </span>
                  <SizeBar
                    bytes={c.bytes}
                    alloc={c.alloc_bytes}
                    max={maxChild}
                    color={c.is_store ? 'bg-blue-800' : 'bg-neutral-500'}
                  />
                  <span className="w-20 shrink-0 text-right font-mono text-[11px] tabular-nums text-neutral-900">
                    {fmtBytes(c.alloc_bytes)}
                  </span>
                  <span className="hidden w-32 shrink-0 text-right font-mono text-[10px] tabular-nums text-neutral-500 sm:block">
                    {fmtInt(c.files)} files
                  </span>
                </div>
              ))}
              <p className="mt-1 font-mono text-[10px] leading-relaxed text-neutral-500">
                Solid bar is data; the lighter extension is block slack — disk
                paid for but not filled. A kind whose bar is mostly slack is
                paying for file <em>count</em>, not volume.
              </p>
            </div>

            <SectionLabel>by file</SectionLabel>
            <div className="flex flex-col gap-1.5 px-3 py-2">
              {snap.kinds.map((k: StoreKindRow) => (
                <div key={k.name} className="flex items-center gap-3">
                  <span className="w-40 shrink-0 truncate font-mono text-[11px] text-neutral-700">
                    {k.name}
                  </span>
                  <SizeBar
                    bytes={k.bytes}
                    alloc={k.alloc_bytes}
                    max={maxKind}
                    color="bg-neutral-500"
                  />
                  <span className="w-20 shrink-0 text-right font-mono text-[11px] tabular-nums text-neutral-900">
                    {fmtBytes(k.alloc_bytes)}
                  </span>
                  <span className="hidden w-32 shrink-0 text-right font-mono text-[10px] tabular-nums text-neutral-500 sm:block">
                    {fmtInt(k.files)} files
                  </span>
                </div>
              ))}
              <p className="mt-1 font-mono text-[10px] leading-relaxed text-neutral-500">
                Rolled up by filename, not extension — a fits tree is ~98% .tsv,
                which locates nothing.
              </p>
            </div>

            <SectionLabel>
              largest runs · {snap.store.split('/').slice(-1)[0]}
            </SectionLabel>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse font-mono text-[11px]">
                <thead>
                  <tr className="text-[10px] uppercase tracking-[0.1em] text-neutral-500">
                    <th className="w-[46%] px-3 py-1 text-left font-medium">run</th>
                    <th className="px-3 py-1 text-left font-medium">status</th>
                    <th className="px-3 py-1 text-right font-medium">on disk</th>
                    <th className="hidden px-3 py-1 text-right font-medium sm:table-cell">
                      files
                    </th>
                    <th className="w-32 px-3 py-1" />
                  </tr>
                </thead>
                <tbody>
                  {shownRuns.map((r) => (
                    <tr
                      key={r.run_id}
                      className="border-t border-neutral-100 align-middle"
                    >
                      <td
                        className="max-w-[22rem] truncate px-3 py-1 text-neutral-900"
                        title={r.run_id}
                      >
                        {r.label ?? r.run_id}
                      </td>
                      <td className="px-3 py-1">
                        {r.status ? (
                          <StatusBadge status={r.status} />
                        ) : (
                          <span className="text-orange-700">not a run</span>
                        )}
                      </td>
                      <td className="px-3 py-1 text-right tabular-nums text-neutral-900">
                        {fmtBytes(r.alloc_bytes)}
                      </td>
                      <td className="hidden px-3 py-1 text-right tabular-nums text-neutral-500 sm:table-cell">
                        {fmtInt(r.files)}
                      </td>
                      <td className="py-1 pr-3">
                        <SizeBar
                          bytes={r.bytes}
                          alloc={r.alloc_bytes}
                          max={maxRun}
                          color={r.status ? 'bg-neutral-500' : 'bg-orange-600'}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {runs.length > RUN_PAGE && (
              <button
                type="button"
                onClick={() => setShowAllRuns((v) => !v)}
                className="w-full border-t border-neutral-100 px-3 py-1.5 text-left font-mono text-[10px] text-neutral-600 hover:bg-neutral-50"
              >
                {showAllRuns
                  ? 'show fewer'
                  : `show all ${fmtInt(runs.length)} runs`}
              </button>
            )}

            {snap.unreadable > 0 && (
              <p className="border-t border-neutral-100 px-3 py-2 font-mono text-[10px] text-orange-700">
                {fmtInt(snap.unreadable)} paths could not be read and are not
                counted in these totals.
              </p>
            )}
          </>
        )}
      </Card>
    </div>
  )
}
