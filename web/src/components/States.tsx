import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/**
 * Quiet, centered notice for empty / error / unreachable states. `bordered`
 * (default) draws its own dashed frame for standalone use; pass `bordered={false}`
 * when it sits inside an existing panel and shouldn't double up borders.
 */
export function MutedNotice({
  title,
  detail,
  className,
  bordered = true,
}: {
  title: string
  detail?: ReactNode
  className?: string
  bordered?: boolean
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-1 px-6 py-12 text-center',
        bordered &&
          'rounded-none border border-dashed border-neutral-200 bg-[#fafafa]',
        className,
      )}
    >
      <p className="text-sm font-medium text-neutral-500">{title}</p>
      {detail && <p className="max-w-sm text-xs text-neutral-400">{detail}</p>}
    </div>
  )
}

/**
 * Empty-state for a posterior-dependent tab (Predictive, Quantities) that has
 * nothing to show. The message depends on *why* the posterior is absent: a
 * stalled or failed run never produced one, and a warming run hasn't yet — so
 * pointing the user at `camdl fit predict` (the `whenDone` case) would be wrong,
 * predict has nothing to draw from. Falls back to `whenDone` for a finished run
 * that simply hasn't had predict run, or whose model declares no such artifact.
 */
export function NoPosteriorNotice({
  status,
  whenDone,
  bordered = true,
}: {
  status: string | undefined
  whenDone: { title: string; detail: ReactNode }
  bordered?: boolean
}) {
  if (status === 'stalled') {
    return (
      <MutedNotice
        bordered={bordered}
        title="No posterior to predict from"
        detail="This run stopped before it finished sampling — it left partial chains but never wrote a pooled posterior, so there is nothing to predict from."
      />
    )
  }
  if (status === 'failed') {
    return (
      <MutedNotice
        bordered={bordered}
        title="No posterior to predict from"
        detail="This run failed before producing a posterior, so there is nothing to predict from."
      />
    )
  }
  if (status === 'warming' || status === 'running') {
    return (
      <MutedNotice
        bordered={bordered}
        title="Still sampling"
        detail={
          <>
            Posterior-predictive checks appear once the fit finishes and you run{' '}
            <span className="font-mono">camdl fit predict</span>.
          </>
        }
      />
    )
  }
  return (
    <MutedNotice bordered={bordered} title={whenDone.title} detail={whenDone.detail} />
  )
}

/** Placeholder rows that occupy the same rhythm as the forest while loading. */
export function ForestSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="divide-y divide-neutral-100">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="px-3 py-2.5">
          <div className="h-[5.5rem] animate-pulse rounded-none bg-neutral-100" />
        </div>
      ))}
    </div>
  )
}
