import { cn } from '@/lib/utils'

/**
 * A run's lifecycle state as a terminal status light: a small SQUARE swatch
 * (not a pill) plus a lowercase mono label. Color is sparing and muted —
 * running=amber, warming=blue, done=emerald, failed=red, stalled=neutral.
 */
const STATUS: Record<string, { swatch: string; label: string }> = {
  running: { swatch: 'bg-warn-vivid', label: 'running' },
  warming: { swatch: 'bg-accent-vivid', label: 'warming' },
  done: { swatch: 'bg-good-vivid', label: 'done' },
  failed: { swatch: 'bg-danger-vivid', label: 'failed' },
  stalled: { swatch: 'bg-fill', label: 'stalled' },
}

export function StatusBadge({
  status,
  className,
}: {
  status: string
  className?: string
}) {
  const meta = STATUS[status] ?? { swatch: 'bg-fill', label: status }
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 font-mono text-[11px] lowercase text-body-dim',
        className,
      )}
    >
      <span className={cn('size-2 shrink-0', meta.swatch)} aria-hidden />
      {meta.label}
    </span>
  )
}
