import { useEffect, useRef, useState } from 'react'
import { writeClipboard } from '@/lib/clipboard'
import { cn } from '@/lib/utils'

/**
 * A quiet copy affordance for a value the reader needs to paste elsewhere —
 * a run id into a shell, a ticket, a message.
 *
 * It confirms only on a copy that actually happened: over plain HTTP to a
 * hostname the clipboard can genuinely be unavailable, and a button that
 * flashes "copied" regardless would send someone to paste nothing.
 */
export function CopyButton({
  text,
  label,
  className,
}: {
  text: string
  /** What is being copied, for the tooltip and screen readers. */
  label: string
  className?: string
}) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle')
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)

  // Clear on unmount so a pending reset never fires into a gone component.
  useEffect(() => () => clearTimeout(timer.current), [])

  const onClick = async () => {
    const ok = await writeClipboard(text)
    setState(ok ? 'copied' : 'failed')
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setState('idle'), 1400)
  }

  return (
    <button
      type="button"
      onClick={onClick}
      title={state === 'failed' ? `Couldn't copy ${label}` : `Copy ${label}`}
      aria-label={`Copy ${label}`}
      className={cn(
        'inline-flex size-4 shrink-0 cursor-pointer items-center justify-center align-middle transition-colors',
        state === 'copied'
          ? 'text-emerald-600'
          : state === 'failed'
            ? 'text-red-600'
            : 'text-neutral-400 hover:text-neutral-800',
        className,
      )}
    >
      {state === 'copied' ? (
        <svg viewBox="0 0 16 16" className="size-3" aria-hidden="true">
          <path
            d="M3.5 8.5 6.5 11.5 12.5 5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ) : (
        <svg viewBox="0 0 16 16" className="size-3" aria-hidden="true">
          <rect
            x="5.5"
            y="5.5"
            width="8"
            height="8"
            rx="1.2"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.3"
          />
          <path
            d="M10.5 3.5v-1a1 1 0 0 0-1-1h-7a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h1"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      )}
    </button>
  )
}
