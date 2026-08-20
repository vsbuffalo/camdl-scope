import { useLayoutEffect, useRef, useState } from 'react'
import { cn } from '@/lib/utils'

/**
 * A parameter's doc comment, wrapped rather than clipped.
 *
 * Model doc comments are prose — often several sentences explaining why a
 * parameter is coordinated the way it is — so truncating to one line with an
 * ellipsis hid the part that carries the reasoning, and a `title` tooltip is
 * not readable for a paragraph. This wraps to {@link CLAMP_LINES} lines and,
 * only when there is genuinely more, offers a `more` / `less` toggle.
 *
 * The overflow test measures the clamped element against its own scroll height
 * after layout, so the toggle appears exactly when text is actually hidden —
 * no character-count guess that misfires at different widths.
 *
 * NOTE the clamp class is written out literally (`line-clamp-2`): Tailwind
 * scans source text, so an interpolated class name would never be generated.
 */

export function Description({
  text,
  className,
}: {
  text: string
  className?: string
}) {
  const ref = useRef<HTMLSpanElement>(null)
  const [open, setOpen] = useState(false)
  const [clipped, setClipped] = useState(false)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const measure = () => {
      // Only meaningful while clamped; when open the element is its full height.
      if (open) return
      setClipped(el.scrollHeight > el.clientHeight + 1)
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [text, open])

  return (
    <span className="flex min-w-0 flex-wrap items-baseline gap-x-1.5">
      <span
        ref={ref}
        className={cn(
          'min-w-0 whitespace-pre-line break-words',
          !open && 'line-clamp-2',
          className,
        )}
      >
        {text}
      </span>
      {(clipped || open) && (
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="shrink-0 text-[10px] font-medium uppercase tracking-wider text-neutral-400 underline-offset-2 transition-colors hover:text-neutral-600 hover:underline"
          aria-expanded={open}
        >
          {open ? 'less' : 'more'}
        </button>
      )}
    </span>
  )
}
