import { useState, type RefObject } from 'react'
import { downloadNodePng } from '@/lib/plot-export'
import { Download } from '@/components/ui/icons'
import { cn } from '@/lib/utils'

/**
 * A hover-revealed "download this plot as PNG" button. `targetRef` points at the
 * DOM node to rasterize — the plot's figure container (for a wide/scrolling
 * figure like the pair grid, ref the full-width inner node so the capture isn't
 * clipped, and place this button on a viewport-width `relative` ancestor so it
 * stays visible). The button is a *sibling* of the target, never inside it, so
 * it never appears in the exported image.
 *
 * Absolute-positioned top-right; the host wraps the figure in a
 * `group/fig relative` container so the button fades in on hover. `name` is the
 * file's base name (`<name>.png`).
 */
export function PlotDownloadButton({
  targetRef,
  name,
  className,
}: {
  targetRef: RefObject<HTMLElement | null>
  name: string
  className?: string
}) {
  const [busy, setBusy] = useState(false)

  const onClick = async () => {
    const el = targetRef.current
    if (!el || busy) return
    setBusy(true)
    try {
      await downloadNodePng(el, name)
    } finally {
      setBusy(false)
    }
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      title="Download PNG"
      aria-label={`Download ${name} as PNG`}
      className={cn(
        'absolute right-1 top-1 z-10 inline-flex items-center gap-1 rounded-sm border border-neutral-200',
        'bg-white/85 px-1.5 py-0.5 font-mono text-[10px] text-neutral-500 backdrop-blur',
        'opacity-0 transition-opacity hover:bg-neutral-50 hover:text-neutral-800',
        'focus-visible:opacity-100 group-hover/fig:opacity-100',
        busy && 'opacity-100',
        className,
      )}
    >
      <Download className="size-3" />
      {busy ? 'saving…' : 'PNG'}
    </button>
  )
}
