import { useState, type RefObject } from 'react'
import { downloadNodePng } from '@/lib/plot-export'
import { Download } from '@/components/ui/icons'
import { cn } from '@/lib/utils'

/**
 * A "download this plot as PNG" button. `targetRef` points at the DOM node to
 * rasterize — the plot's figure container (for a wide/scrolling figure like the
 * pair grid, ref the full-width inner node so the capture isn't clipped, and
 * place this button on a viewport-width `relative` ancestor so it stays
 * visible). The button is a *sibling* of the target, never inside it, so it
 * never appears in the exported image.
 *
 * Two things about where it sits, both learned from it being unreachable on a
 * phone:
 *
 * **It is only hover-revealed where hovering exists.** Fading in on
 * `group-hover` means a touch device, which never hovers, renders it at
 * `opacity: 0` forever — present and tappable but invisible, which is worse
 * than absent. The fade is therefore gated behind `(hover: hover)` and the
 * button is plainly visible everywhere else.
 *
 * **It tracks the viewport rather than the top of the figure.** Pinned with
 * `absolute top-1`, the button on a 12x12 pair grid sits thousands of pixels
 * above the part of the plot being read, so exporting meant scrolling back to
 * the top to find it. A `sticky` row inside a full-height overlay keeps it in
 * reach for as long as any of the figure is on screen. The overlay takes no
 * layout space and passes pointer events through, so the plot underneath still
 * receives its own hover and clicks.
 *
 * `name` is the file's base name (`<name>.png`).
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
    <div className="pointer-events-none absolute inset-0 z-10">
      <div className="sticky top-1 flex justify-end pr-1">
        <button
          type="button"
          onClick={onClick}
          disabled={busy}
          title="Download PNG"
          aria-label={`Download ${name} as PNG`}
          className={cn(
            'pointer-events-auto inline-flex items-center gap-1 rounded-sm border border-neutral-200',
            'bg-white/85 px-1.5 py-0.5 font-mono text-[10px] text-neutral-500 backdrop-blur',
            'transition-opacity hover:bg-neutral-50 hover:text-neutral-800',
            // Visible by default; only a device that can hover gets the fade.
            'opacity-100 [@media(hover:hover)]:opacity-0',
            'focus-visible:opacity-100 group-hover/fig:opacity-100',
            busy && 'opacity-100',
            className,
          )}
        >
          <Download className="size-3" />
          {busy ? 'saving…' : 'PNG'}
        </button>
      </div>
    </div>
  )
}
