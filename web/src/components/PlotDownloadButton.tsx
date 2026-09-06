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
 * **It is only hover-revealed where hovering exists.** Fading in on
 * `group-hover` means a touch device, which never hovers, renders it at
 * `opacity: 0` forever — present and tappable but invisible, which is worse
 * than absent. The fade is therefore gated behind `(hover: hover)` and the
 * button is plainly visible everywhere else.
 *
 * Positioning is deliberately plain: `absolute` top-right, no overlay and no
 * `pointer-events` juggling above it. A sticky variant that followed the
 * viewport down a tall pair grid worked in every browser it could be tested in
 * and failed on the one device it was written for, so the tall-figure problem
 * stays open rather than being solved with a construct that cannot be verified
 * here. Reaching this on a long plot still means scrolling to the top of it.
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
  const [failed, setFailed] = useState(false)

  const onClick = async () => {
    const el = targetRef.current
    if (!el || busy) return
    setBusy(true)
    setFailed(false)
    try {
      // A rasterization that produces nothing used to return silently, so a tap
      // that failed and a tap that never registered looked identical — with no
      // way to tell which from a phone. Say which.
      const result = await downloadNodePng(el, name)
      if (result === 'failed') setFailed(true)
    } catch {
      setFailed(true)
    } finally {
      setBusy(false)
    }
  }

  const label = busy ? 'saving…' : failed ? 'failed' : 'PNG'

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      title="Download PNG"
      aria-label={`Download ${name} as PNG`}
      className={cn(
        'absolute right-1 top-1 z-10 inline-flex items-center gap-1 rounded-sm border',
        'px-1.5 py-0.5 font-mono text-[10px] backdrop-blur',
        'transition-opacity',
        // Visible by default; only a device that can hover gets the fade.
        'opacity-100 [@media(hover:hover)]:opacity-0',
        'focus-visible:opacity-100 group-hover/fig:opacity-100',
        busy && 'opacity-100',
        failed
          ? 'border-red-300 bg-red-50/90 text-red-700'
          : 'border-neutral-200 bg-white/85 text-neutral-500 hover:bg-neutral-50 hover:text-neutral-800',
        className,
      )}
    >
      <Download className="size-3" />
      {label}
    </button>
  )
}
