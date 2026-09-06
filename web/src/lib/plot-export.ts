import { toBlob } from 'html-to-image'

/**
 * Background for an exported PNG. Deliberately fixed rather than themed: a
 * figure leaving this app goes into a manuscript, a slide, or an issue thread,
 * and it should not carry the appearance of whoever exported it.
 */
const EXPORT_BACKGROUND = '#ffffff'

/**
 * Ceiling on the rasterized image, in pixels of area.
 *
 * Safari — including every browser on iOS, which all run WebKit — refuses to
 * allocate a canvas beyond roughly 16.7 million pixels and returns a blank or
 * null result rather than an error. At `pixelRatio: 3` a figure only 1,400 px
 * square is already over it, so a large pair grid on a phone silently produced
 * nothing. Desktop Chrome allows far more, which is why this never showed up
 * anywhere but on the device where it mattered.
 */
const MAX_RASTER_PX = 16_000_000

/** Device pixels per CSS pixel to rasterize at: 3 for the crisp mono tick
 *  labels, reduced for a figure large enough to breach the canvas ceiling.
 *  A slightly softer big export beats a blank one. */
function rasterScale(node: HTMLElement): number {
  const { width, height } = node.getBoundingClientRect()
  const area = Math.max(1, width * height)
  return Math.max(1, Math.min(3, Math.sqrt(MAX_RASTER_PX / area)))
}

/** What happened, so the caller can say so instead of appearing to do nothing. */
export type ExportResult = 'shared' | 'downloaded' | 'cancelled' | 'failed'

/**
 * Rasterize a plot's DOM node to a PNG and hand it to the user. Works for any
 * plot — a single Observable Plot `<svg>` or the composite pair/traces grid
 * (many SVGs + HTML labels) — since it captures whatever DOM is inside the node.
 *
 * Two delivery routes, because "download" does not mean the same thing on a
 * phone. Where the platform can share files (iOS, Android) this opens the share
 * sheet, which is what puts an image in Photos — the anchor route lands it in
 * Files, where it is no use for pasting into a message or a slide. Everywhere
 * else it falls back to a download.
 *
 * The anchor route defers `revokeObjectURL` rather than calling it in a
 * `finally`. Revoking in the same tick as `click()` is a race the browser is
 * entitled to lose: Chrome starts the download synchronously and survives it,
 * WebKit does not, and the symptom is a tap that does nothing at all.
 */
export async function downloadNodePng(
  node: HTMLElement,
  name: string,
): Promise<ExportResult> {
  const blob = await toBlob(node, {
    pixelRatio: rasterScale(node),
    backgroundColor: EXPORT_BACKGROUND,
  })
  if (!blob) return 'failed'

  const filename = `${name}.png`
  const file = new File([blob], filename, { type: 'image/png' })

  // Share sheet first where it exists and accepts a file. `canShare` is the
  // feature test that matters — Safari exposes `share` for text/URLs on
  // platforms where it will still reject files.
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: filename })
      return 'shared'
    } catch (err) {
      // Dismissing the sheet is a choice, not a failure.
      if (err instanceof Error && err.name === 'AbortError') return 'cancelled'
      // Anything else (no transient activation, unsupported target) falls
      // through to the download below rather than stranding the user.
    }
  }

  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.rel = 'noopener'
  // In the document: Safari ignores a click on a detached anchor.
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  window.setTimeout(() => {
    a.remove()
    URL.revokeObjectURL(url)
  }, 30_000)
  return 'downloaded'
}
