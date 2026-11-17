import { toBlob } from 'html-to-image'
import { isDarkNow } from '@/lib/theme'

/**
 * Background for an exported PNG. Deliberately fixed rather than themed: a
 * figure leaving this app goes into a manuscript, a slide, or an issue thread,
 * and it should not carry the appearance of whoever exported it. The figure's
 * own colours follow suit — see `lightCopy`.
 */
const EXPORT_BACKGROUND = '#ffffff'

/** Text properties a figure inherits from the panels around it. A copy moved
 *  out to the end of <body> would otherwise lose them and set its HTML labels
 *  in the page default. */
const INHERITED_TEXT = [
  'font-family',
  'font-size',
  'font-weight',
  'line-height',
  'letter-spacing',
  'font-feature-settings',
] as const

/**
 * A light-theme copy of `node`, attached off-screen, for capture while the
 * page is dark. Every colour in a figure is a `var(--…)` from index.css, and
 * the `theme-light` class redeclares that palette under a light colour scheme,
 * so the copy resolves the light values while the page on screen does not
 * change. Capturing the on-screen node instead would put near-white axis text
 * on the white export background. Call `remove()` on the result when done.
 */
function lightCopy(node: HTMLElement): { copy: HTMLElement; remove: () => void } {
  const style = getComputedStyle(node)
  const wrap = document.createElement('div')
  wrap.setAttribute('aria-hidden', 'true')
  wrap.style.position = 'fixed'
  wrap.style.left = '-100000px'
  wrap.style.top = '0'
  wrap.style.width = `${node.getBoundingClientRect().width}px`
  wrap.style.pointerEvents = 'none'
  wrap.className = 'theme-light'
  wrap.style.color = 'var(--palette-fg)'
  for (const p of INHERITED_TEXT) wrap.style.setProperty(p, style.getPropertyValue(p))
  const copy = node.cloneNode(true) as HTMLElement
  wrap.appendChild(copy)
  document.body.appendChild(wrap)
  return { copy, remove: () => wrap.remove() }
}

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
  const options = {
    pixelRatio: rasterScale(node),
    backgroundColor: EXPORT_BACKGROUND,
  }
  let blob: Blob | null
  if (isDarkNow()) {
    const { copy, remove } = lightCopy(node)
    try {
      blob = await toBlob(copy, options)
    } finally {
      remove()
    }
  } else {
    blob = await toBlob(node, options)
  }
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
