import { toBlob } from 'html-to-image'

/**
 * Rasterize a plot's DOM node to a PNG and trigger a download. Works for any
 * plot — a single Observable Plot `<svg>` or the composite pair/traces grid
 * (many SVGs + HTML labels) — since it captures whatever DOM is inside the node.
 *
 * White background (the app is light, plots draw on `transparent`, so a bare
 * capture would be see-through on a dark slide) and 3× device pixels so the tiny
 * mono tick labels stay crisp. A no-op if the node didn't rasterize.
 */
export async function downloadNodePng(
  node: HTMLElement,
  name: string,
): Promise<void> {
  const blob = await toBlob(node, { pixelRatio: 3, backgroundColor: '#ffffff' })
  if (!blob) return
  const url = URL.createObjectURL(blob)
  try {
    const a = document.createElement('a')
    a.href = url
    a.download = `${name}.png`
    a.click()
  } finally {
    URL.revokeObjectURL(url)
  }
}
