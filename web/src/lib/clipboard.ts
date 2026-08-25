/**
 * Copy text to the clipboard, in the contexts this app is actually read in.
 *
 * `navigator.clipboard` exists only in a SECURE context. Served over plain
 * HTTP to a LAN IP or a Tailscale hostname — which is how the watcher is read
 * from a phone — it is `undefined`, so the modern API alone would leave the
 * button silently dead exactly where copying an id by hand is most painful.
 * The legacy `execCommand` path is deprecated but has no such requirement, so
 * it stands in.
 *
 * Returns whether the text actually reached the clipboard, so the caller can
 * withhold "copied" confirmation rather than claim a copy that never happened.
 */
export async function writeClipboard(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // Permission denied or non-secure context — try the legacy path below
      // rather than reporting failure while a working route remains.
    }
  }
  const ta = document.createElement('textarea')
  ta.value = text
  ta.setAttribute('readonly', '')
  // Off-screen but focusable: `display:none` cannot be selected, and a visible
  // textarea would scroll the page to itself on focus.
  ta.style.position = 'fixed'
  ta.style.top = '0'
  ta.style.opacity = '0'
  ta.style.pointerEvents = 'none'
  document.body.appendChild(ta)
  try {
    ta.select()
    return document.execCommand('copy')
  } catch {
    return false
  } finally {
    ta.remove()
  }
}
