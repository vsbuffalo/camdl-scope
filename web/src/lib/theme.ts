/**
 * The palette, readable from JavaScript.
 *
 * Utility classes cover markup, but Observable Plot and the hand-drawn SVG
 * diagrams take colours as strings, so they cannot go through a class. Those
 * call `token('muted')` and get whatever `--color-muted` currently resolves
 * to, which keeps one definition of the palette (`index.css`) rather than a
 * second one transcribed into the plotting code — where it had drifted:
 * the diagram's greys came from Tailwind's blue-tinted `gray` ramp while the
 * interface around it used `neutral`.
 *
 * Values are read from the document rather than imported, so a theme swap is a
 * change of custom properties and a re-render, with no plotting code involved.
 * `Figure` re-renders on `deps`, so a future theme toggle belongs in that list.
 */

/** Role names in `index.css`. Chrome only: a mark that encodes *data* — a
 *  chain's colour, a level's colour, a diverging scale — is a palette decision
 *  belonging to the view that draws it, not a theme token. */
export type TokenName =
  | 'page'
  | 'surface'
  | 'sunken'
  | 'raised'
  | 'ink'
  | 'ink-dim'
  | 'body'
  | 'body-dim'
  | 'muted'
  | 'faint'
  | 'ghost'
  | 'hairline'
  | 'line'
  | 'line-strong'
  | 'line-heavy'
  | 'fill'
  | 'fill-soft'
  | 'fill-faint'
  | 'fill-strong'
  | 'inverse'
  | 'on-inverse'
  | 'warn'
  | 'warn-strong'
  | 'danger'
  | 'danger-strong'
  | 'good'
  | 'good-strong'
  | 'notice'
  | 'accent'
  | 'accent-deep'
  | 'accent-ink'

/** Fallbacks for a document that has not painted yet (and for any non-browser
 *  context). Same values as `index.css`; used only when the custom property
 *  reads back empty, which would otherwise hand Plot an empty string and draw
 *  the mark in black. */
const FALLBACK: Record<TokenName, string> = {
  page: '#ffffff',
  surface: '#ffffff',
  sunken: '#fafafa',
  raised: '#f5f5f5',
  ink: '#171717',
  'ink-dim': '#262626',
  body: '#404040',
  'body-dim': '#525252',
  muted: '#737373',
  faint: '#a3a3a3',
  ghost: '#d4d4d4',
  hairline: '#f5f5f5',
  line: '#e5e5e5',
  'line-strong': '#d4d4d4',
  'line-heavy': '#a3a3a3',
  fill: '#a3a3a3',
  'fill-soft': '#d4d4d4',
  'fill-faint': '#e5e5e5',
  'fill-strong': '#737373',
  inverse: '#171717',
  'on-inverse': '#ffffff',
  warn: '#d97706',
  'warn-strong': '#b45309',
  danger: '#dc2626',
  'danger-strong': '#b91c1c',
  good: '#059669',
  'good-strong': '#047857',
  notice: '#c2410c',
  accent: '#1d4ed8',
  'accent-deep': '#1e40af',
  'accent-ink': '#1e3a8a',
}

/** Resolved values, cleared by :func:`invalidateTheme`. Reading a custom
 *  property forces style resolution, and the plotting code asks for the same
 *  half-dozen colours inside per-mark loops, so the uncached version showed up
 *  as layout thrash on the wider trace grids. */
let cache: Partial<Record<TokenName, string>> = {}

/** The current value of a palette role, as a colour string Plot/SVG accepts. */
export function token(name: TokenName): string {
  const hit = cache[name]
  if (hit !== undefined) return hit
  let value = ''
  if (typeof document !== 'undefined') {
    value = getComputedStyle(document.documentElement)
      .getPropertyValue(`--color-${name}`)
      .trim()
  }
  const resolved = value || FALLBACK[name]
  cache[name] = resolved
  return resolved
}

/** Drop the cache after the palette changes. A theme toggle calls this before
 *  triggering the re-render that redraws the figures. */
export function invalidateTheme(): void {
  cache = {}
}
