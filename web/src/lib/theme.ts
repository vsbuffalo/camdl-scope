import { useSyncExternalStore } from 'react'

/**
 * The palette, readable from JavaScript, and the viewer's theme choice.
 *
 * Utility classes cover markup, but Observable Plot and the hand-drawn SVG
 * diagrams take colours as strings, so they cannot go through a class. Those
 * call `token('muted')`, which is `var(--palette-muted)`: one definition of the
 * palette (`index.css`) rather than a second one transcribed into the plotting
 * code.
 *
 * A reference rather than a resolved value, so a theme switch needs no
 * re-render. The browser re-resolves the variable (each role is a
 * `light-dark()` pair) the moment `color-scheme` changes, including inside
 * figures already drawn and in module-level constants such as
 * `const AXIS = token('muted')`, which a resolved string would have frozen at
 * import. Plot accepts `var(…)` as a constant colour (its `isColor` test lists
 * it), and html-to-image inlines computed — that is, resolved — styles, so PNG
 * export sees concrete colours.
 */

/** Role names in `index.css`. Chrome only: a mark that encodes *data* — a
 *  chain's colour, a level's colour, a diverging scale — is a palette decision
 *  belonging to the view that draws it, not a theme token. Those use the
 *  `--chain-*`, `--scenario-*`, `--level-*` and `--mark-*` inks instead. */
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

/** A palette role as a colour string Plot/SVG/CSS accepts. */
export function token(name: TokenName): string {
  return `var(--palette-${name})`
}

/** A data ink from `index.css` (`--chain-3`, `--mark-red`, …) as a colour
 *  string. Same reasoning as `token`: a reference, so it follows the theme. */
export function ink(name: string): string {
  return `var(--${name})`
}

// ---------------------------------------------------------------------------
// The theme choice.
//
// `system` follows the OS (no `data-theme` on <html>, so `color-scheme: light
// dark` decides); `light` / `dark` pin it. The choice is stored per browser.
// index.html carries an inline copy of `applyChoice` that runs before first
// paint, so a dark-theme viewer never sees a white flash; the storage key and
// the attribute must stay in step with it.
// ---------------------------------------------------------------------------

export type ThemeChoice = 'light' | 'dark' | 'system'

export const THEME_STORAGE_KEY = 'camdl-scope:theme'

function isChoice(v: unknown): v is ThemeChoice {
  return v === 'light' || v === 'dark' || v === 'system'
}

/** The stored choice, or `system` when there is none or storage is blocked
 *  (a private window, or site data disabled). */
function readChoice(): ThemeChoice {
  try {
    const v = window.localStorage.getItem(THEME_STORAGE_KEY)
    return isChoice(v) ? v : 'system'
  } catch {
    return 'system'
  }
}

function applyChoice(choice: ThemeChoice): void {
  const root = document.documentElement
  if (choice === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', choice)
}

let current: ThemeChoice | null = null
const listeners = new Set<() => void>()

function getChoice(): ThemeChoice {
  if (current === null) current = readChoice()
  return current
}

/** Record and apply a theme choice. Persisting is best-effort: if storage
 *  throws, the choice still holds for this page load. */
export function setThemeChoice(choice: ThemeChoice): void {
  current = choice
  applyChoice(choice)
  try {
    if (choice === 'system') window.localStorage.removeItem(THEME_STORAGE_KEY)
    else window.localStorage.setItem(THEME_STORAGE_KEY, choice)
  } catch {
    // Storage unavailable; the in-memory choice is enough for this session.
  }
  for (const l of listeners) l()
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb)
  return () => listeners.delete(cb)
}

/** Whether the page is drawing its dark theme right now: the pinned choice,
 *  or the OS preference under `system`. */
export function isDarkNow(): boolean {
  const pinned = document.documentElement.getAttribute('data-theme')
  if (pinned === 'dark') return true
  if (pinned === 'light') return false
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
}

/** The current theme choice, re-rendering when it changes. */
export function useThemeChoice(): ThemeChoice {
  return useSyncExternalStore(subscribe, getChoice, () => 'system')
}
