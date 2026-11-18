import { useProfiles } from '@/api/queries'
import { cn } from '@/lib/utils'
import { DEMO } from '@/lib/demo'
import { setThemeChoice, useThemeChoice, type ThemeChoice } from '@/lib/theme'

/** Top-level workspaces. Each owns its own selector + sub-navigation; the run
 *  dropdown belongs to `explore`, not this global chrome. Adding a future mode
 *  (a fit-queue monitor, a residual explorer) is one entry here + one view. */
export type Workspace = 'explore' | 'compare' | 'profile' | 'sims' | 'store'

const ALL_WORKSPACES: { value: Workspace; label: string }[] = [
  { value: 'explore', label: 'Explore fit' },
  { value: 'compare', label: 'Compare models' },
  { value: 'profile', label: 'Profile' },
  { value: 'sims', label: 'Sims' },
  { value: 'store', label: 'Store' },
]

// The static demo has no prequential data, so Compare is dropped from the nav.
const WORKSPACES = DEMO
  ? ALL_WORKSPACES.filter((w) => w.value !== 'compare')
  : ALL_WORKSPACES

/**
 * Persistent top bar: the wordmark on the left, then the workspace nav — a flat,
 * underline-active primary nav in the financial-terminal idiom. This is the
 * outer navigation level; each workspace renders its own inner tabs below.
 */
export function GlobalHeader({
  workspace,
  onWorkspace,
}: {
  workspace: Workspace
  onWorkspace: (w: Workspace) => void
}) {
  // A running profile is not a fit, so the run list cannot show it; the
  // Profile tab carries a running light instead, wherever you are.
  const profiles = useProfiles()
  const nRunning = (profiles.data ?? []).filter((p) => p.status === 'running').length
  return (
    // Height is mirrored by --app-header-h (index.css) so anything sticking
    // below this bar can offset by it; keep them in step.
    <header className="sticky top-0 z-40 border-b border-line bg-surface">
      {/* Match the main content container so the wordmark lines up with panels. */}
      <div className="mx-auto w-full max-w-6xl px-4 sm:px-6">
        <div className="flex items-stretch gap-5 sm:gap-8">
          {/* Two-line wordmark + tall portrait mark. */}
          <div className="flex shrink-0 items-center gap-1.5 py-2.5">
            <span className="w-2 shrink-0 self-stretch bg-accent-ink" aria-hidden />
            <span className="flex flex-col text-sm font-semibold leading-[0.95] tracking-tight text-ink">
              <span>camdl</span>
              {/* '_scope — the tick nods to camdl's #' doc syntax and reads as the
                  instrument clipping ('scope, as in oscilloscope/microscope). */}
              <span className="text-faint">
                <span className="text-accent-ink">{"'"}</span>scope
              </span>
            </span>
          </div>

          <nav className="flex items-stretch gap-4" aria-label="Workspace">
            {WORKSPACES.map((w) => {
              const active = w.value === workspace
              return (
                <button
                  key={w.value}
                  type="button"
                  onClick={() => onWorkspace(w.value)}
                  aria-current={active ? 'page' : undefined}
                  className={cn(
                    // -mb-px drops the active underline onto the header's own
                    // bottom border so the two read as one rule.
                    'relative -mb-px flex items-center border-b-2 px-0.5 text-[13px] font-medium tracking-tight transition-colors',
                    active
                      ? 'border-accent-ink text-ink'
                      : 'border-transparent text-faint hover:text-body-dim',
                  )}
                >
                  {w.label}
                  {w.value === 'profile' && nRunning > 0 && (
                    <span
                      className="ml-1.5 inline-flex items-center gap-1 font-mono text-[10px] text-warn"
                      title={`${nRunning} profile${nRunning === 1 ? '' : 's'} running`}
                    >
                      <span className="size-1.5 bg-warn-vivid" aria-hidden />
                      {nRunning} running
                    </span>
                  )}
                </button>
              )
            })}
          </nav>

          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}

const THEME_OPTIONS: { value: ThemeChoice; label: string }[] = [
  { value: 'light', label: 'light' },
  { value: 'dark', label: 'dark' },
  { value: 'system', label: 'system' },
]

/**
 * Light / dark / follow-the-OS, in the nav's own register: muted mono words,
 * the chosen one in ink. On a phone the three words do not fit beside the nav,
 * so one word shows the current choice and a tap moves to the next.
 */
function ThemeToggle() {
  const choice = useThemeChoice()
  const i = THEME_OPTIONS.findIndex((o) => o.value === choice)
  const here = THEME_OPTIONS[i]!
  const next = THEME_OPTIONS[(i + 1) % THEME_OPTIONS.length]!
  return (
    <div className="ml-auto flex shrink-0 items-center font-mono text-[11px]">
      <div
        role="radiogroup"
        aria-label="Colour theme"
        className="hidden items-center gap-2 sm:flex"
      >
        {THEME_OPTIONS.map((o) => {
          const on = o.value === choice
          return (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={on}
              onClick={() => setThemeChoice(o.value)}
              className={cn(
                'transition-colors',
                on ? 'text-ink' : 'text-faint hover:text-body-dim',
              )}
            >
              {o.label}
            </button>
          )
        })}
      </div>
      <button
        type="button"
        onClick={() => setThemeChoice(next.value)}
        aria-label={`Colour theme: ${here.label}. Switch to ${next.label}.`}
        className="text-faint transition-colors hover:text-body-dim sm:hidden"
      >
        {here.label}
      </button>
    </div>
  )
}
