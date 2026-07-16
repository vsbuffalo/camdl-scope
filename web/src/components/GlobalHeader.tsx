import { cn } from '@/lib/utils'

/** Top-level workspaces. Each owns its own selector + sub-navigation; the run
 *  dropdown belongs to `explore`, not this global chrome. Adding a future mode
 *  (a fit-queue monitor, a residual explorer) is one entry here + one view. */
export type Workspace = 'explore' | 'compare' | 'profile' | 'sims'

const WORKSPACES: { value: Workspace; label: string }[] = [
  { value: 'explore', label: 'Explore fit' },
  { value: 'compare', label: 'Compare models' },
  { value: 'profile', label: 'Profile' },
  { value: 'sims', label: 'Sims' },
]

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
  return (
    <header className="sticky top-0 z-40 border-b border-neutral-200 bg-white">
      {/* Match the main content container so the wordmark lines up with panels. */}
      <div className="mx-auto w-full max-w-6xl px-4 sm:px-6">
        <div className="flex items-stretch gap-5 sm:gap-8">
          {/* Two-line wordmark + tall portrait mark. */}
          <div className="flex shrink-0 items-center gap-1.5 py-2.5">
            <span className="w-2 shrink-0 self-stretch bg-blue-900" aria-hidden />
            <span className="flex flex-col text-sm font-semibold leading-[0.95] tracking-tight text-neutral-900">
              <span>camdl</span>
              <span className="text-neutral-400">watch</span>
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
                      ? 'border-blue-900 text-neutral-900'
                      : 'border-transparent text-neutral-400 hover:text-neutral-600',
                  )}
                >
                  {w.label}
                </button>
              )
            })}
          </nav>
        </div>
      </div>
    </header>
  )
}
