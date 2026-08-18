import { useState } from 'react'

/** Above this many scenarios the checkbox row starts collapsed behind a
 *  `▸ Scenario (k/n)` summary — a big scenario sweep otherwise floods the
 *  control strip. */
const COLLAPSE_OVER = 6

/**
 * Multi-select scenario checkboxes with colored labels — the overlay axis,
 * shared by the Predictive and Quantities tabs. `all` / `none` quick actions
 * set the whole selection at once; with more than {@link COLLAPSE_OVER}
 * scenarios the list collapses behind a count summary (click to expand).
 */
export function ScenarioChecks({
  options,
  selected,
  colorOf,
  onToggle,
  onSetAll,
}: {
  options: readonly string[]
  selected: readonly string[]
  colorOf: (s: string) => string
  onToggle: (s: string) => void
  /** Replace the whole selection (the `all` / `none` quick actions). */
  onSetAll: (names: string[]) => void
}) {
  const many = options.length > COLLAPSE_OVER
  // User toggle overrides the size-based default; null = no override yet.
  const [userOpen, setUserOpen] = useState<boolean | null>(null)
  const open = userOpen ?? !many

  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      {many ? (
        <button
          type="button"
          onClick={() => setUserOpen(!open)}
          aria-expanded={open}
          className="flex items-baseline gap-1 text-[10px] font-medium uppercase tracking-wider text-neutral-400 transition-colors hover:text-neutral-600"
        >
          <span>{open ? '▾' : '▸'}</span>
          <span>Scenario</span>
          <span className="font-mono tabular-nums normal-case">
            ({selected.length}/{options.length})
          </span>
        </button>
      ) : (
        <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
          Scenario
        </span>
      )}

      <span className="flex items-center gap-2 font-mono text-[10px] text-neutral-400">
        <button
          type="button"
          onClick={() => onSetAll([...options])}
          className="underline-offset-2 transition-colors hover:text-neutral-700 hover:underline"
        >
          all
        </button>
        <button
          type="button"
          onClick={() => onSetAll([])}
          className="underline-offset-2 transition-colors hover:text-neutral-700 hover:underline"
        >
          none
        </button>
      </span>

      {open && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          {options.map((opt) => {
            const on = selected.includes(opt)
            return (
              <label
                key={opt}
                className="flex cursor-pointer items-center gap-1.5 font-mono text-xs"
              >
                <input
                  type="checkbox"
                  checked={on}
                  onChange={() => onToggle(opt)}
                  className="size-3"
                  style={{ accentColor: colorOf(opt) }}
                />
                <span className={on ? 'text-neutral-900' : 'text-neutral-500'}>{opt}</span>
              </label>
            )
          })}
        </div>
      )}
    </div>
  )
}
