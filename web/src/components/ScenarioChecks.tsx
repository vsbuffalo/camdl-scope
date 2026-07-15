/** Multi-select scenario checkboxes with colored labels — the overlay axis,
 *  shared by the Predictive and Quantities tabs. */
export function ScenarioChecks({
  options,
  selected,
  colorOf,
  onToggle,
}: {
  options: readonly string[]
  selected: readonly string[]
  colorOf: (s: string) => string
  onToggle: (s: string) => void
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
        Scenario
      </span>
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
    </div>
  )
}
