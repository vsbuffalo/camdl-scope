import { cn } from '@/lib/utils'

/** Flat underline segmented control — matches the tab register. A labelled row
 *  of mutually-exclusive options; the active one carries the underline. */
export function Segmented({
  label,
  options,
  value,
  onChange,
}: {
  label: string
  options: readonly string[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <span className="text-[10px] font-medium uppercase tracking-wider text-faint">
        {label}
      </span>
      <div className="flex flex-wrap items-center gap-3">
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            className={cn(
              '-mb-px border-b-2 border-transparent pb-0.5 font-mono text-xs transition-colors',
              opt === value
                ? 'border-line-ink text-ink'
                : 'text-muted hover:text-ink-dim',
            )}
          >
            {opt || '∅'}
          </button>
        ))}
      </div>
    </div>
  )
}
