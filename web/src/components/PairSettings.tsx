import { useState, type ReactNode } from 'react'
import type { RunDetail } from '@/api/client'
import { cn } from '@/lib/utils'

/** Selection groups come straight off the run detail — no new wire type. */
type ParamGroups = RunDetail['groups']

interface PairSettingsProps {
  groups: ParamGroups
  /** The set of currently-visible param names (controlled by the parent). */
  selection: Set<string>
  onChange: (next: Set<string>) => void
}

/**
 * Display-settings strip for the corner plot: a `⚙ params (N/total)` toggle that
 * opens an inline bordered panel listing scalars (flat checkboxes) and each
 * indexed family (a tri-state master checkbox over collapsible per-member
 * checkboxes). This is the lever that keeps a hierarchical fit's many
 * coordinates from opening as a wall of panels — hide leaves, show what matters.
 */
export function PairSettings({
  groups,
  selection,
  onChange,
}: PairSettingsProps) {
  const [open, setOpen] = useState(false)
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())

  const total =
    groups.scalars.length +
    groups.families.reduce((acc, f) => acc + f.members.length, 0)
  const count = selection.size

  const allParams = [
    ...groups.scalars,
    ...groups.families.flatMap((f) => f.members),
  ]

  const toggle = (name: string) => {
    const next = new Set(selection)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    onChange(next)
  }

  const setMany = (names: string[], on: boolean) => {
    const next = new Set(selection)
    for (const n of names) {
      if (on) next.add(n)
      else next.delete(n)
    }
    onChange(next)
  }

  const toggleExpanded = (base: string) => {
    const next = new Set(expanded)
    if (next.has(base)) next.delete(base)
    else next.add(base)
    setExpanded(next)
  }

  return (
    <div className="border-b border-neutral-100 px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex items-center gap-1.5 rounded-sm border border-neutral-200 px-2 py-1 font-mono text-[11px] text-neutral-600 transition-colors hover:bg-neutral-50"
        >
          <span className="text-neutral-400">{open ? '▾' : '▸'}</span>
          <span>{'⚙ params'}</span>
          <span className="tabular-nums text-neutral-400">
            ({count}/{total})
          </span>
        </button>

        {open && (
          <div className="flex items-center gap-2 font-mono text-[10px] text-neutral-400">
            <Quick
              label="defaults"
              onClick={() => onChange(new Set(groups.default_selection))}
            />
            <span className="text-neutral-300">·</span>
            <Quick
              label="scalars only"
              onClick={() => onChange(new Set(groups.scalars))}
            />
            <span className="text-neutral-300">·</span>
            <Quick label="all" onClick={() => onChange(new Set(allParams))} />
          </div>
        )}
      </div>

      {open && (
        <div className="mt-2 rounded-sm border border-neutral-200 bg-[#fafafa] p-2.5">
          {groups.scalars.length > 0 && (
            <div>
              <SectionLabel>scalars</SectionLabel>
              <div className="flex flex-wrap gap-x-4 gap-y-1">
                {groups.scalars.map((name) => (
                  <Check
                    key={name}
                    label={name}
                    checked={selection.has(name)}
                    onChange={() => toggle(name)}
                  />
                ))}
              </div>
            </div>
          )}

          {groups.families.length > 0 && (
            <div
              className={cn(
                groups.scalars.length > 0 &&
                  'mt-2.5 border-t border-neutral-200 pt-2.5',
              )}
            >
              <SectionLabel>families</SectionLabel>
              <div className="flex flex-col gap-1.5">
                {groups.families.map((fam) => {
                  const sel = fam.members.filter((m) =>
                    selection.has(m),
                  ).length
                  const all = sel === fam.members.length
                  const none = sel === 0
                  const isOpen = expanded.has(fam.base)
                  return (
                    <div key={fam.base}>
                      <div className="flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() => toggleExpanded(fam.base)}
                          aria-expanded={isOpen}
                          aria-label={`${isOpen ? 'collapse' : 'expand'} ${fam.base}`}
                          className="w-3 font-mono text-[10px] text-neutral-400 hover:text-neutral-700"
                        >
                          {isOpen ? '▾' : '▸'}
                        </button>
                        <Check
                          label={
                            <span className="font-mono text-[11px]">
                              {fam.base}{' '}
                              <span className="tabular-nums text-neutral-400">
                                ({sel}/{fam.members.length})
                              </span>
                            </span>
                          }
                          checked={all}
                          indeterminate={!all && !none}
                          onChange={() => setMany(fam.members, !all)}
                        />
                      </div>
                      {isOpen && (
                        <div className="ml-[1.375rem] mt-1 flex flex-wrap gap-x-4 gap-y-1 border-l border-neutral-200 pl-2">
                          {fam.members.map((m) => (
                            <Check
                              key={m}
                              label={m}
                              checked={selection.has(m)}
                              onChange={() => toggle(m)}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  )
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-neutral-400">
      {children}
    </div>
  )
}

function Quick({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="underline-offset-2 transition-colors hover:text-neutral-700 hover:underline"
    >
      {label}
    </button>
  )
}

/**
 * A flat mono checkbox row. `indeterminate` drives the native tri-state look on
 * a family master (set imperatively — the DOM property has no React attribute).
 */
function Check({
  label,
  checked,
  indeterminate,
  onChange,
}: {
  label: ReactNode
  checked: boolean
  indeterminate?: boolean
  onChange: () => void
}) {
  return (
    <label className="flex cursor-pointer select-none items-center gap-1.5 font-mono text-[11px] text-neutral-700">
      <input
        type="checkbox"
        checked={checked}
        ref={(el) => {
          if (el) el.indeterminate = indeterminate ?? false
        }}
        onChange={onChange}
        className="h-3 w-3 accent-neutral-700"
      />
      {typeof label === 'string' ? <span>{label}</span> : label}
    </label>
  )
}
