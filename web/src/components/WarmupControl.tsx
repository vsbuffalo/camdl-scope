import { Slider } from '@/components/ui/slider'
import type { WarmupSetting } from '@/lib/chains'
import { DEMO } from '@/lib/demo'

interface WarmupControlProps {
  /** The chosen share, or `null` for the run's declared warm-up. */
  value: WarmupSetting
  onChange: (pct: WarmupSetting) => void
  /** Where the declared warm-up sits, `null` when the run declares none. */
  declaredPct: number | null
  /** `warmup_cutoff` from the latest response — the iteration draws start at. */
  cutoff: number | null
  /** `n_tail` from the latest response — post-warmup draws per chain. */
  nTail: number | null
}

/**
 * Flat control strip — not a card. Shares the parent frame, set off by a single
 * bottom hairline. Label + percentage on the left, mono readouts on the right,
 * slider underneath. Wraps cleanly at phone widths.
 */
export function WarmupControl({
  value,
  onChange,
  declaredPct,
  cutoff,
  nTail,
}: WarmupControlProps) {
  const declared = value == null
  const pct = value ?? declaredPct ?? 50
  const shown = Math.round(pct)
  return (
    <div className="border-b border-neutral-100 px-3 py-2.5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
          Warm-up
        </span>
        <span className="font-mono text-sm tabular-nums text-neutral-900">
          {shown}%
        </span>
        <span className="text-[11px] text-neutral-400">
          {declared && declaredPct != null
            ? `the run's declared burn-in (${shown}% of each chain), as camdl uses`
            : declared
              ? 'the run declares no burn-in; discarding the first half'
              : `discard first ${shown}% of each chain`}
        </span>
        {!declared && declaredPct != null && !DEMO && (
          <button
            type="button"
            className="text-[11px] text-blue-700 underline-offset-2 hover:underline"
            onClick={() => onChange(null)}
          >
            use declared ({Math.round(declaredPct)}%)
          </button>
        )}

        <dl className="ml-auto flex flex-wrap items-baseline gap-x-4 gap-y-0.5">
          <Readout label="cutoff" value={cutoff} />
          <Readout label="post-warmup draws" value={nTail} />
        </dl>
      </div>

      {/* Static demo: the slider refetches at other cutoffs a static host can't
          serve, so it's hidden — the snapshot is frozen at this warm-up. */}
      {!DEMO && (
        <Slider
          className="mt-2"
          value={[pct]}
          onValueChange={(v) => onChange(v[0] ?? 0)}
          min={0}
          max={95}
          step={5}
        />
      )}
    </div>
  )
}

function Readout({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <dt className="text-[10px] uppercase tracking-wider text-neutral-400">
        {label}
      </dt>
      <dd className="font-mono text-xs tabular-nums text-neutral-700">
        {value ?? '—'}
      </dd>
    </div>
  )
}
