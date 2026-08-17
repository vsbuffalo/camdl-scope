import { Slider } from '@/components/ui/slider'
import { DEMO } from '@/lib/demo'

interface WarmupControlProps {
  value: number
  onChange: (pct: number) => void
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
  cutoff,
  nTail,
}: WarmupControlProps) {
  return (
    <div className="border-b border-neutral-100 px-3 py-2.5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-[10px] font-medium uppercase tracking-wider text-neutral-400">
          Warm-up
        </span>
        <span className="font-mono text-sm tabular-nums text-neutral-900">
          {value}%
        </span>
        <span className="text-[11px] text-neutral-400">
          discard first {value}% of each chain
        </span>

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
          value={[value]}
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
