/**
 * The band-layer vocabulary shared by every ribbon view (the Predictive check
 * and the Quantities trajectories): a median line, the 50% interval, the 90%
 * interval, each independently toggleable.
 *
 * Shared because the *vocabulary* is genuinely common — same three layers, same
 * swatches, same meaning — while each tab composes its own control strip around
 * it (Predictive adds "connect observed"; Quantities does not have observations
 * to connect). Unifying the strip itself would mean a component with a toggle
 * per tab, which is the leak this stops short of.
 */
import { cn } from '@/lib/utils'

export type BandLayer = 'median' | 'p50' | 'p90'

export const BAND_LAYERS: { key: BandLayer; label: string }[] = [
  { key: 'median', label: 'median' },
  { key: 'p50', label: '50%' },
  { key: 'p90', label: '90%' },
]

/** Swatch hinting a layer: a line for the median, a filled band for the
 *  intervals (opacity scaled so 90% reads lighter than 50%). */
export function LayerSwatch({ layer, on }: { layer: BandLayer; on: boolean }) {
  const base = on ? '#525252' : '#a3a3a3'
  if (layer === 'median') {
    return (
      <span
        className="inline-block h-[2px] w-3 rounded-full"
        style={{ background: base }}
      />
    )
  }
  return (
    <span
      className="inline-block h-2 w-3 rounded-[1px]"
      style={{ background: base, opacity: layer === 'p90' ? 0.3 : 0.55 }}
    />
  )
}

/**
 * The three layer checkboxes. Checked = drawn; unchecking a layer removes its
 * mark entirely, so the panel's y-domain rescales to what remains — dropping a
 * wild 90% band is how you get the data off the floor.
 *
 * Renders the labels only: the caller supplies the surrounding strip (and its
 * "Show" heading), so this composes into either tab's controls.
 */
export function BandLayerChecks({
  hidden,
  onToggle,
  className,
}: {
  hidden: ReadonlySet<BandLayer>
  onToggle: (layer: BandLayer) => void
  className?: string
}) {
  return (
    <div className={cn('flex flex-wrap items-center gap-x-3 gap-y-1', className)}>
      {BAND_LAYERS.map(({ key, label }) => {
        const on = !hidden.has(key)
        return (
          <label
            key={key}
            className="flex cursor-pointer items-center gap-1.5 font-mono text-xs"
          >
            <input
              type="checkbox"
              checked={on}
              onChange={() => onToggle(key)}
              className="size-3 accent-neutral-800"
            />
            <LayerSwatch layer={key} on={on} />
            <span className={on ? 'text-neutral-900' : 'text-neutral-500'}>
              {label}
            </span>
          </label>
        )
      })}
    </div>
  )
}
