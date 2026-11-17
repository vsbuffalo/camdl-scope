import { ink } from '@/lib/theme'

/**
 * Subtle categorical palette for chains — muted 700-ish hues, not a rainbow.
 * Shared so every chain-coloured view (the pair-plot scatter, the trace grid)
 * agrees on which colour is which chain. At low opacity these read as quiet
 * tints; the point is to *detect* separation between chains (poor mixing), not
 * to dazzle.
 *
 * The values are `--chain-0` … `--chain-7` in index.css, one light and one dark
 * each: in light, slate-600, teal-700, amber-700, rose-800, indigo-700,
 * green-700, purple-700, yellow-700; in dark, the same hues at lighter steps.
 */
export const CHAIN_COLORS = Array.from({ length: 8 }, (_, i) => ink(`chain-${i}`))

/**
 * The colour for a chain, keyed by its **id** — not its position in whatever
 * subset happens to be on screen. This is the invariant that lets a
 * visibly-wrong chain in the pair plot be matched to its checkbox and dropped:
 * chain N is the same colour in the scatter, the trace grid, and the chain
 * selector, and it does NOT shift when other chains are filtered out. Feed a
 * Plot categorical scale via `range: domain.map(chainColor)` so each id maps to
 * its own colour regardless of the domain's contents or order.
 */
export function chainColor(id: number): string {
  const n = CHAIN_COLORS.length
  return CHAIN_COLORS[((id % n) + n) % n]!
}
