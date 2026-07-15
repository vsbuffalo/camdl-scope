/**
 * Display formatting helpers. These never *compute* statistics — every number
 * shown is shipped authoritative from the Python core; we only render it.
 */

function trimZeros(s: string): string {
  if (!s.includes('.')) return s
  return s.replace(/\.?0+$/, '')
}

/** Drop a ``.0``/``.00`` mantissa tail from an exponential string
 *  (``2.00e-5`` → ``2e-5``, ``2.50e-5`` → ``2.5e-5``, ``2.2e+4`` unchanged). */
function trimExp(s: string): string {
  return s.replace(/\.?0+e/, 'e')
}

/**
 * Magnitude-aware fixed/scientific formatting with a target number of
 * significant figures. The decimal count adapts to the value's scale so small
 * magnitudes keep their figures (``0.002`` not ``0.00``) instead of collapsing;
 * only genuine extremes fall back to scientific notation.
 */
function fmtSig(x: number, sig: number, expDigits: number): string {
  if (x === 0) return '0'
  const a = Math.abs(x)
  if (a < 1e-4 || a >= 1e6) return trimExp(x.toExponential(expDigits))
  const decimals = Math.min(12, Math.max(0, sig - 1 - Math.floor(Math.log10(a))))
  return trimZeros(x.toFixed(decimals))
}

/**
 * Format a posterior summary value with magnitude-aware precision (~4
 * significant figures). Small/huge magnitudes fall back to scientific notation
 * so a forest readout never shows a wall of zeros.
 */
export function fmtValue(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return '—'
  return fmtSig(x, 4, 2)
}

/** Compact form for axis ticks: a touch coarser than {@link fmtValue} (~3
 *  significant figures), adapting to the axis scale so tiny-magnitude ticks
 *  (e.g. a force of infection ~0.003) stay distinct instead of all reading the
 *  same rounded value. */
export function fmtTick(x: number): string {
  if (!Number.isFinite(x)) return ''
  return fmtSig(x, 3, 1)
}

/** R-hat to three decimals; the convergence threshold lives in the caller. */
export function fmtRhat(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return '—'
  return x.toFixed(3)
}

/** Effective sample size as a rounded integer. */
export function fmtEss(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return '—'
  return Math.round(x).toLocaleString()
}
