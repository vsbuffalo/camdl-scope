/**
 * Defensive normalization of model-supplied KaTeX strings.
 *
 * `model.render.json` / `model.graph.json` carry symbols written by the
 * upstream renderer, and some are not valid TeX for what they mean: a
 * multi-character subscript emitted unbraced (`R_eff`) is read by TeX as
 * `R_e` followed by a literal `ff`, so the parameter renders as *R‑sub‑e ff*
 * rather than *R_eff*. Every artifact already on disk carries the defect, so
 * the viewer cannot wait for an upstream fix — it normalizes on the way in.
 *
 * The rewrite is deliberately narrow: an unescaped `_` followed by two or more
 * alphanumerics gets braced and set upright (`\mathrm`), which is the
 * convention for a word-like subscript. An already-braced subscript (`_{...}`),
 * a single-character subscript (`I_s`), and an escaped underscore inside a
 * `\mathrm{after\_control}` are all left exactly as they are.
 */
export function normalizeTex(s: string): string {
  if (!s) return s
  // (?<!\\) — skip `\_`, the escaped underscore used inside \mathrm{…} names.
  // ([A-Za-z][A-Za-z0-9]+) — a word-like subscript, two chars or more.
  return s.replace(
    /(?<!\\)_([A-Za-z][A-Za-z0-9]+)/g,
    (_m, word: string) => `_{\\mathrm{${word}}}`,
  )
}

/**
 * Rough on-screen width (px) of a rendered TeX string, for layout decisions
 * made before the DOM exists. Counts the visible glyphs — control sequences,
 * braces and spacing macros contribute nothing — and treats a fraction as the
 * wider of its two rows rather than their sum, since `\frac` stacks.
 *
 * An estimate, not a measurement: it decides whether a rate label is short
 * enough to sit on its arrow, where being off by a few pixels is harmless.
 */
export function estimateTexWidth(tex: string, fontPx: number): number {
  if (!tex) return 0
  // Split a top-level \frac{a}{b} into its two rows and take the wider.
  const frac = tex.match(/^\\frac\{(.*)\}\{(.*)\}$/)
  if (frac) {
    return Math.max(
      estimateTexWidth(frac[1] ?? '', fontPx),
      estimateTexWidth(frac[2] ?? '', fontPx),
    )
  }
  const visible = tex
    .replace(/\\[a-zA-Z]+/g, 'x') // a control sequence renders as ~one glyph
    .replace(/[{}\\,;!\s]/g, '')
  // ~0.55 em per glyph for the mixed italic/roman math faces KaTeX uses.
  return visible.length * fontPx * 0.55
}
