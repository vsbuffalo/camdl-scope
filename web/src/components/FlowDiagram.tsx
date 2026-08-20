import { Fragment, useMemo, useState } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import type { GraphEdge, ModelGraph, ModelRender } from '@/api/client'
import { Description } from '@/components/Description'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { estimateTexWidth, normalizeTex } from '@/lib/tex'

/**
 * The compartmental flow diagram (`model.graph.json`): a hand-rolled SVG
 * node-link view. The base graph is small (a near-chain of compartments), so a
 * layered left-to-right layout suffices — columns by topological depth, with
 * back-edges (e.g. recover: I→S) drawn as curved return arcs. Plates are
 * collapsed to one enclosing box (a stratified model is thousands of cells, not
 * one node per level); exogenous flows (birth/death) become source/sink chips;
 * plate-family edges that step every compartment (aging) and mean-field pools an
 * edge reads are summarized in the caption. Python emits the graph; this only
 * lays it out.
 */

const NODE_W = 96
const NODE_H = 36
const COL_GAP = 104
const ROW_GAP = 24
const PAD = 28
const CHIP_W = 52

/** How far a back-edge arcs above the top row, by column span. One source of
 *  truth so the reserved top headroom and the drawn arc can't drift. */
function backLift(fromCol: number, toCol: number): number {
  return 34 + Math.abs(fromCol - toCol) * 8
}

/** KaTeX → sanitized HTML for a trusted, model-derived string. Normalized
 *  first: upstream emits word-like subscripts unbraced (`R_eff`), which TeX
 *  reads as `R_e` + literal `ff` — see lib/tex. */
function katexHtml(tex: string): string {
  try {
    return katex.renderToString(normalizeTex(tex), {
      throwOnError: false,
      output: 'html',
    })
  } catch {
    return tex
  }
}

/** A rate expression wider than this (px, estimated) is not drawn on its
 *  arrow: it would run across neighbouring compartments and be unreadable.
 *  The arrow then carries the reaction's name and the full expression appears
 *  in the reactions table below, where it has a whole row to itself. */
const RATE_LABEL_MAX_W = 190

/** An edge is "structural" (defines the layout DAG) iff both endpoints are real
 *  compartments — not the `"c"` iterator and not an exogenous `null`. */
function isStructural(e: GraphEdge): boolean {
  return (
    e.from != null &&
    e.to != null &&
    e.from !== 'c' &&
    e.to !== 'c' &&
    e.from !== e.to
  )
}

type Placed = {
  id: string
  label: string
  col: number
  row: number
  x: number
  y: number
}

type LaidOut = {
  placed: Map<string, Placed>
  forward: GraphEdge[]
  back: GraphEdge[]
  backLiftById: Map<string, number> // per back-edge arc height (staggered if parallel)
  births: GraphEdge[] // from == null
  sinks: GraphEdge[] // to == null, real source (drawn as a chip)
  iterator: GraphEdge[] // from/to == "c" — plate-family (aging/death-all)
  width: number
  height: number
}

/** Longest-path column assignment over the structural edges. Cycles (e.g. a
 *  waning R→S closing the SEIR loop) are broken first: a DFS classifies the
 *  edges that close a cycle as back-edges, depth is a longest-path over the
 *  remaining DAG, and the back-edges are drawn as return arcs. */
function computeLayout(graph: ModelGraph): LaidOut {
  const nodes = graph.nodes
  const structural = graph.edges.filter(isStructural)
  const adj = new Map<string, GraphEdge[]>(nodes.map((n) => [n.id, []]))
  for (const e of structural) adj.get(e.from!)?.push(e)

  // DFS back-edge detection: an edge to a node currently on the stack closes a
  // cycle. Root the walk at exogenous-inflow targets (births) when present, else
  // nodes with no structural in-edge, else all nodes — so a fully-cyclic graph
  // still gets a stable root.
  const hasIn = new Set(structural.map((e) => e.to!))
  const birthTargets = graph.edges
    .filter((e) => e.from == null && e.to != null && e.to !== 'c')
    .map((e) => e.to!)
  const roots = [
    ...birthTargets,
    ...nodes.filter((n) => !hasIn.has(n.id)).map((n) => n.id),
    ...nodes.map((n) => n.id),
  ]
  const backSet = new Set<string>()
  const state = new Map<string, 0 | 1 | 2>() // 0 unseen · 1 on-stack · 2 done
  const visit = (u: string) => {
    state.set(u, 1)
    for (const e of adj.get(u) ?? []) {
      const s = state.get(e.to!) ?? 0
      if (s === 1) backSet.add(e.id)
      else if (s === 0) visit(e.to!)
    }
    state.set(u, 2)
  }
  for (const r of roots) if ((state.get(r) ?? 0) === 0) visit(r)

  // Longest path over the DAG (structural minus back-edges) — converges since
  // it's acyclic; iterate to a fixed point.
  const dag = structural.filter((e) => !backSet.has(e.id))
  const depth = new Map<string, number>(nodes.map((n) => [n.id, 0]))
  for (let i = 0; i < nodes.length; i++) {
    let changed = false
    for (const e of dag) {
      const nd = (depth.get(e.from!) ?? 0) + 1
      if ((depth.get(e.to!) ?? 0) < nd) {
        depth.set(e.to!, nd)
        changed = true
      }
    }
    if (!changed) break
  }

  const forward = dag
  const back = structural.filter((e) => backSet.has(e.id))

  // Per back-edge arc height. Multiple back-edges between the SAME pair (e.g.
  // recover_climb + recover_top, both I_rec→S) would draw the same arc and stack
  // their labels — stagger each parallel one higher so arcs and labels separate.
  const backLiftById = new Map<string, number>()
  const parallelSeen = new Map<string, number>()
  for (const e of back) {
    const key = `${e.from}->${e.to}`
    const rank = parallelSeen.get(key) ?? 0
    parallelSeen.set(key, rank + 1)
    const lift = backLift(depth.get(e.from!) ?? 0, depth.get(e.to!) ?? 0) + rank * 40
    backLiftById.set(e.id, lift)
  }

  // Back-edges arc above the top row; reserve headroom so their apex and rate
  // label aren't clipped at the top of the SVG. Offset every row down by it.
  const maxLift = Math.max(0, ...backLiftById.values())
  const top = maxLift ? maxLift + NODE_H + 8 : PAD

  // Rows: order of appearance within each column.
  const perCol = new Map<number, number>()
  const placed = new Map<string, Placed>()
  for (const n of nodes) {
    const col = depth.get(n.id) ?? 0
    const row = perCol.get(col) ?? 0
    perCol.set(col, row + 1)
    placed.set(n.id, {
      id: n.id,
      label: n.label,
      col,
      row,
      x: PAD + CHIP_W + col * (NODE_W + COL_GAP),
      y: top + row * (NODE_H + ROW_GAP),
    })
  }

  const births = graph.edges.filter((e) => e.from == null && e.to != null && e.to !== 'c')
  const sinks = graph.edges.filter((e) => e.to == null && e.from != null && e.from !== 'c')
  const iterator = graph.edges.filter((e) => e.from === 'c' || e.to === 'c')

  const nCols = Math.max(1, ...[...depth.values()].map((d) => d + 1))
  const nRows = Math.max(1, ...[...perCol.values()])
  const width = PAD * 2 + CHIP_W + nCols * NODE_W + (nCols - 1) * COL_GAP + CHIP_W
  const height = top + nRows * NODE_H + (nRows - 1) * ROW_GAP + PAD

  return { placed, forward, back, backLiftById, births, sinks, iterator, width, height }
}

/** A KaTeX label to overlay on the SVG at absolute pixel coordinates.
 *
 *  Labels are HTML divs positioned over the (unscaled) SVG rather than
 *  ``<foreignObject>`` elements inside it: WebKit/Safari applies page zoom to a
 *  foreignObject's x/y on top of the SVG's own coordinate mapping, so every
 *  label drifted right/down proportionally to its distance from the origin at
 *  any zoom ≠ 100%. The SVG here is drawn 1:1 (width == viewBox), so an HTML
 *  overlay at the same coordinates is pixel-identical — and zoom scales both
 *  layers together in every browser. ``size`` stays an inline px so KaTeX
 *  doesn't scale with any inherited font-size. */
type LabelSpec = {
  key: string
  x: number
  y: number
  w: number
  h: number
  tex: string
  size: number
  color: string
  chip?: boolean // white backing chip (tight to the text) for on-edge rates
  /** Render as plain text rather than KaTeX — used for the reaction-name
   *  fallback when a rate expression is too wide to sit on its arrow. */
  plain?: boolean
}

function TexLabel({ l }: { l: LabelSpec }) {
  return (
    <div
      style={{
        position: 'absolute',
        left: l.x,
        top: l.y,
        width: l.w,
        height: l.h,
        fontSize: `${l.size}px`,
        lineHeight: 1,
      }}
      className={`pointer-events-none flex items-center justify-center whitespace-nowrap text-center ${l.color}`}
    >
      {l.plain ? (
        <span
          className={cn(
            'font-mono',
            l.chip ? 'rounded bg-white/85 px-0.5' : undefined,
          )}
        >
          {l.tex}
        </span>
      ) : (
        <span
          className={l.chip ? 'rounded bg-white/85 px-0.5' : undefined}
          // KaTeX output is sanitized markup for a trusted, model-derived string.
          dangerouslySetInnerHTML={{ __html: katexHtml(l.tex) }}
        />
      )}
    </div>
  )
}

/** Shared geometry of a forward (left→right) edge: line endpoints + midpoint. */
function forwardGeom(from: Placed, to: Placed) {
  const x1 = from.x + NODE_W
  const y1 = from.y + NODE_H / 2
  const x2 = to.x
  const y2 = to.y + NODE_H / 2
  return { x1, y1, x2, y2, mx: (x1 + x2) / 2, my: (y1 + y2) / 2 }
}

/** Shared geometry of a back/return edge: the arc path + its apex height.
 *  ``lift`` is supplied by the layout so parallel back-edges get distinct
 *  heights and the reserved top headroom matches exactly. */
function backGeom(from: Placed, to: Placed, lift: number) {
  const x1 = from.x + NODE_W / 2
  const y1 = from.y
  const x2 = to.x + NODE_W / 2
  const y2 = to.y
  const top = Math.min(y1, y2) - lift
  const d = `M ${x1} ${y1} C ${x1} ${top}, ${x2} ${top}, ${x2} ${y2 - 6}`
  return { x1, x2, top, d }
}

/**
 * Horizontally-scrolling canvas for the diagram. `scroll-x-visible` (index.css)
 * gives it a persistent, styled scrollbar rather than the macOS overlay one
 * that stays hidden until you scroll — with `overflow-x: auto` the bar appears
 * only when the diagram is actually wider than its column.
 */
function ScrollCanvas({
  width,
  height,
  children,
}: {
  width: number
  height: number
  children: React.ReactNode
}) {
  return (
    <div className="scroll-x-visible overflow-x-auto px-3 py-3">
      <div className="relative" style={{ width, height }}>
        {children}
      </div>
    </div>
  )
}

export function FlowDiagram({
  graph,
  render,
}: {
  graph: ModelGraph
  /** The equations artifact, when present: supplies the parameter glossary the
   *  legend explains the rate symbols with. The diagram renders fine without. */
  render?: ModelRender
}) {
  const L = useMemo(() => computeLayout(graph), [graph])
  // Focused compartment: clicking one highlights its flows and names them
  // below, so a reader can follow one compartment at a time through a busy
  // diagram. Click again (or the same chip) to clear.
  const [focus, setFocus] = useState<string | null>(null)

  // The reactions touching the focused compartment, by direction. Edge ids are
  // the model's reaction names (`infection`, `progression`), so this reads as
  // the model's own vocabulary rather than invented labels.
  const flows = useMemo(() => {
    if (!focus) return null
    const into = graph.edges.filter((e) => e.to === focus && e.from !== focus)
    const out = graph.edges.filter((e) => e.from === focus && e.to !== focus)
    return { into, out }
  }, [focus, graph.edges])

  /** Is this edge attached to the focused compartment? */
  const lit = (e: GraphEdge) =>
    focus == null || e.from === focus || e.to === focus

  // Every reaction that carries a rate, for the table below.
  const reactions = useMemo(
    () => graph.edges.filter((e) => e.rate),
    [graph.edges],
  )

  // Glossary for the symbols the arrows are labelled with: a model parameter
  // whose KaTeX symbol (or bare name) occurs in some rate string. Substring
  // matching against the rendered TeX is approximate, so when it matches
  // nothing we fall back to the full parameter list rather than showing an
  // empty key under a heading that promises one.
  const glossary = useMemo(() => {
    const params = render?.parameters ?? []
    if (params.length === 0) return []
    const rates = graph.edges.map((e) => e.rate).join(' ')
    const used = params.filter(
      (p) =>
        (p.symbol && rates.includes(p.symbol)) || (p.name && rates.includes(p.name)),
    )
    return used.length > 0 ? used : params
  }, [render?.parameters, graph.edges])

  // Tight bounding box of the node cluster — the plate enclosure wraps this, not
  // the (headroom-inflated) full canvas, so it doesn't balloon up into the arcs.
  const cells = [...L.placed.values()]
  const bx0 = Math.min(...cells.map((n) => n.x))
  const by0 = Math.min(...cells.map((n) => n.y))
  const bx1 = Math.max(...cells.map((n) => n.x + NODE_W))
  const by1 = Math.max(...cells.map((n) => n.y + NODE_H))

  const plateLabel =
    graph.plates.length > 0
      ? graph.plates.map((p) => `${p.name} (${p.levels.length})`).join(' · ')
      : null

  // Caption: plate-family (iterator) edges and the mean-field pools read.
  const pools = useMemo(
    () => [...new Set(graph.couplings.map((c) => c.aggregate))],
    [graph.couplings],
  )
  const readsPool = new Set(graph.edges.filter((e) => e.reads_pool).map((e) => e.id))

  // Every KaTeX label (node names + edge rates), as HTML-overlay specs sharing
  // the edges' geometry — see TexLabel for why these aren't foreignObjects.
  const labels: LabelSpec[] = []
  for (const e of L.forward) {
    const from = L.placed.get(e.from!)
    const to = L.placed.get(e.to!)
    if (!from || !to || !e.rate) continue
    const g = forwardGeom(from, to)
    // Long rate expressions ran across their neighbours; those arrows carry the
    // reaction name instead and the full rate is in the table below.
    const wide = estimateTexWidth(e.rate, 11) > RATE_LABEL_MAX_W
    labels.push({
      key: `rate-${e.id}`,
      x: g.mx - COL_GAP,
      y: g.my - NODE_H - 2,
      w: COL_GAP * 2,
      h: NODE_H,
      tex: wide ? e.id : e.rate,
      plain: wide,
      size: wide ? 10 : 11,
      color: wide ? 'text-neutral-500' : 'text-neutral-600',
      chip: true,
    })
  }
  for (const e of L.back) {
    const from = L.placed.get(e.from!)
    const to = L.placed.get(e.to!)
    if (!from || !to || !e.rate) continue
    const g = backGeom(from, to, L.backLiftById.get(e.id) ?? 0)
    const wide = estimateTexWidth(e.rate, 11) > RATE_LABEL_MAX_W
    labels.push({
      key: `rate-${e.id}`,
      x: (g.x1 + g.x2) / 2 - COL_GAP,
      y: g.top - NODE_H / 2,
      w: COL_GAP * 2,
      h: NODE_H,
      tex: wide ? e.id : e.rate,
      plain: wide,
      size: wide ? 10 : 11,
      color: 'text-violet-500',
    })
  }
  for (const n of cells) {
    labels.push({
      key: `node-${n.id}`,
      x: n.x,
      y: n.y,
      w: NODE_W,
      h: NODE_H,
      tex: n.label,
      size: 13,
      color: 'text-neutral-900',
    })
  }

  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-baseline gap-x-3 border-b border-neutral-200 px-3 py-2">
        <span className="text-sm font-semibold text-neutral-900">{graph.model}</span>
        <span className="font-mono text-[11px] text-neutral-400">flow diagram</span>
        {plateLabel && (
          <span className="ml-auto rounded-sm bg-neutral-100 px-1.5 py-0.5 font-mono text-[10px] text-neutral-500">
            plates: {plateLabel}
          </span>
        )}
      </div>

      {/* The SVG is drawn 1:1 and the KaTeX labels overlay it as absolutely-
          positioned HTML at the same coordinates (see TexLabel). */}
      <ScrollCanvas width={L.width} height={L.height}>
        <svg
          width={L.width}
          height={L.height}
          viewBox={`0 0 ${L.width} ${L.height}`}
          className="block"
        >
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 9 5 L 0 9 z" fill="#6b7280" />
            </marker>
            <marker
              id="arrow-back"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 9 5 L 0 9 z" fill="#a78bca" />
            </marker>
          </defs>

          {/* Plate enclosure behind the compartment cluster. */}
          {plateLabel && (
            <rect
              x={bx0 - 14}
              y={by0 - 14}
              width={bx1 - bx0 + 28}
              height={by1 - by0 + 28}
              rx={10}
              fill="none"
              stroke="#e5e7eb"
              strokeDasharray="4 4"
            />
          )}

          {/* Edges under nodes. */}
          {L.forward.map((e) => {
            const from = L.placed.get(e.from!)
            const to = L.placed.get(e.to!)
            if (!from || !to) return null
            const g = forwardGeom(from, to)
            return (
              <line
                key={e.id}
                x1={g.x1}
                y1={g.y1}
                x2={g.x2 - 7}
                y2={g.y2}
                stroke={lit(e) ? '#9ca3af' : '#e5e7eb'}
                strokeWidth={lit(e) && focus ? 2 : 1.25}
                markerEnd="url(#arrow)"
              />
            )
          })}
          {L.back.map((e) => {
            const from = L.placed.get(e.from!)
            const to = L.placed.get(e.to!)
            if (!from || !to) return null
            const g = backGeom(from, to, L.backLiftById.get(e.id) ?? 0)
            return (
              <path
                key={e.id}
                d={g.d}
                fill="none"
                stroke={lit(e) ? '#c7b9e0' : '#eee9f5'}
                strokeWidth={lit(e) && focus ? 2 : 1.25}
                markerEnd="url(#arrow-back)"
              />
            )
          })}

          {/* Birth chips (exogenous inflow) → target. */}
          {L.births.map((e) => {
            const to = L.placed.get(e.to!)
            if (!to) return null
            const cx = to.x - CHIP_W - 18
            const cy = to.y + NODE_H / 2
            return (
              <g key={e.id}>
                <rect
                  x={cx}
                  y={cy - 11}
                  width={CHIP_W}
                  height={22}
                  rx={11}
                  fill="#f8fafc"
                  stroke="#d1d5db"
                />
                <text
                  x={cx + CHIP_W / 2}
                  y={cy + 3.5}
                  textAnchor="middle"
                  className="fill-neutral-500 text-[9px]"
                >
                  birth
                </text>
                <line
                  x1={cx + CHIP_W}
                  y1={cy}
                  x2={to.x - 7}
                  y2={cy}
                  stroke="#9ca3af"
                  strokeWidth={1.25}
                  markerEnd="url(#arrow)"
                />
              </g>
            )
          })}

          {/* Sink chips for edges with a specific source and no target. */}
          {L.sinks.map((e) => {
            const from = L.placed.get(e.from!)
            if (!from) return null
            const cx = from.x + NODE_W + 18
            const cy = from.y + NODE_H / 2
            return (
              <g key={e.id}>
                <line
                  x1={from.x + NODE_W}
                  y1={cy}
                  x2={cx - 7}
                  y2={cy}
                  stroke="#9ca3af"
                  strokeWidth={1.25}
                  markerEnd="url(#arrow)"
                />
                <rect
                  x={cx}
                  y={cy - 11}
                  width={CHIP_W}
                  height={22}
                  rx={11}
                  fill="#f8fafc"
                  stroke="#d1d5db"
                />
                <text
                  x={cx + CHIP_W / 2}
                  y={cy + 3.5}
                  textAnchor="middle"
                  className="fill-neutral-500 text-[9px]"
                >
                  exit
                </text>
              </g>
            )
          })}

          {/* Nodes on top — clickable: focusing one lights its flows. */}
          {cells.map((n) => {
            const on = focus === n.id
            const dim = focus != null && !on
            return (
              <rect
                key={n.id}
                x={n.x}
                y={n.y}
                width={NODE_W}
                height={NODE_H}
                rx={7}
                fill={on ? '#eff6ff' : '#ffffff'}
                stroke={dim ? '#c7d2e4' : '#1e3a8a'}
                strokeWidth={on ? 2.25 : 1.25}
                className="cursor-pointer"
                onClick={() => setFocus(on ? null : n.id)}
              >
                <title>{n.id}</title>
              </rect>
            )
          })}
        </svg>

        {/* KaTeX labels over the SVG (node names last so they sit on top). */}
        {labels.map((l) => (
          <TexLabel key={l.key} l={l} />
        ))}
      </ScrollCanvas>

      {/* Compartment key — click to follow one compartment's flows. Upstream
          gives compartments a label but no prose, so rather than invent
          definitions we name the reactions into and out of the focused one,
          in the model's own vocabulary. */}
      <div className="border-t border-neutral-100 px-3 py-2">
        <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-neutral-400">
          compartments
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {cells.map((n) => {
            const on = focus === n.id
            return (
              <button
                key={n.id}
                type="button"
                onClick={() => setFocus(on ? null : n.id)}
                aria-pressed={on}
                className={cn(
                  'rounded-sm border px-1.5 py-0.5 font-mono text-[11px] transition-colors',
                  on
                    ? 'border-blue-300 bg-blue-50 text-blue-900'
                    : 'border-neutral-200 bg-white text-neutral-600 hover:border-neutral-300 hover:text-neutral-900',
                )}
              >
                {n.id}
              </button>
            )
          })}
          {focus && (
            <button
              type="button"
              onClick={() => setFocus(null)}
              className="ml-1 font-mono text-[10px] text-neutral-400 underline-offset-2 hover:text-neutral-600 hover:underline"
            >
              clear
            </button>
          )}
        </div>
        {flows && (
          <div className="mt-1.5 space-y-0.5 text-[11px] leading-snug text-neutral-600">
            {flows.into.length === 0 && flows.out.length === 0 && (
              <div>
                <span className="font-mono text-neutral-900">{focus}</span> — no
                reactions touch this compartment.
              </div>
            )}
            {flows.into.length > 0 && (
              <div>
                <span className="font-mono text-neutral-900">{focus}</span>{' '}
                <span className="text-neutral-400">in ←</span>{' '}
                {flows.into
                  .map((e) => `${e.id}${e.from ? ` (from ${e.from})` : ' (exogenous)'}`)
                  .join(', ')}
              </div>
            )}
            {flows.out.length > 0 && (
              <div>
                <span className="font-mono text-neutral-900">{focus}</span>{' '}
                <span className="text-neutral-400">out →</span>{' '}
                {flows.out
                  .map((e) => `${e.id}${e.to ? ` (to ${e.to})` : ' (exit)'}`)
                  .join(', ')}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Reactions in full. The arrows carry a rate expression only when it is
          narrow enough to sit legibly on them; every rate is here regardless,
          with a row to itself, so nothing is lost to the layout. */}
      {reactions.length > 0 && (
        <div className="border-t border-neutral-100 px-3 py-2">
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-neutral-400">
            reactions
          </div>
          <div className="scroll-x-visible overflow-x-auto">
            <table className="text-[11px]">
              <tbody>
                {reactions.map((e) => {
                  const on =
                    focus != null && (e.from === focus || e.to === focus)
                  return (
                    <tr
                      key={e.id}
                      className={cn(
                        'align-baseline',
                        focus != null && !on && 'opacity-40',
                      )}
                    >
                      <td className="py-0.5 pr-3 font-mono text-[10px] text-neutral-400">
                        {e.id}
                      </td>
                      <td className="whitespace-nowrap py-0.5 pr-4 font-mono text-neutral-700">
                        {e.from ?? '•'}
                        <span className="mx-1 text-neutral-400">→</span>
                        {e.to ?? '•'}
                      </td>
                      <td className="py-0.5 text-neutral-800">
                        <span
                          // Trusted, model-derived KaTeX.
                          dangerouslySetInnerHTML={{ __html: katexHtml(e.rate) }}
                        />
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Rate symbol glossary, from the equations artifact when present: the
          diagram's arrows are labelled with symbols, and this is where the
          reader learns what each one means without leaving the view. */}
      {glossary.length > 0 && (
        <div className="border-t border-neutral-100 px-3 py-2">
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-neutral-400">
            rate parameters
          </div>
          <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-3 gap-y-1">
            {glossary.map((p) => (
              <Fragment key={p.name}>
                <dt className="text-right text-[13px] text-neutral-900">
                  <span
                    // KaTeX output for a trusted, model-derived symbol.
                    dangerouslySetInnerHTML={{ __html: katexHtml(p.symbol) }}
                  />
                </dt>
                <dd className="min-w-0 text-[11px] text-neutral-600">
                  <span className="font-mono text-[10px] text-neutral-400">
                    {p.name}
                  </span>
                  {p.description && (
                    <Description
                      text={p.description}
                      className="text-[11px] text-neutral-600"
                    />
                  )}
                </dd>
              </Fragment>
            ))}
          </dl>
        </div>
      )}

      {(L.iterator.length > 0 || pools.length > 0) && (
        <div className="space-y-1 border-t border-neutral-100 px-3 py-2 text-[11px] text-neutral-500">
          {L.iterator.map((e) => (
            <div key={e.id}>
              <span className="font-mono text-[10px] text-neutral-400">{e.id}</span>{' '}
              applies to every compartment
              {e.advances ? `, stepping the ${e.advances} plate` : ''}.
            </div>
          ))}
          {pools.length > 0 && (
            <div>
              mean-field pools:{' '}
              <span className="font-mono text-[10px] text-neutral-600">
                {pools.join(', ')}
              </span>{' '}
              — read by {[...readsPool].join(', ') || 'coupled edges'}.
            </div>
          )}
        </div>
      )}
    </Card>
  )
}
