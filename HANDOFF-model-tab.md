# Handoff — "Model" tab: compartmental flow diagram

Status: ready to build. Scope for now is the **flow diagram** only. (An
identifiability panel is a possible future addition — see the note at the end —
but is not part of this handoff.)

This is a viewer-only change: one new **compiler-emitted sidecar artifact**
(`model.graph.json`) gets surfaced as a new top-level **Model** tab holding a
flow diagram. It's purely additive — no change to `ingest.py`/`state.py`
architecture, the store layout, or existing artifacts. All paths below are
verified against the current codebase.

## The artifact (dropped in each run dir, `meta.run_dir` root)

`model.graph.json` — a per-run, identity-neutral sidecar with model-pure content
(byte-identical across runs of the same model), a sibling of `model.render.json`.
**The emitter is implemented — build against the real samples in
`model-graph-samples/` (this repo), not a sketch.** `ctl_bb.graph.json` (a garki
base model) is the best reference; the others are `sir_basic` (minimal),
`ajura_compound_re` (4 plates), `seir_age`. Realized shape:

```jsonc
{
  "model": "ctl_bb",
  "nodes":  [{ "id": "S_naive", "label": "S_{naive}" }, ...],   // base compartments; label is a KaTeX string
  "plates": [{ "name": "village", "levels": ["kwaru","ajura"] },
             { "name": "age", "levels": ["a1_4","a5_9",...] }],
  "edges":  [
    { "id": "inoc_naive", "from": "S_naive", "to": "E_naive",
      "rate": "h_{v}\\,S_{naive,v,a}", "advances": null, "reads_pool": true },   // rate is a KaTeX string
    { "id": "aging", "from": "c", "to": "c",                                     // "c" = compartment iterator
      "rate": "...", "advances": "age", "reads_pool": false },                   //   → "every node", stepping the age plate
    { "id": "birth", "from": null, "to": "S_naive", ... },   // from:null = exogenous inflow
    { "id": "death", "from": "c",  "to": null, ... } ],       // to:null   = outflow
  "couplings": [
    { "edge": "inoc_naive", "aggregate": "inf_vil", "over": ["age"] },   // the FOI reads the inf_vil pool
    { "edge": "inoc_naive", "aggregate": "Nvil",    "over": ["age"] } ]  //   and the Nvil pool
}
```

Three conventions the emitter settled (flag if you'd prefer otherwise):
- **`rate` and `label` are KaTeX strings** (e.g. `h_{v}\,S_{naive,v,a}`) — render
  with `katex.renderToString`, same as the existing `ModelView`.
- **Plate-family edges use `"c"` (the compartment iterator) as `from`/`to`** —
  `aging`/`death` apply to *every* compartment; treat `from/to == "c"` as "all
  nodes" and use `advances` for the stepped plate. Draw `aging` as a plate-step
  glyph on the plate box rather than an edge per node.
- **`couplings.aggregate` carries the pool's *name*** (`inf_vil`, `Nvil`), not a
  generic `"sum"` — more meaningful for a label ("reads the inf_vil pool over
  age"). Nested sums flatten into `over`.

## Backend wiring (mirror the `model.render.json` pipeline exactly)

1. **Reader** — copy `camdl_watch/model_render.py` (~40 lines, pure JSON
   pass-through, reads from `run_dir` root): `camdl_watch/model_graph.py` with
   `read_model_graph(run_dir)`, `has_model_graph(run_dir)`. (Note:
   `model_render.py`'s docstring says "segment-level" but the code reads `run_dir`
   root — the artifact lands there.)
2. **Pydantic** (`camdl_watch/api/models.py`): add `ModelGraph { model, nodes,
   edges, plates, couplings }`. Add `has_model_graph: bool` to `RunDetail`
   (declare at models.py:254, set at routes.py:365 — the two places
   `has_model_render` uses).
3. **Route** (`camdl_watch/api/routes.py`, copy the `get_model_render` block at
   ~:797): `GET /api/runs/{run_id}/model-graph` (`_store()` → `_find_meta` → read →
   404-when-absent → `model_validate`).
4. `make types` — regenerates `web/openapi.json` + `web/src/api/types.ts`.

## Frontend wiring

5. **Client + hook**: add `getModelGraph` (`web/src/api/client.ts`, alongside
   `getModelRender` at :151) and `useModelGraph` with a `qk` key
   (`web/src/api/queries.ts`, mirror `useModelRender` at :163 / `qk.modelRender`
   at :31).
6. **Model tab**: in `web/src/components/ExploreWorkspace.tsx`, add
   `{ value: 'model', label: 'Model' }` to the `tabs` arrays in **both**
   `PosteriorTabs` (~:80) and `MleTabs` (~:33), gated on
   `useRun(...).data?.has_model_graph` (mirror the `hasQuantities` conditional
   spread at :40/:127 + the conditional `<TabsContent>` at :67/:162). Its
   `ModelTab` renders the diagram in a `Card`; optionally promote the existing
   `ModelView` (from `SourceTab.tsx`) into it too.

## The component — `FlowDiagram.tsx`

A **hand-rolled SVG** node-link diagram from `model.graph.json`. There's no
graph-layout lib in the app and Observable Plot can't do node-link, so this is
new — but the *base* graph is tiny (4–8 nodes, nearly a chain), so a simple
**layered left-to-right layout** suffices (columns by topological depth; back-edges
like `recover: I_rec→S` as a curved return arc). Details:
- nodes = rounded rects; labels via KaTeX (`Tex` is currently local to
  `ModelView.tsx:13` — export it, or `import katex; katex.renderToString(...)`);
- edges = arrows labeled with `rate`; `from:null`/`to:null` render as small
  source/sink chips ("births" / "deaths");
- `plates` = a light nested rectangle enclosing the compartment cluster, labeled
  `age · imm · compound` — do **not** draw one node per stratum, collapse to the
  base + a plate annotation (a stratified model can be 1000s of cells);
- `couplings` / `reads_pool` = a dashed edge from a "shared pool" chip into the
  coupled edge — this is the mean-field force of infection, worth surfacing;
- responsive sizing: follow the `ResizeObserver`-seeded self-measuring idiom in
  `DiagnosticsTab.tsx`'s `MixingChart` / `ProfileSurface.tsx`.

## Design constraints

- **Light theme only.** The app has no dark mode anywhere (`index.html` pins
  `color-scheme: light`; no `dark:` utilities). Match the existing neutral +
  `blue-900`-accent, small-type, `Card`-based idiom. (An early mockup was dark —
  ignore that styling.)
- **Python computes, TS renders** — the frontend derives nothing; it lays out the
  artifact as given.

## Size

Backend reader/route/model/field + frontend client/hook/tab are near-mechanical
copies (**small**). The `FlowDiagram` SVG is the one **moderate**,
genuinely-new piece.

## Note — a later identifiability panel

A second `Card` in this tab may later show an **identifiability** view, but it is
not built yet and its data source is still being decided on the camdl side. It
will be *practical* identifiability (profile likelihood from `camdl profile`,
and/or a structural summary exported to StructuralIdentifiability.jl) — **not** a
bespoke scaling lint (that idea was dropped as unsound). Build the flow diagram
now; the identifiability panel is a separate, later slot.
