/**
 * Static-demo build flag (set at build time with `VITE_DEMO=1`).
 *
 * In demo mode there is no live backend: the API client reads a pre-baked JSON
 * snapshot (see `scripts/make_demo_snapshot.py`), and controls that would vary a
 * request's query params — the warm-up slider, the chain selector — are hidden,
 * as is the Compare workspace (the snapshot has no prequential data). Everything
 * renders at its default, snapshotted state.
 */
export const DEMO = import.meta.env.VITE_DEMO === '1'
