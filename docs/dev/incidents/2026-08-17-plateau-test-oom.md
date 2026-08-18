# Plateau test allocated O(n²) memory per request — took the host into swap

Date: 2026-08-17
Project: camdl-watcher
Status: Fixed
Fix: `camdl_watch/diagnostics.py` (`_PLATEAU_MAX_POINTS` thinning in `_plateau_test`)

## What happened

Two watcher instances serving the ebola-bdbv store (82 MB, four PGAS fits of a
small national model) were killed after ~3 minutes: one at 4.0 GB RSS and
growing, the other at 0.7 GB, with system swap at 50.8/51.2 GB and the machine
thrashing. The store's largest stage directory is 34 MB.

## Root cause

`diagnostics._plateau_test` pools every chain's finite log-likelihood values
and hands the trailing window to `scipy.stats.theilslopes`. Theil–Sen
materializes **all pairwise differences** — several n×n float64 intermediates —
so memory is quadratic in the pooled point count.

The bdbv fits are the first in any watched store with 6,000-sweep PGAS traces
across 4 chains: ~24,000 pooled ll points, ~12,000 after windowing, and
12,000² × 8 B ≈ 1.15 GB *per intermediate*. Measured: **7.35 GB peak for one
call** (205 MB baseline). Every request to `/posterior` or `/diagnostics` runs
it, so a browser tab sitting on either tab re-triggered multi-GB transients;
two instances of that put the host into swap. Earlier stores (garki, old
ebola) had shorter traces, keeping n below the quadratic cliff — the code was
long wrong, but never fed a large n.

Two details from the report that measurement corrected:

- **Idle is flat.** 40 s with zero requests: 243 MB, no growth. The
  "~12 MB/s steady leak against an idle store" was request-driven transients
  (plus allocator ratchet) seen through coarse sampling — the store was idle,
  the browser wasn't.
- **The day's commits were innocent.** The suspect surface (Posterior-tab
  scalar quantities, FlowDiagram) is frontend-only; `_plateau_test` predates
  them. The trigger was the new store's trace length, not new code.

## Diagnosis method

Endpoint bisection in fresh subprocesses (`ru_maxrss` per single request):
`/runs`, detail, `/draws`, `/traces` all ≈ 230 MB; `/posterior` 7.4 GB. Stage
bisection inside the handler: `build_run_state` 224 MB, `+compute_diagnostics`
7.4 GB. Per-param arviz calls (R̂, bulk/tail-ESS, MCSE on (4, 3001) arrays):
flat at 226 MB — leaving `_plateau_test`, confirmed directly (205 → 7,351 MB).

## Fix

Thin the pooled series to `_PLATEAU_MAX_POINTS = 2000` with an even stride
before Theil–Sen. The test asks only "is the trailing slope ≈ 0 relative to
the ll scale" — a 2,000-point subsample answers that identically, and caps the
pairwise intermediates at ~32 MB regardless of trace length.

After: `/posterior` and `/diagnostics` peak ≈ 425 MB on the same run; 120 s of
continuous polling of every endpoint holds flat at ~450–475 MB.

Regression test: `test_plateau_test_memory_is_bounded_on_long_traces`
(tracemalloc peak < 200 MB on a synthetic 4 × 30,000-draw run — uncapped, that
input would allocate ~29 GB).

## Lesson

Any per-request statistic must have cost independent of trace length — traces
grow unboundedly during a fit, and the watcher polls. Theil–Sen was chosen for
robustness with no attention to its O(n²) memory; the store that finally
crossed the cliff was 40× smaller than the RSS it induced. Audit candidates:
anything else calling into scipy/numpy with pooled full-trace inputs
(`theilslopes` was the only pairwise-materializing call found).
