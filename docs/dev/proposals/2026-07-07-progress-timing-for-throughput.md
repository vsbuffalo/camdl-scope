# Progress timing for throughput & efficiency benchmarks

Date: 2026-07-07
Project: camdl-watcher
Status: proposed (upstream ask to camdl)
Tags: monitoring, progress, benchmarks, upstream

## Problem

The watcher's live-monitoring view can show *where* a fit is (phase, `step`
of `total`) but not *how fast* it's going. Three benchmarks would sharpen the
picture:

1. **Throughput** — iterations/second (or seconds/iteration for slow samplers).
2. **ETA** — time to `total` at the current rate.
3. **Cost efficiency** — effective samples per second, `ESS / elapsed`.

All three need one ingredient the store does not currently persist: **wall-clock
elapsed time**. `progress.json` records only `updated_at` (unix seconds of the
last heartbeat write) plus `RunState::Running { phase, step, total }`; the
trace rows carry no per-row timestamp, and `pgas_summary.json` carries no
timing. There is no start time on disk.

## What is already possible without any upstream change

- **ESS/iteration** (dimensionless sampling efficiency, `ess / (n_tail ·
  n_chains)`) needs no timing and is **shipped** — a derived column in the
  Diagnostics table. Hardware-independent; the honest per-parameter efficiency.
- **Live it/s + ETA** can be had by *differentiating the heartbeat*: the run
  summary already carries `step` and `updated_at`, refreshed every 5 s while a
  run is live, so the frontend can compute `Δstep / Δupdated_at` (smoothed) and
  `ETA = (total − step) / rate`. This is live-only and noisy over short
  windows, and it vanishes the moment a run finishes (no more heartbeats).

The gap is the **average** throughput and **ESS/sec**, which need a fixed
reference time and must survive into finished runs.

## The ask: one field

Add `started_at` (unix epoch seconds) to `progress.json`, set once at run start
and preserved across heartbeat rewrites.

```rust
// rust/crates/io/src/progress.rs
pub struct Progress {
    pub updated_at: u64,
    pub started_at: u64,   // NEW: unix secs, stamped at run start, sticky
    pub pid: u32,
    pub state: RunState,
}
```

This is symmetric with what the CLI already computes: the fit table derives an
`age_seconds` per run (`rust/crates/cli/tests/fit_experiment_management.rs`
exercises it as "the ONLY wall-clock-derived field in the row"), so run age is
already a known quantity — it simply isn't persisted where a viewer can read
it per-run.

### What it unlocks

With `started_at` on disk, `elapsed = updated_at − started_at` and:

- **Average throughput** `step / elapsed` — stable, unlike the heartbeat
  derivative, and defined for finished runs.
- **ESS/second** `ess / elapsed` — the number that actually compares
  algorithms/backends on a given machine (PGAS vs PMMH, chain-binomial vs ODE).
- **Robust ETA** from the average rate rather than the last-interval slope.

## Alternatives considered

- **Per-write cumulative wall-time field** (e.g. `elapsed_s` written into each
  heartbeat). Equivalent information; `started_at` is simpler (one immutable
  stamp) and lets the consumer compute elapsed against its own clock if a
  heartbeat is stale.
- **Derive start from the earliest file mtime** (run dir or first trace row).
  Works without an upstream change but is fragile: mtimes drift on copy/rsync,
  and a resumed fit's dir predates its current sampling window.
- **Frontend heartbeat-diff only.** Already viable for the live view; does not
  give a stable average or any post-hoc ESS/sec. Complementary, not a
  substitute.

## Watcher-side consumption (once the field lands)

`RunProgress` (`camdl_watch/state.py`) and the `ProgressInfo` wire model gain a
`started_at`; the summary route projects `elapsed` and derives it/s + ESS/sec
next to the existing progress blurb. No new artifact, no new file — one field
threads through the existing progress path.

## Recommendation

Ship ESS/iteration now (done). Add the live heartbeat-diff it/s + ETA as a
frontend-only step. Open the upstream issue for `started_at` to unlock the
average throughput and ESS/sec benchmarks — a one-field change that makes the
run leaf self-describing about its own timing.
