# Killed / crashed fits classified as `done`

Date: 2026-08-09
Project: camdl-watcher
Status: Fixed
Fix: `camdl_watch/ingest.py` (`stage_completed`, `_pick_posterior_dir`), `camdl_watch/assembly.py` (`classify`)

## What happened

A merged-fit run in the ebola-camdl store (`fit_merged_pmmh-2ab50eab`) was
killed mid-sampling. The watcher listed it as `done`. Opening it and asking for
predictives failed with no useful message, because there was no posterior to
draw from: the stage held only partial per-chain traces
(2,094 / 2,094 / 2,098 rows of a 90,000-step target), no `draws.tsv`, no
`fit_state.toml`, no summary or diagnostics JSON.

## Root cause

Status classification, in the no-heartbeat fallback of `assembly.classify`:

```python
live = ingest.stage_is_live(rs.meta.posterior_dir)
has_draws = any(buf.n for buf in rs.chains.values())
if has_draws:
    return Status.RUNNING if live else Status.DONE   # <- here
```

`has_draws` meant "any chain trace has ≥1 row" — not "the stage produced its
pooled `draws.tsv`". So the only completion evidence consulted was *did any
chain write a line*. A stage that was killed, crashed, or OOM'd is
indistinguishable under this test from one that finished: dead process + trace
rows → `done`.

A camdl posterior stage writes its completion artifacts only at the end, and
they co-occur: `draws.tsv` (the pooled, thinned posterior — what
`camdl fit predict` re-simulates from), `fit_state.toml`, and the
`*_summary.json` / `diagnostics.json`. A killed stage has the trace files and
none of these. Confirmed by surveying both stores: every seed-leaf has all of
`{draws.tsv, fit_state.toml, *_summary.json}` or none.

The heartbeat path was already correct — a stage whose `progress.json` carries a
terminal or stale-`running` state is classified from that, and none of this
applies. The gap was only in the legacy PID fallback, which is what older /
externally-launched runs hit (they write no `progress.json`).

### Second defect: stage selection could mask a completed stage

`_pick_posterior_dir` ranked candidate stages by `(non-empty chain count,
mtime)`. A crashed *relaunch* can leave a stub with as many started chains as
the real run and a newer mtime, so a run with one completed stage plus one
partial stub could surface the stub. Under the old classifier that stub still
read `done` (wrong reason, right answer). Fixing only the classifier would then
have flipped such a run to `stalled` — a regression. In every multi-stage run
observed today the completed stage happened to also be the most recent, so the
tie-break masked this by luck; nothing guaranteed it.

## Fix

1. `ingest.stage_completed(seed_dir)` — true iff the stage wrote a non-empty
   `draws.tsv`. The load-bearing completion marker: no `draws.tsv` means no
   posterior to predict from, whatever the trace files contain.
2. `_pick_posterior_dir` — rank completed stages above partial stubs
   (`(completed, non-empty chains, mtime)`), so a finished stage is always
   surfaced when one exists.
3. `classify` fallback — a dead process with trace rows returns `done` only if
   `stage_completed`, else `stalled`.

`stalled` (existing status; neutral swatch) is reused rather than adding an
`incomplete` state — semantically it is the same family as a stale heartbeat
(terminated without finishing), and it avoids touching the enum, the wire
model, and the TS regen.

## Impact

Runs re-classified `done → stalled` after the fix, on the current stores:

- ebola `data/build/camdl/runs/fits`: **1** (`fit_merged_pmmh-2ab50eab`).
  (`fit_merged_pgas-2ab50eab` was already `stalled` via its stale heartbeat —
  unaffected.)
- garki `results/fits`: **19** (crashed / abandoned fits — several
  `joint_capacity_gam`, `host_cutc`, `ctl_bb_immladder_*`, `ctl_bb_spray_*`).

Sanity-checked: no run reads `stalled` while a completed sibling stage exists —
the selection fix keeps genuinely-finished runs `done`; only all-incomplete runs
flip.

## Follow-up: honest empty-state on the posterior-dependent tabs

Opening a stalled run's Predictive (or Quantities) tab previously showed
"No predictive artifact — run `camdl fit predict`", which is wrong: predict has
no posterior to draw from, so following that advice just fails. The tabs now key
their empty-state on run status (`web/src/components/States.tsx`
`NoPosteriorNotice`, used by `PredictiveTab` / `QuantitiesTab`): a `stalled` /
`failed` run reports "No posterior to predict from — this run stopped before it
finished sampling"; a `warming` / `running` run reports "Still sampling"; a
`done` run keeps the original run-predict guidance.

## Note on the store contract

This is another instance of the standing contract lesson: derive run state from
camdl's *written artifacts* (`draws.tsv`, `progress.json`, `run.json`), not from
weaker proxies (a trace row exists; a `.lock` file is present). A future
sanctioned `camdl watch` API would make this explicit; until then `ingest.py` is
the single module that encodes it.
