# camdl 'scope

A browser-based results viewer (and live monitor) for
[camdl](https://github.com/vsbuffalo/camdl) fits. Point it at a fit store and it
serves a fast local web app: per-fit posterior summaries, a pair/corner plot, a
posterior-predictive check, generated quantities, traces, a convergence verdict,
and the syntax-highlighted model + fit config — plus a workspace for comparing
models by prequential score. Concurrent runs are auto-discovered, and a run
still sampling updates live.

Read-only on the store — it never touches your fits.

## Install

Installing from source builds the browser UI, so [Node.js](https://nodejs.org)
(any recent version, for `npm`) must be on your PATH alongside
[uv](https://docs.astral.sh/uv/). Without it the install fails with that
message rather than producing a server with no UI.

```sh
# Install the `camdl-scope` command on your PATH:
uv tool install git+https://github.com/vsbuffalo/camdl-scope

# Or run it once without installing:
uvx --from git+https://github.com/vsbuffalo/camdl-scope camdl-scope --port 8800

# Or add it to a project's dev dependencies:
uv add --dev git+https://github.com/vsbuffalo/camdl-scope
```

From a local checkout, swap the git URL for the path (add `--editable` to track edits).

## Usage

Run from your camdl project root and open the printed URL:

```sh
camdl-scope --port 8800
```

Flags:

- `--port` / `-p` — TCP port (default `8800`).
- `--host` — interface to bind; use `--host 0.0.0.0` to view from a phone over
  the LAN / Tailscale (default `127.0.0.1`).
- `--store` / `-s` — the fit store to watch (the directory of run dirs).
  Defaults to `results/fits` under the current directory.

```sh
camdl-scope --port 8800 --host 0.0.0.0 --store /path/to/results/fits
```

`--store` just sets the `CAMDL_SCOPE_STORE` env var, so that works too;
`python -m camdl_scope ...` is equivalent to the `camdl-scope` command.

### Linking to one fit

`?run=<token>` opens that fit directly, so a fit can be handed over in a
message, a ticket, or an agent's output instead of described:

```
http://thuja:8800/?run=fit_national_base-abde5fe0
http://thuja:8800/?run=abde5fe0&tab=diagnostics
```

`<token>` is whatever you have to hand, since whoever writes the link rarely has
the run id itself:

| token | example |
| --- | --- |
| the run id | `fit_national_base-abde5fe0` |
| the fit hash, short or full 64-hex | `abde5fe0`, `abde5fe0ecd7…33c1` |
| a leaf path, straight from camdl's stdout | `results/fits/fit_national_base-abde5fe0/pgas-34f379b8/seed_1-06cbd6b3` |
| a config stem, when the store holds one fit of it | `fit_national_base` |

`&tab=` opens an inner tab: `posterior`, `prior`, `pair`, `traces`,
`diagnostics`, `predictive`, `quantities`, `source` — or `estimate` /
`restarts` for a point-estimate fit.

A token matching several fits opens **none** of them and says how many it
matched, rather than guessing: silently opening a fit other than the one the
link named is not something the reader could detect. Add the hash to
disambiguate.

The address bar tracks whatever is on screen, so the current URL is always the
link to hand over; the chain-link button beside the run id copies it.

## What it shows

**Explore a fit** — one fit, across tabs:

- **Posterior** — a doc-labelled forest of marginal posteriors (symbol,
  description, citation from the model's `#'` docs), median [90%], R̂ / ESS.
- **Pair** — a corner plot: per-chain scatter, diagonal marginals with a smooth
  prior overlay, and a *fit-to-posterior | show-prior-breadth* axis toggle.
- **Predictive** — posterior-predictive ribbons vs the observed series; overlays
  forecast horizons and, for a scenario-aware predict, every scenario.
- **Quantities** — generated quantities (`camdl fit predict`'s sidecar): scalar
  reductions in a censoring-aware table, series as banded ribbons, both faceted
  by scenario.
- **Traces** — per-parameter and log-posterior traces, all chains overlaid, with
  an axis-trimming warm-up control.
- **Diagnostics** — camdl's verdict on the run, read from what camdl persisted
  (`events.jsonl`, `diagnostics.json`) and never recomputed: the dashboard has
  no convergence threshold of its own. A run camdl has not assessed (still
  sampling, or written before camdl persisted its verdict) says so. Every
  method leaf of the folder is listed with its own status and verdict, failed
  and running ones included. Per-chain mixing and a per-parameter R̂ / ESS
  table show the numbers; a number is coloured only where camdl flagged that
  parameter, and a non-finite R̂ reads `inf` or `NaN`.
- **Source** — the syntax-highlighted `.camdl` model and `fit.toml`.

**Compare models** — select fits that carry a `prequential.json` and compare
their out-of-sample predictive accuracy via the authoritative `camdl compare`:
elpd, Δelpd ± paired SE, evidence (decibans), CRPS, PIT — as a table and a Δelpd
error-bar plot.

While a fit is still sampling, the open run shows a live progress blurb and its
tabs refresh.

## What to run so a fit is fully viewable

The 'scope is read-only: it shows the artifacts a run directory contains and
nothing else. A tab is missing because its artifact is missing, not because the
viewer is hiding it. Everything here is a camdl command run from your project
root — the 'scope never generates anything.

| Run this | Artifacts it writes | Tabs it lights up |
|---|---|---|
| `camdl fit run <fit.toml>` | `chain_*/trace.tsv`, `diagnostics.json`, `events.jsonl`, `<stage>_summary.json`, `model.render.json`, `model.graph.json` | Posterior, Pair, Traces, Diagnostics, Model |
| `camdl fit predict <run>` | `predict/<run_id12>/` holding `predictive/`, `observed/`, `predictive.json`, `quantities/` (`predict/<run_id12>-<opts8>/` under non-default options) | Predictive, Quantities |
| `camdl simulate <model> --draws prior --fit <fit.toml> --obs-dir <run_dir>/prior_predictive` | `prior_predictive/<stream>.tsv` | Prior (the predictive-check plot) |

The Prior tab's prior→posterior contraction table needs no extra command — it
reads the priors from the fit config and the draws from the trace. Only the
plot needs the third row.

Two traps worth knowing, both of which produce a viewer that looks wrong when
the artifact is:

- **A predict with other options is a separate predict.** `camdl fit predict
  --stream X` (or `--horizon`, `--n-draws`, `--scenario`, …) writes
  `predict/<run_id12>-<opts8>/` beside the default one rather than over it. The
  viewer shows the default-options predict and offers the others in a picker
  labelled by their options. Predict output an older camdl wrote directly in
  the run directory (`predictive/`, `observed/`, `quantities/`) is not read;
  rerun `camdl fit predict`.
- **Naming scenarios drops the fitted arm.** `camdl fit predict --scenario A
  --scenario B` emits A and B and *not* the as-fitted posterior predictive, so
  the scenarios have nothing to be compared against
  ([camdl#625](https://github.com/vsbuffalo/camdl/issues/625)). Running both
  horizons restores a `fitted` block via the one-step section — unless one-step
  aborts, which is how a sidecar ends up scenario-only.

If a tab is empty, the fastest diagnosis is `ls` on the run directory (and its
`predict/`): the table above says which name should be there.

## Develop

```sh
# Python tests
uv run --extra dev pytest -q

# Frontend (the SPA lives under web/)
make types          # regenerate web/src/api/types.ts from the FastAPI schema
cd web && npm install && npm run build
```
