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

With [uv](https://docs.astral.sh/uv/):

```sh
# Install the `camdl-watch` command on your PATH:
uv tool install git+https://github.com/vsbuffalo/camdl-scope

# Or run it once without installing:
uvx --from git+https://github.com/vsbuffalo/camdl-scope camdl-watch --port 8800

# Or add it to a project's dev dependencies:
uv add --dev git+https://github.com/vsbuffalo/camdl-scope
```

From a local checkout, swap the git URL for the path (add `--editable` to track edits).

## Usage

Run from your camdl project root and open the printed URL:

```sh
camdl-watch --port 8800
```

Flags:

- `--port` / `-p` — TCP port (default `8800`).
- `--host` — interface to bind; use `--host 0.0.0.0` to view from a phone over
  the LAN / Tailscale (default `127.0.0.1`).
- `--store` / `-s` — the fit store to watch (the directory of run dirs).
  Defaults to `results/fits` under the current directory.

```sh
camdl-watch --port 8800 --host 0.0.0.0 --store /path/to/results/fits
```

`--store` just sets the `CAMDL_WATCH_STORE` env var, so that works too;
`python -m camdl_watch ...` is equivalent to the `camdl-watch` command.

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
- **Diagnostics** — camdl's authoritative end-of-stage verdict, or a synthesized
  live one while sampling; per-chain mixing and a per-parameter R̂ / ESS table.
- **Source** — the syntax-highlighted `.camdl` model and `fit.toml`.

**Compare models** — select fits that carry a `prequential.json` and compare
their out-of-sample predictive accuracy via the authoritative `camdl compare`:
elpd, Δelpd ± paired SE, evidence (decibans), CRPS, PIT — as a table and a Δelpd
error-bar plot.

While a fit is still sampling, the open run shows a live progress blurb and its
tabs refresh.

## Develop

```sh
# Python tests
uv run --extra dev pytest -q

# Frontend (the SPA lives under web/)
make types          # regenerate web/src/api/types.ts from the FastAPI schema
cd web && npm install && npm run build
```
