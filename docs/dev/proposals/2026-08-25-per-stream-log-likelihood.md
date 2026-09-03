# Per-stream log-likelihood views

Date: 2026-08-25
Project: camdl-watcher
Status: proposed
Tags: diagnostics, predictive, prequential, model-checking

Rendered mockups from real fits:
<https://claude.ai/code/artifact/91a872db-5c59-4707-ab81-ba45a56fb86f>

## Problem

The watcher can say a fit is bad but not *which observation stream* it is bad
at. A model fitted to eight streams reports one `obs_ll`, and the Compare tab
reports one `elpd` per model. Neither localises the misfit, so the question
"where is the tension between this model and the data" has no answer in the UI.

camdl already writes the answer in two places. Both are read by the watcher
today and neither reaches a screen.

## The two sources are not one quantity

|  | `obs_ll_<stream>` (trace) | `prequential.json` `per_stream` |
|---|---|---|
| One value is | `log p(y_stream \| x, θ)` for that stream, summed over all its times, at one sweep's conditioned latent path | the one-step-ahead log score for that stream at one time, at a single plug-in θ |
| Axes | draw × stream | time × stream |
| Written by | PGAS, one column per declared stream (camdl gh#742) | `pfilter --save-prequential` (camdl gh#269) |
| Also carries | nothing — one scalar per draw | CRPS, PIT, and the q05–q95 predictive interval per point |
| Comparable across models | no: complete-data, path-conditioned, gameable | yes, via `camdl compare` |
| In the watcher now | parsed into `ChainBuffer.aux` every refresh, then dropped | only the whole-model `elpd` total reaches Compare |

One has a draw axis and no time axis; the other has a time axis and no draw
axis. Neither has both, so this is two views rather than one view with a
toggle. Presenting them as a single "per-stream log-likelihood" number would
silently mix a complete-data quantity with a predictive score.

The trace columns need no ingest change: camdl declares
`obs_ll_<stream>` with `role: "diagnostic"` in the stage's `run.json`, and
`read_column_roles` already files them into `aux`. The gap is entirely at the
presentation end.

### Availability is disjoint in practice

| sampler | `obs_ll_<stream>` | prequential `per_stream` |
|---|---|---|
| PGAS (post gh#742) | yes, one per declared stream | when a pfilter stage scored it |
| PMMH / MH / NUTS | no — only scalar `log_likelihood` | when a pfilter stage scored it |

On the current stores the two do not overlap. `fit_national_careflow_24c`
(ebola-bdbv-camdl, PGAS) carries eight `obs_ll_` columns but its prequential
scored only `deaths_national`. `ctl_bb_immladder_anchored` (camdl-garki) has
ten streams in its prequential but is MH/ODE, so no per-stream trace breakdown
exists. Shipping one view alone leaves half the fits blank; each should render
when its own artifact is present and be absent otherwise.

## Types

A new module of pure readers, mirroring how `predictive.py` reads the ribbons —
no ingest dependency, no cycle.

```python
# camdl_watch/streams.py

@dataclass(frozen=True)
class StreamFit:
    """One stream's share of a fit's data fit, from obs_ll_<stream>.
    A posterior distribution over draws, not a point."""
    stream: str
    n_obs: int                # non-null rows in observed/<stream>.tsv
    mean: float               # posterior mean of the summed log-density
    sd: float
    q05: float
    q95: float
    between_chain_sd: float   # sd of the per-chain means — chains that
                              # disagree about ONE stream localise a mode


@dataclass(frozen=True)
class StreamScore:
    """One stream's one-step-ahead score at one time, from prequential.json."""
    stream: str
    t: float
    y_obs: float
    log_score: float
    crps: float | None
    pit: float | None
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
```

Two endpoints:

```
GET /runs/{run_id}/stream-fit
  -> {run_id, loglik_type, total, rows: [StreamFit]}
  404 when the trace carries no obs_ll_<stream> columns.

GET /runs/{run_id}/stream-scores
  -> {run_id, provenance, conditioning, joint, streams, cells: [StreamScore]}
  404 when the run has no prequential.json.
```

`n_obs` must come from the stream's non-null count in
`observed/<stream>.tsv`, not its row count, or a stream with holes ranks
falsely well.

`loglik_type` is the field not to drop. It is the seam that stops view 1 being
read as a model-comparison number, camdl already computes it as a property of
the inference method, and re-deriving it here would duplicate an authority that
then goes stale.

## Four views

**1. Streams ranked by log-density per observation** — Diagnostics, beside the
per-chain mixing panel. Ranking by a stream's *total* log-density ranks streams
by how many observations they have; per observation is the comparable quantity
and reorders the list. On `fit_national_careflow_24c`: `cases_national` is
hardest at −4.40 nats/obs and carries 31% of the total penalty,
`care_absconded` easiest at −2.14, while by raw total `deaths_national` (92
points) would have looked comparably bad to `care_absconded` (14 points).

The `between_chain_sd` column earns its place independently: 5.72 nats for
`deaths_national` against 0.30 for `community_deaths_national` says the chains
found different explanations of the deaths series specifically. That localises
a multimodality to one stream, which per-parameter R̂ cannot do.

**2. A tension map over time × stream** — Predictive, under the ribbon, sharing
its time axis. Position gives stream and date, area gives the penalty, hue
gives the PIT direction. Sign is what makes it actionable: a stream the model
consistently under-predicts is a different repair from one whose dispersion is
too tight. On `ctl_bb_immladder_anchored`, pooling the five age bands per
village, Ajura scores −3.17 nats/obs at mean PIT 0.605 and Kwaru −2.49 at 0.444
— one systematic, opposite-signed miss per village.

A dot matrix is the form that survives scale. Ten streams already exceeds what
a categorical palette should carry as line colors, and a district model has
hundreds.

**3. Cumulative log score, one line per stream** — the per-point score against
time is spiky; its running sum is monotone, so slope is the local misfit rate
and a kink is a date. On the Garki fit the rates are flat within stream
(`prevalence_ajura_a25_44` runs −3.88, −3.85, −3.61 across thirds of the
series), which diagnoses a persistent per-stream level offset rather than a
missed event — a read that points at the observation model, not the epidemic
timeline.

**4. PIT calibration per stream** — log score confounds bias with dispersion;
the PIT separates them. Plot the ECDF *difference* from uniform rather than the
ECDF, for dynamic range.

## Three caveats that belong in the UI, not in this doc

Each can produce a confident wrong conclusion, so each should be a line of text
on the surface that shows it.

- **The trace decomposition is complete-data and gameable.** `obs_ll` is
  `log p(y | x, θ)` at the sweep's sampled path, not a marginal likelihood; a
  chain can raise `transition_ll` at the cost of `obs_ll`. Carry
  `loglik_type` onto the view and never let a stream total read as a
  model-comparison quantity.
- **The prequential scores are optimistic in level.** `conditioning:
  in_sample` means θ was fitted to the whole series, so every "one-step-ahead"
  score has already seen the future through θ; `provenance: plug_in` ignores
  parameter uncertainty on top of that. Both are honest for comparing streams
  *within* one fit, which is the whole use here, and neither is honest as an
  absolute forecast score. camdl reserves `Lfo` and `RollingOrigin` in the same
  enum; when they land these views get honest out-of-sample scores with no
  schema or UI change.
- **The PIT is in-sample, which biases it toward the centre.** A clean PIT
  panel is therefore not evidence of calibration. The direction of a gross miss
  survives this; a subtle one does not.

## Prior art

This is the standard pointwise-elpd toolkit with "stream" as the grouping.
*Bayesian Workflow* §8.4 is titled "Assessing the influence of individual data
points **and subsets of the data**" — a stream is such a subset — and its
figure 8.10b plots the pointwise LOO contribution against a key predictor to
find which points drive a difference. View 2 is that move with time as the
predictor and stream as a second axis. §8.2 (fig. 8.6) is the PIT ECDF and the
ECDF-difference plot; §8.3 (fig. 8.9) is the in-sample-PIT bias that caveat 3
records.

The honest gap: their pointwise quantity is cross-validated and both of ours
are in-sample. That is a difference in what the number licenses, not a
presentational one.

### References

- Gelman, A., Vehtari, A., McElreath, R., with Simpson, D., Margossian, C. C.,
  Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., Modrák, M., and Leos
  Barajas, V. (2026). *Bayesian Workflow*, corrected edition. CRC Press.
  §8.2, §8.3, §8.4.
- Vehtari, A., Gelman, A., and Gabry, J. (2017). Practical Bayesian model
  evaluation using leave-one-out cross-validation and WAIC. *Statistics and
  Computing* 27, 1413–1432.
- Gneiting, T., Balabdaoui, F., and Raftery, A. E. (2007). Probabilistic
  forecasts, calibration and sharpness. *JRSS B* 69, 243–268.
- Dawid, A. P. (1984). Statistical theory: the prequential approach (with
  discussion). *JRSS A* 147, 278–292.
- Säilynoja, T., Bürkner, P.-C., and Vehtari, A. (2022). Graphical test for
  discrete uniformity and its applications in goodness-of-fit evaluation and
  multiple sample comparison. *Statistics and Computing* 32, 32. The
  simultaneous ECDF band view 4 should use.
- Tesso, H. and Vehtari, A. (2026). LOO-PIT predictive model checking.
  arXiv:2603.02928. The POT uniformity test, valid under dependence.
- Säilynoja, T., Johnson, A. R., Martin, O. A., and Vehtari, A. (2025).
  Recommendations for visual predictive checks in Bayesian workflow.
  arXiv:2503.01509. Discrete-data checks — most camdl streams are counts.
