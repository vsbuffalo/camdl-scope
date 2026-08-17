# camdl-watch code-smell review

Date: 2026-07-17
Scope: the whole tree — Python core (`camdl_watch/`), API (`camdl_watch/api/`),
React SPA (`web/src/`).

## Reading of the whole

The types are good and the intent — "the middle layer is the program, the UI is
a projection, every number is computed once in Python and shipped authoritative"
— is stated in the module docstrings and mostly honoured. `schema.py` and
`model_render.py` are the models to imitate: tolerant parsing, tight types, no
layout assumptions, explicit `None`/`[]` degradation.

The dominant smell is the same one camdl's own `ARCHITECTURE.md` names as camdl's
unifying problem: **one truth kept in agreement by hand across N copies**. It
shows up here as (a) camdl's on-disk layout and column/schema conventions
re-encoded as bare literals in many readers, (b) two generations of store-reader
living side by side, and (c) small helpers copy-pasted rather than shared. In
several places the copies have already drifted, and a few of those drifts are
live latent bugs that surface as silently-wrong numbers rather than as errors —
the dangerous direction for a scientific viewer.

Below: the real bugs first (drift that already produces wrong output), then the
through-line themes.

---

## Real bugs (drift that yields silently-wrong output)

**B1 — `diagnostics.py:53-54`: the promised finite mask is missing → silent NaN
R̂/ESS.** `_tail_arrays` carries the comment "Keep only finite tail (early -inf
ll etc. would poison arviz)", but the next line appends the raw masked values
with no finite filter. `compute_diagnostics:175` computes `finite =
arr[np.isfinite(arr)]` and uses it only for mean/sd; the arviz calls at 177-182
get the unfiltered `arr`. A single early `-inf`/NaN (routine for
`log_posterior`/`log_likelihood`, which *are* diagnosed) makes `az.rhat`/`az.ess`
throw, and `_az_safe` swallows it to NaN. The objective's convergence silently
reads "not estimable". Either restore the mask (per the comment) or drop the
comment — the mask is the right fix.

**B2 — `routes.py:868-869,931-932` vs `968-970`: NaN/blank quantile → `0.0` on
ribbons, → `null` on the table.** `_fnum` (827) coerces non-finite/blank to
`0.0`; `_band_cell`→`_finite_or_none` (894, 244) coerces to `None`. Predictive
ribbons and series-quantity ribbons use `_fnum`; scalar-quantity cells use
`_band_cell`. So a missing/fully-censored quantile plots as a hard `0` inside a
posterior-predictive interval on the ribbon, but correctly shows as a hole in the
table. The quantities layer already documents that camdl writes blank `q*` for a
fully-censored scalar; a forecast horizon or empty stratum can do the same on the
predictive side. One band-cell policy should cover both.

**B3 — `compare.py:157`: commensurability inverts on schema drift.** With
`allow_mismatched=True`, `commensurable` is `len({row.get("t_score") for row in
rows}) <= 1`. If camdl renames `t_score` or reshapes rows, every `.get` returns
`None`, the set collapses to `{None}`, and the function reports
`commensurable=True` unconditionally — i.e. it asserts the models *are*
comparable exactly when it can no longer verify they are. Compounded by
`compare.py:132-134`, where exit code `2` is hard-coded to mean "horizon
mismatch"; any future exit-2 gets silently reclassified and retried with
`--allow-mismatched-horizon`. `commensurable` is also computed two different ways
on the two code paths (exit-2 refusal vs the `t_score` set), which can disagree.

**B4 — `profiles.py`: the failure sentinel (≈ −1e100) leaks into `mle` and
`points`.** `load_profile` admits any *finite* `best_loglik` (224-225), and the
sentinel is finite. `ci_bounds_1d` filters `p.loglik > SENTINEL_LOGLIK` (85), but
`mle` (74) and the `points` list do not — so a grid cell whose only restart
failed becomes a real `ProfilePoint(loglik≈-1e100)` that the UI plots as a giant
downward spike, and can even win the argmax for `mle`. Filter the sentinel once,
in `load_profile`, not per-method.

**B5 — root-resolution helpers have already drifted.** The "walk from the store
to a sibling `results/` tree" ternary exists in five places and they disagree:
`sims.py:37` and `sims.py:196` guard on `store.name == "fits"`; `profiles.py:116`
(`store.parent / "profiles"`) does **not**. When the store is not the `fits` leaf
(e.g. `--store results/`), `sims_root` resolves to `results/sims` but
`profiles_root` resolves to `<results-parent>/profiles` — different trees.
`ingest.py:403` (one level up, for `camdl --root`) and `ingest.py:817` /
`routes.py:720` (two levels up, for relative model paths) are two genuinely
different quantities wearing the same ternary, so a reader can't tell which
"root" is meant. Wants one `project_root(store)` and one
`results_sibling(store, name)`.

---

## Theme 1 — camdl's layout & schema re-encoded by hand (the big one)

`ingest.py`'s docstring claims it is "deliberately isolated: if camdl grows a
sanctioned `camdl watch` API later, swap this file and nothing downstream
changes." That is no longer true on two counts.

**Layout knowledge is scattered, not isolated.** Direct globbing/parsing of
camdl's on-disk tree lives in `ingest.py` (`[0-9]*-posterior-*`, `seed_*`,
`chain_*`), `mle.py:59-62` (`[0-9]*-*-*`, excludes `-posterior-` by substring),
`compare.py:35-40,61` (`_PREQUENTIAL_GLOBS` depth-scan), `predictive.py:52`
(`glob("*.tsv")`), `sims.py`, and `profiles.py`. Each re-derives the layout
independently; `mle.find_mle_seed` and `ingest._pick_posterior_dir` encode the
stage-naming convention twice.

**Two generations of reader coexist.** camdl has shipped the content-addressed
store: `sims/`, `profiles/`, `surveys/`, `pfilters/` and each fit seed leaf now
carry a `run.json` `RunRecord` (authoritative `run_id`, factored `levels`,
`status`, `provenance`). `sims.py` and `profiles.py` correctly read `run.json`
(`levels`, `kind`, `status`, `provenance`). The fit path in `ingest.py` does not:
it derives run identity by string-splitting the **directory name**
(`RunMeta.run_id = run_dir.name`; `.hash`/`.derived_label` do
`run_id.rsplit("-",1)`), reads status from the `.lock` PID + `progress.json`, and
reads `run.json` only for `wall_time_seconds` (299). The CAS path-shape contract
explicitly asks downstream consumers (it names camdl-viewer) to "resolve runs by
reading `run.json` (`run_id`, `levels`, `kind`), not by parsing path segments."
The watcher's own `run_id` is therefore a path segment, not camdl's 64-hex
`run_id`, so it can't round-trip with `camdl show <prefix>`.

Nuance worth keeping: `progress.json`'s freshness is a *better* liveness signal
than `run.json`'s static `status` (a crashed run leaves `status:running`
forever), so this isn't "just read `run.json` status". The clean split is:
`run.json` is authoritative for **identity** (`run_id`, `levels`) and terminal
status; `progress.json` for **liveness**. A single `resolve_leaf(dir) ->
RunRecord` reader (shared with sims/profiles) plus the existing heartbeat logic
would retire the path-parsing without losing stall detection.

**Column/CLI contracts as bare literals with silent failure:**
- `compare.py` — `t_score`/`rows` keys and exit code `2` (B3).
- `mle.py:99-108` — `loglik`/`chain`/`status`/`n_evals` columns; only `loglik`
  is guarded, the rest `row.get(..., 0)` so a renamed column reads as `0`.
- `mle.py:24-25` — `FAILED_LOGLIK = -1e99` with a comment that says "≈ −1e100";
  the two are presented as the same value. If camdl's sentinel moves,
  converged/failed classification flips silently. (`isinstance(v,(int,float))` at
  88 also accepts `bool` as a coordinate — `bool ⊂ int`.)
- `predictive.py:11-19` — the predictive TSV column order pinned only in prose.
- `highlight.py:26-53` — camdl's keyword/builtin/distribution grammar hand-copied
  from `highlights.scm`/`camdl.xml`; a new distribution under-colours silently
  (cosmetic, low stakes, but a real N-copies-of-one-truth).

**`predictive.discover_streams` globs, against the manifest discipline
`quantities.py` sets.** `quantities.py:9-12` deliberately reads only what the
manifest lists "never the directory glob," because a renamed block leaves a stale
TSV behind. `predictive.py:42-54` enumerates streams by `glob("*.tsv")` — the
exact hazard, with `schema.ObsSchema.streams` (the authoritative stream index)
available right there. Two sibling readers, opposite stances.

## Theme 2 — "which objective is present" computed six ways

The backend decides which pooled objectives (`log_posterior`/`log_likelihood`) to
show with two different rules: `_present_objectives` (`routes.py:429`) requires
the column in **all** draw-bearing chains; `get_traces` (`routes.py:1002-1004`)
inlines **any**. So a run where only some chains carry `log_posterior` shows lp__
in Traces but not in the forest/pair. The frontend then adds three more:
`TracesTab.tsx:24` hard-codes `OBJECTIVE_NAMES`, `PosteriorTab.tsx:85` trusts the
backend `is_objective` flag, `PairTab.tsx:162` trusts the backend `objectives`
list. One predicate (`all`, backend-side) that the API ships as the single list
would collapse all six.

## Theme 3 — small helpers copy-pasted rather than shared

- **`pl.read_csv(sep="\t", infer_schema_length=10000)` banded-TSV read — 4
  copies** (`predictive.py:64`, `quantities.py:103`, `mle.py:96`,
  `ingest.py:588`), with inconsistent guards (`mle` uses bare `except Exception`;
  the others `except (OSError, PolarsError)`). One `read_banded_tsv` helper.
- **Post-warmup-tail + finite-mean-over-aux loop — 4 copies in `diagnostics.py`**
  (`_aux_tail_mean` 215-228, the inner `_live` in `per_chain_mixing` 375-387,
  plus the `values`-or-`aux` column lookup at 48 and 137). Genuinely-shared,
  bug-prone substrate (it's where B1 hid).
- **Convergence thresholds `RHAT_HIGH=1.1`, `RHAT_OK=1.05`, `ESS_LOW=100` +
  `rhatClass`/`essClass`** duplicated byte-for-byte in `ForestRow.tsx:6-24` and
  `DiagnosticsTab.tsx:22-50`.
- **Failure sentinel** `-1e99`/`-1e100` as a bare literal in `mle.py` and four
  frontend files (`ProfileWorkspace.tsx:163`, `ProfilePlot.tsx:45`,
  `ProfileSurface.tsx:24`, `RestartsTab.tsx:10`).
- **`docs.py`** — `DOC_CATEGORIES` (26-32) is dead (no usages) *and* stale (lists
  5; the model has 6 — `quantities` was added everywhere but here). The category
  list is hand-synced across four sites (fields 64-69, `from_meta` 98-104,
  `is_empty` 106-114, the dead tuple). `for_param` (122-138) reimplements the
  `<base>_<Level>` longest-prefix expansion that
  `ingest._resolve_ir_for_param` owns.
- **Self-measuring `ResizeObserver` scaffold — ~11 copies** in the frontend
  (`PredictiveTab`, `QuantitiesTab`, `DiagnosticsTab`, `PairPlot`,
  `MarginalDensity`, `ProfilePlot`, `ProfileSurface`, `DeltaElpdPlot`,
  `TracesTab`, `RestartsTab`, and `Figure` which already abstracts the pattern).
  A `useMeasuredWidth()` hook folds all of them.
- **`PredictivePanel` (`PredictiveTab.tsx:62`) and `BandPanel`
  (`QuantitiesTab.tsx:62`)** are the same ribbon scaffold (identical axis/style
  config), differing only in the marks. `stratumLabel` and the `dimLevels`
  builder are also copied between those two files.
- **Route preamble** `store=_store(); meta=_find_meta(store,run_id); if meta is
  None: raise HTTPException(404)` — 15×. A FastAPI `Depends(resolve_run)`
  centralizes it, and lets Theme-1's shared reader land in one place.

## Theme 4 — stringly-typed where an ADT belongs

- **`Warning_` has no `kind`, so `routes._warning_kind` (1049-1065)
  reverse-engineers the tag from the human message string** (`"ESS" in message`,
  `">" in message`). `Finding` already carries a typed `kind`; `Warning_` should
  too. Reword a message and the live-verdict grouping silently mis-buckets.
- **Backend closed-sets typed as bare `string`.** `source`, `severity`,
  `status`, `mode`, `shape`, `fit_kind`, `origin` are Pydantic `str`, so OpenAPI
  emits `string` and every UI branch (`data.source === 'camdl'`, `q.shape ===
  'series'`, `run.fit_kind === 'mle'`, …) is an unchecked, non-exhaustive
  comparison. One middle-layer fix — `Literal`/`Enum` on the Pydantic models —
  gives the ~dozen UI comparisons exhaustiveness for free.
- **`sims.py` `role` (`"time"|"state"|"flow"`), `status`, and `schema: dict`**
  are bare strings/bags; `resolve_roles` coerces any unknown role to `"state"`
  (246). A `Role` enum + reuse of the existing `Status` enum. `quantities.py`
  `shape`/`source` are the same closed sets typed as `str`.
- **`diagnostics.effective_rhat/ess`** return a `"camdl"|"live"` source tag that
  is stringly-typed *and* discarded at every call site (`routes.py:406-407,
  1161-1162` all `x, _ = …`). Either surface it (the UI could show provenance) or
  drop it. `per_chain_mixing` returns a positional 4-tuple with an optional band
  in slot 4 — a small `MixingBar` dataclass.
- **`grouping.ParamGroups._order` is a frozen-dataclass field set by
  post-construction mutation** (`object.__setattr__` at 149). A `ParamGroups`
  built directly (not via `group_params`) silently has `_order=()`, so
  `default_selection`/`all_params` return `[]`. Make order a real constructor
  argument so the frozen invariant holds.

## Theme 5 — scientific thresholds in the projection layer

The compare route treats camdl as the single source of truth for the elpd math —
yet re-hardcodes two verdicts beside it: `gap_is_real = |Δ| > 2·se`
(`routes.py:1277`) and `overconfident = pit < 0.70` (1279). camdl already emits
`evidence_label` from `delta_elpd_db` (it owns the evidence scale in
`compare.rs`/`evidence.rs`), so these should come from camdl too. The frontend
goes further and *computes statistics* despite `lib/format.ts`'s own stated rule
("these never compute statistics — every number is shipped authoritative"): PIT
(`PredictiveTab.tsx:231-252`, quantile grid hard-coded 973-974) and R²
(`214-224`) are derived client-side, and the R̂/ESS colour thresholds re-decide
pass/fail instead of keying off camdl's `findings.severity` (which renders in the
same panel and can disagree).

## Theme 6 — broad exception handling that can hide wrong science

- `diagnostics._az_safe` (73) `except Exception → NaN` — narrow to the arviz/numpy
  exceptions actually expected, so a shape bug or arviz API change surfaces
  instead of silently blanking a diagnostic. (`_plateau_test`'s `except Exception`
  at 121 is the same category, lower stakes.)
- `mle.py:97` bare `except Exception` reading `chain_results.tsv`, inconsistent
  with the narrow guards elsewhere; degrades to "no restarts", dropping the MLE
  identifiability view.
- (The `health` probe's broad guard in `app.py:65` is deliberate and scoped —
  fine.)

## Theme 7 — performance shape

Every route calls `_find_meta` → `ingest.discover_runs(store,
include_warming=True)`, a full store rescan that reads every run's
`fit.meta.json`/`fit.toml.original` *and shells out to `camdl list`* (via
`_native_labels`), just to resolve one run by id — on every poll. `list_runs`
additionally builds a full `RunState` (tail-reading every chain of every run) per
request. The module docstring acknowledges "run state rebuilt per request," but
the O(runs) rescan + subprocess per single-run request exceeds that. A
signature-keyed cache (mtime/size) on discovery, and skipping `camdl list` when
no label is needed, are the clean wins. Not a correctness issue.

## Small / stale

- `cli.py:39` comment "the app reads `CAMDL_WATCH_STORE` at import" — the app
  reads it *fresh per call* (`app.py:28-34`); stale.
- `routes.py:108` `_warmup_cutoff` docstring "(app.py's rule)" — no such rule in
  `app.py`.
- `client.ts:49-54` `RunStatus` union — a hand-maintained duplicate of the
  backend status set, imported nowhere (`StatusBadge` takes `string`). Dead.
- `SourceTab.tsx:704-710` — a JSDoc describing `SourceTab` is stranded above
  `RenderedModel`.
- Frontend live-refresh uses two mechanisms: react-query `refetchInterval`
  (`queries.ts`) and a bespoke `setInterval`+`invalidateQueries` with a broad
  `queryKey.includes(liveId)` predicate (`ExploreWorkspace.tsx:212-222`). Unify
  on `refetchInterval` gated by run status.
- Giant components with clean seams: `PredictiveTab.tsx` (1327 lines — stats →
  `lib/`, the ~200-line derivation block → a `usePredictiveModel` hook, the
  four-branch `view` switch → four view components, eight plot sub-components →
  their own files); `QuantitiesTab.tsx` (622).
- Magic numbers worth a name/comment: `diagnostics.py` `0.5*sd` (125),
  `bulk_ess/4` (257), the `<4` draw floor duplicated at 58 & 247, the `10**9`
  n_tail sentinel; `grouping.py` `len(primary) >= 2` (60).
- `sims.read_sim_model` parses `.camdl` with three regexes (184-215); a `}` or
  `#` inside a string mis-extracts `states`, which changes which columns sum into
  a compartment total (a silently-wrong path, even if a real parser is out of
  scope). `pl.sum_horizontal` (355) treats nulls as 0, undercounting a total with
  missing cells.

---

## Suggested order of attack

1. The five real bugs (B1–B5) — small diffs, each removes a silent wrong number.
2. One shared `resolve_leaf(dir) -> RunRecord` reader over `run.json`, adopted by
   the fit path so identity/status stop coming from path segments; fold the five
   root-resolution ternaries into `project_root` / `results_sibling`. This is the
   Theme-1 consolidation and it retires the most drift surface.
3. Ship objective-presence and the compare/convergence verdicts as
   backend-authoritative fields; delete the client-side stats and duplicated
   thresholds. (Theme 2 + 5.)
4. `Literal`/`Enum` on the Pydantic closed-sets — one change, ~dozen UI
   comparisons become checked. (Theme 4.)
5. Extract the shared helpers (`read_banded_tsv`, the diagnostics tail loop,
   `useMeasuredWidth`, `RibbonPanel`, the route `Depends`). (Theme 3.)
6. Narrow the broad `except`s (Theme 6); the discovery cache (Theme 7); the stale
   comments and dead code last.
