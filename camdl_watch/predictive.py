"""Posterior-predictive artifacts — the ``camdl fit predict`` output, if run.

A fit only has these once ``camdl fit predict`` has been run against it; a live
or never-predicted fit has none and every reader here returns ``None`` / ``[]``.

``fit predict`` replays one run, and writes into a directory of the problem
folder named for that run and the options it was predicted with (camdl gh#896,
run spec §2.2.2)::

    <run_dir>/predict/<run_id12>/              # default predict options
    <run_dir>/predict/<run_id12>-<opts8>/      # any other --horizon, --n-draws, …
        predictive/<stream>.tsv   predictive.json   # quantile ribbons
        observed/<stream>.tsv     observed.json     # the observed series
        quantities/<name>.tsv     quantities.json   # see quantities.py

``<run_id12>`` is the first 12 hex characters of the run's ``run_id`` (the leaf's
``run.json``); ``<opts8>`` hashes the options, which ``predictive.json`` records
under ``options``. :func:`predict_dirs` finds these for one run; every other
reader here takes one such directory. Predict output an older camdl wrote at the
folder level is not read: camdl no longer reads it either, and it cannot say
which run it replays.

Column layout (read straight off camdl's renderer):

* ``predictive/<stream>.tsv`` —
  ``time | <index dims…> | horizon | treatment | rhat_max | ess_min | n_draws |
  q05 | q25 | q50 | q75 | q95``. The ``<index dims…>`` columns are the stream's
  stratifying dimensions (none for a single national series); several horizons
  stack under the one header.
* ``observed/<stream>.tsv`` — ``time | <index dims…> | value``; the value is an
  empty cell where the observed series has a hole.

We return the frames verbatim (polars, schema inferred) — interpretation
(which columns are dims vs. quantiles) belongs to the consumer, which can read
the stream's ``index_dims`` from the :class:`camdl_watch.schema.ObsSchema`.
Pure readers: no dependency on ingest, no cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

PREDICT_DIR = "predict"
PREDICTIVE_DIR = "predictive"
OBSERVED_DIR = "observed"
# Sidecars that carry the stream time calendar (origin epoch + unit), in order of
# preference — both should agree; predictive is the one the ribbons come from.
_CALENDAR_SIDECARS = ("predictive.json", "observed.json")
#: How many hex characters of a run_id name its predict directory.
RUN_ID_PREFIX = 12


@dataclass(frozen=True)
class PredictDir:
    """One ``fit predict`` output of one run.

    ``key`` is the directory name (``<run_id12>`` or ``<run_id12>-<opts8>``) and
    is what the API takes to select it. ``options`` are the predict options
    camdl recorded for it; ``default`` marks the one predicted with camdl's
    default options, which is the one shown unless another is asked for."""

    key: str
    path: Path
    default: bool
    options: dict


def _read_options(d: Path) -> dict:
    for name in _CALENDAR_SIDECARS:
        try:
            opts = json.loads((d / name).read_text()).get("options")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(opts, dict):
            return opts
    return {}


def predict_dirs(run_dir: Path, run_id: str | None) -> list[PredictDir]:
    """Every ``fit predict`` output of run ``run_id`` in problem folder
    ``run_dir``: the default-options one first, then each option-keyed one by
    name. Empty when the run was never predicted or its ``run_id`` is unknown
    (a leaf with no readable ``run.json``)."""
    if not run_id or len(run_id) < RUN_ID_PREFIX:
        return []
    root = Path(run_dir) / PREDICT_DIR
    if not root.is_dir():
        return []
    id12 = run_id[:RUN_ID_PREFIX]
    out: list[PredictDir] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name == id12:
            out.insert(0, PredictDir(d.name, d, True, _read_options(d)))
        elif d.name.startswith(id12 + "-"):
            out.append(PredictDir(d.name, d, False, _read_options(d)))
    return out


def resolve_predict_dir(
    run_dir: Path, run_id: str | None, key: str | None = None
) -> Path | None:
    """The predict directory to read: the one named ``key`` if given, else the
    default-options one, else (a run only ever predicted with non-default
    options) the first option-keyed one. ``None`` when there is no such predict
    for this run — a ``key`` belonging to another run is not found, never
    substituted."""
    dirs = predict_dirs(run_dir, run_id)
    if key is not None:
        return next((p.path for p in dirs if p.key == key), None)
    return dirs[0].path if dirs else None


def discover_streams(run_dir: Path) -> list[str]:
    """The logical stream names a run has predictive/observed artifacts for —
    the union of ``predictive/*.tsv`` and ``observed/*.tsv`` stems, sorted.
    Empty when ``camdl fit predict`` was never run for this fit."""
    run_dir = Path(run_dir)
    stems: set[str] = set()
    for sub in (PREDICTIVE_DIR, OBSERVED_DIR):
        d = run_dir / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.tsv"):
            stems.add(p.stem)
    return sorted(stems)


def _read_tsv(path: Path) -> pl.DataFrame | None:
    """Read a tidy TSV, or ``None`` if it is absent / empty / unparseable. An
    empty (0-byte) file raises ``NoDataError``; a header-only file reads back as
    a zero-row frame, which is a valid (if empty) artifact and is returned."""
    if not path.is_file():
        return None
    try:
        return pl.read_csv(path, separator="\t", infer_schema_length=10000)
    except (OSError, pl.exceptions.PolarsError):
        return None


@dataclass(frozen=True)
class Calendar:
    """The stream time axis's calendar: a numeric ``time`` value is the date
    ``origin + time × days_per_unit`` days. Lets a consumer show real dates
    instead of raw day-indices."""

    origin: str  # ISO date the time axis counts from, e.g. "1910-01-01"
    time_unit: str = "days"
    days_per_unit: float = 1.0


def read_calendar(run_dir: Path) -> Calendar | None:
    """The time calendar camdl records in the predictive/observed sidecar, so a
    numeric ``time`` column can be rendered as a date. Reads ``predictive.json``
    (falling back to ``observed.json``); ``None`` when neither declares an origin
    (a relative-time model — the time axis stays numeric)."""
    run_dir = Path(run_dir)
    for name in _CALENDAR_SIDECARS:
        p = run_dir / name
        if not p.is_file():
            continue
        try:
            cal = json.loads(p.read_text()).get("calendar")
        except (OSError, ValueError):
            continue
        if isinstance(cal, dict) and cal.get("origin"):
            return Calendar(
                origin=str(cal["origin"]),
                time_unit=str(cal.get("time_unit") or "days"),
                days_per_unit=float(cal.get("days_per_unit") or 1.0),
            )
    return None


@dataclass(frozen=True)
class PredictiveSeries:
    """One stream's posterior-predictive quantile ribbons (``predictive/`` TSV)."""

    stream: str
    table: pl.DataFrame


@dataclass(frozen=True)
class ObservedSeries:
    """One stream's observed series to overlay (``observed/`` TSV)."""

    stream: str
    table: pl.DataFrame


#: Where a prior predictive lands inside a run dir. camdl has no prior-predictive
#: writer of its own (camdl#711), but `camdl simulate --draws prior --obs-dir
#: <run>/prior_predictive` writes exactly one TSV per stream, named for the
#: stream — the same naming `predictive/` and `observed/` use — so pointing that
#: flag here makes the artifact discoverable by the same store walk.
PRIOR_PREDICTIVE_DIR = "prior_predictive"

#: The quantiles a predictive ribbon is drawn from, matching what `fit predict`
#: writes so a prior band and a posterior band are the same object.
_BAND_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def discover_prior_streams(run_dir: Path) -> list[str]:
    """Stream names with a prior-predictive replicate file, sorted. Empty when
    the run has none (the normal case until one is generated)."""
    d = Path(run_dir) / PRIOR_PREDICTIVE_DIR
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.tsv"))


def read_prior_bands(run_dir: Path, stream: str) -> pl.DataFrame | None:
    """Band a prior-predictive replicate file into the ribbon contract.

    ``simulate --obs-dir`` writes RAW draws — ``replicate | draw | time |
    <stream>`` — because only `fit predict` bands observation streams
    (camdl#711). Reducing them here rather than in the browser keeps one
    quantile convention across the prior and posterior ribbons; a consumer that
    banded client-side would silently disagree with `fit predict` at the tails.

    Returns ``time | q05 | q25 | q50 | q75 | q95``, or ``None`` when the file is
    absent or carries no value column."""
    table = _read_tsv(Path(run_dir) / PRIOR_PREDICTIVE_DIR / f"{stream}.tsv")
    if table is None or table.height == 0 or "time" not in table.columns:
        return None
    # The value column is named for the stream; fall back to the single
    # remaining numeric column so a renamed stream still reads.
    value = stream if stream in table.columns else None
    if value is None:
        rest = [c for c in table.columns if c not in ("replicate", "draw", "time")]
        if len(rest) != 1:
            return None
        value = rest[0]
    return (
        table.group_by("time")
        .agg(
            [
                pl.col(value).quantile(q, interpolation="linear").alias(name)
                for q, name in zip(_BAND_QUANTILES, ("q05", "q25", "q50", "q75", "q95"))
            ]
        )
        .sort("time")
    )


def read_predictive(run_dir: Path, stream: str) -> PredictiveSeries | None:
    """The predictive quantile ribbons for ``stream``, or ``None`` if absent."""
    table = _read_tsv(Path(run_dir) / PREDICTIVE_DIR / f"{stream}.tsv")
    return PredictiveSeries(stream=stream, table=table) if table is not None else None


def read_observed(run_dir: Path, stream: str) -> ObservedSeries | None:
    """The observed series for ``stream``, or ``None`` if absent."""
    table = _read_tsv(Path(run_dir) / OBSERVED_DIR / f"{stream}.tsv")
    return ObservedSeries(stream=stream, table=table) if table is not None else None
