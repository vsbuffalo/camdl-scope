"""Posterior-predictive artifacts — the ``camdl fit predict`` output, if run.

A fit only has these once ``camdl fit predict`` has been run against it; a live
or never-predicted fit has neither directory and every reader here returns
``None`` / ``[]``. The verb writes two tidy, plot-ready TSV families under the
fit (run) directory, one file per *logical* stream::

    <run_dir>/predictive/<stream>.tsv   # quantile ribbons
    <run_dir>/observed/<stream>.tsv     # the observed series to overlay

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

PREDICTIVE_DIR = "predictive"
OBSERVED_DIR = "observed"
# Sidecars that carry the stream time calendar (origin epoch + unit), in order of
# preference — both should agree; predictive is the one the ribbons come from.
_CALENDAR_SIDECARS = ("predictive.json", "observed.json")


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
