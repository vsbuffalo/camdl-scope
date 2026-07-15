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


def read_predictive(run_dir: Path, stream: str) -> PredictiveSeries | None:
    """The predictive quantile ribbons for ``stream``, or ``None`` if absent."""
    table = _read_tsv(Path(run_dir) / PREDICTIVE_DIR / f"{stream}.tsv")
    return PredictiveSeries(stream=stream, table=table) if table is not None else None


def read_observed(run_dir: Path, stream: str) -> ObservedSeries | None:
    """The observed series for ``stream``, or ``None`` if absent."""
    table = _read_tsv(Path(run_dir) / OBSERVED_DIR / f"{stream}.tsv")
    return ObservedSeries(stream=stream, table=table) if table is not None else None
