"""Generated-quantities sidecar — the ``camdl fit predict`` quantities output.

A fit has these once ``camdl fit predict`` ran on a model with a ``quantities {}``
block. They are written sibling to predictive/observed, inside the run's predict
directory (``<run_dir>/predict/<run_id12>[-<opts8>]/``, found by
:func:`camdl_watch.predictive.predict_dirs`); every reader here takes that
directory as ``run_dir``::

    <predict dir>/quantities.json         # manifest (schema camdl.quantities/v1)
    <predict dir>/quantities/<name>.tsv   # one banded TSV per logical quantity

The MANIFEST is the authoritative index. The ``quantities/`` dir can also hold
stale TSVs from a prior predict whose block named quantities differently (a
renamed quantity leaves its old file behind), so we read only what the manifest
lists — never the directory glob.

Each entry's ``shape`` drives the rendering:

* ``series`` (no reduction) — a banded trajectory, ``time | <dims…> | n_draws |
  q05 | q25 | q50 | q75 | q95`` — a ribbon.
* ``scalar`` (a reduction) — a banded point, ``<dims…> | n_draws | q05…q95`` — a
  table row. A *censorable* scalar (a ``time_of_*`` / ``first_*`` reduction that
  can fail to fire) inserts ``n_value | n_censored | p_censored`` before the
  band, and the band is conditional on the event firing (empty q* when every
  draw censored).

Pure readers (polars, schema inferred); no dependency on ingest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

QUANTITIES_DIR = "quantities"
MANIFEST = "quantities.json"


@dataclass(frozen=True)
class QuantityMeta:
    """One *logical* quantity — what it is and how to render it. Scenario-agnostic
    (shape/source/reduce/docs are identical across scenarios); the scenario axis
    lives on :class:`Manifest` and in each TSV's ``scenario`` column."""

    name: str
    shape: str  # "series" | "scalar"
    source: str  # "state" | "observations" | "derived"
    index_dims: list[str]
    reduce: str | None
    unit: str | None
    censorable: bool  # the reduction can fail to fire → a censoring trio + p_censored


@dataclass(frozen=True)
class Manifest:
    """A fit's quantities manifest, denormalized then collapsed: ``quantities``
    are the *logical* quantities (deduped by name — the scenario-aware
    ``fit predict`` emits one manifest entry per quantity × scenario), and
    ``scenarios`` is the distinct scenario set (``[]`` for an old, scenario-less
    sidecar — its TSVs carry no ``scenario`` column)."""

    quantities: list[QuantityMeta]
    scenarios: list[str]


def read_manifest(run_dir: Path) -> Manifest:
    """The fit's quantities manifest, collapsed to logical quantities + the
    scenario set. Empty (no quantities, no scenarios) when the fit has no
    sidecar (never predicted, or no ``quantities {}`` block)."""
    try:
        raw = json.loads((Path(run_dir) / MANIFEST).read_text())
    except (OSError, json.JSONDecodeError):
        return Manifest(quantities=[], scenarios=[])
    by_name: dict[str, QuantityMeta] = {}
    order: list[str] = []
    scenarios: list[str] = []
    for q in raw.get("quantities", []):
        name = q.get("name")
        if not name:
            continue
        sc = q.get("scenario")
        if sc is not None and sc not in scenarios:
            scenarios.append(sc)
        if name not in by_name:
            order.append(name)
            by_name[name] = QuantityMeta(
                name=name,
                shape=q.get("shape", "scalar"),
                source=q.get("source", "state"),
                index_dims=list(q.get("index_dims", [])),
                reduce=q.get("reduce"),
                unit=q.get("unit"),
                censorable=isinstance(q.get("censoring"), dict),
            )
    return Manifest(quantities=[by_name[n] for n in order], scenarios=scenarios)


def _read_tsv(path: Path) -> pl.DataFrame | None:
    """Read a banded TSV, or ``None`` if absent / empty / unparseable."""
    if not path.is_file():
        return None
    try:
        return pl.read_csv(path, separator="\t", infer_schema_length=10000)
    except (OSError, pl.exceptions.PolarsError):
        return None


def read_quantity(run_dir: Path, name: str) -> pl.DataFrame | None:
    """One quantity's banded TSV, or ``None`` if absent. ``name`` must come from
    the manifest (the authoritative index — don't glob the directory)."""
    return _read_tsv(Path(run_dir) / QUANTITIES_DIR / f"{name}.tsv")
