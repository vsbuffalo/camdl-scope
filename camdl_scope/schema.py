"""Observation schema — the machine-readable shape of a fit's data streams.

Each fit's ``fit.meta.json`` carries an optional ``schema`` block: a faithful
projection of the model's *expanded* observation structure, derived by camdl as
a pure fold over the model's observation leaves (the same IR the particle filter
binds, so it cannot disagree with what was fit). It is the thing a consumer
needs to facet a posterior-predictive stream by its index dimensions and to
label panels by level name, without parsing any DSL.

Shape::

    "schema": {
      "dimensions": { "<dim>": { "levels": [<level>, …] }, … },
      "streams":    [ { "name", "index_dims": [<dim>, …],
                        "value_column", "value_kind", "likelihood" }, … ]
    }

``streams`` is one entry per *logical* stream (grouped by data-source key, so a
stratified ``cases[p in patch]`` is a single entry with ``index_dims =
["patch"]``, never one per expanded leaf). ``value_kind`` is absent for models
predating the explicit ``columns {}`` block. The whole ``schema`` key is absent
(``None``) for a sidecar written without a model in hand (CLI-only profile fits,
test fixtures).

Parsing is deliberately tolerant: any field may be missing on a partial schema
and nothing here raises. Stdlib only — no dependency on ingest, no cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _str_list(v: object) -> list[str]:
    """A list-of-strings field, coerced; non-lists degrade to ``[]``."""
    return [str(x) for x in v] if isinstance(v, list) else []


def _opt_str(v: object) -> str | None:
    return v if isinstance(v, str) else None


@dataclass(frozen=True)
class DimensionSpec:
    """One indexing dimension and its ordered levels (e.g. ``patch -> [Bo,
    Bombali, …]``), the union of stratum levels seen across all streams."""

    name: str
    levels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StreamSpec:
    """One logical observation stream's structure.

    ``index_dims`` are the dimensions it is stratified over (``[]`` for a single
    national series — a consumer facets by these). ``value_column`` is the
    scored ``~`` LHS; ``value_kind`` its DSL role (``count`` / ``real`` /
    ``probability`` / …, ``None`` when undeclared); ``likelihood`` the family
    (``poisson`` / ``neg_binomial`` / …)."""

    name: str
    index_dims: list[str] = field(default_factory=list)
    value_column: str | None = None
    value_kind: str | None = None
    likelihood: str | None = None


@dataclass(frozen=True)
class ObsSchema:
    """A fit's observation/dimension schema: ``streams`` × ``dimensions``."""

    dimensions: dict[str, DimensionSpec] = field(default_factory=dict)
    streams: list[StreamSpec] = field(default_factory=list)

    @classmethod
    def from_meta(cls, meta: dict) -> "ObsSchema | None":
        """Read the ``schema`` block off a parsed ``fit.meta.json``.

        Returns ``None`` when the sidecar carries no ``schema`` (the key is
        absent or ``null``). A present-but-partial schema parses tolerantly:
        missing ``dimensions`` / ``streams`` become empty, and a malformed entry
        is skipped rather than raising.
        """
        raw = meta.get("schema")
        if not isinstance(raw, dict):
            return None

        dims: dict[str, DimensionSpec] = {}
        raw_dims = raw.get("dimensions")
        if isinstance(raw_dims, dict):
            for name, spec in raw_dims.items():
                levels = spec.get("levels") if isinstance(spec, dict) else None
                dims[str(name)] = DimensionSpec(name=str(name), levels=_str_list(levels))

        streams: list[StreamSpec] = []
        raw_streams = raw.get("streams")
        if isinstance(raw_streams, list):
            for s in raw_streams:
                if not isinstance(s, dict):
                    continue
                streams.append(
                    StreamSpec(
                        name=str(s.get("name", "")),
                        index_dims=_str_list(s.get("index_dims")),
                        value_column=_opt_str(s.get("value_column")),
                        value_kind=_opt_str(s.get("value_kind")),
                        likelihood=_opt_str(s.get("likelihood")),
                    )
                )

        return cls(dimensions=dims, streams=streams)
