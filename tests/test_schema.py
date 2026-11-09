"""ObsSchema tests: tolerant parsing of the ``schema`` block from a synthetic
meta and from real fit.meta.json sidecars (which currently carry ``schema:
null`` -> None)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from camdl_scope.schema import DimensionSpec, ObsSchema, StreamSpec

# Real fit.meta.json sidecars live under the camdl sibling repo; mirrors
# test_ingest's relative store derivation.
CAMDL = Path(os.environ.get("CAMDL_REPO", Path(__file__).resolve().parents[2] / "camdl"))


def _real_metas() -> list[Path]:
    if not CAMDL.is_dir():
        return []
    return sorted(CAMDL.rglob("fit.meta.json"))


def test_from_meta_synthetic_full():
    meta = {
        "schema": {
            "dimensions": {
                "patch": {"levels": ["Bo", "Bombali", "Kailahun"]},
            },
            "streams": [
                {
                    "name": "cases",
                    "index_dims": ["patch"],
                    "value_column": "cases",
                    "value_kind": "count",
                    "likelihood": "neg_binomial",
                },
                {
                    # A national series: no index dims, value_kind undeclared.
                    "name": "deaths",
                    "index_dims": [],
                    "value_column": "deaths",
                    "likelihood": "poisson",
                },
            ],
        }
    }
    sch = ObsSchema.from_meta(meta)
    assert sch is not None
    assert sch.dimensions["patch"] == DimensionSpec(name="patch", levels=["Bo", "Bombali", "Kailahun"])
    assert len(sch.streams) == 2
    cases = sch.streams[0]
    assert isinstance(cases, StreamSpec)
    assert cases.name == "cases"
    assert cases.index_dims == ["patch"]
    assert cases.value_column == "cases"
    assert cases.value_kind == "count"
    assert cases.likelihood == "neg_binomial"
    deaths = sch.streams[1]
    assert deaths.index_dims == []
    assert deaths.value_kind is None  # undeclared role -> None
    assert deaths.likelihood == "poisson"


def test_from_meta_none_when_absent_or_null():
    assert ObsSchema.from_meta({}) is None
    assert ObsSchema.from_meta({"schema": None}) is None
    # A non-dict schema also yields None rather than raising.
    assert ObsSchema.from_meta({"schema": []}) is None


def test_from_meta_partial_is_tolerant():
    # Missing streams / dimensions, and a malformed stream entry, all degrade.
    sch = ObsSchema.from_meta({"schema": {"streams": [{"name": "cases"}, 42]}})
    assert sch is not None
    assert sch.dimensions == {}
    assert len(sch.streams) == 1
    assert sch.streams[0].name == "cases"
    assert sch.streams[0].value_column is None


def test_real_meta_round_trips():
    """Every real fit.meta.json parses without error: those with a ``schema``
    yield an ObsSchema, those without (the current sidecars carry ``schema:
    null``) yield None. Skips if the sibling repo isn't present."""
    metas = _real_metas()
    if not metas:
        pytest.skip("no real fit.meta.json under the camdl sibling repo")
    saw_schema = False
    for mp in metas:
        meta = json.loads(mp.read_text())
        sch = ObsSchema.from_meta(meta)
        if meta.get("schema") is None:
            assert sch is None, f"{mp}: null schema must parse to None"
        else:
            saw_schema = True
            assert sch is not None
            # streams is always a list; dimensions always a dict.
            assert isinstance(sch.streams, list)
            assert isinstance(sch.dimensions, dict)
    if not saw_schema:
        # Documents the current on-disk reality without failing the suite.
        pytest.skip("no on-disk fit.meta.json carries a schema block yet")
