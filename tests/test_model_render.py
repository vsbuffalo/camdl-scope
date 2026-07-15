"""Reader tests for the structured-model artifact (``model.render.json``).

No real ``model.render.json`` exists in the reference store yet (the contract is
merging upstream), so these use a synthetic fixture matching the agreed schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from camdl_watch import model_render
from camdl_watch.api.models import ModelRender

_EXAMPLE = {
    "model": "sir_basic",
    "mode": "indexed",
    "states": ["S", "I", "R"],
    "dimensions": [{"name": "region", "levels": ["north", "south"]}],
    "parameters": [
        {"name": "beta", "symbol": "\\beta", "description": "per-capita transmission rate"},
        {"name": "gamma", "symbol": "\\gamma", "description": "recovery rate"},
    ],
    "definitions": [{"name": "N", "tex": "N = S + I + R"}],
    "transitions": [
        {"name": "infection", "reactants": "S", "products": "I", "rate": "\\frac{\\beta\\,S\\,I}{N}"}
    ],
    "dynamics": [{"state": "S", "tex": "\\dot{S} = -\\frac{\\beta\\,S\\,I}{N}"}],
}


def test_has_model_render(tmp_path: Path):
    assert model_render.has_model_render(tmp_path) is False
    (tmp_path / "model.render.json").write_text(json.dumps(_EXAMPLE))
    assert model_render.has_model_render(tmp_path) is True


def test_read_absent_is_none(tmp_path: Path):
    assert model_render.read_model_render(tmp_path) is None


def test_read_malformed_is_none(tmp_path: Path):
    (tmp_path / "model.render.json").write_text("not json{")
    assert model_render.read_model_render(tmp_path) is None
    # A JSON array (not an object) is also rejected — the contract is a map.
    (tmp_path / "model.render.json").write_text("[1, 2, 3]")
    assert model_render.read_model_render(tmp_path) is None


def test_read_roundtrips_and_validates(tmp_path: Path):
    (tmp_path / "model.render.json").write_text(json.dumps(_EXAMPLE))
    raw = model_render.read_model_render(tmp_path)
    assert raw is not None
    m = ModelRender.model_validate(raw)
    assert m.model == "sir_basic" and m.mode == "indexed"
    assert m.states == ["S", "I", "R"]
    assert m.dimensions[0].name == "region"
    assert m.parameters[0].symbol == "\\beta"
    assert m.transitions[0].rate == "\\frac{\\beta\\,S\\,I}{N}"
    assert m.dynamics[0].tex.startswith("\\dot{S}")


def test_validate_tolerates_missing_optional_sections(tmp_path: Path):
    # A minimal render (only the required identity) validates with empty sections
    # — the contract stays forward/backward compatible.
    minimal = {"model": "m", "mode": "flat"}
    m = ModelRender.model_validate(minimal)
    assert m.parameters == [] and m.transitions == [] and m.dynamics == []
