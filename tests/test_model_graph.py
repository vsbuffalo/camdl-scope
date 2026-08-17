"""Reader + contract tests for the flow-graph artifact (``model.graph.json``).

The synthetic cases exercise the reader's absent/malformed handling; the sample
cases validate the real emitter output in ``model-graph-samples/`` against the
:class:`ModelGraph` wire contract (in particular the ``from``/``to`` aliasing).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from camdl_watch import model_graph
from camdl_watch.api.models import ModelGraph

_SAMPLES_DIR = Path(__file__).resolve().parents[1] / "model-graph-samples"
_SAMPLES = sorted(_SAMPLES_DIR.glob("*.graph.json"))

_MINIMAL = {
    "model": "sir_basic",
    "nodes": [{"id": "S", "label": "S"}, {"id": "I", "label": "I"}],
    "plates": [],
    "edges": [
        {"id": "infection", "from": "S", "to": "I",
         "rate": "\\frac{β\\,S\\,I}{N}", "advances": None, "reads_pool": False}
    ],
    "couplings": [],
}


def test_has_model_graph(tmp_path: Path):
    assert model_graph.has_model_graph(tmp_path) is False
    (tmp_path / "model.graph.json").write_text(json.dumps(_MINIMAL))
    assert model_graph.has_model_graph(tmp_path) is True


def test_read_absent_is_none(tmp_path: Path):
    assert model_graph.read_model_graph(tmp_path) is None


def test_read_malformed_is_none(tmp_path: Path):
    (tmp_path / "model.graph.json").write_text("not json{")
    assert model_graph.read_model_graph(tmp_path) is None
    # A JSON array (not an object) is rejected — the contract is a map.
    (tmp_path / "model.graph.json").write_text("[1, 2, 3]")
    assert model_graph.read_model_graph(tmp_path) is None


def test_edge_from_to_alias_and_exogenous(tmp_path: Path):
    """``from``/``to`` map onto ``source``/``target``; ``None`` marks an
    exogenous flow (birth has no source, death no target)."""
    graph = {
        "model": "m",
        "nodes": [{"id": "S", "label": "S"}],
        "edges": [
            {"id": "birth", "from": None, "to": "S", "rate": "\\delta", "reads_pool": True},
            {"id": "death", "from": "c", "to": None, "rate": "\\mu"},
        ],
    }
    m = ModelGraph.model_validate(graph)
    birth, death = m.edges
    assert birth.source is None and birth.target == "S" and birth.reads_pool is True
    assert death.source == "c" and death.target is None
    # Round-trips back to the wire spelling FastAPI serializes with.
    assert m.model_dump(by_alias=True)["edges"][0]["from"] is None
    assert m.model_dump(by_alias=True)["edges"][0]["to"] == "S"


def test_validate_tolerates_missing_optional_sections():
    m = ModelGraph.model_validate({"model": "m"})
    assert m.nodes == [] and m.edges == [] and m.plates == [] and m.couplings == []


@pytest.mark.skipif(not _SAMPLES, reason="no model-graph-samples present")
@pytest.mark.parametrize("path", _SAMPLES, ids=lambda p: p.stem)
def test_real_emitter_samples_validate(path: Path):
    """Every committed emitter sample validates against the wire contract."""
    raw = json.loads(path.read_text())
    m = ModelGraph.model_validate(raw)
    assert m.model
    assert len(m.nodes) == len(raw["nodes"])
    assert len(m.edges) == len(raw["edges"])
    # Every edge endpoint is a known node id, the iterator "c", or None
    # (exogenous). This is the invariant the diagram layout relies on.
    ids = {n.id for n in m.nodes} | {"c", None}
    for e in m.edges:
        assert e.source in ids, f"{path.stem}:{e.id} bad source {e.source}"
        assert e.target in ids, f"{path.stem}:{e.id} bad target {e.target}"
    # Every coupling references a real edge id.
    edge_ids = {e.id for e in m.edges}
    for c in m.couplings:
        assert c.edge in edge_ids, f"{path.stem}: coupling on unknown edge {c.edge}"
