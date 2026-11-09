"""ModelDocs tests: parse the ``#'`` doc dictionary from a golden IR envelope
(and from a synthetic meta), and resolve expanded coordinates to their base
parameter doc."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from camdl_scope.docs import DocBlock, ModelDocs

# The camdl sibling repo (its golden IR envelopes carry a top-level ``docs``
# block in the same shape fit.meta.json embeds). Mirrors test_ingest's relative
# derivation; override with CAMDL_REPO when the sibling lives elsewhere.
CAMDL = Path(os.environ.get("CAMDL_REPO", Path(__file__).resolve().parents[2] / "camdl"))
GOLDEN = CAMDL / "ocaml" / "golden"


def _golden(name: str) -> dict:
    p = GOLDEN / name
    if not p.is_file():
        pytest.skip(f"golden IR {name} not present")
    return json.loads(p.read_text())


def test_docs_from_golden_sir_basic():
    docs = ModelDocs.from_meta(_golden("sir_basic.ir.json"))
    assert not docs.is_empty()
    beta = docs.parameters["beta"]
    assert beta.symbol == "β"
    assert "transmission" in (beta.text or "")
    gamma = docs.parameters["gamma"]
    assert gamma.symbol == "γ"
    assert "recovery" in (gamma.text or "")
    # A documented-but-symbol-less param: text present, symbol absent.
    n0 = docs.parameters["N0"]
    assert n0.symbol is None
    assert n0.text is not None
    # Only the parameters category is populated here.
    assert docs.compartments == {}


def test_doc_block_is_empty():
    assert DocBlock().is_empty()
    assert not DocBlock(text="x").is_empty()
    assert not DocBlock(symbol="β").is_empty()
    assert not DocBlock(reference="Anderson & May 1991").is_empty()


def test_empty_when_no_docs_key():
    assert ModelDocs.from_meta({}).is_empty()
    assert ModelDocs.from_meta({"docs": None}).is_empty()
    # A non-dict docs value degrades to empty rather than raising.
    assert ModelDocs.from_meta({"docs": []}).is_empty()


def test_ref_json_key_maps_to_reference():
    meta = {"docs": {"parameters": {"beta": {"text": "t", "symbol": "β", "ref": "AM91"}}}}
    blk = ModelDocs.from_meta(meta).parameters["beta"]
    assert blk.reference == "AM91"
    assert blk.symbol == "β"
    assert blk.text == "t"


def test_for_param_exact_and_prefix():
    meta = {
        "docs": {
            "parameters": {
                "k_raw": {"text": "raw per-patch effect", "symbol": "k"},
                "beta": {"symbol": "β"},
            }
        }
    }
    docs = ModelDocs.from_meta(meta)
    # Exact match.
    assert docs.for_param("beta").symbol == "β"
    # Expanded coordinate resolves to its base via longest-prefix.
    assert docs.for_param("k_raw_Bo").text == "raw per-patch effect"
    # Nothing matches -> None.
    assert docs.for_param("rho") is None


def test_for_param_prefers_longest_base():
    meta = {
        "docs": {
            "parameters": {
                "k": {"symbol": "k"},
                "k_raw": {"symbol": "kr"},
            }
        }
    }
    docs = ModelDocs.from_meta(meta)
    # k_raw_Bo starts with both "k_" and "k_raw_"; the longest base wins.
    assert docs.for_param("k_raw_Bo").symbol == "kr"
