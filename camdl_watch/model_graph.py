"""Compartmental flow-graph artifact — ``model.graph.json``.

camdl writes a per-run ``model.graph.json`` (a sibling of ``model.render.json``,
at the ``run_dir`` root) with model-pure content: the base compartments
(``nodes``), the stratifying ``plates``, the ``edges`` (transitions, with a
KaTeX rate string), and the mean-field ``couplings`` (which aggregate pools an
edge's rate reads). It is byte-identical across runs of the same model.

This reader is a pure pass-through: it reads and JSON-parses the file (or returns
``None`` when absent / unreadable). Interpretation into typed fields belongs to
the API layer's :class:`camdl_watch.api.models.ModelGraph`, which validates the
parsed dict — additive and forward-compatible: unknown keys are ignored, missing
optional sections default to empty.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_GRAPH_FILE = "model.graph.json"


def has_model_graph(run_dir: Path) -> bool:
    """Whether the run carries a ``model.graph.json`` to draw."""
    return (Path(run_dir) / MODEL_GRAPH_FILE).is_file()


def read_model_graph(run_dir: Path) -> dict | None:
    """The parsed ``model.graph.json`` for a run, or ``None`` when it's absent or
    unparseable (an older run predating the artifact — the viewer falls back to
    the equations / raw source)."""
    p = Path(run_dir) / MODEL_GRAPH_FILE
    if not p.is_file():
        return None
    try:
        parsed = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
