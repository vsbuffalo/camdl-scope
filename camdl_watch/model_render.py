"""Structured model math for display — the ``model.render.json`` artifact.

camdl writes a segment-level ``model.render.json`` (a sibling of
``model.ir.json``, present for any run — fit or sim) whose every math string is a
standalone KaTeX-renderable expression. The viewer renders the leaves client-side
and owns the layout, so no server-side math rendering is needed.

This reader is a pure pass-through: it reads and JSON-parses the file (or returns
``None`` when absent / unreadable). Interpretation into typed fields belongs to
the API layer's :class:`camdl_watch.api.models.ModelRender`, which validates the
parsed dict — additive and forward-compatible: unknown keys are ignored, missing
optional sections default to empty.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_RENDER_FILE = "model.render.json"


def has_model_render(run_dir: Path) -> bool:
    """Whether the run carries a ``model.render.json`` to render."""
    return (Path(run_dir) / MODEL_RENDER_FILE).is_file()


def read_model_render(run_dir: Path) -> dict | None:
    """The parsed ``model.render.json`` for a run, or ``None`` when it's absent
    or unparseable (an older run predating the artifact — the viewer falls back
    to the raw source)."""
    p = Path(run_dir) / MODEL_RENDER_FILE
    if not p.is_file():
        return None
    try:
        parsed = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
