"""Model comparison via the authoritative ``camdl compare``.

The watcher never recomputes prequential scores. It locates each run's
``prequential.json`` (written by a pfilter stage) and shells out to ``camdl
compare`` — the single source of truth for the elpd / Δelpd paired-SE math and
the Jeffreys/decibans evidence scale — then projects its JSON onto the wire.

Models are passed through a generated ``compare.toml`` so each carries a stable
name (the run id). Passing bare paths would name *every* model
``prequential.json`` (camdl derives the row name from the file name), colliding
the rows and the ``--baseline`` lookup.

The comparability guard is camdl's: it refuses (exit 2) when ``T_score`` differs
across models — "Δelpd and Δcrps are not commensurable" — unless
``--allow-mismatched-horizon``. We surface that refusal as ``commensurable=False``
rather than hiding it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# The camdl binary to shell out to; override for a non-PATH install / tests.
CAMDL_BIN = os.environ.get("CAMDL_BIN", "camdl")

# Search depths under a run dir for a pfilter stage's prequential.json. Bounded
# (no full-tree rglob) so listing many runs stays cheap; camdl writes the stage
# as a child of the run dir, so depth ≤4 covers stage/seed/chain layouts.
_PREQUENTIAL_GLOBS = (
    "prequential.json",
    "*/prequential.json",
    "*/*/prequential.json",
    "*/*/*/prequential.json",
)


class CompareError(RuntimeError):
    """``camdl compare`` failed for a reason worth showing the user."""


@dataclass(frozen=True)
class CompareSpec:
    """One model in a comparison: a stable ``name`` (the run id, which becomes
    the row name) and its ``prequential.json`` ``path``."""

    name: str
    path: Path


def camdl_available() -> bool:
    """Whether the ``camdl`` binary is on PATH (the compare backend needs it)."""
    return shutil.which(CAMDL_BIN) is not None


def find_prequential(run_dir: Path) -> Path | None:
    """The run's ``prequential.json`` (from a pfilter stage); newest wins, or
    ``None`` if the run never scored one. Bounded-depth search (see globs)."""
    for pat in _PREQUENTIAL_GLOBS:
        try:
            cands = sorted(
                run_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True
            )
        except OSError:
            continue
        if cands:
            return cands[0]
    return None


def _write_compare_toml(
    specs: list[CompareSpec], baseline: str | None, metrics: list[str] | None
) -> str:
    """A ``compare.toml`` with ``format = "json"`` and one ``[[model]]`` per spec.
    String values go through ``json.dumps`` — a valid TOML basic string for
    filesystem paths (double-quoted, backslash/quote escaped)."""
    lines = ['format = "json"']
    if baseline is not None:
        lines.append(f"baseline = {json.dumps(baseline)}")
    if metrics:
        lines.append("metrics = [" + ", ".join(json.dumps(m) for m in metrics) + "]")
    lines.append("")
    for s in specs:
        lines += [
            "[[model]]",
            f"name = {json.dumps(s.name)}",
            f"path = {json.dumps(str(s.path.resolve()))}",
            "",
        ]
    return "\n".join(lines)


def run_compare(
    specs: list[CompareSpec],
    baseline: str | None = None,
    allow_mismatched: bool = False,
    metrics: list[str] | None = None,
) -> tuple[dict, bool, list[str]]:
    """Invoke ``camdl compare`` over ``specs``; return ``(json, commensurable,
    notes)``.

    Runs once without the horizon override. If camdl refuses on a ``T_score``
    mismatch (exit 2), retries WITH ``--allow-mismatched-horizon`` so the caller
    still receives absolute scores (Δ columns ``null``) and learns the models
    were not commensurable. ``notes`` carries stderr advisories (the optimism
    caveat). Raises :class:`CompareError` on any other failure.
    """
    if len(specs) < 2:
        raise CompareError("need at least two models to compare")

    toml_text = _write_compare_toml(specs, baseline, metrics)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".compare.toml", delete=False
    ) as fh:
        fh.write(toml_text)
        toml_path = fh.name

    def invoke(allow: bool) -> subprocess.CompletedProcess[str]:
        cmd = [CAMDL_BIN, "compare", "--config", toml_path, "--no-progress"]
        if allow:
            cmd.append("--allow-mismatched-horizon")
        return subprocess.run(cmd, capture_output=True, text=True)

    try:
        proc = invoke(allow_mismatched)
        commensurable = True
        if proc.returncode == 2 and not allow_mismatched:
            commensurable = False  # T_score mismatch refusal
            proc = invoke(True)
        if proc.returncode != 0:
            raise CompareError(
                proc.stderr.strip() or f"camdl compare exited {proc.returncode}"
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise CompareError(f"could not parse camdl compare output: {e}") from e
    finally:
        try:
            os.unlink(toml_path)
        except OSError:
            pass

    notes = [
        line.strip()[len("note:") :].strip()
        for line in proc.stderr.splitlines()
        if line.strip().startswith("note:")
    ]
    # If the caller allowed mismatches up front, derive commensurability from the
    # actual T_scores (the refusal path that would have set it was skipped).
    if allow_mismatched:
        commensurable = len({row.get("t_score") for row in data.get("rows", [])}) <= 1
    return data, commensurable, notes
