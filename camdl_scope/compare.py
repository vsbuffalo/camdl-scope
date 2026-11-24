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
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import ingest, store

# The camdl binary to shell out to; override for a non-PATH install / tests.
CAMDL_BIN = os.environ.get("CAMDL_BIN", "camdl")

#: The score a particle-filter run (or a fit run with ``record_prequential``)
#: writes in its own folder.
PREQUENTIAL_FILE = "prequential.json"


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


def find_prequential(leaf: Path) -> Path | None:
    """The fit run's ``prequential.json``, or ``None`` if it scored none."""
    p = Path(leaf) / PREQUENTIAL_FILE
    return p if p.is_file() else None


@dataclass(frozen=True)
class ScoredRun:
    """A scoring run: a fit run whose method is the particle filter, which has
    a ``prequential.json`` and neither draws nor a point estimate, so the run
    list never shows it; this finds it by its score. ``run_id`` is
    ``<problem folder>~<run folder>``, the form the run list's ids take."""

    run_id: str
    label: str
    leaf: Path
    path: Path
    mtime: float


def discover_scored(root: Path) -> list[ScoredRun]:
    """Every scoring run in the store (newest first). Its label is the stem of
    the fit file that made it."""
    out: list[ScoredRun] = []
    for kind_dir in store.kind_dirs(root, ingest.FIT_KIND):
        for problem in store.walk(kind_dir, 1):
            for leaf in store.walk(problem / ingest.RUNS_DIR, ingest.RUN_DEPTH):
                pq = find_prequential(leaf)
                ident = store.read_identity(leaf)
                record = store.read_run_record(leaf)
                if pq is None or ident is None or record is None:
                    continue
                method = ident.body.get("chain", {}).get("method", {})
                if method.get("algorithm") not in ingest.SCORING_ALGORITHMS:
                    continue
                config = record.given.get("config")
                stem = Path(config["path"]).stem if isinstance(config, dict) and config.get("path") else problem.name
                out.append(ScoredRun(
                    f"{problem.name}{ingest.LEAF_SEP}{leaf.name}", stem, leaf, pq, pq.stat().st_mtime,
                ))
    return sorted(out, key=lambda r: r.mtime, reverse=True)


# camdl's JSON carries its advisories as named top-level fields (null when they
# do not apply), in this reading order: what the scores are, whether the data
# and filter settings were checked to match, then how to read the SE and the
# evidence words.
_CAVEAT_FIELDS = (
    "optimism_caveat",
    "data_mismatch",
    "data_changed",
    "data_unchecked",
    "data_change_unchecked",
    "data_compared_on_prefix",
    "filter_settings_mismatch",
    "filter_settings_unchecked",
    "filter_noise_unassessed",
    "se_caveat",
    "evidence_scale_note",
)


def json_notes(data: dict) -> list[str]:
    """camdl's advisories from its JSON: the top-level caveat fields that are
    set, each row's lag-1 autocorrelation of per-step Δelpd (which makes the
    paired SE optimistic when large), and each row's warnings by kind. Read and
    worded here, never judged: camdl decides which caveats apply."""
    out = [str(data[k]) for k in _CAVEAT_FIELDS if isinstance(data.get(k), str) and data[k]]
    for r in data.get("rows", []):
        name = r.get("name", "?")
        lag1 = r.get("lag1_autocorr_delta_elpd")
        if isinstance(lag1, (int, float)):
            out.append(
                f"lag-1 autocorrelation of the per-step Δelpd vs baseline: {name} {lag1:+.2f}"
            )
        for w in r.get("warnings") or []:
            kind = w.get("kind") if isinstance(w, dict) else w
            if kind:
                out.append(f"{name}: {str(kind).replace('_', ' ')}")
    return out


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
    were not commensurable. ``notes`` carries camdl's advisories: its JSON caveat
    fields and row warnings (:func:`json_notes`), plus any stderr ``note:`` lines. Raises :class:`CompareError` on any other failure.
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
    ] + json_notes(data)
    # If the caller allowed mismatches up front, derive commensurability from the
    # actual T_scores (the refusal path that would have set it was skipped).
    if allow_mismatched:
        commensurable = len({row.get("t_score") for row in data.get("rows", [])}) <= 1
    return data, commensurable, notes
