"""camdl's verdict on a method run, read from what camdl persisted.

camdl decides, once, whether a run converged: the verdict, the checks it made
and the findings it raised are computed by camdl, written into the method leaf,
and read back from there. This module is the one place camdl-scope reads them.
It holds no threshold: a colour or a status word here is a presentation of a
category camdl wrote (``converged``, ``not_converged``, a finding's severity),
never of a number compared against a bar of our own.

Where the verdict lives
-----------------------
Per method leaf (``<method dir>/seed_*``), camdl writes

* ``events.jsonl`` (schema ``camdl.events/v1``, run spec 10.11): one JSON line
  per event. The last line of a finished run is a ``method_report``, either
  ``completed`` with the method's report (the object ``camdl fit summary
  --format json`` gives under ``methods``, verdict included) or ``failed`` with
  the sentence the command printed. This is what we read: it is a file in the
  store like every other artifact the viewer reads, it exists for failed runs
  too, and it needs no subprocess while a poll is in flight.
* ``diagnostics.json``: the findings (read by :mod:`camdl_scope.ingest`).

A leaf written before ``events.jsonl`` existed carries no persisted verdict.
For those, and only those, we ask the installed ``camdl fit summary <leaf>
--format json`` (:func:`summary_for_leaf`), which is camdl computing the same
verdict; when camdl is not installed the verdict is reported as not recorded,
never filled in locally.

Number grammar
--------------
Summary schema v5, ``events.jsonl`` and ``diagnostics.json`` write a number as a
JSON number when finite, the string ``"NaN"``, ``"inf"`` or ``"-inf"`` when not,
and ``null`` when no value was computed (camdl inference spec 7.2.2; run spec
10.7). :func:`parse_number` reads exactly that. Summary schema v4 wrote ``null``
for a non-finite number too, so a v4 ``null`` could be NaN, an infinity or
absent; it is read as *unknown* (:data:`UNKNOWN`) and never as a number.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

EVENTS_SCHEMA = "camdl.events/v1"
#: Summary schema versions this reader accepts. 5 is the current grammar; 4 is
#: accepted because its only difference that matters here is the ambiguous null.
SUMMARY_VERSIONS_KNOWN = (4, 5)
SUMMARY_VERSION_CURRENT = 5

#: How a missing value is shown. v5 ``null`` means camdl did not compute it;
#: v4 ``null`` could also stand for a non-finite number.
NOT_COMPUTED = "not computed"
UNKNOWN = "unknown"

Status = Literal["ok", "warn", "error", "neutral"]


class UnsupportedSchema(ValueError):
    """A camdl artifact carries a schema version this reader does not know."""


def parse_number(raw: Any) -> float | None:
    """One value of camdl's number grammar: a JSON number, ``"NaN"``, ``"inf"``,
    ``"-inf"``, or ``None`` for not computed. Anything else is refused by name
    rather than coerced, so a changed grammar fails visibly."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(f"expected a number, got the boolean {raw!r}")
    if isinstance(raw, (int, float)):
        return float(raw)
    if raw == "NaN":
        return math.nan
    if raw == "inf":
        return math.inf
    if raw == "-inf":
        return -math.inf
    raise ValueError(f"expected a number, 'NaN', 'inf', '-inf' or null, got {raw!r}")


def number_text(v: float | None, *, legacy_null: bool = False, digits: int = 3) -> str:
    """A number for display. Non-finite values are named, a missing one is
    ``not computed`` (``unknown`` for a v4 null, which cannot say which)."""
    if v is None:
        return UNKNOWN if legacy_null else NOT_COMPUTED
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{digits}g}"


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What camdl concluded about one method leaf.

    ``kind`` is camdl's own tag for a completed run (``converged``,
    ``not_converged``, ``unassessable``, ``not_assessed``, ``not_applicable``)
    plus the states a leaf can be in without a report: ``failed`` (camdl
    recorded an error), ``running`` (no end-of-run report yet), ``not_recorded``
    (no persisted verdict and none obtainable) and ``unreadable`` (a camdl
    artifact this reader refused). ``status`` is the colour class of that kind.
    ``text`` is the words shown; for a run camdl reported on it is camdl's
    ``verdict_text`` when a summary supplied one.
    """

    kind: str
    status: Status
    text: str
    max_rhat: float | None = None
    param: str | None = None
    band: str | None = None
    params: tuple[str, ...] = ()
    error: str | None = None


def _status_of(kind: str, band: str | None) -> Status:
    # A presentation of camdl's category, not a comparison against a bar.
    if kind == "converged":
        return "ok"
    if kind == "not_converged":
        return "error" if band == "severe" else "warn"
    if kind == "unassessable":
        return "warn"
    return "neutral"


def verdict_from_report(raw: Any, *, legacy_null: bool = False) -> Verdict:
    """camdl's ``Verdict`` object (``{"kind": ..., ...}``) as it appears in a
    method report."""
    if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
        raise ValueError(f"a verdict is an object with a string 'kind', got {raw!r}")
    kind = raw["kind"]
    max_rhat = parse_number(raw.get("max_rhat"))
    param = raw.get("param")
    band = raw.get("band")
    params = tuple(str(p) for p in raw.get("params") or ())
    shown = number_text(max_rhat, legacy_null=legacy_null)
    on = f" on {param}" if param else ""
    if kind == "converged":
        text = f"converged: max R̂ {shown}{on}"
    elif kind == "not_converged":
        text = f"not converged: max R̂ {shown}{on}" + (f" ({band})" if band else "")
    elif kind == "unassessable":
        text = "R̂ could not be computed for " + (", ".join(params) or "some parameters")
    elif kind == "not_assessed":
        text = f"not assessed: {raw.get('reason', 'no reason recorded')}"
    elif kind == "not_applicable":
        text = "no convergence verdict for this method (no chains to compare)"
    else:
        raise UnsupportedSchema(f"camdl wrote a verdict kind this reader does not know: {kind!r}")
    return Verdict(
        kind=kind, status=_status_of(kind, band if isinstance(band, str) else None),
        text=text, max_rhat=max_rhat,
        param=param if isinstance(param, str) else None,
        band=band if isinstance(band, str) else None, params=params,
    )


def _verdict_with_gate(report: dict, base: Verdict) -> Verdict:
    """An optimizer leaf (IF2, NLopt) has no R-hat verdict; camdl's gate result
    (``gate.overall_pass``) is its verdict. ``overall_pass`` null means the gate
    did not decide."""
    gate = report.get("gate")
    if base.kind != "not_applicable" or not isinstance(gate, dict):
        return base
    passed = gate.get("overall_pass")
    if passed is True:
        return Verdict("gate_pass", "ok", "camdl's gate passed")
    if passed is False:
        return Verdict("gate_fail", "error", "camdl's gate failed")
    return base


# ---------------------------------------------------------------------------
# events.jsonl
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Events:
    """A parsed ``events.jsonl``: every event, plus the end-of-run report if the
    log has one."""

    events: tuple[dict, ...]
    method_report: dict | None


def read_events(leaf: Path) -> Events | None:
    """``leaf/events.jsonl``, or ``None`` when the file is absent.

    Every line must carry ``schema == "camdl.events/v1"``; a line of another
    schema refuses the file, naming the version found. A final line that does
    not parse is a write in progress (camdl writes each line whole, but the
    reader may arrive between two) and is skipped; an unparseable line anywhere
    else is an error."""
    path = leaf / "events.jsonl"
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    events: list[dict] = []
    for i, ln in enumerate(lines):
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError as e:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{path}: line {i + 1} is not JSON ({e})") from e
        if not isinstance(ev, dict):
            raise ValueError(f"{path}: line {i + 1} is not an object")
        if ev.get("schema") != EVENTS_SCHEMA:
            raise UnsupportedSchema(
                f"{path}: line {i + 1} has schema {ev.get('schema')!r}; "
                f"camdl-scope reads {EVENTS_SCHEMA!r}"
            )
        events.append(ev)
    report = next((e for e in reversed(events) if e.get("event") == "method_report"), None)
    return Events(tuple(events), report)


def verdict_from_events(ev: Events) -> Verdict:
    rep = ev.method_report
    if rep is None:
        return Verdict("running", "neutral",
                       "no end-of-run report yet (still running, or stopped without one)")
    outcome = rep.get("outcome")
    if outcome == "failed":
        err = str(rep.get("error") or "failed, no error recorded")
        return Verdict("failed", "error", f"failed: {err}", error=err)
    if outcome == "completed":
        report = rep.get("report")
        if not isinstance(report, dict) or "verdict" not in report:
            raise ValueError("method_report 'completed' carries no report.verdict")
        return _verdict_with_gate(report, verdict_from_report(report["verdict"]))
    raise UnsupportedSchema(f"method_report outcome {outcome!r} is not known")


# ---------------------------------------------------------------------------
# fit summary --format json
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeafEntry:
    """One row of a summary's ``leaves`` array: a method leaf and what camdl
    concluded about it, whether or not it completed."""

    leaf: str
    method: str
    run_id: str
    status: str  # completed | running | failed
    verdict: Verdict
    error: str | None = None


@dataclass(frozen=True)
class FitSummary:
    version: int
    camdl_version: str
    leaves: tuple[LeafEntry, ...]
    #: ``methods[].verdict`` per completed leaf, keyed by method name.
    reports: tuple[dict, ...] = ()
    failures: tuple[dict, ...] = ()


def parse_fit_summary(text: str) -> FitSummary:
    """A ``camdl fit summary --format json`` document.

    Refuses a schema version outside :data:`SUMMARY_VERSIONS_KNOWN`, naming it.
    Under v4 a ``null`` number is shown as *unknown*, under v5 as *not
    computed*; neither is ever turned into a number or into an ``ok``."""
    doc = json.loads(text)
    schema = doc.get("schema") if isinstance(doc, dict) else None
    version = schema.get("version") if isinstance(schema, dict) else None
    if version not in SUMMARY_VERSIONS_KNOWN:
        raise UnsupportedSchema(
            f"fit summary schema version {version!r} is not supported; camdl-scope reads "
            f"versions {', '.join(map(str, SUMMARY_VERSIONS_KNOWN))}"
        )
    legacy = version == 4
    leaves: list[LeafEntry] = []
    for row in doc.get("leaves", []):
        status = str(row.get("status"))
        err = row.get("error")
        raw_v = row.get("verdict")
        st = row.get("state")
        state = st.get("state") if isinstance(st, dict) else None
        if raw_v is None:
            if state == "failed" and isinstance(st.get("failure"), dict):
                err = st["failure"].get("message") or err
            if state == "interrupted":
                v = Verdict("interrupted", "neutral",
                            "interrupted: stopped without recording an outcome")
            elif state == "failed_reason_unknown":
                v = Verdict("failed", "error", "failed (camdl recorded no reason)")
            else:
                kind = "failed" if (status == "failed" or state == "failed") else (
                    "running" if status == "running" else "not_recorded")
                v = Verdict(kind, "error" if kind == "failed" else "neutral",
                            str(row.get("verdict_text") or kind), error=err)
        else:
            base = verdict_from_report(raw_v, legacy_null=legacy)
            text = str(row.get("verdict_text") or base.text)
            if base.kind in ("converged", "not_converged") and base.max_rhat is None:
                text += f" (max R\u0302 {number_text(None, legacy_null=legacy)})"
            v = Verdict(**{**base.__dict__, "text": text})
        leaves.append(LeafEntry(
            leaf=str(row.get("leaf")), method=str(row.get("method")),
            run_id=str(row.get("run_id")), status=status, verdict=v,
            error=err if isinstance(err, str) else None,
        ))
    return FitSummary(
        version=version, camdl_version=str(schema.get("camdl_version", "")),
        leaves=tuple(leaves), reports=tuple(doc.get("methods", [])),
        failures=tuple(doc.get("failures", [])),
    )


@lru_cache(maxsize=256)
def _summary_text(leaf: str, stamp: float, camdl: str) -> str:
    out = subprocess.run(
        [camdl, "fit", "summary", leaf, "--format", "json"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip().splitlines()[-1] if out.stderr.strip()
                           else f"camdl fit summary exited {out.returncode}")
    return out.stdout


def summary_for_leaf(leaf: Path) -> LeafEntry | None:
    """camdl's own summary of a leaf, for a leaf with no ``events.jsonl``.

    ``None`` when camdl is not on PATH. Results are cached on the leaf's
    ``run.json`` mtime, so a poll does not re-run camdl for an unchanged leaf.
    Raises on a camdl failure or an unknown schema; the caller turns that into a
    visible ``unreadable`` verdict."""
    camdl = shutil.which("camdl")
    if camdl is None:
        return None
    try:
        stamp = (leaf / "run.json").stat().st_mtime
    except OSError:
        stamp = 0.0
    doc = parse_fit_summary(_summary_text(str(leaf), stamp, camdl))
    if len(doc.leaves) != 1:
        raise ValueError(f"expected one leaf from camdl fit summary, got {len(doc.leaves)}")
    return doc.leaves[0]


# ---------------------------------------------------------------------------
# One leaf, as the dashboard shows it
# ---------------------------------------------------------------------------


def read_run_status(leaf: Path) -> str:
    """``completed`` / ``running`` / ``failed`` as the leaf's ``run.json`` records
    it, or ``unknown`` when it is absent or unreadable."""
    try:
        rec = json.loads((leaf / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        return "unknown"
    s = rec.get("status") if isinstance(rec, dict) else None
    return s if s in ("completed", "running", "failed") else "unknown"


def read_run_failure(leaf: Path) -> str | None:
    """``failure.message`` from a leaf's ``run.json`` when it records
    ``status: failed``: the error camdl stopped on, written before it exited.
    ``None`` otherwise, including a ``failed`` record with no ``failure`` (a
    camdl before the field, or a panic)."""
    try:
        rec = json.loads((leaf / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rec, dict) or rec.get("status") != "failed":
        return None
    f = rec.get("failure")
    msg = f.get("message") if isinstance(f, dict) else None
    return msg if isinstance(msg, str) and msg else None


def leaf_verdict(leaf: Path, *, ask_camdl: bool = True) -> Verdict:
    """The verdict for one method leaf. Never raises: a camdl artifact this
    reader refuses becomes a visible ``unreadable`` verdict carrying the
    message, so one bad leaf neither hides itself nor breaks its neighbours.

    ``ask_camdl`` allows the ``camdl fit summary`` fallback for a leaf with no
    ``events.jsonl``. A listing of many leaves passes ``False`` so that opening
    a folder never starts one subprocess per old leaf."""
    # camdl records a failure in run.json before it exits; that record is
    # preferred over the events log's failed report.
    failure = read_run_failure(leaf)
    if failure is not None:
        return Verdict("failed", "error", f"failed: {failure}", error=failure)
    try:
        ev = read_events(leaf)
        if ev is not None:
            return verdict_from_events(ev)
        entry = summary_for_leaf(leaf) if ask_camdl else None
        if entry is not None:
            return entry.verdict
    except (UnsupportedSchema, ValueError, RuntimeError, OSError,
            subprocess.SubprocessError, json.JSONDecodeError) as e:
        return Verdict("unreadable", "neutral", f"verdict not read: {e}", error=str(e))
    status = read_run_status(leaf)
    if status == "failed":
        return Verdict("failed", "error",
                       "failed (camdl recorded no error text: this leaf has no events.jsonl)")
    if status == "running":
        return Verdict("running", "neutral", "running")
    return Verdict(
        "not_recorded", "neutral",
        "camdl's verdict was not persisted for this leaf (no events.jsonl) and the camdl "
        "binary was not available to compute it; run `camdl fit summary` on the leaf",
    )


@dataclass(frozen=True)
class LeafRow:
    """A method leaf of a problem folder, for the leaf table."""

    leaf: str  # path under the folder, e.g. ``pmmh-83841c1b/seed_1-06cbd6b3``
    method: str
    run_id: str | None
    status: str
    verdict: Verdict


def _method_of(leaf: Path) -> str:
    """The method label: the leaf's method dir name up to its hash suffix
    (``pmmh-83841c1b`` is ``pmmh``). Display only; the verdict never depends
    on it."""
    name = leaf.parent.name
    head, _, tail = name.rpartition("-")
    return head if head and len(tail) == 8 else name


def leaf_rows(folder: Path) -> list[LeafRow]:
    """Every method leaf under a problem folder: completed, running or failed,
    each with its own status and verdict. None is chosen over another; a folder
    holding an IF2 run and a PMMH run lists both."""
    rows: list[LeafRow] = []
    try:
        method_dirs = sorted(e for e in folder.iterdir() if e.is_dir())
    except OSError:
        return rows
    for md in method_dirs:
        for sd in sorted(md.glob("seed_*")):
            if not sd.is_dir():
                continue
            try:
                rid = json.loads((sd / "run.json").read_text()).get("run_id")
            except (OSError, json.JSONDecodeError, AttributeError):
                rid = None
            verdict = leaf_verdict(sd, ask_camdl=False)
            status = read_run_status(sd)
            # camdl's end-of-run report is authoritative about a failure: a
            # leaf that failed during init keeps ``status = "running"`` in its
            # run.json (seen with camdl 0.1.0+0553b5a1) while events.jsonl
            # records the failed method_report.
            if verdict.kind == "failed":
                status = "failed"
            rows.append(LeafRow(
                leaf=f"{md.name}/{sd.name}", method=_method_of(sd),
                run_id=rid if isinstance(rid, str) else None,
                status=status, verdict=verdict,
            ))
    return rows


# ---------------------------------------------------------------------------
# Per-parameter colour, from camdl's findings
# ---------------------------------------------------------------------------

#: The finding camdl raises for a parameter whose R-hat is high. Its severity is
#: camdl's; this module only looks the parameter up.
RHAT_FINDING = "rhat_high"


def rhat_status(param: str, findings: list, verdict: Verdict | None) -> Status:
    """The colour class of one parameter's R-hat cell, from camdl alone.

    A parameter camdl raised an ``rhat_high`` finding for takes that finding's
    severity. Otherwise the parameter the verdict names as the worst takes the
    verdict's status. Every other parameter is ``neutral``: camdl raised nothing
    for it, which is not the same as camdl having certified it. ``findings`` are
    :class:`camdl_scope.state.Finding` objects."""
    worst: Status = "neutral"
    rank = {"neutral": 0, "ok": 0, "warn": 1, "error": 2}
    for f in findings:
        if f.kind == RHAT_FINDING and f.param == param:
            s: Status = "error" if f.severity.value == "error" else "warn"
            if rank[s] > rank[worst]:
                worst = s
    if worst == "neutral" and verdict is not None and verdict.param == param \
            and verdict.status in ("warn", "error"):
        return verdict.status
    return worst


# ---------------------------------------------------------------------------
# The acceptance band camdl applies
# ---------------------------------------------------------------------------


def band_from(raw: Any) -> tuple[float, float] | None:
    """``(band_lo, band_hi)`` from an object holding both, else ``None``."""
    if not isinstance(raw, dict):
        return None
    try:
        lo, hi = parse_number(raw.get("band_lo")), parse_number(raw.get("band_hi"))
    except ValueError:
        return None
    return (lo, hi) if lo is not None and hi is not None else None


def acceptance_band(leaf: Path, findings: list) -> tuple[float, float] | None:
    """The healthy acceptance band camdl applied to this leaf, or ``None``.

    camdl publishes it as values on the ``begin`` line of ``events.jsonl``
    (``acceptance: {kernel, band_lo, band_hi}``, or ``null`` for a method with
    no acceptance diagnostics) and on each ``acceptance_rate_unhealthy`` finding
    (``band_lo``, ``band_hi``). Both are optional: a leaf written by an older
    camdl carries neither, and then no band is drawn. Nothing is assumed."""
    try:
        ev = read_events(leaf)
    except (UnsupportedSchema, ValueError, OSError):
        ev = None
    if ev is not None:
        begin = next((e for e in ev.events if e.get("event") == "begin"), None)
        band = band_from(begin.get("acceptance")) if begin else None
        if band is not None:
            return band
    for f in findings:
        if f.kind == "acceptance_rate_unhealthy":
            band = band_from(f.detail)
            if band is not None:
                return band
    return None
