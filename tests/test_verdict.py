"""camdl's verdict is read, never recomputed.

The dashboard shows what camdl persisted (``events.jsonl``, ``diagnostics.json``,
or ``camdl fit summary --format json``) and holds no threshold of its own. These
tests pin that: a converged verdict stands over an R-hat the dashboard would
once have called an error, a missing verdict is shown as missing, and camdl's
number grammar (number | "NaN" | "inf" | "-inf" | null) is read without loss.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from camdl_scope import verdict as V
from camdl_scope.state import Finding, Severity
from tests.fixtures.make_golden_store import POSTERIOR_DIR, RUN_DIR, SEED_DIR, build

EV = {"schema": "camdl.events/v1"}


def _write_events(leaf: Path, *events: dict) -> None:
    leaf.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({**EV, "seq": i, "elapsed": float(i), **e}) for i, e in enumerate(events)]
    (leaf / "events.jsonl").write_text("\n".join(lines) + "\n")


def _report(verdict: dict, **extra) -> dict:
    return {"event": "method_report", "outcome": "completed",
            "report": {"name": "pgas", "method": "pgas", "verdict": verdict, **extra}}


# ---------------------------------------------------------------------------
# The number grammar
# ---------------------------------------------------------------------------


def test_number_grammar():
    assert V.parse_number(1.25) == 1.25
    assert V.parse_number(3) == 3.0
    assert math.isnan(V.parse_number("NaN"))
    assert V.parse_number("inf") == math.inf
    assert V.parse_number("-inf") == -math.inf
    assert V.parse_number(None) is None


@pytest.mark.parametrize("bad", ["1.2", "Infinity", "nan", True, [1], {"v": 1}])
def test_number_grammar_refuses_anything_else(bad):
    with pytest.raises(ValueError):
        V.parse_number(bad)


def test_missing_and_non_finite_values_read_differently():
    assert V.number_text(math.nan) == "NaN"
    assert V.number_text(math.inf) == "inf"
    assert V.number_text(-math.inf) == "-inf"
    assert V.number_text(None) == "not computed"
    # A v4 null cannot say which of the three it was, nor that it was absent.
    assert V.number_text(None, legacy_null=True) == "unknown"
    assert V.number_text(1.0123456) == "1.012"
    assert V.number_text(1.0022154450301082) == "1.002"  # not "1"


# ---------------------------------------------------------------------------
# fit summary --format json
# ---------------------------------------------------------------------------


def _summary_doc(version: int, max_rhat) -> str:
    return json.dumps({
        "schema": {"version": version, "camdl_version": "0.2.0+test"},
        "fit_dir": "fit/x",
        "leaves": [
            {"leaf": "fit/x/pmmh-aaaaaaaa/seed_1-bbbbbbbb", "method": "pmmh",
             "run_id": "r1", "status": "completed",
             "verdict": {"kind": "not_converged", "max_rhat": max_rhat,
                         "param": "beta", "band": "severe"},
             "verdict_text": "not converged"},
            {"leaf": "fit/x/if2-cccccccc/seed_1-dddddddd", "method": "if2",
             "run_id": "r2", "status": "failed", "verdict": None,
             "verdict_text": "failed", "error": "every chain failed to start"},
            {"leaf": "fit/x/pgas-eeeeeeee/seed_1-ffffffff", "method": "pgas",
             "run_id": "r3", "status": "running", "verdict": None,
             "verdict_text": "running"},
        ],
        "table_rows": [], "methods": [], "failures": [],
    })


def test_summary_v5_lists_every_leaf_and_reads_non_finite_numbers():
    doc = V.parse_fit_summary(_summary_doc(5, "inf"))
    assert [(e.method, e.status) for e in doc.leaves] == [
        ("pmmh", "completed"), ("if2", "failed"), ("pgas", "running")]
    pmmh, if2, pgas = (e.verdict for e in doc.leaves)
    assert pmmh.max_rhat == math.inf and pmmh.status == "error"
    assert if2.kind == "failed" and if2.error == "every chain failed to start"
    assert pgas.kind == "running" and pgas.status == "neutral"


def test_summary_v5_null_is_not_computed():
    doc = V.parse_fit_summary(_summary_doc(5, None))
    assert doc.leaves[0].verdict.max_rhat is None
    assert "not computed" in doc.leaves[0].verdict.text


def test_summary_v4_null_is_unknown_never_a_number_or_ok():
    doc = V.parse_fit_summary(_summary_doc(4, None))
    v = doc.leaves[0].verdict
    assert v.max_rhat is None and v.status != "ok"
    assert "unknown" in v.text


@pytest.mark.parametrize("version", [3, 6, None, "5"])
def test_summary_refuses_an_unknown_schema_version_by_name(version):
    with pytest.raises(V.UnsupportedSchema) as e:
        V.parse_fit_summary(_summary_doc(version, 1.0))
    assert repr(version) in str(e.value) and "4, 5" in str(e.value)


def test_unknown_verdict_kind_is_refused_not_guessed():
    with pytest.raises(V.UnsupportedSchema, match="mostly_fine"):
        V.verdict_from_report({"kind": "mostly_fine"})


# ---------------------------------------------------------------------------
# events.jsonl
# ---------------------------------------------------------------------------


def test_completed_report_is_camdls_verdict(tmp_path):
    _write_events(tmp_path, {"event": "begin", "method": "pgas"},
                  _report({"kind": "converged", "max_rhat": 1.03, "param": "beta"}))
    v = V.leaf_verdict(tmp_path)
    assert (v.kind, v.status, v.param) == ("converged", "ok", "beta")
    assert v.max_rhat == pytest.approx(1.03)


def test_not_converged_severe_band_is_error_and_plain_band_is_warn(tmp_path):
    _write_events(tmp_path / "a", _report(
        {"kind": "not_converged", "max_rhat": 2.0, "param": "b", "band": "severe"}))
    _write_events(tmp_path / "b", _report(
        {"kind": "not_converged", "max_rhat": 1.07, "param": "b", "band": "not_converged"}))
    assert V.leaf_verdict(tmp_path / "a").status == "error"
    assert V.leaf_verdict(tmp_path / "b").status == "warn"


def test_infinite_rhat_in_a_verdict_is_read_as_infinity(tmp_path):
    _write_events(tmp_path, _report(
        {"kind": "not_converged", "max_rhat": "inf", "param": "b", "band": "severe"}))
    v = V.leaf_verdict(tmp_path)
    assert v.max_rhat == math.inf and "inf" in v.text


def test_failed_report_carries_camdls_error(tmp_path):
    _write_events(tmp_path, {"event": "begin", "method": "pmmh"},
                  {"event": "method_report", "outcome": "failed",
                   "error": "no chain reached a finite log-likelihood"})
    v = V.leaf_verdict(tmp_path)
    assert v.kind == "failed" and v.status == "error"
    assert v.error == "no chain reached a finite log-likelihood"


def test_log_without_report_is_running_not_a_verdict(tmp_path):
    _write_events(tmp_path, {"event": "begin", "method": "pgas"})
    v = V.leaf_verdict(tmp_path)
    assert v.kind == "running" and v.status == "neutral"


def test_a_line_being_written_is_skipped_but_a_corrupt_one_is_not(tmp_path):
    _write_events(tmp_path / "a", {"event": "begin", "method": "pgas"})
    with (tmp_path / "a" / "events.jsonl").open("a") as f:
        f.write('{"schema": "camdl.events/v1", "seq": 1, "ev')
    assert V.leaf_verdict(tmp_path / "a").kind == "running"

    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "events.jsonl").write_text(
        'not json\n' + json.dumps({**EV, "seq": 1, "event": "begin"}) + "\n")
    assert V.leaf_verdict(tmp_path / "b").kind == "unreadable"


def test_unknown_events_schema_is_refused_naming_it(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"schema": "camdl.events/v2", "seq": 0, "event": "begin"}) + "\n")
    v = V.leaf_verdict(tmp_path)
    assert v.kind == "unreadable" and "camdl.events/v2" in v.text


def test_optimizer_leaf_takes_camdls_gate_not_a_rhat_verdict(tmp_path):
    _write_events(tmp_path, _report({"kind": "not_applicable"}, gate={"overall_pass": False}))
    v = V.leaf_verdict(tmp_path)
    assert (v.kind, v.status) == ("gate_fail", "error")


def test_no_events_and_no_camdl_is_not_recorded_never_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(V.shutil, "which", lambda _: None)
    (tmp_path / "run.json").write_text(json.dumps({"status": "completed"}))
    v = V.leaf_verdict(tmp_path)
    assert v.kind == "not_recorded" and v.status == "neutral"


def test_no_events_falls_back_to_camdls_own_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(V.shutil, "which", lambda _: "/fake/camdl")
    V._summary_text.cache_clear()
    doc = json.loads(_summary_doc(5, 1.02))
    doc["leaves"] = [doc["leaves"][0]]
    monkeypatch.setattr(V, "_summary_text", lambda leaf, stamp, camdl: json.dumps(doc))
    v = V.leaf_verdict(tmp_path)
    assert v.kind == "not_converged" and v.max_rhat == pytest.approx(1.02)
    # a listing never starts a subprocess per leaf
    assert V.leaf_verdict(tmp_path, ask_camdl=False).kind == "not_recorded"


def test_an_unknown_summary_version_from_camdl_is_shown_as_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(V.shutil, "which", lambda _: "/fake/camdl")
    monkeypatch.setattr(V, "_summary_text", lambda leaf, stamp, camdl: _summary_doc(9, 1.0))
    v = V.leaf_verdict(tmp_path)
    assert v.kind == "unreadable" and "9" in v.text


# ---------------------------------------------------------------------------
# Every leaf of a folder
# ---------------------------------------------------------------------------


def test_leaf_rows_lists_completed_failed_and_running_leaves(tmp_path):
    ok = tmp_path / "pgas-11111111" / "seed_1-aaaaaaaa"
    bad = tmp_path / "pmmh-22222222" / "seed_1-bbbbbbbb"
    live = tmp_path / "if2-33333333" / "seed_1-cccccccc"
    _write_events(ok, _report({"kind": "converged", "max_rhat": 1.01, "param": "b"}))
    (ok / "run.json").write_text(json.dumps({"run_id": "rid-ok", "status": "completed"}))
    _write_events(bad, {"event": "method_report", "outcome": "failed", "error": "boom"})
    (bad / "run.json").write_text(json.dumps({"run_id": "rid-bad", "status": "failed"}))
    _write_events(live, {"event": "begin", "method": "if2"})
    (live / "run.json").write_text(json.dumps({"run_id": "rid-live", "status": "running"}))
    rows = V.leaf_rows(tmp_path)
    assert [(r.method, r.status, r.verdict.kind) for r in rows] == [
        ("if2", "running", "running"),
        ("pgas", "completed", "converged"),
        ("pmmh", "failed", "failed"),
    ]
    assert {r.run_id for r in rows} == {"rid-ok", "rid-bad", "rid-live"}


# ---------------------------------------------------------------------------
# Per-parameter colour
# ---------------------------------------------------------------------------


def _finding(param, severity):
    return Finding(kind="rhat_high", severity=severity, message="", param=param, detail={})


def test_param_colour_comes_from_camdls_finding_severity():
    fs = [_finding("a", Severity.ERROR), _finding("b", Severity.WARN)]
    assert V.rhat_status("a", fs, None) == "error"
    assert V.rhat_status("b", fs, None) == "warn"
    # camdl raised nothing for c: neutral, not certified
    assert V.rhat_status("c", fs, None) == "neutral"


def test_param_colour_has_no_threshold_of_ours():
    # R-hat 1.5 is not in the arguments at all: with no finding and a converged
    # verdict the parameter is neutral whatever its number.
    conv = V.Verdict("converged", "ok", "converged", max_rhat=1.01, param="a")
    assert V.rhat_status("a", [], conv) == "neutral"
    worst = V.Verdict("not_converged", "warn", "x", max_rhat=1.08, param="a",
                      band="not_converged")
    assert V.rhat_status("a", [], worst) == "warn"


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    build(tmp_path)
    monkeypatch.setenv("CAMDL_SCOPE_STORE", str(tmp_path))
    return tmp_path


def _client() -> TestClient:
    from camdl_scope.api.app import app

    return TestClient(app)


def _seed(store: Path) -> Path:
    return store / RUN_DIR / POSTERIOR_DIR / SEED_DIR


def test_golden_run_shows_the_verdict_camdl_wrote(store):
    c = _client()
    d = c.get(f"/api/runs/{RUN_DIR}/diagnostics").json()
    assert d["verdict"]["kind"] == "converged" and d["verdict"]["status"] == "ok"
    assert [(r["method"], r["status"], r["is_open"]) for r in d["leaves"]] == [
        ("pgas", "completed", True)]
    assert c.get(f"/api/runs/{RUN_DIR}").json()["verdict"]["kind"] == "converged"


def test_a_converged_verdict_stands_over_a_high_rhat_number(store):
    """The dashboard once coloured R-hat above 1.1 as an error on its own terms.
    Here camdl says converged and its summary carries an R-hat of 1.5: the
    number is shown, and no status is derived from it."""
    p = _seed(store) / "pgas_summary.json"
    s = json.loads(p.read_text())
    s["rhat"]["beta"] = 1.5
    p.write_text(json.dumps(s))
    c = _client()
    params = c.get(f"/api/runs/{RUN_DIR}/diagnostics?warmup_pct=50").json()["params"]
    beta = next(x for x in params if x["name"] == "beta")
    assert beta["rhat"] == pytest.approx(1.5)
    assert beta["rhat_status"] == "neutral"
    post = c.get(f"/api/runs/{RUN_DIR}/posterior").json()["params"]
    assert next(x for x in post if x["name"] == "beta")["rhat_status"] == "neutral"


def test_a_flagged_parameter_takes_camdls_severity(store):
    (_seed(store) / "diagnostics.json").write_text(json.dumps([{
        "kind": {"type": "rhat_high", "param": "beta", "rhat": 1.2, "threshold": 1.1},
        "severity": "warning", "message": "Rhat for 'beta' is 1.2.",
        "method": "pgas", "timestamp": "2026-11-10T00:00:00Z"}]))
    c = _client()
    post = c.get(f"/api/runs/{RUN_DIR}/posterior").json()["params"]
    by = {x["name"]: x["rhat_status"] for x in post if not x["is_objective"]}
    assert by["beta"] == "warn"
    assert all(v == "neutral" for k, v in by.items() if k != "beta")


def test_an_infinite_rhat_reaches_the_wire_as_inf_text(store):
    p = _seed(store) / "pgas_summary.json"
    s = json.loads(p.read_text())
    s["rhat"]["beta"] = "inf"
    p.write_text(json.dumps(s))
    params = _client().get(f"/api/runs/{RUN_DIR}/diagnostics?warmup_pct=50").json()["params"]
    beta = next(x for x in params if x["name"] == "beta")
    assert beta["rhat"] is None and beta["rhat_text"] == "inf"


def test_a_leaf_without_a_persisted_verdict_is_shown_as_unrecorded(store, monkeypatch):
    (_seed(store) / "events.jsonl").unlink()
    monkeypatch.setattr(V.shutil, "which", lambda _: None)
    d = _client().get(f"/api/runs/{RUN_DIR}/diagnostics").json()
    assert d["verdict"]["kind"] == "not_recorded" and d["verdict"]["status"] == "neutral"


def test_a_run_camdl_has_not_assessed_gets_no_local_findings(store):
    """No summary and no end-of-run report (still sampling): the dashboard shows
    that camdl has not assessed it, and invents no warnings from its own live
    R-hat or ESS."""
    seed = _seed(store)
    (seed / "pgas_summary.json").unlink()
    _write_events(seed, {"event": "begin", "method": "pgas"})
    d = _client().get(f"/api/runs/{RUN_DIR}/diagnostics").json()
    assert d["verdict"]["kind"] == "running"
    assert d["findings"] == []
    assert all(p["rhat_status"] == "neutral" for p in d["params"])


def test_a_failed_report_overrides_a_run_json_still_saying_running(tmp_path):
    """camdl 0.1.0+0553b5a1 leaves ``status = "running"`` in the run.json of a
    leaf that failed at chain init, while events.jsonl records the failed
    method_report. The leaf is listed as failed."""
    leaf = tmp_path / "pmmh-11111111" / "seed_1-aaaaaaaa"
    _write_events(leaf, {"event": "method_report", "outcome": "failed", "error": "boom"})
    (leaf / "run.json").write_text(json.dumps({"run_id": "r", "status": "running"}))
    [row] = V.leaf_rows(tmp_path)
    assert row.status == "failed" and row.verdict.error == "boom"


def test_run_json_failure_is_preferred_to_the_events_report(tmp_path):
    leaf = tmp_path / "pmmh-11111111" / "seed_1-aaaaaaaa"
    _write_events(leaf, {"event": "method_report", "outcome": "failed", "error": "from events"})
    (leaf / "run.json").write_text(json.dumps({
        "status": "failed", "failure": {"message": "from run.json", "operation": "pmmh"}}))
    v = V.leaf_verdict(leaf)
    assert v.kind == "failed" and v.error == "from run.json"
    # a failed record with no failure falls back to the events report
    (leaf / "run.json").write_text(json.dumps({"status": "failed"}))
    assert V.leaf_verdict(leaf).error == "from events"


def test_summary_leaf_state_is_read_when_there_is_no_verdict():
    doc = json.loads(_summary_doc(5, 1.0))
    doc["leaves"][1]["state"] = {"state": "failed",
                                 "failure": {"message": "no chain scored", "operation": "pmmh"}}
    doc["leaves"][1].pop("error")
    doc["leaves"][2]["state"] = {"state": "interrupted", "pid": None}
    doc["leaves"][2]["status"] = "running"
    got = V.parse_fit_summary(json.dumps(doc)).leaves
    assert got[1].verdict.error == "no chain scored" and got[1].verdict.kind == "failed"
    assert got[2].verdict.kind == "interrupted" and got[2].verdict.status == "neutral"
