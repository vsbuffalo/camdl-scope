"""camdl's frozen-chain verdict, findings and per-chain log-likelihood
comparison are read from what camdl persisted and shown as written.

The payload shapes are those camdl wrote for the frozen PGAS fixture of its own
test ``chain_frozen.rs`` (2 chains, 20 particles, 60 sweeps, burn_in 10, seed 1).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from camdl_scope import chainrecord as CR
from camdl_scope import verdict as V
from tests.fixtures.make_golden_store import POSTERIOR_DIR, RUN_DIR, SEED_DIR, build

EV = {"schema": "camdl.events/v1"}


def _frozen_finding(chain: int, n_div: int, step: float) -> dict:
    return {
        "kind": {"type": "chain_frozen", "chain_id": chain, "n_draws": 50, "burn_in": 10,
                 "n_steps": 50,
                 "sampler": {"kernel": "nuts", "step_size": step, "n_divergent": n_div}},
        "severity": "error",
        "message": (f"took no step after burn-in: all 50 retained draws are one parameter "
                    f"vector. {n_div} of its 50 post-burn-in NUTS steps diverged."),
        "method": "pgas", "timestamp": "2026-11-11T16:34:42Z",
    }


def _agreement(flag_chain: int | None = None) -> dict:
    def row(c: int, mean: float) -> dict:
        flagged = c == flag_chain
        return {"chain": c, "mean": mean, "outcome": {
            "status": "assessed", "diff": -9.1 if flagged else 0.06, "se": 1.75 if flagged else 0.36,
            "z": 5.2 if flagged else 0.17, "cutoff": 3.48, "flagged": flagged}}
    return {"column": "obs_ll", "family_alpha": 0.001, "min_ess": 10.0,
            "chains": [row(0, -41.59), row(1, -50.7 if flag_chain == 1 else -41.65)]}


def _write_events(leaf: Path, verdict: dict, diagnostics: dict | None = None) -> None:
    report = {"name": "pgas", "method": "pgas", "n_chains": 2, "checks": [], "findings": [],
              "verdict": verdict, "method_result": {"diagnostics": diagnostics or {}}}
    lines = [{**EV, "seq": 0, "elapsed": 1.0, "event": "method_report",
              "outcome": "completed", "report": report}]
    (leaf / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))


@pytest.fixture
def store(tmp_path, monkeypatch):
    build(tmp_path)
    monkeypatch.setenv("CAMDL_SCOPE_STORE", str(tmp_path))
    return tmp_path


@pytest.fixture
def leaf(store) -> Path:
    return store / RUN_DIR / POSTERIOR_DIR / SEED_DIR


def _client() -> TestClient:
    from camdl_scope.api.app import app

    return TestClient(app)


def _make_frozen(leaf: Path) -> None:
    _write_events(leaf, {"kind": "chains_frozen", "chains": [0, 1]},
                  {"frozen_chains": [0, 1], "chain_logliks": _agreement()})
    summary = json.loads((leaf / "pgas_summary.json").read_text())
    summary["frozen_chains"] = [0, 1]
    summary["chain_loglik_agreement"] = _agreement()
    (leaf / "pgas_summary.json").write_text(json.dumps(summary))
    (leaf / "diagnostics.json").write_text(json.dumps([
        _frozen_finding(0, 49, 0.4998), _frozen_finding(1, 36, 0.2829)]))


# --- the verdict ------------------------------------------------------------


def test_chains_frozen_verdict_is_not_converged_and_names_the_chains():
    v = V.verdict_from_report({"kind": "chains_frozen", "chains": [0, 1]})
    assert (v.kind, v.status, v.chains) == ("chains_frozen", "error", (0, 1))
    assert v.text == "not converged (chains 0 and 1 took no step after burn-in)"
    one = V.verdict_from_report({"kind": "chains_frozen", "chains": [2]})
    assert one.text == "not converged (chain 2 took no step after burn-in)"
    three = V.verdict_from_report({"kind": "chains_frozen", "chains": [0, 1, 3]})
    assert three.text == "not converged (chains 0, 1 and 3 took no step after burn-in)"


@pytest.mark.parametrize("bad", [None, "0", [0, "1"], [True]])
def test_a_malformed_frozen_chain_list_is_refused_not_guessed(bad):
    with pytest.raises(ValueError):
        V.verdict_from_report({"kind": "chains_frozen", "chains": bad})


def test_fit_summary_row_text_is_camdl_s_own():
    doc = json.dumps({
        "schema": {"version": 5, "camdl_version": "0.1.0+5bc19b89"},
        "leaves": [{"leaf": "x/pgas-aaaaaaaa/seed_1-bbbbbbbb", "method": "pgas",
                    "run_id": "r", "status": "completed",
                    "verdict": {"kind": "chains_frozen", "chains": [0, 1]},
                    "verdict_text": "not converged (chains 0 and 1 took no step after burn-in)"}],
    })
    row = V.parse_fit_summary(doc).leaves[0]
    assert row.verdict.kind == "chains_frozen" and row.verdict.chains == (0, 1)
    assert row.verdict.text == "not converged (chains 0 and 1 took no step after burn-in)"


def test_an_unknown_verdict_kind_is_still_refused():
    with pytest.raises(V.UnsupportedSchema):
        V.verdict_from_report({"kind": "no_such_kind"})


# --- the persisted records --------------------------------------------------


def test_frozen_chains_empty_is_not_the_same_as_absent(leaf):
    assert CR.read_frozen_chains(leaf) is None
    summary = json.loads((leaf / "pgas_summary.json").read_text())
    summary["frozen_chains"] = []
    (leaf / "pgas_summary.json").write_text(json.dumps(summary))
    assert CR.read_frozen_chains(leaf) == []
    summary["frozen_chains"] = [1, 0]
    (leaf / "pgas_summary.json").write_text(json.dumps(summary))
    assert CR.read_frozen_chains(leaf) == [0, 1]


def test_frozen_chains_fall_back_to_the_method_report(leaf):
    _write_events(leaf, {"kind": "chains_frozen", "chains": [1]}, {"frozen_chains": [1]})
    assert CR.read_frozen_chains(leaf) == [1]


def test_comparison_is_read_from_the_summary_then_the_report_then_fit_state(leaf):
    assert CR.read_chain_logliks(leaf) is None
    _write_events(leaf, {"kind": "not_applicable"}, {"chain_logliks": _agreement()})
    from_report = CR.read_chain_logliks(leaf)
    assert from_report is not None and from_report.column == "obs_ll"
    (leaf / "events.jsonl").unlink()
    (leaf / "fit_state.toml").write_text(
        '[chain_loglik_agreement]\ncolumn = "clean_loglik"\nfamily_alpha = 0.001\nmin_ess = 10.0\n\n'
        '[[chain_loglik_agreement.chains]]\nchain = 0\nmean = -44.2\n\n'
        '[chain_loglik_agreement.chains.outcome]\nstatus = "assessed"\ndiff = -0.02\nse = 0.016\n'
        'z = 1.5\ncutoff = 3.66\nflagged = false\n')
    from_toml = CR.read_chain_logliks(leaf)
    assert from_toml is not None and from_toml.column == "clean_loglik"
    assert from_toml.chains[0].flagged is False


def test_not_assessed_reasons_and_non_finite_numbers_are_read_without_loss():
    rec = CR.parse_chain_logliks({
        "column": "log_likelihood", "column_absent": True, "family_alpha": 0.001, "min_ess": 10.0,
        "chains": [
            {"chain": 0, "mean": "NaN", "outcome": {"status": "not_assessed",
                                                    "reason": "non_finite_mean"}},
            {"chain": 1, "mean": -3.0, "outcome": {"status": "not_assessed",
                                                   "reason": "too_few_effective_draws", "ess": 4.5}},
            {"chain": 2, "mean": -3.0, "outcome": {"status": "assessed", "diff": "-inf",
                                                   "se": 0.1, "z": "inf", "cutoff": 3.5,
                                                   "flagged": True}},
        ]})
    assert rec is not None and rec.column_absent
    assert math.isnan(rec.chains[0].mean) and rec.chains[0].reason == "non_finite_mean"
    assert rec.chains[1].ess == 4.5
    assert rec.chains[2].diff == -math.inf and rec.chains[2].z == math.inf


def test_an_unknown_comparison_status_is_refused():
    with pytest.raises(ValueError):
        CR.parse_chain_logliks({"column": "x", "chains": [
            {"chain": 0, "mean": 1.0, "outcome": {"status": "something_new"}}]})


# --- through the API --------------------------------------------------------


def test_diagnostics_shows_the_frozen_verdict_findings_and_comparison(leaf):
    _make_frozen(leaf)
    c = _client()
    d = c.get(f"/api/runs/{RUN_DIR}/diagnostics").json()
    assert d["verdict"]["kind"] == "chains_frozen" and d["verdict"]["status"] == "error"
    assert d["verdict"]["chains"] == [0, 1]
    assert d["verdict"]["text"] == "not converged (chains 0 and 1 took no step after burn-in)"
    assert d["frozen_chains"] == [0, 1]
    (g,) = [f for f in d["findings"] if f["kind"] == "chain_frozen"]
    assert g["severity"] == "error"
    assert [m["chain"] for m in g["chain_messages"]] == [0, 1]
    assert g["chain_messages"][0]["message"].startswith("took no step after burn-in")
    rec = d["chain_logliks"]
    assert rec["column"] == "obs_ll" and rec["chains"][0]["status"] == "assessed"
    assert rec["chains"][0]["flagged"] is False and rec["chains"][0]["z"] == "0.17"
    assert d["chain_logliks_error"] is None
    assert c.get(f"/api/runs/{RUN_DIR}").json()["frozen_chain_ids"] == [0, 1]


def test_a_flagged_chain_shows_camdl_flag_and_the_finding(leaf):
    _write_events(leaf, {"kind": "converged", "max_rhat": 1.01, "param": "beta"})
    summary = json.loads((leaf / "pgas_summary.json").read_text())
    summary["frozen_chains"] = []
    summary["chain_loglik_agreement"] = _agreement(flag_chain=1)
    (leaf / "pgas_summary.json").write_text(json.dumps(summary))
    (leaf / "diagnostics.json").write_text(json.dumps([{
        "kind": {"type": "chain_loglik_differs", "chain_id": 1, "column": "obs_ll",
                 "diff": -9.1, "z": 5.2, "cutoff": 3.48},
        "severity": "warning",
        "message": "chain 1's mean obs_ll differs from the others'", "method": "pgas"}]))
    d = _client().get(f"/api/runs/{RUN_DIR}/diagnostics").json()
    assert d["frozen_chains"] == []
    flagged = [r["chain"] for r in d["chain_logliks"]["chains"] if r["flagged"]]
    assert flagged == [1]
    (g,) = [f for f in d["findings"] if f["kind"] == "chain_loglik_differs"]
    assert g["severity"] == "warn" and g["chain_messages"][0]["chain"] == 1


def test_a_leaf_that_predates_the_records_says_not_recorded(leaf):
    d = _client().get(f"/api/runs/{RUN_DIR}/diagnostics").json()
    assert d["frozen_chains"] is None
    assert d["chain_logliks"] is None and d["chain_logliks_error"] is None
    assert _client().get(f"/api/runs/{RUN_DIR}").json()["frozen_chain_ids"] == []


def test_a_record_this_reader_cannot_parse_is_reported_not_hidden(leaf):
    summary = json.loads((leaf / "pgas_summary.json").read_text())
    summary["chain_loglik_agreement"] = {"column": "obs_ll", "chains": "nope"}
    (leaf / "pgas_summary.json").write_text(json.dumps(summary))
    d = _client().get(f"/api/runs/{RUN_DIR}/diagnostics").json()
    assert d["chain_logliks"] is None
    assert "was not read" in d["chain_logliks_error"]
