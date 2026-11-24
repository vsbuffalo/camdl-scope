"""Each optimiser search's progress from camdl's live ``opt_trace.tsv``: one
file per leaf, a ``chain`` column, a row every 25 evaluations or 2 s and one at
the search's end (``eval``, ``loglik`` of that evaluation, ``best_loglik`` so
far, the parameters at the best point)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from camdl_scope import ingest
from camdl_scope.mle import read_mle, read_opt_trace
from tests.test_mle_live import HEADER, _beat, _leaf, _problem

TRACE_HEADER = "chain\teval\tloglik\tbest_loglik\ta\tb\n"


def _rows(chain: int, points: list[tuple[int, str, str, str, str]]) -> str:
    return "".join(f"{chain}\t{e}\t{ll}\t{best}\t{a}\t{b}\n" for e, ll, best, a, b in points)


# chain 0 climbs and is still running; chain 1 started at -inf and has found a
# finite value; chain 2 has no finite value yet.
C0 = _rows(0, [(1, "-900.5", "-900.5", "0.6", "2.0"), (25, "-80.0", "-40.0", "1.4", "2.4"),
               (50, "-30.0", "-12.5", "1.5", "2.5")])
C1 = _rows(1, [(1, "-inf", "", "", ""), (25, "-55.0", "-55.0", "1.2", "2.8")])
C2 = _rows(2, [(1, "nan", "", "", "")])


def _running(tmp_path, trace: str, *, alive: bool = True):
    """A running optimiser with no start finished and ``trace`` as its live
    ``opt_trace.tsv``; ``alive=False`` gives it an owner that has exited."""
    store = tmp_path / "camdl-store"
    kit, problem = _problem(store)
    leaf = _leaf(kit, problem, alive=alive)
    (leaf / "chain_results.tsv").write_text(HEADER)
    (leaf / "opt_trace.tsv").write_text(trace)
    _beat(kit, leaf, step=50, total=400)
    return store, leaf


def test_trace_rows_group_by_search(tmp_path):
    _, leaf = _running(tmp_path, TRACE_HEADER + C0 + C1 + C2)
    by = {t.chain: t for t in read_opt_trace(leaf, ["a", "b"], live=True)}
    assert by[0].evals == 50 and by[0].best_loglik == -12.5
    assert [p.best_loglik for p in by[0].points] == [-900.5, -40.0, -12.5]
    assert by[0].best_values == {"a": 1.5, "b": 2.5} and not by[0].started_nonfinite
    assert by[1].started_nonfinite and by[1].points[0].best_loglik is None
    assert by[1].best_loglik == -55.0
    assert by[2].started_nonfinite and by[2].best_loglik is None


def test_torn_last_row_is_dropped_while_live(tmp_path):
    _, leaf = _running(tmp_path, TRACE_HEADER + C0 + "0\t75\t-20.0\t-12")
    [t] = read_opt_trace(leaf, ["a", "b"], live=True)
    assert t.evals == 50


def test_running_search_supplies_the_provisional_best_before_any_finishes(tmp_path):
    _, leaf = _running(tmp_path, TRACE_HEADER + C0 + C1 + C2)
    fit = read_mle(leaf, ["a", "b"])
    assert fit is not None and fit.provisional and fit.restarts == []
    assert fit.loglik == -12.5 and fit.provisional_from_running
    assert {p.name: p.value for p in fit.params} == {"a": 1.5, "b": 2.5}


def test_a_finished_start_beats_a_worse_running_search(tmp_path):
    _, leaf = _running(tmp_path, TRACE_HEADER + C0 + C1)
    (leaf / "chain_results.tsv").write_text(HEADER + "3\t-10.0\txtol_reached\t300\t1.1\t3.0\n")
    fit = read_mle(leaf, ["a", "b"])
    assert fit.loglik == -10.0 and not fit.provisional_from_running


def test_a_running_search_ahead_of_every_finished_start_leads(tmp_path):
    _, leaf = _running(tmp_path, TRACE_HEADER + C0)
    (leaf / "chain_results.tsv").write_text(HEADER + "3\t-20.0\txtol_reached\t300\t1.1\t3.0\n")
    fit = read_mle(leaf, ["a", "b"])
    assert fit.loglik == -12.5 and fit.provisional_from_running


def test_no_trace_reads_as_before(tmp_path):
    _, leaf = _running(tmp_path, TRACE_HEADER)
    (leaf / "opt_trace.tsv").unlink()
    fit = read_mle(leaf, ["a", "b"])
    assert fit.searches == [] and fit.loglik is None and not fit.provisional_from_running


def test_finished_run_keeps_its_traces(tmp_path):
    _, leaf = _running(tmp_path, TRACE_HEADER + C0)
    (leaf / "chain_results.tsv").write_text(HEADER + "0\t-12.5\txtol_reached\t50\t1.5\t2.5\n")
    (leaf / "mle_params.toml").write_text("a = 1.5\nb = 2.5\n\n[provenance]\nlog_likelihood = -12.5\n")
    fit = read_mle(leaf, ["a", "b"])
    assert not fit.provisional and [t.chain for t in fit.searches] == [0]
    assert fit.searches[0].best_loglik == fit.loglik


def test_endpoint_carries_searches_and_marks_finished(tmp_path, monkeypatch):
    store, leaf = _running(tmp_path, TRACE_HEADER + C0 + C1 + C2)
    (leaf / "chain_results.tsv").write_text(HEADER + "1\t-50.0\tftol_reached\t60\t1.2\t2.8\n")
    monkeypatch.setenv("CAMDL_STORE", str(store))
    from camdl_scope.api.app import app

    c = TestClient(app)
    [run] = c.get("/api/runs").json()
    d = c.get(f"/api/runs/{run['run_id']}/mle").json()
    by = {s["chain"]: s for s in d["searches"]}
    assert by[0] == {"chain": 0, "evals": 50, "best_loglik": -12.5, "finished": False,
                     "started_nonfinite": False,
                     "trace": [[1, -900.5], [25, -40.0], [50, -12.5]]}
    assert by[1]["finished"] and by[1]["started_nonfinite"]
    assert by[2]["best_loglik"] is None
    assert d["loglik"] == -12.5 and d["provisional_from_running"]


def test_dead_owner_reads_as_stalled_with_its_traces(tmp_path):
    store, leaf = _running(tmp_path, TRACE_HEADER + C0, alive=False)
    (leaf / "chain_results.tsv").write_text(HEADER + "1\t-50.0\tftol_reached\t60\t1.2\t2.8\n")
    [meta] = ingest.discover_runs(store, include_warming=True)
    from camdl_scope.assembly import build_run_overview

    assert build_run_overview(meta).status.value == "stalled"
    fit = read_mle(leaf, ["a", "b"])
    assert fit.searches[0].evals == 50 and fit.provisional_from_running
