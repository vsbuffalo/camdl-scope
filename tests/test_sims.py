"""Sim reader tests over a StoreKit store: one ``camdl simulate`` command whose
cells are ``sim`` results with a wide ``traj.tsv`` each — no real sim needed."""

from __future__ import annotations

from pathlib import Path

from camdl_scope import sims
from tests.storekit import StoreKit

# Two compartments (S, E) over two village strata (va, vb); state-major columns.
_HEADER = "t\tS_va\tS_vb\tE_va\tE_vb\tflow_inoc_va\tflow_inoc_vb"
_ROWS_A = ["0\t10\t20\t1\t2\t0\t0", "1\t8\t18\t3\t4\t1\t1"]
_ROWS_B = ["0\t100\t200\t0\t0\t0\t0", "1\t90\t180\t10\t20\t5\t5"]

_SOURCE = """\
# toy model
time_unit = days
origin = date("2026-01-05")
compartments { S, E }
"""


def _traj(rows: list[str]) -> str:
    return _HEADER + "\n" + "\n".join(rows) + "\n"


def _sim(root: Path, *, status: str = "ended") -> int:
    """A two-cell sweep over ``beta`` with ``gamma`` held fixed; returns the
    command's invocation sequence number."""
    kit = StoreKit.create(root)
    return kit.sim([
        {"params": {"beta": 0.1, "gamma": 0.2}, "traj": _traj(_ROWS_A)},
        {"params": {"beta": 0.3, "gamma": 0.2}, "traj": _traj(_ROWS_B)},
    ], model="toy", source=_SOURCE, status=status)


def test_discover_sims(tmp_path: Path):
    seq = _sim(tmp_path)
    found = sims.discover_sims(tmp_path)
    assert len(found) == 1
    s = found[0]
    assert s.sim_id == str(seq) and s.model == "toy" and s.status == "done"
    assert len(s.members) == 2  # the sweep
    # Each member is labelled by the one parameter that varies.
    assert {m.label for m in s.members} == {"beta=0.1", "beta=0.3"}


def test_running_sim_lists_the_cells_that_name_it(tmp_path: Path):
    # A running command's record lists no results yet; its finished cells are
    # found by the invocation their run.json names.
    seq = _sim(tmp_path, status="running")
    [s] = sims.discover_sims(tmp_path)
    assert s.sim_id == str(seq) and s.status == "running"
    assert {m.label for m in s.members} == {"beta=0.1", "beta=0.3"}


def test_model_facts_come_from_the_compiled_models_source(tmp_path: Path):
    _sim(tmp_path)
    model = sims.read_sim_model(sims.discover_sims(tmp_path)[0])
    assert model.origin == "2026-01-05"
    assert model.time_unit == "days"
    assert model.states == ["S", "E"]


def test_roles_follow_column_names(tmp_path: Path):
    _sim(tmp_path)
    m = sims.discover_sims(tmp_path)[0].members[0]
    roles = sims.resolve_roles(m)
    assert roles["t"] == "time"
    assert roles["S_va"] == "state" and roles["E_vb"] == "state"
    assert roles["flow_inoc_va"] == "flow"


def test_compartment_totals(tmp_path: Path):
    _sim(tmp_path)
    sim = sims.discover_sims(tmp_path)[0]
    by_label = {m.label: m for m in sim.members}
    roles = sims.resolve_roles(sim.members[0])
    assert sims.available_series(roles, "state") == ["S", "E"]
    # beta=0.3's S total = S_va + S_vb = 100+200 at t0, 90+180 at t1.
    ser = sims.read_member_total(by_label["beta=0.3"], "S", roles, "state")
    assert ser is not None
    assert ser.time == [0.0, 1.0]
    assert ser.value == [300.0, 270.0]
    # E total for beta=0.3 = 0 then 30.
    e = sims.read_member_total(by_label["beta=0.3"], "E", roles, "state")
    assert e is not None and e.value == [0.0, 30.0]


def test_store_without_sims_is_empty(tmp_path: Path):
    StoreKit.create(tmp_path)
    assert sims.discover_sims(tmp_path) == []
