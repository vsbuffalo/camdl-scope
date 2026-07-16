"""Sim-tree reader tests. Synthetic ``sims/`` fixtures (wide ``traj.tsv`` + a
``run.json`` per leaf) matching camdl's on-disk layout — no real sim needed."""

from __future__ import annotations

import json
from pathlib import Path

from camdl_watch import sims


def _member(leaf: Path, params_label: str, header: str, rows: list[str], schema=None):
    """Write one sim leaf: run.json (levels + optional output_schema) + traj.tsv."""
    leaf.mkdir(parents=True, exist_ok=True)
    rj = {
        "kind": "sim",
        "status": "completed",
        "levels": [
            {"name": "model", "label": "toy"},
            {"name": "config", "label": "ode-dt1"},
            {"name": "params", "label": params_label},
            {"name": "scenario", "label": "baseline"},
            {"name": "seed", "label": "seed_1"},
        ],
    }
    if schema is not None:
        rj["output_schema"] = {"traj.tsv": schema}
    (leaf / "run.json").write_text(json.dumps(rj))
    (leaf / "traj.tsv").write_text(header + "\n" + "\n".join(rows) + "\n")


# Two compartments (S, E) over two village strata (va, vb); state-major columns.
_HEADER = "t\tS_va\tS_vb\tE_va\tE_vb\tflow_inoc_va\tflow_inoc_vb"
_ROWS_A = ["0\t10\t20\t1\t2\t0\t0", "1\t8\t18\t3\t4\t1\t1"]
_ROWS_B = ["0\t100\t200\t0\t0\t0\t0", "1\t90\t180\t10\t20\t5\t5"]


def _sim(store: Path):
    base = store.parent / "sims" / "toy-abc123" / "ode-dt1-cfg"
    _member(base / "base-aaaa" / "baseline-s" / "seed_1-x", "base·1", _HEADER, _ROWS_A)
    _member(base / "base-bbbb" / "baseline-s" / "seed_1-y", "base·2", _HEADER, _ROWS_B)


def test_discover_sims(tmp_path: Path):
    store = tmp_path / "results" / "fits"
    store.mkdir(parents=True)
    _sim(store)
    found = sims.discover_sims(store)
    assert len(found) == 1
    s = found[0]
    assert s.sim_id == "toy-abc123" and s.model == "toy"
    assert len(s.members) == 2  # the sweep
    assert {m.label for m in s.members} == {"base·1", "base·2"}


def test_sims_root_sibling(tmp_path: Path):
    store = tmp_path / "results" / "fits"
    assert sims.sims_root(store) == tmp_path / "results" / "sims"
    # A non-fits store gets a sims/ child.
    assert sims.sims_root(tmp_path / "store") == tmp_path / "store" / "sims"


def test_roles_inferred_without_schema(tmp_path: Path):
    store = tmp_path / "results" / "fits"
    store.mkdir(parents=True)
    _sim(store)
    m = sims.discover_sims(store)[0].members[0]
    roles = sims.resolve_roles(m)
    assert roles["t"] == "time"
    assert roles["S_va"] == "state" and roles["E_vb"] == "state"
    assert roles["flow_inoc_va"] == "flow"


def test_compartment_totals(tmp_path: Path):
    store = tmp_path / "results" / "fits"
    store.mkdir(parents=True)
    _sim(store)
    sim = sims.discover_sims(store)[0]
    by_label = {m.label: m for m in sim.members}
    roles = sims.resolve_roles(sim.members[0])
    assert sims.available_series(roles, "state") == ["S", "E"]
    # base·2's S total = S_va + S_vb = 100+200 at t0, 90+180 at t1.
    ser = sims.read_member_total(by_label["base·2"], "S", roles, "state")
    assert ser is not None
    assert ser.time == [0.0, 1.0]
    assert ser.value == [300.0, 270.0]
    # E total for base·2 = 0 then 30.
    e = sims.read_member_total(by_label["base·2"], "E", roles, "state")
    assert e is not None and e.value == [0.0, 30.0]


def test_absent_sims_tree_is_empty(tmp_path: Path):
    store = tmp_path / "results" / "fits"
    store.mkdir(parents=True)
    assert sims.discover_sims(store) == []
