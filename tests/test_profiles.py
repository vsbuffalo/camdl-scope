"""Profile-likelihood core and endpoints, over a run store built by StoreKit.

A profile is one ``camdl profile`` command (one invocation record); its points
are the ``profile_point`` results that command listed or claimed, and its grid
is ``given.args.sweep`` expanded as camdl's ``Grid::expand`` expands it."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from camdl_scope.profiles import discover_profiles, expand_grid, load_profile
from tests.storekit import StoreKit

CI_DROP_1D = 1.9207  # ½·χ²₁,₀.₉₅


def _lin_cross(x0: float, y0: float, x1: float, y1: float, y: float) -> float:
    """Where the straight line through (x0, y0) and (x1, y1) reaches ``y``."""
    return x0 + (y - y0) / (y1 - y0) * (x1 - x0)


def _sweep(name: str, values: list[float]) -> list[dict]:
    return [{"name": name, "grid": {"list": values}}]


def _peak_profile(kit: StoreKit, **kw) -> int:
    """A symmetric peak at g=1.0 (loglik −0.5), two starts per cell: the second
    start is 4.5 worse, so the max-over-starts reduction is exercised."""
    curve = {0.1: -10.0, 0.5: -3.0, 1.0: -0.5, 1.5: -3.0, 2.0: -10.0}
    points = [{"focal": {"g": g}, "loglik": ll - d, "mle": {"beta": 0.9}}
              for g, ll in curve.items() for d in (0.0, 4.5)]
    return kit.profile(_sweep("g", list(curve)), points, n_starts=2, **kw)


def _grid_2d(kit: StoreKit) -> int:
    """A 2×2 (g, Cscale) grid with the peak at (0.03, 2.0); each cell has two
    starts, and the worse start at the MLE cell has different nuisance values."""
    best = {(0.01, 0.5): (-8.0, {"r1": 0.01}), (0.01, 2.0): (-6.0, {"r1": 0.02}),
            (0.03, 0.5): (-3.0, {"r1": 0.03}), (0.03, 2.0): (-0.5, {"r1": 0.042, "phi_inv": 0.2})}
    points = []
    for (g, c), (ll, nu) in best.items():
        points.append({"focal": {"g": g, "Cscale": c}, "loglik": ll, "mle": nu})
        points.append({"focal": {"g": g, "Cscale": c}, "loglik": ll - 4.5, "mle": {"r1": 9.9}})
    sweep = [{"name": "g", "grid": {"list": [0.01, 0.03]}},
             {"name": "Cscale", "grid": {"list": [0.5, 2.0]}}]
    return kit.profile(sweep, points, n_starts=2, label="surf")


# --- grid expansion, as camdl's Grid::expand (cli/src/args/types.rs) ----------


def test_expand_list_is_the_listed_values_in_order():
    assert expand_grid({"list": [0.3, 0.1, 2]}) == [0.3, 0.1, 2.0]


def test_expand_linear_includes_both_end_points():
    assert expand_grid({"linear": {"min": 0.0, "max": 1.0, "n": 5}}) == [0.0, 0.25, 0.5, 0.75, 1.0]
    vals = expand_grid({"linear": {"min": 0.1, "max": 0.5, "n": 5}})
    assert len(vals) == 5 and vals[0] == 0.1 and vals[-1] == 0.5
    # camdl computes min + (max − min)·i/(n−1), so the middle value is not 0.3:
    assert vals[2] == 0.1 + (0.5 - 0.1) * 0.5 == 0.30000000000000004


def test_expand_log10_is_evenly_spaced_in_log_with_end_points():
    vals = expand_grid({"log10": {"min": 1.0, "max": 1000.0, "n": 4}})
    assert vals == pytest.approx([1.0, 10.0, 100.0, 1000.0], rel=1e-12)
    assert vals[0] == 1.0
    ratios = [b / a for a, b in zip(vals, vals[1:])]
    assert ratios == pytest.approx([10.0] * 3, rel=1e-12)


@pytest.mark.parametrize("kind", ["linear", "log10"])
def test_expand_one_point_is_the_minimum(kind):
    assert expand_grid({kind: {"min": 0.2, "max": 5.0, "n": 1}}) == [0.2]


def test_expand_unknown_spec_is_empty():
    assert expand_grid({"cubic": {"min": 0.0, "max": 1.0, "n": 3}}) == []
    assert expand_grid({"list": [1.0], "linear": {"min": 0.0, "max": 1.0, "n": 2}}) == []


# --- the curve ----------------------------------------------------------------


def test_load_profile_aggregates_starts_and_orders(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = _peak_profile(kit)
    c = load_profile(kit.root, str(seq))
    assert c is not None
    assert c.base_id == str(seq) and c.label == "sir"  # no --label: the model's name
    assert c.params == ("g",) and c.ndim == 1
    assert c.method == "nl-sbplx" and c.loglik_type == "ode_marginal"
    assert [p.coords[0] for p in c.points] == [0.1, 0.5, 1.0, 1.5, 2.0]  # ascending
    peak = next(p for p in c.points if p.coords == (1.0,))
    assert peak.loglik == -0.5 and peak.n_starts == 2  # max over the two starts
    assert peak.loglik_spread == pytest.approx(4.5)
    assert c.mle.coords == (1.0,) and c.mle.loglik == -0.5
    assert c.status() == "done" and not c.provisional and c.pending == []


def test_label_names_the_profile(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = _peak_profile(kit, label="m4l_profile")
    assert load_profile(kit.root, str(seq)).label == "m4l_profile"


def test_load_2d_profile_surface(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = _grid_2d(kit)
    c = load_profile(kit.root, str(seq))
    assert c is not None
    assert c.params == ("g", "Cscale") and c.ndim == 2
    assert sorted(p.coords for p in c.points) == [(0.01, 0.5), (0.01, 2.0), (0.03, 0.5), (0.03, 2.0)]
    assert all(p.n_starts == 2 and p.complete for p in c.points)
    assert c.mle.coords == (0.03, 2.0) and c.mle.loglik == -0.5
    # The MLE cell reports the winning start's nuisance MLE, not the −5.0 one's.
    assert c.mle.nuisance == {"r1": 0.042, "phi_inv": 0.2}
    # A 2D region uses ½·χ²₂,₀.₉₅ = 2.996, not the 1D 1.92; there is no 1D bracket.
    assert c.ci_drop == pytest.approx(0.5 * 5.9915, abs=1e-4)
    assert c.ci_bounds_1d() == (None, None)


def test_ci_bounds_interpolated_both_sides(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    c = load_profile(kit.root, str(_peak_profile(kit)))
    thresh = -0.5 - CI_DROP_1D
    lo, hi = c.ci_bounds_1d()
    assert lo == pytest.approx(_lin_cross(0.5, -3.0, 1.0, -0.5, thresh), abs=1e-9)
    assert hi == pytest.approx(_lin_cross(1.5, -3.0, 1.0, -0.5, thresh), abs=1e-9)
    assert lo == pytest.approx(0.61586, abs=1e-5) and hi == pytest.approx(1.38414, abs=1e-5)


def test_ci_open_when_mle_at_grid_edge(tmp_path):
    # Monotone-increasing loglik: the MLE is at the largest value, so the upper
    # limit is not bracketed by the grid.
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = kit.profile(_sweep("g", [0.1, 1.0, 2.0]), [
        {"focal": {"g": 0.1}, "loglik": -20.0}, {"focal": {"g": 1.0}, "loglik": -5.0},
        {"focal": {"g": 2.0}, "loglik": -1.0}])
    c = load_profile(kit.root, str(seq))
    lo, hi = c.ci_bounds_1d()
    assert c.mle.coords == (2.0,)
    assert lo == pytest.approx(_lin_cross(1.0, -5.0, 2.0, -1.0, -1.0 - CI_DROP_1D), abs=1e-9)
    assert hi is None


def test_ci_ignores_sentinel_points(tmp_path):
    # A clean peak plus a failed cell (camdl's −1e100 sentinel) at the edge: the
    # sentinel neither wins the MLE nor drags the upper crossing toward 2.0.
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = kit.profile(_sweep("g", [0.5, 1.0, 1.5, 2.0]), [
        {"focal": {"g": 0.5}, "loglik": -3.0}, {"focal": {"g": 1.0}, "loglik": -0.5},
        {"focal": {"g": 1.5}, "loglik": -3.0}, {"focal": {"g": 2.0}, "loglik": -1e100}])
    c = load_profile(kit.root, str(seq))
    assert len(c.points) == 4
    assert c.mle.coords == (1.0,)
    lo, hi = c.ci_bounds_1d()
    assert lo == pytest.approx(0.61586, abs=1e-5)
    assert hi == pytest.approx(1.38414, abs=1e-5)


def test_all_sentinel_profile_has_no_ci(tmp_path):
    # Every cell failed: the profile still loads (the UI shows "no valid optimum
    # yet"), but there is no interval to bracket.
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = kit.profile(_sweep("g", [0.5, 1.0]), [
        {"focal": {"g": 0.5}, "loglik": -1e100}, {"focal": {"g": 1.0}, "loglik": -1e100}])
    c = load_profile(kit.root, str(seq))
    assert c is not None and len(c.points) == 2
    assert c.ci_bounds_1d() == (None, None)


def test_coordinate_is_the_exact_focal_value_not_the_folder_name(tmp_path):
    """StoreKit, like camdl, names the cell folder with the value rounded to
    four decimals (``gamma_0.2313``); the curve carries the exact 0.23125."""
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = kit.profile(_sweep("gamma", [0.23125]),
                      [{"focal": {"gamma": 0.23125}, "loglik": -61.47, "mle": {"beta": 0.839}}])
    [folder] = list((kit.root / kit.build / "profile_point").glob("*/*/*/*"))
    assert "0.2313" in folder.parent.parent.name  # the folder carries the rounded value
    [p] = load_profile(kit.root, str(seq)).points
    assert p.coords == (0.23125,) and p.nuisance == {"beta": 0.839}


def test_two_arithmetic_routes_to_one_value_are_one_cell(tmp_path):
    """A linear grid reaches 0.30000000000000004 where a listed grid has 0.3;
    camdl stores two cells, the curve shows one, at the command's grid value."""
    kit = StoreKit.create(tmp_path / "camdl-store")
    sweep = [{"name": "g", "grid": {"linear": {"min": 0.1, "max": 0.5, "n": 5}}}]
    seq = kit.profile(sweep, [
        {"focal": {"g": g}, "loglik": -float(i)} for i, g in enumerate(expand_grid(sweep[0]["grid"]))
    ] + [{"focal": {"g": 0.3}, "loglik": -1.5}], n_starts=1)
    c = load_profile(kit.root, str(seq))
    assert len(c.points) == 5 and c.pending == []
    mid = [p for p in c.points if math.isclose(p.coords[0], 0.3)]
    assert len(mid) == 1
    assert mid[0].coords == (0.30000000000000004,) and mid[0].n_starts == 2
    assert mid[0].loglik == -1.5  # the better of the two routes' starts


def test_a_point_off_the_grid_has_no_expected_starts(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = kit.profile(_sweep("g", [0.1, 0.2]), [
        {"focal": {"g": 0.1}, "loglik": -2.0}, {"focal": {"g": 0.2}, "loglik": -1.0},
        {"focal": {"g": 0.15}, "loglik": -0.5}])
    c = load_profile(kit.root, str(seq))
    off = next(p for p in c.points if p.coords == (0.15,))
    assert off.n_expected is None and off.complete
    assert c.starts_progress() == (2, 2)  # only on-grid cells count


def test_reused_points_listed_by_a_later_command_count_for_it(tmp_path):
    """A finished command lists every point it asked for, computed or reused;
    a second command that reused the first's points has them as its own."""
    kit = StoreKit.create(tmp_path / "camdl-store")
    first = _peak_profile(kit)
    rec = json.loads((kit.root / ".invocations" / f"{first}.json").read_text())
    reused = [{**r, "outcome": "reused"} for r in rec["results"]]
    second = kit.invocation(["profile", "sir.camdl"], reused, label="again", given=rec["given"])
    c = load_profile(kit.root, str(second))
    assert c.label == "again" and len(c.points) == 5
    assert c.mle.coords == (1.0,) and all(p.n_starts == 2 for p in c.points)


def test_points_of_another_command_are_not_this_ones(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    _peak_profile(kit)
    seq = kit.profile(_sweep("g", [0.5, 1.0]), [{"focal": {"g": 1.0}, "loglik": -7.0}])
    c = load_profile(kit.root, str(seq))
    assert [(p.coords, p.n_starts, p.loglik) for p in c.points] == [((1.0,), 1, -7.0)]
    assert [p.coords for p in c.pending] == [(0.5,)]


def test_discover_lists_profile_commands_newest_first(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    first = _peak_profile(kit)
    problem = kit.problem()
    kit.fit_run(problem)  # a fit command is not a profile
    no_sweep = kit.invocation(["profile", "sir.camdl"], [], given={"args": {"model": "sir.camdl"}})
    last = kit.profile(_sweep("g", [0.5]), [{"focal": {"g": 0.5}, "loglik": -2.0}], label="solo")
    found = discover_profiles(kit.root)
    assert [c.base_id for c in found] == [str(last), str(first)]  # no grid → not a profile
    assert str(no_sweep) not in {c.base_id for c in found}
    assert load_profile(kit.root, str(no_sweep)) is None


def test_unknown_profile_and_non_store_are_empty(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    _peak_profile(kit)
    assert load_profile(kit.root, "99") is None
    assert discover_profiles(tmp_path / "not-a-store") == []
    assert load_profile(tmp_path / "not-a-store", "1") is None


def test_failed_point_marks_the_profile_failed(tmp_path):
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = _peak_profile(kit)
    # StoreKit.profile records every point as computed; mark one failed, as an
    # ended camdl record lists a point whose optimisation failed.
    path = kit.root / ".invocations" / f"{seq}.json"
    rec = json.loads(path.read_text())
    rec["results"][0]["outcome"] = "failed"
    path.write_text(json.dumps(rec, indent=1))
    assert load_profile(kit.root, str(seq)).status() == "failed"


# --- API endpoints ------------------------------------------------------------


def _client(kit: StoreKit, monkeypatch) -> TestClient:
    monkeypatch.setenv("CAMDL_STORE", str(kit.root))
    from camdl_scope.api.app import app

    return TestClient(app)


def test_profile_endpoints(tmp_path, monkeypatch):
    """`/api/profiles` lists, `/api/profiles/{seq}` returns the curve and CI."""
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = _peak_profile(kit)
    c = _client(kit, monkeypatch)
    lst = c.get("/api/profiles").json()
    assert [p["base_id"] for p in lst] == [str(seq)]
    assert lst[0]["params"] == ["g"] and lst[0]["n_points"] == 5 and lst[0]["status"] == "done"
    assert (lst[0]["n_starts_finished"], lst[0]["n_starts_expected"]) == (10, 10)

    r = c.get(f"/api/profiles/{seq}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["params"] == ["g"]
    assert body["mle_coords"] == [1.0] and body["mle_loglik"] == -0.5
    assert body["ci_lo"] == pytest.approx(0.61586, abs=1e-5)
    assert body["ci_hi"] == pytest.approx(1.38414, abs=1e-5)
    assert len(body["points"]) == 5 and body["provisional"] is False
    assert body["heartbeat_at"] is None  # an ended command has no heartbeat

    assert c.get("/api/profiles/nope").status_code == 404


def test_2d_profile_endpoint(tmp_path, monkeypatch):
    """A 2D profile surfaces as `params=[g, Cscale]`, coordinate-carrying points
    each with its nuisance MLE, and no 1D bracket (a region, not an interval)."""
    kit = StoreKit.create(tmp_path / "camdl-store")
    seq = _grid_2d(kit)
    c = _client(kit, monkeypatch)
    [s] = c.get("/api/profiles").json()
    assert s["params"] == ["g", "Cscale"] and s["mle_coords"] == [0.03, 2.0] and s["label"] == "surf"

    r = c.get(f"/api/profiles/{seq}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mle_coords"] == [0.03, 2.0] and body["mle_loglik"] == -0.5
    assert body["ci_lo"] is None and body["ci_hi"] is None
    assert body["ci_drop"] == pytest.approx(2.9957, abs=1e-4)
    assert all(len(p["coords"]) == 2 for p in body["points"])
    mle_pt = next(p for p in body["points"] if p["coords"] == [0.03, 2.0])
    assert mle_pt["nuisance"] == {"r1": 0.042, "phi_inv": 0.2}
