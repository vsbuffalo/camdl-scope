"""Profile-likelihood core + endpoints — over a crafted ``profiles/`` tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from camdl_watch.profiles import discover_profiles, load_profile


def _leaf(
    profiles_dir: Path,
    base: str,
    base_label: str,
    point_seg: str,
    point_label: str,
    best_loglik: float,
    *,
    method: str = "nl-sbplx",
    seed: int = 0,
    start: int = 0,
    nuisance: dict[str, float] | None = None,
) -> None:
    """Write one ``profile_point`` leaf (run.json, plus an mle.toml when
    ``nuisance`` is given) under profiles/<base>/..."""
    leaf = (
        profiles_dir / base / point_seg / f"{method}_-hh"
        / f"seed_{seed}-hh" / f"start_{start}-hh"
    )
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "run.json").write_text(json.dumps({
        "kind": "profile_point",
        "levels": [
            {"name": "profile", "label": base_label},
            {"name": "point", "label": point_label},
            {"name": "stage", "label": method},
        ],
        "inputs": {
            "best_loglik": best_loglik, "method": method,
            "loglik_type": "ode_marginal", "grid_point": 0,
        },
        "status": "completed",
    }))
    if nuisance is not None:
        body = "".join(f"{k} = {v}\n" for k, v in nuisance.items())
        (leaf / "mle.toml").write_text(f"[mle]\n{body}")


def _peak_profile(profiles_dir: Path, base: str = "demo-aa") -> None:
    """A symmetric peak at g=1.0 (loglik −0.5); the g=1.0 point has two restarts
    so the max-over-starts reduction is exercised."""
    pts = [("g_0.1000", "g=0.1000", -10.0), ("g_0.5000", "g=0.5000", -3.0),
           ("g_1.0000", "g=1.0000", -0.5), ("g_1.5000", "g=1.5000", -3.0),
           ("g_2.0000", "g=2.0000", -10.0)]
    for seg, label, ll in pts:
        _leaf(profiles_dir, base, "demo", seg, label, ll, start=0)
    # A worse restart at the peak — best_loglik must win.
    _leaf(profiles_dir, base, "demo", "g_1.0000", "g=1.0000", -5.0, start=1)


def _grid_2d(profiles_dir: Path, base: str = "surf-ee") -> None:
    """A 2×2 (g, Cscale) grid with the peak at the (0.03, 2.0) cell — the 2D
    surface analogue of `_peak_profile`. Param names come from the label's
    capitalization (`Cscale`), not the lower-cased dir segment."""
    cells = [
        ("g_0.0100__cscale_0.5000", "g=0.0100__Cscale=0.5000", -8.0, {"r1": 0.01}),
        ("g_0.0100__cscale_2.0000", "g=0.0100__Cscale=2.0000", -6.0, {"r1": 0.02}),
        ("g_0.0300__cscale_0.5000", "g=0.0300__Cscale=0.5000", -3.0, {"r1": 0.03}),
        # MLE cell, with the nuisance params it optimized to:
        ("g_0.0300__cscale_2.0000", "g=0.0300__Cscale=2.0000", -0.5,
         {"r1": 0.042, "phi_inv": 0.2}),
    ]
    for seg, label, ll, nu in cells:
        _leaf(profiles_dir, base, base.split("-")[0], seg, label, ll, nuisance=nu)
    # A worse restart at the MLE cell with DIFFERENT nuisance — the reported
    # nuisance must come from the winning (best_loglik) restart, not this one.
    _leaf(profiles_dir, base, base.split("-")[0], "g_0.0300__cscale_2.0000",
          "g=0.0300__Cscale=2.0000", -5.0, start=1, nuisance={"r1": 9.9})


def test_load_profile_aggregates_starts_and_orders(tmp_path):
    store = tmp_path / "fits"
    _peak_profile(tmp_path / "profiles")

    c = load_profile(store, "demo-aa")
    assert c is not None
    assert c.params == ("g",) and c.ndim == 1 and c.method == "nl-sbplx"
    assert [p.coords[0] for p in c.points] == [0.1, 0.5, 1.0, 1.5, 2.0]  # ascending
    peak = next(p for p in c.points if p.coords[0] == 1.0)
    assert peak.loglik == -0.5 and peak.n_starts == 2  # max over 2 restarts
    assert c.mle.coords == (1.0,) and c.mle.loglik == -0.5


def test_load_2d_profile_surface(tmp_path):
    _grid_2d(tmp_path / "profiles")
    c = load_profile(tmp_path / "fits", "surf-ee")
    assert c is not None
    assert c.params == ("g", "Cscale") and c.ndim == 2
    assert len(c.points) == 4 and all(len(p.coords) == 2 for p in c.points)
    assert c.mle.coords == (0.03, 2.0) and c.mle.loglik == -0.5
    # The MLE cell reports the WINNING restart's nuisance MLE, not the -5.0 one.
    assert c.mle.nuisance == {"r1": 0.042, "phi_inv": 0.2}
    # 2D region uses ½·χ²(2) = 2.996, not the 1D 1.92; no 1D CI bracket.
    assert c.ci_drop == pytest.approx(2.9957, abs=1e-3)
    assert c.ci_bounds_1d() == (None, None)


def test_ci_bounds_interpolated_both_sides(tmp_path):
    _peak_profile(tmp_path / "profiles")
    c = load_profile(tmp_path / "fits", "demo-aa")
    lo, hi = c.ci_bounds_1d()
    # thresh = -0.5 − 1.9207 = −2.4207; crossings interpolated on [0.5,1.0] & [1.0,1.5]:
    #   lo = 0.5 + (−2.4207 − −3)/(−0.5 − −3)·0.5 = 0.6159 (hi symmetric).
    assert lo == pytest.approx(0.6159, abs=0.01)
    assert hi == pytest.approx(1.3841, abs=0.01)


def test_ci_open_when_mle_at_grid_edge(tmp_path):
    # Monotone-increasing loglik → MLE at the largest value → upper CI unbounded.
    for seg, label, ll in [("g_0.1000", "g=0.1000", -20.0),
                           ("g_1.0000", "g=1.0000", -5.0),
                           ("g_2.0000", "g=2.0000", -1.0)]:
        _leaf(tmp_path / "profiles", "edge-bb", "edge", seg, label, ll)
    c = load_profile(tmp_path / "fits", "edge-bb")
    lo, hi = c.ci_bounds_1d()
    assert c.mle.coords == (2.0,)
    assert lo is not None and hi is None  # bounded below, open above (grid edge)


def test_ci_ignores_sentinel_points(tmp_path):
    # A clean peak plus a failed (−1e100 sentinel) cell at the edge. The CI must
    # be read off the FITTED points, not dragged toward the sentinel, and the
    # sentinel must not win the MLE.
    pd = tmp_path / "profiles"
    for seg, label, ll in [("g_0.5000", "g=0.5000", -3.0),
                           ("g_1.0000", "g=1.0000", -0.5),
                           ("g_1.5000", "g=1.5000", -3.0),
                           ("g_2.0000", "g=2.0000", -1e100)]:  # failed cell
        _leaf(pd, "sent-ff", "sent", seg, label, ll)
    c = load_profile(tmp_path / "fits", "sent-ff")
    assert c.mle.coords == (1.0,)  # the sentinel doesn't win the argmax
    lo, hi = c.ci_bounds_1d()
    assert lo == pytest.approx(0.6159, abs=0.01)  # same as the sentinel-free case
    assert hi == pytest.approx(1.3841, abs=0.01)


def test_all_sentinel_profile_has_no_ci(tmp_path):
    # A just-started profile whose every point failed: it still loads (so the UI
    # can show "no valid optimum yet"), but there's no CI to bracket.
    pd = tmp_path / "profiles"
    for seg, label in [("g_0.5000", "g=0.5000"), ("g_1.0000", "g=1.0000")]:
        _leaf(pd, "dead-gg", "dead", seg, label, -1e100)
    c = load_profile(tmp_path / "fits", "dead-gg")
    assert c is not None and len(c.points) == 2
    assert c.ci_bounds_1d() == (None, None)


def test_discover_orders_and_skips_empty(tmp_path):
    _peak_profile(tmp_path / "profiles", base="demo-aa")
    _leaf(tmp_path / "profiles", "solo-cc", "solo", "g_0.5000", "g=0.5000", -2.0)
    (tmp_path / "profiles" / "empty-dd").mkdir(parents=True)  # no leaves → skipped
    found = discover_profiles(tmp_path / "fits")
    ids = {c.base_id for c in found}
    assert ids == {"demo-aa", "solo-cc"}  # empty-dd dropped


def test_no_profiles_dir_is_empty(tmp_path):
    assert discover_profiles(tmp_path / "fits") == []


# --- API endpoints -----------------------------------------------------------


def test_profile_endpoints(tmp_path, monkeypatch):
    """`/api/profiles` lists, `/api/profiles/{id}` returns the curve + CI; the
    store's sibling `profiles/` tree is read (results/fits ↔ results/profiles)."""
    store = tmp_path / "results" / "fits"
    store.mkdir(parents=True)
    _peak_profile(tmp_path / "results" / "profiles")
    monkeypatch.setenv("CAMDL_WATCH_STORE", str(store))
    from camdl_watch.api.app import app

    c = TestClient(app)
    lst = c.get("/api/profiles").json()
    assert [p["base_id"] for p in lst] == ["demo-aa"]
    assert lst[0]["params"] == ["g"] and lst[0]["n_points"] == 5

    body = c.get("/api/profiles/demo-aa").json()
    assert body["params"] == ["g"]
    assert body["mle_coords"] == [1.0] and body["mle_loglik"] == -0.5
    assert body["ci_lo"] == pytest.approx(0.6159, abs=0.01)
    assert body["ci_hi"] == pytest.approx(1.3841, abs=0.01)
    assert len(body["points"]) == 5

    assert c.get("/api/profiles/nope").status_code == 404


def test_2d_profile_endpoint(tmp_path, monkeypatch):
    """A 2D profile surfaces as `params=[g, Cscale]`, coord-carrying points, and
    no 1D CI bracket (a region, not an interval)."""
    store = tmp_path / "results" / "fits"
    store.mkdir(parents=True)
    _grid_2d(tmp_path / "results" / "profiles")
    monkeypatch.setenv("CAMDL_WATCH_STORE", str(store))
    from camdl_watch.api.app import app

    c = TestClient(app)
    lst = c.get("/api/profiles").json()
    assert lst[0]["params"] == ["g", "Cscale"] and lst[0]["mle_coords"] == [0.03, 2.0]

    body = c.get("/api/profiles/surf-ee").json()
    assert body["params"] == ["g", "Cscale"]
    assert body["mle_coords"] == [0.03, 2.0] and body["mle_loglik"] == -0.5
    assert body["ci_lo"] is None and body["ci_hi"] is None  # 2D → region, no bracket
    assert body["ci_drop"] == pytest.approx(2.9957, abs=1e-3)
    assert all(len(p["coords"]) == 2 for p in body["points"])
    # The clicked-cell payload: each point carries its conditional nuisance MLE.
    mle_pt = next(p for p in body["points"] if p["coords"] == [0.03, 2.0])
    assert mle_pt["nuisance"] == {"r1": 0.042, "phi_inv": 0.2}
