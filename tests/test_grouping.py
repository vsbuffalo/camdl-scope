"""Parameter-grouping tests: indexed-family detection, scalar separation,
default selection, against synthetic lists and the real estimated lists."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from camdl_watch import ingest
from camdl_watch.grouping import group_params

STORE = Path(os.environ.get("CAMDL_WATCH_STORE", Path(__file__).resolve().parents[3] / "results" / "fits"))


def test_hierk_families_and_scalars():
    # The real nc_pgas_long estimated list: 7 scalars/hypers + 14 k_raw leaves.
    params = [
        "R0", "kappa", "rho", "D50", "theta", "mu_k", "tau_k",
        "k_raw_Kailahun", "k_raw_Kenema", "k_raw_Kono", "k_raw_Bo",
        "k_raw_Bonthe", "k_raw_Pujehun", "k_raw_Moyamba", "k_raw_Tonkolili",
        "k_raw_Bombali", "k_raw_Koinadugu", "k_raw_Kambia", "k_raw_Port_Loko",
        "k_raw_Western_Area_Rural", "k_raw_Western_Area_Urban",
    ]
    g = group_params(params)
    # exactly one family: k_raw, with all 14 leaves
    assert set(g.families) == {"k_raw"}
    assert len(g.families["k_raw"]) == 14
    # mu_k, tau_k are scalars (singleton bases mu/tau), NOT a `k` family
    assert "mu_k" in g.scalars and "tau_k" in g.scalars
    assert set(g.scalars) == {"R0", "kappa", "rho", "D50", "theta", "mu_k", "tau_k"}
    # default selection hides the 14 leaves
    dflt = g.default_selection()
    assert "k_raw_Bo" not in dflt
    assert set(dflt) == set(g.scalars)
    # default selection preserves canonical order (scalars in original order)
    assert dflt == ["R0", "kappa", "rho", "D50", "theta", "mu_k", "tau_k"]


def test_all_scalar_run_has_no_families():
    # natbc_mh_long: 5 scalar params, no indexed leaves.
    params = ["R0", "kappa", "rho", "k", "D50"]
    g = group_params(params)
    assert g.families == {}
    assert g.scalars == params
    assert g.default_selection() == params


def test_multi_family_longest_base_wins():
    # Two distinct families plus a deeper-nested one: longest-base assignment.
    params = [
        "alpha", "beta",
        "k_raw_A", "k_raw_B", "k_raw_C",
        "sigma_x", "sigma_y",
    ]
    g = group_params(params)
    assert set(g.families) == {"k_raw", "sigma"}
    assert len(g.families["k_raw"]) == 3
    assert len(g.families["sigma"]) == 2
    assert set(g.scalars) == {"alpha", "beta"}


def test_singleton_indexed_stays_scalar():
    # A base with only one matching member is not a family.
    params = ["foo_bar", "baz"]
    g = group_params(params)
    assert g.families == {}
    assert set(g.scalars) == {"foo_bar", "baz"}


def test_fully_indexed_run_falls_back_to_all():
    # Every param is a family leaf (no scalars) — hiding leaves would leave the
    # pair plot's default selection empty. Fall back to the params themselves.
    params = ["abar_kwaru", "abar_ajura", "mu_kwaru", "mu_ajura"]
    g = group_params(params)
    assert g.scalars == []
    assert g.default_selection() == params


def test_default_selection_capped():
    # A wide fit opens on a legible grid, not an N² wall (the ⚙ control adds more).
    from camdl_watch.grouping import DEFAULT_PAIR_CAP

    params = [f"s{i}" for i in range(20)]  # 20 scalars
    dflt = group_params(params).default_selection()
    assert dflt == params[:DEFAULT_PAIR_CAP] and len(dflt) == DEFAULT_PAIR_CAP


def test_grouping_on_real_run():
    metas = {m.run_id: m for m in ingest.discover_runs(STORE)}
    meta = next((m for k, m in metas.items()
                 if k.startswith("natbc_dens_hierk_nc_pgas_long")), None)
    if meta is None:
        pytest.skip("nc_pgas_long not present")
    g = group_params(list(meta.estimated))
    assert "k_raw" in g.families
    assert len(g.families["k_raw"]) >= 10
    # default view is small (scalars only), << full param count
    assert len(g.default_selection()) < len(meta.estimated)
