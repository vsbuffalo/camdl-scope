"""Diagnostics tests: verify our arviz-backed numbers match a direct
``az.rhat`` / ``az.ess`` call on the same post-warmup arrays."""

from __future__ import annotations

import os
from pathlib import Path

import arviz as az
import numpy as np
import pytest

from camdl_scope import diagnostics as dmod
from camdl_scope import ingest
from camdl_scope.state import (
    ChainBuffer,
    ChainSummary,
    Finding,
    PriorFamily,
    PriorSpec,
    RunMeta,
    RunState,
    Severity,
)
from camdl_scope.state import Backend

STORE = Path(os.environ.get("CAMDL_SCOPE_STORE", Path(__file__).resolve().parents[3] / "results" / "fits"))


def _load_run(prefix: str) -> RunState:
    metas = {m.run_id: m for m in ingest.discover_runs(STORE)}
    meta = next((m for k, m in metas.items() if k.startswith(prefix)), None)
    if meta is None:
        pytest.skip(f"{prefix} not present")
    rs = RunState(meta=meta)
    for cid, path in meta.chain_paths.items():
        buf = ChainBuffer(cid=cid, path=path)
        ingest.tail_chain(buf)
        rs.chains[cid] = buf
    rs.priors = ingest.extract_priors(meta)
    return rs


def test_diagnostics_match_arviz_pgas():
    rs = _load_run("natbc_dens_hierk_nc_pgas_long")
    lo, hi = rs.min_iter(), rs.max_iter()
    warmup = int(lo + 0.5 * (hi - lo))
    diag = dmod.compute_diagnostics(rs, warmup)

    # Reproduce the exact post-warmup, common-length array the diagnostics use
    # and call arviz directly.
    param = "R0"
    per_chain = []
    for cid in sorted(rs.chains):
        buf = rs.chains[cid]
        per_chain.append(buf.values[param][buf.iters >= warmup])
    m = min(len(v) for v in per_chain)
    arr = np.stack([v[-m:] for v in per_chain])

    assert diag.per_param[param].rhat == pytest.approx(float(az.rhat(arr)), rel=1e-9, nan_ok=True)
    assert diag.per_param[param].bulk_ess == pytest.approx(
        float(az.ess(arr, method="bulk")), rel=1e-9, nan_ok=True)
    assert diag.per_param[param].tail_ess == pytest.approx(
        float(az.ess(arr, method="tail", prob=(0.05, 0.95))), rel=1e-9, nan_ok=True)
    assert diag.per_param[param].mcse == pytest.approx(float(az.mcse(arr)), rel=1e-9, nan_ok=True)
    assert diag.per_param[param].mean == pytest.approx(float(np.mean(arr[np.isfinite(arr)])))


def test_diagnostics_acceptance_mh():
    rs = _load_run("natbc_mh_long")
    lo, hi = rs.min_iter(), rs.max_iter()
    warmup = int(lo + 0.5 * (hi - lo))
    diag = dmod.compute_diagnostics(rs, warmup)
    # MH traces carry an `accepted` column -> acceptance is computed.
    assert diag.acceptance is not None
    assert 0.0 <= diag.acceptance <= 1.0


def _finding(kind, sev, msg, param=None, **detail):
    return Finding(kind=kind, severity=sev, message=msg, param=param, detail=detail)


def test_summarize_findings_aggregates_repetition():
    # camdl repeats acceptance per param×chain; aggregation collapses to one line.
    findings = [
        _finding("acceptance_rate_unhealthy", Severity.ERROR,
                 "rate 99% outside healthy range [15%, 50%].", param="R0", rate=0.99,
                 band_lo=0.15, band_hi=0.5),
        _finding("acceptance_rate_unhealthy", Severity.ERROR,
                 "rate 99% outside healthy range [15%, 50%].", param="k", rate=0.99),
        _finding("acceptance_rate_unhealthy", Severity.ERROR,
                 "rate 86% outside healthy range [15%, 50%].", param="R0", rate=0.862),
        _finding("rhat_high", Severity.ERROR, "Rhat R0", param="R0", rhat=4.7),
        _finding("rhat_high", Severity.WARN, "Rhat k", param="k", rhat=1.18),
        _finding("max_tree_depth_hits", Severity.ERROR, "tree", n_hits=471, n_sweeps=750,
                 pct=62.8, max_depth=8),
    ]
    groups = dmod.summarize_findings(findings)
    by_kind = {g.kind: g for g in groups}
    # one group per kind, errors first
    assert groups[0].severity is Severity.ERROR
    # acceptance: distinct rates collapsed (2 distinct), band recovered
    acc = by_kind["acceptance_rate_unhealthy"].headline
    assert "2 chain rate(s)" in acc and "86%–99%" in acc and "healthy 15–50%" in acc
    # rhat: worst-first, top params named
    rh = by_kind["rhat_high"].headline
    assert rh.startswith("R̂ high: R0 4.70")
    # tree depth: counts surfaced
    assert "471/750" in by_kind["max_tree_depth_hits"].headline


def _run_with_summary(summary):
    meta = RunMeta(run_id="s", run_dir="/tmp", posterior_dir="/tmp", chain_paths={},
                   model="m", algorithm="pgas", backend=Backend.CHAIN_BINOMIAL,
                   estimated=["R0"], target_sweeps=None, declared_burn_in=None)
    rs = RunState(meta=meta)
    rs.summary = summary
    return rs


def test_per_chain_mixing_prefers_camdl_acceptance():
    cs = ChainSummary(stage="pgas", n_chains=2, rhat={}, ess={}, ess_per_chain={},
                      acceptance_rates=[[0.99, 0.99], [0.86, 0.86]])
    label, values, ids, band = dmod.per_chain_mixing(_run_with_summary(cs), warmup=0)
    assert label == "acceptance"
    assert values == pytest.approx([0.99, 0.86])
    # camdl counts chains from 0 (chain_0 …), so with no trace dirs to read the
    # ids from, a summary's positional array maps to 0..n-1.
    assert band is None and ids == [0, 1]  # no finding carries a band: none drawn


def _band_finding(lo=0.15, hi=0.50, **extra):
    from camdl_scope.state import Finding, Severity
    return Finding(
        kind="acceptance_rate_unhealthy", severity=Severity.ERROR, param=None,
        message="chain acceptance rate 99.0% is outside healthy range.",
        detail={"rate": 0.99, **({"band_lo": lo, "band_hi": hi} if lo is not None else {}),
                **extra})


def test_acceptance_band_is_the_one_camdl_published_in_its_finding_fields():
    cs = ChainSummary(stage="pgas", n_chains=2, rhat={}, ess={}, ess_per_chain={},
                      acceptance_rates=[[0.99], [0.86]], findings=[_band_finding(0.60, 0.95)])
    _l, _v, _i, band = dmod.per_chain_mixing(_run_with_summary(cs), warmup=0)
    assert band == pytest.approx((0.60, 0.95))


def test_per_chain_mixing_ids_come_from_the_run_not_the_position():
    """A summary array is positional; the ids must be the chains actually
    present, so a run whose chains are 3 and 7 does not get relabelled 1, 2."""
    cs = ChainSummary(stage="pgas", n_chains=2, rhat={}, ess={}, ess_per_chain={},
                      acceptance_rates=[[0.4], [0.45]])
    rs = _run_with_summary(cs)
    for cid in (3, 7):
        rs.chains[cid] = ChainBuffer(cid=cid, path=Path("/tmp"))
    _label, _values, ids, _band = dmod.per_chain_mixing(rs, warmup=0)
    assert ids == [3, 7]


def test_per_chain_mixing_live_renewal_when_no_summary():
    # No summary, PGAS trace -> trajectory renewal per chain, no band.
    meta = RunMeta(run_id="s", run_dir="/tmp", posterior_dir="/tmp", chain_paths={},
                   model="m", algorithm="pgas", backend=Backend.CHAIN_BINOMIAL,
                   estimated=["R0"], target_sweeps=None, declared_burn_in=None)
    rs = RunState(meta=meta)
    for cid, r in ((1, 0.3), (2, 0.5)):
        buf = ChainBuffer(cid=cid, path=Path("/tmp"))
        buf.iters = np.arange(10)
        buf.aux = {"trajectory_renewal": np.full(10, r)}
        rs.chains[cid] = buf
    label, values, ids, band = dmod.per_chain_mixing(rs, warmup=0)
    assert label == "trajectory renewal" and band is None
    assert values == pytest.approx([0.3, 0.5])
    assert ids == [1, 2]  # the live path carries the real chain ids


def test_effective_rhat_ess_prefer_summary():
    cs = ChainSummary(stage="pgas", n_chains=2, rhat={"R0": 1.5},
                      ess={"R0": None}, ess_per_chain={})
    rs = _run_with_summary(cs)
    diag = dmod.compute_diagnostics  # noqa: F841 (just touching the module)
    # Build a minimal Diagnostics by hand via the live path on empty chains:
    from camdl_scope.state import Diagnostics, ParamDiag
    d = Diagnostics(per_param={"R0": ParamDiag(rhat=9.9, bulk_ess=5.0, tail_ess=5.0,
                                               mcse=0.1, mean=0.0, sd=1.0)},
                    acceptance=None, n_divergent=None, plateaued=None, plateau_slope=None,
                    chain_separation={}, warmup_cutoff=0, n_tail=10)
    rhat, src = dmod.effective_rhat(d, rs.summary, "R0")
    assert rhat == pytest.approx(1.5) and src == "camdl"  # camdl wins over live 9.9
    ess, src2 = dmod.effective_ess(d, rs.summary, "R0")
    assert ess is None and src2 == "camdl"  # camdl's "not estimable" preserved


# ---------------------------------------------------------------------------
# Plateau-test memory guard (docs/dev/incidents/2026-08-17-plateau-test-oom.md)
# ---------------------------------------------------------------------------


def test_plateau_test_memory_is_bounded_on_long_traces():
    """scipy's theilslopes materializes n×n pairwise intermediates; unthinned,
    a 4-chain × 30k-sweep pooled series would allocate ~29 GB and take the host
    into swap (as the 6k-sweep PGAS fits did at 7 GB/request). The pooled input
    must be thinned to a constant cap before Theil–Sen."""
    import tracemalloc
    from types import SimpleNamespace

    import numpy as np

    from camdl_scope import diagnostics as diag_mod
    from camdl_scope.state import ChainBuffer

    rng = np.random.default_rng(0)
    n = 30_000
    chains = {}
    for cid in range(1, 5):
        buf = ChainBuffer(cid=cid, path=Path(f"/nonexistent/chain_{cid}"))
        buf.iters = np.arange(n, dtype=np.int64)
        # A plateaued ll series: flat with noise.
        buf.aux["log_likelihood"] = -500.0 + rng.normal(0, 1.0, n)
        chains[cid] = buf
    run = SimpleNamespace(chains=chains)  # _plateau_test reads only .chains

    tracemalloc.start()
    plateaued, slope = diag_mod._plateau_test(run)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert plateaued is True
    assert slope is not None and abs(slope) < 1.0
    # Capped Theil–Sen intermediates are ~32 MB; leave generous headroom while
    # staying far below the quadratic regime (which would be gigabytes).
    assert peak < 200 * 1024 * 1024, f"plateau test allocated {peak/2**20:.0f} MB"


def test_sampler_telemetry_panel_sums_counters_and_means_rates():
    """Divergences are events: summing is the only reduction that shows them.
    A mean over 10 draws turns 3 divergences into 0.3 and reads as healthy."""
    meta = RunMeta(run_id="s", run_dir="/tmp", posterior_dir="/tmp", chain_paths={},
                   model="m", algorithm="pgas", backend=Backend.CHAIN_BINOMIAL,
                   estimated=["R0"], target_sweeps=None, declared_burn_in=None)
    rs = RunState(meta=meta)
    buf = ChainBuffer(cid=1, path=Path("/tmp"))
    buf.iters = np.arange(10)
    buf.aux = {
        "n_divergent": np.array([1.0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0]),
        "step_size": np.full(10, 0.25),
        "log_posterior": np.full(10, -3.0),  # excluded: shown in Traces
    }
    rs.chains[1] = buf
    panels = {p.id: p for p in dmod.sampler_panels(rs, warmup=0)}
    tel = panels["sampler-telemetry"]
    cols = {c.key: i for i, c in enumerate(tel.columns)}
    assert "log_posterior" not in cols
    assert tel.rows == [1]
    assert tel.values[0][cols["n_divergent"]] == pytest.approx(3.0)  # summed
    assert tel.values[0][cols["step_size"]] == pytest.approx(0.25)  # mean


def test_unknown_diagnostic_column_still_surfaces():
    """A sampler column the watcher has never heard of must appear (mean, no
    note) rather than be dropped — that silence is what hid divergences."""
    meta = RunMeta(run_id="s", run_dir="/tmp", posterior_dir="/tmp", chain_paths={},
                   model="m", algorithm="pgas", backend=Backend.CHAIN_BINOMIAL,
                   estimated=["R0"], target_sweeps=None, declared_burn_in=None)
    rs = RunState(meta=meta)
    buf = ChainBuffer(cid=1, path=Path("/tmp"))
    buf.iters = np.arange(4)
    buf.aux = {"brand_new_metric": np.array([1.0, 2.0, 3.0, 4.0])}
    rs.chains[1] = buf
    tel = {p.id: p for p in dmod.sampler_panels(rs, warmup=0)}["sampler-telemetry"]
    assert [c.key for c in tel.columns] == ["brand_new_metric"]
    assert tel.values[0][0] == pytest.approx(2.5)


def test_block_acceptance_panel_hidden_when_replicated_across_params():
    """PGAS writes one chain-level rate across the parameter axis today; a grid
    of identical columns would restate the mixing bar, so it stays hidden."""
    cs = ChainSummary(stage="pgas", n_chains=2, rhat={}, ess={}, ess_per_chain={},
                      acceptance_rates=[[0.9, 0.9, 0.9], [0.7, 0.7, 0.7]])
    panels = {p.id for p in dmod.sampler_panels(_run_with_summary(cs), warmup=0)}
    assert "block-acceptance" not in panels


def test_block_acceptance_panel_shown_when_blocks_actually_differ():
    cs = ChainSummary(stage="pgas", n_chains=1, rhat={}, ess={}, ess_per_chain={},
                      acceptance_rates=[[0.02, 0.45, 0.99]])
    rs = _run_with_summary(cs)
    rs.meta.estimated = ["a", "b", "c"]
    panel = {p.id: p for p in dmod.sampler_panels(rs, warmup=0)}["block-acceptance"]
    assert [c.key for c in panel.columns] == ["a", "b", "c"]
    assert panel.values[0] == pytest.approx([0.02, 0.45, 0.99])
    assert all(c.band is None for c in panel.columns)  # camdl published none
    cs.findings.append(_band_finding())
    panel = {p.id: p for p in dmod.sampler_panels(rs, warmup=0)}["block-acceptance"]
    assert all(c.band == pytest.approx((0.15, 0.50)) for c in panel.columns)


def _run_with_param(values: np.ndarray, spec: PriorSpec) -> RunState:
    meta = RunMeta(run_id="s", run_dir="/tmp", posterior_dir="/tmp", chain_paths={},
                   model="m", algorithm="pgas", backend=Backend.CHAIN_BINOMIAL,
                   estimated=["R0"], target_sweeps=None, declared_burn_in=None)
    rs = RunState(meta=meta)
    buf = ChainBuffer(cid=1, path=Path("/tmp"))
    buf.iters = np.arange(values.size)
    buf.values = {"R0": values}
    rs.chains[1] = buf
    rs.priors = {"R0": spec}
    return rs


def test_bound_pressure_names_the_edge_the_draws_are_piled_against():
    """Mass at the lower bound and mass at the upper bound call for opposite
    fixes, so the split must attribute the pressure to the right edge and the
    total must stay the union (the 1% edge regions cannot overlap)."""
    # 20 draws in [0, 10]: 4 pinned at the top, 2 at the bottom, rest interior.
    vals = np.concatenate([np.full(4, 9.99), np.full(2, 0.01), np.linspace(3, 7, 14)])
    spec = PriorSpec(param="R0", family=PriorFamily.FLAT, bounds=(0.0, 10.0))
    row = dmod.prior_posterior(_run_with_param(vals, spec), warmup=0)[0]
    assert row.bounds == (0.0, 10.0)
    assert row.bound_pressure_hi == pytest.approx(4 / 20)
    assert row.bound_pressure_lo == pytest.approx(2 / 20)
    assert row.bound_pressure == pytest.approx(6 / 20)
    assert row.post_median == pytest.approx(float(np.median(vals)))


def test_an_infinite_bound_is_not_a_bound():
    """``bounds = [0, inf]`` is writable in the fit TOML. Taking 1% of an
    infinite width puts every draw inside the edge, which would report a
    healthy posterior as pinned to its box."""
    vals = np.linspace(1.0, 5.0, 20)
    spec = PriorSpec(param="R0", family=PriorFamily.FLAT, bounds=(0.0, float("inf")))
    row = dmod.prior_posterior(_run_with_param(vals, spec), warmup=0)[0]
    assert row.bounds is None
    assert row.bound_pressure is None


def _events_begin(leaf, acceptance):
    import json as _json
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "events.jsonl").write_text(_json.dumps(
        {"schema": "camdl.events/v1", "seq": 0, "elapsed": 0.0, "event": "begin",
         "method": "pmmh", "header": {}, "acceptance": acceptance}) + "\n")


def test_acceptance_band_prefers_begin_then_finding_then_none(tmp_path):
    from camdl_scope.verdict import acceptance_band
    leaf = tmp_path / "pmmh-1" / "seed_1-a"
    # nothing published: no band, and none assumed
    assert acceptance_band(leaf, []) is None
    # an older leaf with no begin band but a finding that carries one
    assert acceptance_band(leaf, [_band_finding(0.1, 0.4)]) == (0.1, 0.4)
    # a finding from a camdl that predates the fields
    assert acceptance_band(leaf, [_band_finding(None)]) is None
    # begin wins over the finding; a null acceptance falls through to it
    _events_begin(leaf, {"kernel": "nuts", "band_lo": 0.6, "band_hi": 0.95})
    assert acceptance_band(leaf, [_band_finding(0.1, 0.4)]) == (0.6, 0.95)
    _events_begin(leaf, None)
    assert acceptance_band(leaf, []) is None
    assert acceptance_band(leaf, [_band_finding(0.1, 0.4)]) == (0.1, 0.4)
