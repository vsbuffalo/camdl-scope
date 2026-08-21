"""Diagnostics tests: verify our arviz-backed numbers match a direct
``az.rhat`` / ``az.ess`` call on the same post-warmup arrays."""

from __future__ import annotations

import os
from pathlib import Path

import arviz as az
import numpy as np
import pytest

from camdl_watch import diagnostics as dmod
from camdl_watch import ingest
from camdl_watch.state import (
    ChainBuffer,
    ChainSummary,
    Finding,
    RunMeta,
    RunState,
    Severity,
)
from camdl_watch.state import Backend

STORE = Path(os.environ.get("CAMDL_WATCH_STORE", Path(__file__).resolve().parents[3] / "results" / "fits"))


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
                 "rate 99% outside healthy range [15%, 50%].", param="R0", rate=0.99),
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
    # camdl names chains chain_1…chain_n, so a summary's positional array maps
    # to 1-based ids — never c0, which names no chain.
    assert band == (0.15, 0.50) and ids == [1, 2]


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
    from camdl_watch.state import Diagnostics, ParamDiag
    d = Diagnostics(per_param={"R0": ParamDiag(rhat=9.9, bulk_ess=5.0, tail_ess=5.0,
                                               mcse=0.1, mean=0.0, sd=1.0)},
                    acceptance=None, n_divergent=None, plateaued=None, plateau_slope=None,
                    chain_separation={}, warmup_cutoff=0, n_tail=10)
    rhat, src = dmod.effective_rhat(d, rs.summary, "R0")
    assert rhat == pytest.approx(1.5) and src == "camdl"  # camdl wins over live 9.9
    ess, src2 = dmod.effective_ess(d, rs.summary, "R0")
    assert ess is None and src2 == "camdl"  # camdl's "not estimable" preserved


def test_derive_warnings_defers_rhat_to_camdl():
    # With a summary present, the live R̂/ESS warnings are suppressed (the
    # verdict strip owns them); watcher-only signals still surface.
    rs = _load_run("natbc_dens_hierk_nc_pgas_long")
    lo, hi = rs.min_iter(), rs.max_iter()
    warmup = int(lo + 0.5 * (hi - lo))
    diag = dmod.compute_diagnostics(rs, warmup)
    cs = ChainSummary(stage="pgas", n_chains=len(rs.chains), rhat={}, ess={},
                      ess_per_chain={}, findings=[])
    ws_live = dmod.derive_warnings(diag, rs)
    ws_camdl = dmod.derive_warnings(diag, rs, summary=cs)
    rhat_live = [w for w in ws_live if "R̂" in w.message]
    rhat_camdl = [w for w in ws_camdl if "R̂" in w.message]
    assert len(rhat_camdl) == 0
    assert len(rhat_camdl) <= len(rhat_live)


def test_warnings_shape():
    rs = _load_run("natbc_dens_hierk_nc_pgas_long")
    lo, hi = rs.min_iter(), rs.max_iter()
    warmup = int(lo + 0.5 * (hi - lo))
    diag = dmod.compute_diagnostics(rs, warmup)
    ws = dmod.derive_warnings(diag, rs)
    assert isinstance(ws, list) and ws
    # severities ordered error->warn->info
    order = {"error": 0, "warn": 1, "info": 2}
    sev = [order[w.severity.value] for w in ws]
    assert sev == sorted(sev)


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

    from camdl_watch import diagnostics as diag_mod
    from camdl_watch.state import ChainBuffer

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
