"""Ingest tests: tail-safety against torn lines, discovery against the real
store, prior extraction."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import json

from camdl_watch import ingest
from camdl_watch.state import ChainBuffer, PriorFamily, Severity

STORE = Path(os.environ.get("CAMDL_WATCH_STORE", Path(__file__).resolve().parents[3] / "results" / "fits"))


def _pgas_run():
    metas = {m.run_id: m for m in ingest.discover_runs(STORE)}
    for k in metas:
        if k.startswith("natbc_dens_hierk_nc_pgas_long"):
            return metas[k]
    pytest.skip("nc_pgas_long run not present")


def test_discover_finds_target_runs():
    metas = {m.run_id: m for m in ingest.discover_runs(STORE)}
    targets = [k for k in metas if k.startswith((
        "natbc_dens_hierk_nc_pgas_long",
        "natbc_dens_mh_long",
        "natbc_mh_long",
    ))]
    # The store is a live working dir that gets pruned between sessions; skip
    # (like the other store-backed tests) when the reference runs aren't present.
    if not targets:
        pytest.skip("reference runs not present in store")
    for t in targets:
        m = metas[t]
        assert m.chain_paths, f"{t} has no chains"
        for p in m.chain_paths.values():
            assert p.stat().st_size > 0, f"{t} picked an empty trace"


def test_discover_skips_empty_posterior_dir():
    # mh_long runs have an empty 01-posterior-* stub; the non-empty one wins.
    metas = {m.run_id: m for m in ingest.discover_runs(STORE)}
    for k, m in metas.items():
        if k.startswith("natbc_dens_mh_long"):
            for p in m.chain_paths.values():
                assert p.stat().st_size > 0
            return
    pytest.skip("natbc_dens_mh_long not present")


def test_stage_is_live_pid(tmp_path: Path):
    seed = tmp_path / "seed_1"
    seed.mkdir()
    lock = seed / ".lock"
    # This test process is, by definition, alive.
    lock.write_text(str(os.getpid()))
    assert ingest.stage_is_live(seed) is True
    # An (almost certainly) unused high PID -> dead.
    lock.write_text("2147483646")
    assert ingest.stage_is_live(seed) is False
    # No lock at all -> not live.
    lock.unlink()
    assert ingest.stage_is_live(seed) is False
    # Garbage lock -> not live (and no exception).
    lock.write_text("not-a-pid")
    assert ingest.stage_is_live(seed) is False


def test_pick_posterior_surfaces_only_live_burnin(tmp_path: Path):
    """An empty (burn-in) trace is hidden by default and when dead, but
    surfaced with has_draws=False when include_warming and the PID is live."""
    run = tmp_path / "cfg-deadbeef"
    seed = run / "01-posterior-abc12345" / "seed_1-06cbd6b3"
    (seed / "chain_1").mkdir(parents=True)
    (seed / "chain_1" / "trace.tsv").write_text("")  # 0 bytes, mid burn-in

    # Default: empty stage is invisible regardless of liveness.
    (seed / ".lock").write_text(str(os.getpid()))
    assert ingest._pick_posterior_dir(run) is None

    # include_warming + live PID -> surfaced as a no-draws (warming) stage.
    picked = ingest._pick_posterior_dir(run, include_warming=True)
    assert picked is not None
    _pdir, _seed, paths, has_draws = picked
    assert has_draws is False
    assert 1 in paths and paths[1].name == "trace.tsv"

    # include_warming but dead PID -> still hidden (killed-mid-burn-in stub).
    (seed / ".lock").write_text("2147483646")
    assert ingest._pick_posterior_dir(run, include_warming=True) is None


def test_pick_posterior_prefers_drawn_over_warming(tmp_path: Path):
    """A stage with draws always wins and reports has_draws=True, even with a
    live empty sibling stage present."""
    run = tmp_path / "cfg-cafef00d"
    drawn = run / "01-posterior-aaaa1111" / "seed_1-06cbd6b3" / "chain_1"
    warming = run / "02-posterior-bbbb2222" / "seed_1-06cbd6b3" / "chain_1"
    drawn.mkdir(parents=True)
    warming.mkdir(parents=True)
    (drawn / "trace.tsv").write_text("step\tR0\n0\t1.1\n1\t1.2\n")
    (warming / "trace.tsv").write_text("")
    (warming.parent / ".lock").write_text(str(os.getpid()))

    picked = ingest._pick_posterior_dir(run, include_warming=True)
    assert picked is not None
    _pdir, _seed, paths, has_draws = picked
    assert has_draws is True
    assert paths[1].stat().st_size > 0


def test_tail_reads_pgas_chain():
    m = _pgas_run()
    cid, path = next(iter(m.chain_paths.items()))
    buf = ChainBuffer(cid=cid, path=path)
    n = ingest.tail_chain(buf)
    assert n > 0
    assert buf.header is not None
    # PGAS aux columns present, normalized iter col present.
    assert "log_likelihood" in buf.aux
    assert "log_posterior" in buf.aux
    assert "trajectory_renewal" in buf.aux
    assert buf.iters.shape[0] == n
    # An estimated param made it into values.
    assert "R0" in buf.values
    assert buf.values["R0"].shape[0] == n


def test_tail_torn_last_line(tmp_path: Path):
    """A half-written final line must be dropped and recovered on next read."""
    p = tmp_path / "trace.tsv"
    header = "sweep\tlog_likelihood\tlog_posterior\tR0\n"
    rows = "".join(f"{i}\t-100.{i}\t-110.{i}\t1.{i}\n" for i in range(5))
    torn = "5\t-99.9\t-10"  # no newline -> torn
    p.write_text(header + rows + torn)

    buf = ChainBuffer(cid=1, path=p)
    n1 = ingest.tail_chain(buf)
    assert n1 == 5, f"expected 5 complete rows, got {n1}"
    assert buf.iters.tolist() == [0, 1, 2, 3, 4]

    # Now the torn line completes and a new full line is appended.
    with p.open("a") as fh:
        fh.write("9.9\n6\t-98.0\t-108.0\t1.6\n")
    n2 = ingest.tail_chain(buf)
    assert n2 == 2, f"expected 2 new rows after completion, got {n2}"
    assert buf.iters.tolist()[-2:] == [5, 6]
    assert buf.values["R0"].shape[0] == 7


def test_tail_incremental_matches_full(tmp_path: Path):
    """Two partial tails must accumulate to the same arrays as one full read."""
    p = tmp_path / "trace.tsv"
    header = "step\tlog_likelihood\tlog_posterior\taccepted\tR0\n"
    p.write_text(header + "0\t-1.0\t-2.0\t1\t1.1\n1\t-1.5\t-2.5\t0\t1.2\n")
    buf = ChainBuffer(cid=1, path=p)
    ingest.tail_chain(buf)
    with p.open("a") as fh:
        fh.write("2\t-1.6\t-2.6\t1\t1.3\n")
    ingest.tail_chain(buf)
    assert buf.iters.tolist() == [0, 1, 2]
    assert np.allclose(buf.values["R0"], [1.1, 1.2, 1.3])
    assert np.allclose(buf.aux["accepted"], [1, 0, 1])


def test_extract_priors_pgas():
    m = _pgas_run()
    priors = ingest.extract_priors(m)
    # fit_toml-sourced
    assert priors["D50"].family is PriorFamily.LOGNORMAL
    assert priors["D50"].args["mu"] == pytest.approx(7.6)
    assert priors["D50"].args["sigma"] == pytest.approx(0.6)
    assert priors["rho"].family is PriorFamily.BETA
    assert priors["rho"].args["alpha"] == pytest.approx(4.0)
    # model_ir-sourced (parsed from .camdl)
    assert priors["mu_k"].family is PriorFamily.NORMAL
    assert priors["mu_k"].args["mu"] == pytest.approx(1.8)
    assert priors["mu_k"].args["sigma"] == pytest.approx(0.7)
    assert priors["tau_k"].family is PriorFamily.HALFNORMAL
    assert priors["tau_k"].args["sigma"] == pytest.approx(0.5)
    # indexed expansion: k_raw_Bo inherits k_raw[patch] ~ normal(0,1)
    assert priors["k_raw_Bo"].family is PriorFamily.NORMAL
    assert priors["k_raw_Bo"].args["mu"] == pytest.approx(0.0)
    assert priors["k_raw_Bo"].args["sigma"] == pytest.approx(1.0)


def test_parse_ir_params_survives_brace_in_comment(tmp_path: Path):
    # A `}` inside a doc comment must NOT truncate the parameters{} block — every
    # param after it (mu, here) was silently dropped, so its prior fell to Flat.
    model = tmp_path / "m.camdl"
    model.write_text(
        "parameters {\n"
        "  abar[v] : count in [0.05, 500.0] ~ log_normal(mu = 2.3, sigma = 1.5)\n"
        "  #' Daily mortality (survival p = e^{-mu} per day) — note the brace.\n"
        "  mu[v]   : rate  in [0.02, 2.0]   ~ log_normal(mu = -1.6, sigma = 0.3)\n"
        "}\n"
    )
    ir = ingest._parse_ir_params(model)
    assert set(ir) == {"abar", "mu"}  # not just {"abar"}
    assert ir["mu"][0] is PriorFamily.LOGNORMAL
    assert ir["mu"][1]["mu"] == pytest.approx(-1.6)


def test_extract_priors_reads_leaf_archived_model(tmp_path: Path):
    # The recorded model_path is *relative* (unresolvable from any other CWD), so
    # a model-declared prior would fall to Flat unless we read the leaf-archived
    # model.camdl.original instead.
    seed = tmp_path / "fits" / "m-aa" / "01-posterior-hh" / "seed_1-hh"
    (seed / "chain_1").mkdir(parents=True)
    (seed / "chain_1" / "trace.tsv").write_text("draw\tabar_v1\n0\t10.0\n1\t11.0\n")
    (tmp_path / "fits" / "m-aa" / "fit.meta.json").write_text(json.dumps({
        "model_path": "fits/../models/does_not_resolve.camdl",
        "estimated": ["abar_v1"],
        "resolved_priors": [{"param": "abar_v1", "source": "model_ir"}],
    }))
    (tmp_path / "fits" / "m-aa" / "fit.toml.original").write_text(
        "[estimate]\nabar_v1 = { start = 10.0 }\n"
    )
    (tmp_path / "fits" / "m-aa" / "model.camdl.original").write_text(
        "parameters {\n  abar[v] : count in [0.05, 500.0]"
        " ~ log_normal(mu = 2.3, sigma = 1.5)\n}\n"
    )
    m = {r.run_id: r for r in ingest.discover_runs(tmp_path / "fits")}["m-aa"]
    spec = ingest.extract_priors(m)["abar_v1"]
    assert spec.family is PriorFamily.LOGNORMAL  # not Flat
    assert spec.source == "model_ir" and spec.bounds == (0.05, 500.0)


def test_derived_label_is_readable():
    m = _pgas_run()
    # config stem recovered from fit.meta.json's fit_toml_path
    assert m.fit_toml_stem == "natbc_dens_hierk_nc_pgas_long"
    lbl = m.derived_label
    assert m.fit_toml_stem in lbl
    assert m.algorithm in lbl  # e.g. "pgas"
    assert m.backend.value in lbl  # e.g. "chain_binomial"
    # no raw hash in the readable label
    assert m.hash not in lbl
    assert m.hash  # but the hash is still recoverable from run_id


def test_display_label_prefers_user_label():
    m = _pgas_run()
    # No native label set on disk -> display falls back to derived.
    assert m.display_label == (m.user_label or m.derived_label)
    if m.user_label is None:
        assert m.display_label == m.derived_label


def test_fit_toml_stem_fallback_to_dirname(tmp_path: Path):
    # When fit.meta.json lacks fit_toml_path, fall back to the dir-name prefix.
    run_dir = tmp_path / "my_cfg_stem-deadbeef"
    stem = ingest._fit_toml_stem({}, run_dir)
    assert stem == "my_cfg_stem"
    # And prefers fit_toml_path when present.
    stem2 = ingest._fit_toml_stem({"fit_toml_path": "other_cfg.toml"}, run_dir)
    assert stem2 == "other_cfg"


def test_read_chain_summary_pgas(tmp_path: Path):
    """PGAS: acceptance_rates [chain][param] -> per-chain reduction; findings
    parsed with severity mapping; null combined ESS preserved as None."""
    seed = tmp_path / "seed_1"
    seed.mkdir()
    (seed / "pgas_summary.json").write_text(json.dumps({
        "stage": "pgas",
        "n_chains": 2,
        "rhat": {"R0": 1.02, "k_obs": 3.19},
        "ess": {"R0": 35.8, "k_obs": None},
        "ess_per_chain": {"R0": [4.1, 10.1], "k_obs": [47.0, 17.0]},
        "acceptance_rates": [[0.99, 0.99], [0.86, 0.86]],
    }))
    (seed / "diagnostics.json").write_text(json.dumps([
        {"kind": {"type": "rhat_high", "param": "k_obs", "rhat": 3.19, "threshold": 1.1},
         "severity": "error", "message": "Rhat for 'k_obs' is 3.19."},
        {"kind": {"type": "acceptance_rate_unhealthy", "rate": 0.99, "param": "R0"},
         "severity": "error", "message": "acceptance 99% outside healthy range [15%, 50%]."},
    ]))
    cs = ingest.read_chain_summary(seed)
    assert cs is not None
    assert cs.stage == "pgas" and cs.n_chains == 2
    assert cs.ess["k_obs"] is None  # not estimable -> None preserved
    assert cs.per_chain_acceptance == pytest.approx([0.99, 0.86])
    assert len(cs.findings) == 2
    assert {f.severity for f in cs.findings} == {Severity.ERROR}
    rhat_f = next(f for f in cs.findings if f.kind == "rhat_high")
    assert rhat_f.param == "k_obs" and rhat_f.detail["rhat"] == pytest.approx(3.19)


def test_read_chain_summary_pmmh_scalar_acceptance(tmp_path: Path):
    """PMMH stores a per-chain scalar acceptance_rate + a MAP point estimate."""
    seed = tmp_path / "seed_1"
    seed.mkdir()
    (seed / "pmmh_summary.json").write_text(json.dumps({
        "stage": "pmmh", "n_chains": 3,
        "rhat": {"R0": 1.01}, "ess": {"R0": 371.0},
        "ess_per_chain": {"R0": [71.0, 98.0, 132.0]},
        "acceptance_rate": [0.145, 0.146, 0.139],
        "map_chain": 2, "map_loglik": -407.3, "map_params": {"R0": 2.06},
    }))
    cs = ingest.read_chain_summary(seed)
    assert cs is not None and cs.stage == "pmmh"
    assert cs.per_chain_acceptance == pytest.approx([0.145, 0.146, 0.139])
    assert cs.map_chain == 2 and cs.map_loglik == pytest.approx(-407.3)
    assert cs.map_params["R0"] == pytest.approx(2.06)


def test_read_chain_summary_absent_is_none(tmp_path: Path):
    seed = tmp_path / "seed_1"
    seed.mkdir()
    assert ingest.read_chain_summary(seed) is None


def test_read_algorithm_config_pgas(tmp_path: Path):
    """PGAS records the particle count and sweep budget; every key passes
    through, including the ones the header strip won't end up showing."""
    seed = tmp_path / "seed_1"
    seed.mkdir()
    (seed / "run.json").write_text(json.dumps({
        "inputs": {
            "algorithm": {"algorithm": "pgas", "backend": "chain_binomial",
                          "chains": 8, "particles": 1200, "sweeps": 40000},
            "wall_time_seconds": 13426.3,
        },
    }))
    cfg = ingest._read_algorithm_config(seed)
    assert cfg["particles"] == 1200
    assert cfg["sweeps"] == 40000
    assert cfg["algorithm"] == "pgas" and cfg["chains"] == 8


def test_read_algorithm_config_keys_are_algorithm_specific(tmp_path: Path):
    """An MH stage has no particles and counts `iterations`; an NLopt scout
    carries `max_evals`/`tolerance`. Nothing is invented or dropped."""
    mh = tmp_path / "mh"
    mh.mkdir()
    (mh / "run.json").write_text(json.dumps({"inputs": {"algorithm": {
        "algorithm": "mh", "backend": "ode", "chains": 8, "iterations": 60000,
    }}}))
    cfg = ingest._read_algorithm_config(mh)
    assert "particles" not in cfg
    assert cfg["iterations"] == 60000

    scout = tmp_path / "scout"
    scout.mkdir()
    (scout / "run.json").write_text(json.dumps({"inputs": {"algorithm": {
        "algorithm": "nl-sbplx", "backend": "ode", "chains": 600,
        "max_evals": 60000, "tolerance": 1e-6,
    }}}))
    cfg = ingest._read_algorithm_config(scout)
    assert cfg["max_evals"] == 60000
    assert cfg["tolerance"] == pytest.approx(1e-6)


def test_read_algorithm_config_degrades_to_empty(tmp_path: Path):
    """No run.json, unparseable bytes, or a run predating the field — all read
    as "nothing recorded" rather than raising and taking the run down."""
    seed = tmp_path / "seed_1"
    seed.mkdir()
    assert ingest._read_algorithm_config(seed) == {}

    (seed / "run.json").write_text("{not json")
    assert ingest._read_algorithm_config(seed) == {}

    (seed / "run.json").write_text(json.dumps({"inputs": {"seed": 1}}))
    assert ingest._read_algorithm_config(seed) == {}

    # Present but the wrong shape (a bare string where the block should be).
    (seed / "run.json").write_text(json.dumps({"inputs": {"algorithm": "pgas"}}))
    assert ingest._read_algorithm_config(seed) == {}


def test_sample_prior_respects_bounds():
    from camdl_watch.state import PriorSpec

    spec = PriorSpec("D50", PriorFamily.LOGNORMAL, {"mu": 7.6, "sigma": 0.6},
                     bounds=(200.0, 20000.0))
    x = ingest.sample_prior(spec, 5000)
    assert x.size > 0
    assert x.min() >= 200.0 and x.max() <= 20000.0

    flat = PriorSpec("k_raw", PriorFamily.FLAT, {}, bounds=(-5.0, 5.0))
    xf = ingest.sample_prior(flat, 5000)
    assert xf.min() >= -5.0 and xf.max() <= 5.0
