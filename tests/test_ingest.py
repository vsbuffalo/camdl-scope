"""Ingest tests: tail-safety against torn lines, discovery against the real
store, prior extraction."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import json

from camdl_scope import ingest
from camdl_scope.state import ChainBuffer, PriorFamily, Severity

STORE = Path(os.environ.get("CAMDL_SCOPE_STORE", Path(__file__).resolve().parents[3] / "results" / "fits"))


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


def _leaf(run: Path, method: str, label: str | None, trace: str) -> Path:
    seed = run / method / "seed_1-06cbd6b3"
    (seed / "chain_1").mkdir(parents=True)
    (seed / "chain_1" / "trace.tsv").write_text(trace)
    prov = {"argv": ["camdl", "fit", "run"]} | ({"label": label} if label else {})
    (seed / "run.json").write_text(json.dumps({"run_id": method, "provenance": prov}))
    return seed


def test_label_is_the_shown_leafs_own(tmp_path: Path):
    """camdl gh#896: a label names one run and lives on that leaf's run.json
    (``provenance.label``). A folder with two runs holds two labels; the entry
    shows the label of the leaf it reads, never a folder-level one."""
    run = tmp_path / "fits" / "cfg-cafef00d"
    _leaf(run, "pgas-aaaa1111", "long", "sweep\tR0\n0\t1.1\n1\t1.2\n")
    _leaf(run, "pgas-bbbb2222", "short", "")  # empty sibling: not the one shown
    # An older camdl's folder-level label: camdl no longer reads it, nor do we.
    (run / "fit.meta.json").write_text(json.dumps({"label": "stale"}))
    m = {r.run_id: r for r in ingest.discover_runs(tmp_path / "fits")}["cfg-cafef00d"]
    assert m.posterior_dir.parent.name == "pgas-aaaa1111"
    assert m.user_label == "long"
    assert m.leaf_run_id == "pgas-aaaa1111"
    assert m.display_label == "long"


def test_unlabelled_leaf_falls_back_to_derived(tmp_path: Path):
    run = tmp_path / "fits" / "cfg-deadbeef"
    _leaf(run, "pgas-aaaa1111", None, "sweep\tR0\n0\t1.1\n")
    m = ingest.discover_runs(tmp_path / "fits")[0]
    assert m.user_label is None
    assert m.display_label == m.derived_label
    assert ingest._read_leaf_identity(tmp_path / "absent") == (None, None)


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
    from camdl_scope.state import PriorSpec

    spec = PriorSpec("D50", PriorFamily.LOGNORMAL, {"mu": 7.6, "sigma": 0.6},
                     bounds=(200.0, 20000.0))
    x = ingest.sample_prior(spec, 5000)
    assert x.size > 0
    assert x.min() >= 200.0 and x.max() <= 20000.0

    flat = PriorSpec("k_raw", PriorFamily.FLAT, {}, bounds=(-5.0, 5.0))
    xf = ingest.sample_prior(flat, 5000)
    assert xf.min() >= -5.0 and xf.max() <= 5.0


def test_log_uniform_prior_resolves_and_samples(tmp_path: Path):
    # `~ log_uniform(lower=, upper=)` is camdl's spelling for a scale
    # parameter's prior (kappa in every province model). Unknown to the family
    # table it fell to Flat on the bounds, whose *linear*-uniform draws are
    # invisible binned on a log axis — read as "no prior" in the pair plot.
    model = tmp_path / "m.camdl"
    model.write_text(
        "parameters {\n"
        "  kappa : positive in [1e-9, 1e-2]"
        " ~ log_uniform(lower = 1e-9, upper = 1e-2)\n"
        "}\n"
    )
    ir = ingest._parse_ir_params(model)
    parsed = ir["kappa"]
    assert parsed.family is PriorFamily.LOGUNIFORM
    assert parsed.args["lo"] == pytest.approx(1e-9)
    assert parsed.args["hi"] == pytest.approx(1e-2)

    from camdl_scope.state import PriorSpec

    spec = PriorSpec("kappa", parsed.family, parsed.args, bounds=parsed.bounds)
    x = ingest.sample_prior(spec, 5000)
    assert x.size > 0
    assert x.min() >= 1e-9 and x.max() <= 1e-2
    # Uniform on the log scale: each decade carries ~1/7 of the mass, so the
    # bottom decade is populated — the linear-uniform fallback put ~0 there.
    assert np.mean(x < 1e-8) == pytest.approx(1.0 / 7.0, abs=0.03)

    lp = ingest.log_prior_density(spec, np.array([1e-5, 1e-1]))
    assert lp[0] == pytest.approx(-np.log(1e-5) - np.log(np.log(1e7)))
    assert lp[1] == -np.inf


def test_uniform_prior_accepts_lower_upper_spelling(tmp_path: Path):
    # camdl's canonical uniform args are `lower`/`upper`; the table only knew
    # lo/hi aliases, so the args silently dropped and the prior fell to (0, 1).
    model = tmp_path / "m.camdl"
    model.write_text(
        "parameters {\n"
        "  w : rate in [2.0, 9.0] ~ uniform(lower = 2.0, upper = 9.0)\n"
        "}\n"
    )
    parsed = ingest._parse_ir_params(model)["w"]
    assert parsed.family is PriorFamily.UNIFORM
    assert parsed.args == {"lo": pytest.approx(2.0), "hi": pytest.approx(9.0)}


def test_exponential_prior_reads_rate_not_scale(tmp_path: Path):
    # `~ exponential(rate = R)` is camdl's process-noise prior (sigma_se in the
    # pcnoise models). Missing from the family table it fell to Flat on its box,
    # replacing a prior with mean 1/R by a uniform with mean (lo+hi)/2 — for
    # rate 5.99 on [0, 2], 0.167 became 1.0, and every contraction measured
    # against it overstated what the data had done.
    model = tmp_path / "m.camdl"
    model.write_text(
        "parameters {\n"
        "  sigma_se : positive in [0.0, 2.0] ~ exponential(rate = 5.99)\n"
        "}\n"
    )
    parsed = ingest._parse_ir_params(model)["sigma_se"]
    assert parsed.family is PriorFamily.EXPONENTIAL
    assert parsed.args == {"rate": pytest.approx(5.99)}
    assert parsed.unsupported is None

    from camdl_scope.state import PriorSpec

    spec = PriorSpec("sigma_se", parsed.family, parsed.args, bounds=parsed.bounds)
    x = ingest.sample_prior(spec, 40_000)
    # Mean 1/rate, not rate: the scale reading would put this at ~6, outside
    # the box entirely. Truncation at 2.0 removes exp(-11.98) of the mass, so
    # the truncated mean is the untruncated one to well within tolerance.
    assert x.mean() == pytest.approx(1.0 / 5.99, rel=0.02)
    # The design the model's doc comment states: P(sigma_se > 0.5) = 0.05.
    assert np.mean(x > 0.5) == pytest.approx(0.05, abs=0.005)

    lp = ingest.log_prior_density(spec, np.array([0.1, 3.0]))
    assert lp[0] == pytest.approx(np.log(5.99) - 5.99 * 0.1)
    assert lp[1] == -np.inf  # outside the declared box


def test_truncated_normal_normalises_over_the_declared_box(tmp_path: Path):
    # A .camdl writes `truncated_normal(mean =, sd =)` and takes the interval
    # from `in [lo, hi]`, so the density has to normalise over the box rather
    # than over the line.
    model = tmp_path / "m.camdl"
    model.write_text(
        "parameters {\n"
        "  take : probability in [0.3, 1.0] ~ truncated_normal(mean = 0.7, sd = 0.2)\n"
        "}\n"
    )
    parsed = ingest._parse_ir_params(model)["take"]
    assert parsed.family is PriorFamily.TRUNCNORMAL
    assert parsed.args == {"mu": pytest.approx(0.7), "sigma": pytest.approx(0.2)}

    from camdl_scope.state import PriorSpec

    spec = PriorSpec("take", parsed.family, parsed.args, bounds=parsed.bounds)
    x = ingest.sample_prior(spec, 20_000)
    assert x.size > 19_000  # sampled from the truncation, not rejected into it
    assert x.min() >= 0.3 and x.max() <= 1.0

    grid = np.linspace(0.3, 1.0, 20_001)
    dens = np.exp(ingest.log_prior_density(spec, grid))
    assert np.trapezoid(dens, grid) == pytest.approx(1.0, abs=1e-3)


def test_unknown_prior_family_is_recorded_not_silently_flattened(tmp_path: Path):
    # The failure this whole family table has: a distribution camdl grows and
    # this reader lacks still parses, as Flat on the declared box. That is the
    # right thing to draw and the wrong thing to draw *silently* — the bounds
    # survive, the density does not, and nothing downstream could tell the
    # substitution from a prior the modeller wrote as flat.
    model = tmp_path / "m.camdl"
    model.write_text(
        "parameters {\n"
        "  q : real in [-1.0, 1.0] ~ von_mises(mu = 0.0, kappa = 2.0)\n"
        "}\n"
    )
    parsed = ingest._parse_ir_params(model)["q"]
    assert parsed.family is PriorFamily.FLAT
    assert parsed.bounds == (-1.0, 1.0)
    assert parsed.unsupported == "von_mises"

    # A box with no `~` at all is a flat prior the model declared, not a gap.
    model.write_text("parameters {\n  q : real in [-1.0, 1.0]\n}\n")
    assert ingest._parse_ir_params(model)["q"].unsupported is None


def test_unknown_toml_prior_family_is_recorded(tmp_path: Path):
    # Same gap on the fit.toml side, where an override can name a family the
    # model file never did.
    entry = {"bounds": [0.0, 2.0], "prior": {"von_mises": {"kappa": 2.0}}}
    parsed = ingest._prior_from_toml_entry(entry)
    assert parsed.family is PriorFamily.FLAT
    assert parsed.bounds == (0.0, 2.0)
    assert parsed.unsupported == "von_mises"

    ok = ingest._prior_from_toml_entry(
        {"bounds": [0.0, 2.0], "prior": {"exponential": {"rate": 5.99}}}
    )
    assert ok.family is PriorFamily.EXPONENTIAL
    assert ok.args == {"rate": pytest.approx(5.99)}
    assert ok.unsupported is None


def test_sampler_telemetry_is_diagnostic_without_a_declared_schema():
    """A stage whose ``run.json`` declares no column roles falls back to
    AUX_COLUMNS. When that list omitted the HMC/NUTS block, `n_divergent` was
    filed as a *parameter* — a bucket nothing reads, since RunState.params comes
    from the fit's declared estimands — so divergences were parsed off disk
    every refresh and silently dropped, and the run read as clean."""
    from camdl_scope.state import ChainBuffer

    buf = ChainBuffer(cid=1, path=Path("/tmp"))  # no diagnostic_cols declared
    for col in ("n_divergent", "step_size", "tree_depth", "n_leapfrog",
                "accept_stat", "energy"):
        assert buf.is_diagnostic(col), f"{col} must not be filed as a parameter"
    # A declared schema still wins over the fallback, in both directions.
    declared = ChainBuffer(cid=1, path=Path("/tmp"),
                           diagnostic_cols=frozenset({"custom_metric"}))
    assert declared.is_diagnostic("custom_metric")
    assert not declared.is_diagnostic("n_divergent")


# ----------------------------------------------------------------------------
# The [stages] -> [method] split: one algorithm per config, and a method
# directory named for it with no ordinal and no stage name.
# ----------------------------------------------------------------------------


def _method_leaf(
    run_dir: Path, method_dir: str, *, toml: str, sweeps_ran: int | None = None
) -> Path:
    """A finished posterior leaf under ``run_dir/method_dir``: two chains with
    rows, ``draws.tsv``, and (when ``sweeps_ran`` is given) the leaf's own
    ``run.json`` record of what it ran."""
    seed = run_dir / method_dir / "seed_1-06cbd6b3"
    for cid in (1, 2):
        (seed / f"chain_{cid}").mkdir(parents=True)
        (seed / f"chain_{cid}" / "trace.tsv").write_text(
            "sweep\tbeta\n0\t0.5\n1\t0.52\n"
        )
    (seed / "draws.tsv").write_text("beta\n0.5\n0.52\n")
    (run_dir / "fit.meta.json").write_text(
        json.dumps({"model_path": "models/m.camdl", "estimated": ["beta"]})
    )
    (run_dir / "fit.toml.original").write_text(toml)
    if sweeps_ran is not None:
        (seed / "run.json").write_text(json.dumps({
            "kind": "fit_stage",
            "levels": [{"name": "fit"}, {"name": "method"}, {"name": "seed"}],
            "inputs": {"algorithm": {
                "algorithm": "pgas", "backend": "chain_binomial",
                "chains": 2, "particles": 100, "sweeps": sweeps_ran,
            }},
        }))
    return seed


_METHOD_TOML = (
    '[method]\nalgorithm = "pgas"\nbackend = "chain_binomial"\n'
    "chains = 2\nsweeps = 300\nburn_in = 50\nthin = 2\n"
)


def test_method_layout_is_discovered(tmp_path: Path):
    """The current layout: ``[method]`` in the config and a method directory
    named for the algorithm. The old reader globbed ``[0-9]*-posterior-*`` and
    matched nothing here, so a fit written by a current camdl was invisible."""
    run = tmp_path / "fits" / "pgas_demo-62fdd451"
    _method_leaf(run, "pgas-dd1b21fd", toml=_METHOD_TOML)

    runs = ingest.discover_runs(tmp_path / "fits")
    assert len(runs) == 1
    m = runs[0]
    assert m.posterior_dir.parent.name == "pgas-dd1b21fd"
    assert (m.algorithm, m.backend.value) == ("pgas", "chain_binomial")
    assert (m.target_sweeps, m.declared_burn_in, m.thin) == (300, 50, 2)


def test_stage_named_anything_but_posterior_is_discovered(tmp_path: Path):
    """A pre-split stage the user called something other than ``posterior``.
    Both ``01-post-*`` and ``01-fit-*`` exist in real stores; the name glob
    skipped them and said nothing, so the fit simply never appeared."""
    run = tmp_path / "fits" / "m1-48d2340a"
    _method_leaf(
        run,
        "01-post-1be97327",
        toml=('[stages.post]\nalgorithm = "pgas"\nbackend = "chain_binomial"\n'
              "sweeps = 900\nburn_in = 100\n"),
    )

    runs = ingest.discover_runs(tmp_path / "fits")
    assert len(runs) == 1
    # …and the stage's own name resolves its [stages.<name>] entry, so the
    # settings come through instead of defaulting to "unknown".
    assert runs[0].algorithm == "pgas"
    assert (runs[0].target_sweeps, runs[0].declared_burn_in) == (900, 100)


def test_config_describing_a_sibling_leaf_is_not_attributed(tmp_path: Path):
    """``fit.toml.original`` belongs to the segment, not the leaf. Re-running one
    config after editing it leaves several leaves under a file describing only
    the last, and reporting its sweep budget against an earlier leaf makes a
    finished fit read as half-done forever."""
    run = tmp_path / "fits" / "fit_base-abde5fe0"
    # The config on disk says 40000 sweeps; this leaf ran 20000.
    seed = _method_leaf(
        run, "01-posterior-dd00a557",
        toml=('[stages.posterior]\nalgorithm = "pgas"\nbackend = "chain_binomial"\n'
              "sweeps = 40000\nburn_in = 2000\nthin = 5\n"),
        sweeps_ran=20000,
    )
    assert seed.exists()

    m = ingest.discover_runs(tmp_path / "fits")[0]
    assert m.target_sweeps == 20000, "the leaf's own record must win"
    # Same algorithm, so the config is still attributable for what the leaf's
    # record does not carry.
    assert (m.declared_burn_in, m.thin) == (2000, 5)


def test_config_for_a_different_algorithm_is_refused(tmp_path: Path):
    """Switching a config's algorithm and re-running lands an IF2 leaf beside
    the PGAS one under a config now describing only IF2. None of its numbers
    describe the PGAS leaf, so none of them are reported against it."""
    run = tmp_path / "fits" / "pgas_demo-62fdd451"
    _method_leaf(
        run, "pgas-969fb88c",
        toml=('[method]\nalgorithm = "if2"\nbackend = "chain_binomial"\n'
              "iterations = 25\ncooling = 0.7\n"),
        sweeps_ran=120,
    )

    m = ingest.discover_runs(tmp_path / "fits")[0]
    assert m.algorithm == "pgas"          # from the leaf, not the config
    assert m.target_sweeps == 120         # not the config's 25 iterations
    assert m.declared_burn_in is None     # unknown beats a sibling's number


def test_point_estimate_leaf_is_classified_by_its_artifact(tmp_path: Path):
    """``mle_params.toml`` marks a point estimate. The reader used to find it by
    globbing for a non-``posterior`` stage name; a current optimizer leaf is
    ``if2-<hash>``, which that glob missed."""
    run = tmp_path / "fits" / "if2_demo-62fdd451"
    seed = run / "if2-802e7013" / "seed_1-06cbd6b3"
    seed.mkdir(parents=True)
    (run / "fit.meta.json").write_text(
        json.dumps({"model_path": "models/m.camdl", "estimated": ["beta"]})
    )
    (run / "fit.toml.original").write_text(
        '[method]\nalgorithm = "if2"\nbackend = "chain_binomial"\niterations = 25\n'
    )
    (seed / "mle_params.toml").write_text(
        "beta = 0.61\n\n[provenance]\nlog_likelihood = -543.4\nmethod = \"if2\"\n"
    )

    runs = ingest.discover_runs(tmp_path / "fits")
    assert len(runs) == 1
    assert runs[0].fit_kind == "mle"
    assert runs[0].algorithm == "if2"
    assert runs[0].posterior_dir == seed


def test_summary_is_found_for_any_sampler(tmp_path: Path):
    """The summary was looked up by an enumerated list of algorithm names, so a
    sampler not on it — ``mh``, which is in stores today — reached the viewer
    with no R̂, no ESS and no findings."""
    seed = tmp_path / "mh-aaaa1111" / "seed_1-bbbb"
    seed.mkdir(parents=True)
    (seed / "mh_summary.json").write_text(json.dumps({
        "stage": "mh", "n_chains": 2, "rhat": {"beta": 1.01},
        "ess": {"beta": 420.0}, "thin": 3,
    }))

    cs = ingest.read_chain_summary(seed)
    assert cs is not None
    assert cs.rhat == {"beta": 1.01} and cs.thin == 3


def test_bad_init_chain_ids_are_reported_one_based(tmp_path: Path):
    """camdl numbered ``bad_init.chain_id`` from 0 while numbering the
    ``chain_N/`` directories, its own finding message and ``progress.json`` from
    1; the release that tags summaries with ``schema`` makes them agree. Ingest
    normalizes both vintages to 1-based, so nothing downstream has to know
    which camdl wrote the store."""
    def _leaf(dirname: str, summary: dict) -> Path:
        seed = tmp_path / dirname / "seed_1-bbbb"
        seed.mkdir(parents=True)
        (seed / "pgas_summary.json").write_text(json.dumps(summary))
        (seed / "diagnostics.json").write_text(json.dumps([
            {"kind": {"type": "bad_init", "chain_id": 6, "reason": "non-finite"},
             "message": "Chain 7 skipped", "severity": "error"},
        ]))
        return seed

    untagged = ingest.read_chain_summary(_leaf("pgas-old", {"n_chains": 8}))
    assert untagged is not None
    assert untagged.findings[0].detail["chain_id"] == 7  # 0-based on disk

    tagged = ingest.read_chain_summary(
        _leaf("pgas-new", {"schema": "camdl.fit-summary/v1", "n_chains": 8})
    )
    assert tagged is not None
    assert tagged.findings[0].detail["chain_id"] == 6  # already 1-based


def _seed_leaf(run: Path, n: int, label: str, mtime: float) -> Path:
    """A finished seed run: draws.tsv, a non-empty trace, its own label."""
    seed = run / "pmmh-83841c1b" / f"seed_{n}-{n:08x}"
    (seed / "chain_1").mkdir(parents=True)
    (seed / "chain_1" / "trace.tsv").write_text("step\tR0\n0\t1.1\n10\t1.2\n")
    (seed / "draws.tsv").write_text("chain\tdraw\tR0\n1\t0\t1.2\n")
    (seed / "run.json").write_text(json.dumps(
        {"run_id": f"{n}" * 64, "provenance": {"label": label}}))
    for p in seed.rglob("*"):
        os.utime(p, (mtime, mtime))
    return seed


def test_each_seed_run_is_its_own_entry_lowest_seed_default(tmp_path):
    """A fit run with seeds 1-6 used to open as one entry showing whichever
    seed finished last (seed 6, "synthetic-s6"), with nothing saying so. Each
    seed run is now an entry; the folder name opens seed 1."""
    run = tmp_path / "fits" / "fit_synthetic-02e1fdff"
    for n in (1, 2, 3):
        # Seed 3 finished last: the old pick.
        _seed_leaf(run, n, "synthetic" if n == 1 else f"synthetic-s{n}", 1_000 + n)
    metas = ingest.discover_runs(tmp_path / "fits")
    assert [(m.run_id, m.seed, m.folder_default, m.user_label) for m in metas] == [
        ("fit_synthetic-02e1fdff~seed_1-00000001", 1, True, "synthetic"),
        ("fit_synthetic-02e1fdff~seed_2-00000002", 2, False, "synthetic-s2"),
        ("fit_synthetic-02e1fdff~seed_3-00000003", 3, False, "synthetic-s3"),
    ]
    assert ingest.resolve_run(metas, "fit_synthetic-02e1fdff").seed == 1
    assert ingest.resolve_run(metas, "fit_synthetic-02e1fdff~seed_3-00000003").seed == 3
    # The folder's hash still reads from the folder, not the seed suffix.
    assert {m.hash for m in metas} == {"02e1fdff"}


def test_a_single_run_folder_keeps_the_folder_name_as_its_id(tmp_path):
    run = tmp_path / "fits" / "fit_r0-84761258"
    _seed_leaf(run, 4, "r0", 1_000)
    (m,) = ingest.discover_runs(tmp_path / "fits")
    assert m.run_id == "fit_r0-84761258" and m.seed == 4 and m.folder_default


def _summary_with(tmp_path: Path, extra: dict) -> "ingest.ChainSummary":
    seed = tmp_path / "seed_1"
    seed.mkdir(exist_ok=True)
    (seed / "pgas_summary.json").write_text(json.dumps(
        {"n_chains": 2, "rhat": {"a": 1.01}, "ess": {"a": 400.0}, **extra}))
    return ingest.read_chain_summary(seed)


def test_summary_says_which_rhat_and_ess_it_holds(tmp_path):
    """camdl changed what ``rhat``/``ess`` hold without changing the keys
    (camdl-scope#7): rank-normalized R-hat and bulk-ESS since 2026-08-22,
    classic R-hat and summed ESS before. The schema tag marks the newer
    contract; untagged summaries from between the change and the tag carry
    rhat_bulk/rhat_classic, which only the newer writer emits."""
    assert _summary_with(tmp_path, {"schema": "camdl.fit-summary/v2"}).stats_vintage == "rank_bulk"
    assert _summary_with(tmp_path, {"rhat_classic": {"a": 1.0}}).stats_vintage == "rank_bulk"
    assert _summary_with(tmp_path, {}).stats_vintage == "classic"


def test_summary_carries_camdls_reason_for_a_missing_rhat(tmp_path):
    s = _summary_with(tmp_path, {
        "schema": "camdl.fit-summary/v2", "method": "pgas",
        "rhat_not_reported": {"b": "non_finite_rhat"},
    })
    assert s.rhat_refused == {"b": "non_finite_rhat"}
    assert s.stage == "pgas"  # the method names the stage when no stage is written
