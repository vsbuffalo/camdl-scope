"""Ingest tests: discovery of fit runs in a camdl run store, tail-safety
against torn lines, prior extraction, and the run's own records (summary,
algorithm config, wall time). Every store is built with
:class:`tests.storekit.StoreKit`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from camdl_scope import diagnostics as diag_mod
from camdl_scope import ingest
from camdl_scope.state import ChainBuffer, PriorFamily, Severity
from tests.storekit import StoreKit


@pytest.fixture
def kit(tmp_path: Path) -> StoreKit:
    return StoreKit.create(tmp_path / "camdl-store")


def _trace(leaf: Path, cid: int, text: str) -> Path:
    tp = leaf / f"chain_{cid}" / "trace.tsv"
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(text)
    return tp


def _run_id(leaf: Path) -> str:
    """camdl's run id of the result in ``leaf`` (its ``identity.json``)."""
    return json.loads((leaf / "identity.json").read_text())["run_id"]


def _by_id(root: Path, **kw) -> dict:
    return {m.run_id: m for m in ingest.discover_runs(root, **kw)}


# A hierarchical PGAS fit: two priors overridden in the fit file, the rest
# declared in the model, one of them indexed by patch.
_PGAS_MODEL = """\
parameters {
  D50 : positive in [200.0, 20000.0] ~ log_normal(mu = 1.0, sigma = 1.0)
  rho : probability in [0.0, 1.0]
  mu_k : real in [-5.0, 5.0] ~ normal(mu = 1.8, sigma = 0.7)
  tau_k : positive in [0.0, 5.0] ~ half_normal(sigma = 0.5)
  k_raw[patch] : real in [-5.0, 5.0] ~ normal(mu = 0.0, sigma = 1.0)
}
"""
_PGAS_FIT_TOML = """\
[estimate]
D50 = { bounds = [200.0, 20000.0], prior = { log_normal = { mu = 7.6, sigma = 0.6 } } }
rho = { bounds = [0.0, 1.0], prior = { beta = { alpha = 4.0, beta = 12.0 } } }
"""
_PGAS_ESTIMATED = ["D50", "rho", "mu_k", "tau_k", "k_raw_Bo"]
_PGAS_HEADER = "sweep\tlog_likelihood\tlog_posterior\ttrajectory_renewal\t" + "\t".join(_PGAS_ESTIMATED)


def _pgas_run(kit: StoreKit):
    """A completed PGAS run with two chains of 30 sweeps; its entry."""
    problem = kit.problem("natbc_dens_hierk", source=_PGAS_MODEL)
    leaf = kit.fit_run(
        problem, algorithm="pgas", config="configs/natbc_dens_hierk_nc_pgas_long.toml",
        method={"backend": "chain_binomial", "chains": 2, "sweeps": 40000, "burn_in": 2000},
        estimated=_PGAS_ESTIMATED, fit_toml=_PGAS_FIT_TOML,
        meta={"resolved_priors": [
            {"param": "D50", "source": "fit_toml"}, {"param": "rho", "source": "fit_toml"},
            {"param": "mu_k", "source": "model_ir"}, {"param": "tau_k", "source": "model_ir"},
            {"param": "k_raw_Bo", "source": "model_ir"},
        ]},
    )
    row = "\t-100.0\t-110.0\t1\t" + "\t".join(["0.5"] * len(_PGAS_ESTIMATED))
    for cid in (0, 1):
        _trace(leaf, cid, _PGAS_HEADER + "\n" + "".join(f"{i}{row}\n" for i in range(30)))
    (m,) = ingest.discover_runs(kit.root)
    return m


def test_discovery_reads_the_nonempty_traces(kit):
    # A run with one chain still empty (it has not written its first row)
    # lists only the chain with rows, so nothing downstream plots a header.
    problem = kit.problem("sir")
    leaf = kit.fit_run(problem, algorithm="mh", status="running")
    _trace(leaf, 0, "step\tbeta\n0\t0.5\n1\t0.52\n")
    _trace(leaf, 1, "")
    (m,) = ingest.discover_runs(kit.root)
    assert m.leaf_dir == leaf and m.problem_dir == problem
    assert set(m.chain_paths) == {0}
    assert all(p.stat().st_size > 0 for p in m.chain_paths.values())


def test_live_burn_in_run_is_listed_only_with_include_warming(kit):
    """A live run with nothing written yet is hidden by default, and listed
    with ``include_warming``; once its owner is dead it is hidden either way."""
    live = kit.fit_run(kit.problem("sir", name="warm-aaaa"), algorithm="pgas", status="running")
    _trace(live, 0, "")  # created, mid burn-in
    dead = kit.fit_run(kit.problem("sir", name="dead-bbbb"), algorithm="pgas",
                       status="running", alive=False)
    _trace(dead, 0, "")

    assert ingest.discover_runs(kit.root) == []
    (m,) = ingest.discover_runs(kit.root, include_warming=True)
    assert m.run_id == "warm-aaaa" and m.leaf_dir == live
    # The empty trace is still the chain's path, for the tail to follow.
    assert m.chain_paths[0].name == "trace.tsv"
    assert (m.record_status, m.owner_alive) == ("running", True)


def test_run_with_rows_is_the_default_over_a_live_empty_sibling(kit):
    """A run with draws opens by default, even with a live empty sibling."""
    problem = kit.problem("sir", name="cfg-cafef00d")
    # The empty run has the lower seed, so seed order alone would pick it.
    drawn = kit.fit_run(problem, algorithm="pgas", seed=2)
    _trace(drawn, 0, "step\tR0\n0\t1.1\n1\t1.2\n")
    warming = kit.fit_run(problem, algorithm="pgas", seed=1, status="running")
    _trace(warming, 0, "")
    metas = ingest.discover_runs(kit.root, include_warming=True)
    assert [m.leaf_dir for m in metas] == [drawn, warming]
    assert metas[0].folder_default and not metas[1].folder_default
    assert ingest.resolve_run(metas, "cfg-cafef00d").leaf_dir == drawn
    assert metas[0].chain_paths[0].stat().st_size > 0


def test_tail_reads_pgas_chain(kit):
    m = _pgas_run(kit)
    cid, path = next(iter(m.chain_paths.items()))
    buf = ChainBuffer(cid=cid, path=path)
    n = ingest.tail_chain(buf)
    assert n == 30
    assert buf.header is not None
    # PGAS aux columns present, normalized iter col present.
    assert "log_likelihood" in buf.aux
    assert "log_posterior" in buf.aux
    assert "trajectory_renewal" in buf.aux
    assert buf.iters.shape[0] == n
    # An estimated param made it into values, not aux.
    assert "D50" in buf.values and "D50" not in buf.aux
    assert buf.values["D50"].shape[0] == n


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


def test_extract_priors_pgas(kit):
    m = _pgas_run(kit)
    priors = ingest.extract_priors(m)
    # fit_toml-sourced: the run's fit.toml overrides the model's D50 prior.
    assert priors["D50"].source == "fit_toml"
    assert priors["D50"].family is PriorFamily.LOGNORMAL
    assert priors["D50"].args["mu"] == pytest.approx(7.6)
    assert priors["D50"].args["sigma"] == pytest.approx(0.6)
    assert priors["rho"].family is PriorFamily.BETA
    assert priors["rho"].args["alpha"] == pytest.approx(4.0)
    # model_ir-sourced (parsed from the problem's model.camdl)
    assert priors["mu_k"].source == "model_ir"
    assert priors["mu_k"].family is PriorFamily.NORMAL
    assert priors["mu_k"].args["mu"] == pytest.approx(1.8)
    assert priors["mu_k"].args["sigma"] == pytest.approx(0.7)
    assert priors["tau_k"].family is PriorFamily.HALFNORMAL
    assert priors["tau_k"].args["sigma"] == pytest.approx(0.5)
    # indexed expansion: k_raw_Bo inherits k_raw[patch] ~ normal(0,1)
    assert priors["k_raw_Bo"].family is PriorFamily.NORMAL
    assert priors["k_raw_Bo"].args["mu"] == pytest.approx(0.0)
    assert priors["k_raw_Bo"].args["sigma"] == pytest.approx(1.0)
    assert priors["k_raw_Bo"].bounds == (-5.0, 5.0)


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


def test_extract_priors_reads_the_problems_model(kit):
    # The recorded model_path is *relative* (unresolvable from any other CWD), so
    # a model-declared prior would fall to Flat unless it is read from the
    # model.camdl camdl keeps in the problem folder.
    problem = kit.problem("m", source=(
        "parameters {\n  abar[v] : count in [0.05, 500.0]"
        " ~ log_normal(mu = 2.3, sigma = 1.5)\n}\n"
    ))
    leaf = kit.fit_run(
        problem, algorithm="mh", estimated=["abar_v1"],
        fit_toml="[estimate]\nabar_v1 = { start = 10.0 }\n",
        meta={"model_path": "fits/../models/does_not_resolve.camdl",
              "resolved_priors": [{"param": "abar_v1", "source": "model_ir"}]},
    )
    _trace(leaf, 0, "draw\tabar_v1\n0\t10.0\n1\t11.0\n")
    (m,) = ingest.discover_runs(kit.root)
    spec = ingest.extract_priors(m)["abar_v1"]
    assert spec.family is PriorFamily.LOGNORMAL  # not Flat
    assert spec.args == {"mu": pytest.approx(2.3), "sigma": pytest.approx(1.5)}
    assert spec.source == "model_ir" and spec.bounds == (0.05, 500.0)


def test_derived_label_is_readable(kit):
    m = _pgas_run(kit)
    # The config stem comes from the run's recorded config path.
    assert m.config_stem == "natbc_dens_hierk_nc_pgas_long"
    assert m.model == "natbc_dens_hierk"
    lbl = m.derived_label
    assert lbl == "natbc_dens_hierk_nc_pgas_long · pgas/chain_binomial"
    # no raw hash in the readable label
    assert m.hash not in lbl
    assert m.hash == m.problem_dir.name.rsplit("-", 1)[1]


def _labelled_run(kit: StoreKit, problem: Path, seed: int, label: str | None) -> Path:
    leaf = kit.fit_run(problem, algorithm="pgas", seed=seed, label=label)
    _trace(leaf, 0, "sweep\tR0\n0\t1.1\n1\t1.2\n")
    return leaf


def test_label_is_each_runs_own(kit):
    """A label names one run, through the invocation that carried
    ``--label``. A folder with two runs holds two labels; each entry shows its
    own run's, never a folder-level one."""
    problem = kit.problem("sir", name="cfg-cafef00d")
    long_run = _labelled_run(kit, problem, 1, "long")
    short_run = _labelled_run(kit, problem, 2, "short")
    # A folder-level label in a stray fit.meta.json is not read.
    (problem / "fit.meta.json").write_text(json.dumps({"label": "stale"}))
    by_leaf = {m.leaf_dir: m for m in ingest.discover_runs(kit.root)}
    assert by_leaf[long_run].user_label == "long"
    assert by_leaf[long_run].display_label == "long"
    assert by_leaf[long_run].leaf_run_id == _run_id(long_run)
    assert by_leaf[short_run].user_label == "short"


def test_a_label_moves_to_the_latest_invocation_that_carried_it(kit):
    """camdl's ``@L`` names the latest completed invocation labelled ``L``: a
    re-run with the same label takes it, and the earlier run loses it."""
    problem = kit.problem("sir", name="cfg-beadfeed")
    first = _labelled_run(kit, problem, 1, "best")
    second = _labelled_run(kit, problem, 2, "best")
    by_leaf = {m.leaf_dir: m for m in ingest.discover_runs(kit.root)}
    assert by_leaf[second].user_label == "best"
    assert by_leaf[first].user_label is None
    assert by_leaf[first].display_label == by_leaf[first].derived_label


def test_a_running_invocations_label_is_not_yet_current(kit):
    """A labelled command that has not ended is not ``@L`` yet (camdl counts
    only invocations whose results all completed)."""
    leaf = kit.fit_run(kit.problem("sir"), algorithm="pgas", status="running", label="wip")
    _trace(leaf, 0, "sweep\tR0\n0\t1.1\n")
    (m,) = ingest.discover_runs(kit.root)
    assert m.user_label is None


def test_unlabelled_run_falls_back_to_derived(kit):
    _labelled_run(kit, kit.problem("sir"), 1, None)
    (m,) = ingest.discover_runs(kit.root)
    assert m.user_label is None
    assert m.display_label == m.derived_label == "fit · pgas/ode"


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


def _run_inputs(seed: Path, body: object) -> None:
    seed.mkdir(parents=True, exist_ok=True)
    (seed / "run_inputs.json").write_text(body if isinstance(body, str) else json.dumps(body))


def test_read_algorithm_config_pgas(tmp_path: Path):
    """PGAS records the particle count and sweep budget; every key passes
    through, including the ones the header strip won't end up showing."""
    seed = tmp_path / "seed_1"
    _run_inputs(seed, {"algorithm": {
        "algorithm": "pgas", "backend": "chain_binomial", "chains": 8,
        "particles": 1200, "sweeps": 40000, "adaptive": True,
    }})
    cfg = ingest._read_algorithm_config(seed)
    assert cfg["particles"] == 1200
    assert cfg["sweeps"] == 40000
    assert cfg["algorithm"] == "pgas" and cfg["chains"] == 8
    assert cfg["adaptive"] == "true"  # a bool stays a word, not 1


def test_read_algorithm_config_keys_are_algorithm_specific(tmp_path: Path):
    """An MH run has no particles and counts `iterations`; an NLopt scout
    carries `max_evals`/`tolerance`. Nothing is invented or dropped."""
    mh = tmp_path / "mh"
    _run_inputs(mh, {"algorithm": {
        "algorithm": "mh", "backend": "ode", "chains": 8, "iterations": 60000,
    }})
    cfg = ingest._read_algorithm_config(mh)
    assert "particles" not in cfg
    assert cfg["iterations"] == 60000

    scout = tmp_path / "scout"
    _run_inputs(scout, {"algorithm": {
        "algorithm": "nl-sbplx", "backend": "ode", "chains": 600,
        "max_evals": 60000, "tolerance": 1e-6,
    }})
    cfg = ingest._read_algorithm_config(scout)
    assert cfg["max_evals"] == 60000
    assert cfg["tolerance"] == pytest.approx(1e-6)


def test_read_algorithm_config_degrades_to_empty(tmp_path: Path):
    """No run_inputs.json, unparseable bytes, or one without the block — all
    read as "nothing recorded" rather than raising and taking the run down."""
    seed = tmp_path / "seed_1"
    seed.mkdir()
    assert ingest._read_algorithm_config(seed) == {}

    _run_inputs(seed, "{not json")
    assert ingest._read_algorithm_config(seed) == {}

    _run_inputs(seed, {"seed": 1})
    assert ingest._read_algorithm_config(seed) == {}

    # Present but the wrong shape (a bare string where the block should be).
    _run_inputs(seed, {"algorithm": "pgas"})
    assert ingest._read_algorithm_config(seed) == {}


def test_discovered_run_carries_its_algorithm_config(kit):
    leaf = kit.fit_run(kit.problem("sir"), algorithm="pgas")
    _trace(leaf, 0, "sweep\tR0\n0\t1.1\n")
    _run_inputs(leaf, {"algorithm": {"algorithm": "pgas", "particles": 400}})
    (m,) = ingest.discover_runs(kit.root)
    assert m.algorithm_config == {"algorithm": "pgas", "particles": 400}


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


def test_sampler_telemetry_is_diagnostic():
    """HMC/NUTS telemetry columns are diagnostics, not parameters. When
    AUX_COLUMNS omitted that block, `n_divergent` was filed as a *parameter* —
    a bucket nothing reads, since RunState.params comes from the fit's declared
    estimands — so divergences were parsed off disk every refresh and silently
    dropped, and the run read as clean."""
    buf = ChainBuffer(cid=1, path=Path("unused"))
    for col in ("n_divergent", "step_size", "tree_depth", "n_leapfrog",
                "accept_stat", "energy"):
        assert buf.is_diagnostic(col), f"{col} must not be filed as a parameter"
    assert not buf.is_diagnostic("beta")


# ----------------------------------------------------------------------------
# A run's settings come from its own folder: method.toml for the [method]
# table, the identity for the algorithm.
# ----------------------------------------------------------------------------


def test_method_settings_are_read_from_the_runs_method_toml(kit):
    leaf = kit.fit_run(
        kit.problem("pgas_demo"), algorithm="pgas",
        method={"backend": "chain_binomial", "chains": 2, "sweeps": 300,
                "burn_in": 50, "thin": 2},
    )
    for cid in (0, 1):
        _trace(leaf, cid, "sweep\tbeta\n0\t0.5\n1\t0.52\n")
    (m,) = ingest.discover_runs(kit.root)
    assert m.leaf_dir == leaf
    assert (m.algorithm, m.backend.value) == ("pgas", "chain_binomial")
    assert (m.target_sweeps, m.declared_burn_in, m.thin) == (300, 50, 2)
    assert m.estimated == ["beta", "gamma"]  # from the run's fit.meta.json


def test_each_run_reports_its_own_settings_not_a_siblings(kit):
    """Re-running a problem under edited settings, or another algorithm,
    leaves several runs in one folder. Each reports its own budget: reporting
    a sibling's sweep count against a run makes a finished fit read as
    half-done forever, and an IF2 run's numbers describe no PGAS run."""
    problem = kit.problem("fit_base")
    short = kit.fit_run(problem, algorithm="pgas", seed=1,
                        method={"sweeps": 20000, "burn_in": 2000, "thin": 5})
    long = kit.fit_run(problem, algorithm="pgas", seed=2,
                       method={"sweeps": 40000, "burn_in": 4000})
    if2 = kit.fit_run(problem, algorithm="if2", seed=3, method={"iterations": 25})
    for leaf in (short, long):
        _trace(leaf, 0, "sweep\tbeta\n0\t0.5\n")
    by_leaf = {m.leaf_dir: m for m in ingest.discover_runs(kit.root)}
    assert (by_leaf[short].target_sweeps, by_leaf[short].declared_burn_in,
            by_leaf[short].thin) == (20000, 2000, 5)
    assert (by_leaf[long].target_sweeps, by_leaf[long].declared_burn_in,
            by_leaf[long].thin) == (40000, 4000, 1)
    assert by_leaf[if2].algorithm == "if2" and by_leaf[if2].fit_kind == "mle"
    assert (by_leaf[if2].target_sweeps, by_leaf[if2].declared_burn_in) == (25, None)


def test_point_estimate_run_is_classified_by_its_algorithm(kit):
    """An optimizer's run is a point estimate (``fit_kind == "mle"``)."""
    leaf = kit.fit_run(kit.problem("if2_demo"), algorithm="if2",
                       method={"backend": "chain_binomial", "iterations": 25})
    (leaf / "mle_params.toml").write_text(
        "beta = 0.61\n\n[provenance]\nlog_likelihood = -543.4\nmethod = \"if2\"\n"
    )
    (m,) = ingest.discover_runs(kit.root)
    assert m.fit_kind == "mle"
    assert m.algorithm == "if2"
    assert m.leaf_dir == leaf


def test_scoring_runs_are_not_listed_as_fits(kit):
    """A particle-filter scoring run belongs to Compare, not the fit list."""
    problem = kit.problem("sir")
    kit.fit_run(problem, algorithm="pfilter")
    fit = kit.fit_run(problem, algorithm="mh")
    _trace(fit, 0, "step\tbeta\n0\t0.5\n")
    assert [m.leaf_dir for m in ingest.discover_runs(kit.root)] == [fit]


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


def test_bad_init_chain_ids_follow_camdl_and_the_directories(tmp_path: Path):
    """camdl counts chains from 0 everywhere: its message, ``chain_id`` in
    diagnostics.json, and the ``chain_N/`` directories. A leaf whose chains all
    failed at init writes no ``*_summary.json``; its findings used to be shifted
    by one ("Chain 0" in the message, chain 1 in the dashboard)."""
    seed = tmp_path / "pmmh-bd1183f1" / "seed_1-06cbd6b3"
    seed.mkdir(parents=True)
    (seed / "diagnostics.json").write_text(json.dumps([
        {"kind": {"type": "bad_init", "chain_id": 0, "reason": "no start scored"},
         "severity": "error", "message": "Chain 0 starting parameters were pathological"},
        {"kind": {"type": "bad_init", "chain_id": 1, "reason": "no start scored"},
         "severity": "error", "message": "Chain 1 starting parameters were pathological"},
    ]))
    cs = ingest.read_chain_summary(seed)  # no summary file at all
    assert cs is not None
    assert [f.detail["chain_id"] for f in cs.findings] == [0, 1]
    groups = diag_mod.summarize_findings(cs.findings)
    assert "chains 0, 1" in groups[0].headline

    # The same ids with a tagged summary present: unchanged.
    (seed / "pmmh_summary.json").write_text(
        json.dumps({"schema": "camdl.fit-summary/v2", "n_chains": 2}))
    cs = ingest.read_chain_summary(seed)
    assert [f.detail["chain_id"] for f in cs.findings] == [0, 1]


def _seed_run(kit: StoreKit, problem: Path, n: int, label: str, mtime: float,
              method: str = "pmmh-83841c1b") -> Path:
    """A finished seed run with a non-empty trace and its own label."""
    leaf = kit.fit_run(problem, algorithm="pmmh", seed=n, label=label,
                       method_dir=method, seed_dir=f"seed_{n}-{n:012x}")
    _trace(leaf, 0, "step\tR0\n0\t1.1\n10\t1.2\n")
    for p in leaf.rglob("*"):
        os.utime(p, (mtime, mtime))
    return leaf


def test_each_seed_run_is_its_own_entry_lowest_seed_default(kit):
    """A fit with seeds 1-3: each seed run is an entry with its own label,
    and the folder name opens seed 1 — not whichever seed finished last."""
    problem = kit.problem("sir", name="fit_synthetic-02e1fdff0000")
    for n in (3, 1, 2):
        # Seed 3 finished last (newest mtime) and was written first.
        _seed_run(kit, problem, n, "synthetic" if n == 1 else f"synthetic-s{n}",
                  1_000 + 10 * n)
    metas = ingest.discover_runs(kit.root)
    assert [(m.run_id, m.seed, m.folder_default, m.user_label) for m in metas] == [
        ("fit_synthetic-02e1fdff0000~seed_1-000000000001", 1, True, "synthetic"),
        ("fit_synthetic-02e1fdff0000~seed_2-000000000002", 2, False, "synthetic-s2"),
        ("fit_synthetic-02e1fdff0000~seed_3-000000000003", 3, False, "synthetic-s3"),
    ]
    assert ingest.resolve_run(metas, "fit_synthetic-02e1fdff0000").seed == 1
    assert ingest.resolve_run(metas, "fit_synthetic-02e1fdff0000~seed_3-000000000003").seed == 3
    # The folder's hash still reads from the folder, not the seed suffix.
    assert {m.hash for m in metas} == {"02e1fdff0000"}


def test_runs_under_different_method_dirs_are_not_replicates(kit):
    """One plain run (seed 2) and one warm-started from its posterior (seed 1,
    ``--starts from_posterior=...``). The starts change camdl's method hash, so
    they sit in different method dirs. Each method dir has its own default,
    and ``folder~method`` opens it."""
    problem = kit.problem("sir", name="m0_mh-76aecea70000")
    _seed_run(kit, problem, 2, "m0_mh", 1_000, method="mh-92946094")
    _seed_run(kit, problem, 1, "m0_mh_mode_a", 1_100, method="mh-97f0a52f")
    _seed_run(kit, problem, 3, "m0_mh_mode_a-s3", 1_200, method="mh-97f0a52f")
    metas = ingest.discover_runs(kit.root)
    got = {(m.method, m.seed): (m.folder_default, m.method_default) for m in metas}
    assert got == {
        ("mh-92946094", 2): (False, True),
        ("mh-97f0a52f", 1): (True, True),
        ("mh-97f0a52f", 3): (False, False),
    }
    assert ingest.resolve_run(metas, "m0_mh-76aecea70000~mh-97f0a52f").seed == 1
    assert ingest.resolve_run(metas, "m0_mh-76aecea70000~mh-92946094").seed == 2
    assert ingest.resolve_run(metas, "m0_mh-76aecea70000~mh-00000000") is None


def test_the_same_seed_dir_under_two_method_dirs_gets_the_method_in_its_id(kit):
    """The same seed under two method settings has the same seed dir name; the
    entry id then includes the method dir so the two stay distinct."""
    problem = kit.problem("sir", name="twin-0000aaaa0000")
    a = kit.fit_run(problem, algorithm="mh", method_dir="mh-aaaa", seed_dir="seed_1-000000000001")
    b = kit.fit_run(problem, algorithm="mh", method_dir="mh-bbbb", seed_dir="seed_1-000000000001")
    for leaf in (a, b):
        _trace(leaf, 0, "step\tR0\n0\t1.1\n")
    ids = {m.leaf_dir: m.run_id for m in ingest.discover_runs(kit.root)}
    assert ids == {
        a: "twin-0000aaaa0000~mh-aaaa~seed_1-000000000001",
        b: "twin-0000aaaa0000~mh-bbbb~seed_1-000000000001",
    }


def test_a_single_run_folder_keeps_the_folder_name_as_its_id(kit):
    _seed_run(kit, kit.problem("sir", name="fit_r0-847612580000"), 4, "r0", 1_000)
    (m,) = ingest.discover_runs(kit.root)
    assert m.run_id == "fit_r0-847612580000" and m.seed == 4 and m.folder_default


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


def test_wall_time_reads_the_execution_record(tmp_path: Path):
    """camdl records wall time per execution in execution.json; until the run
    has written it there is no wall time."""
    seed = tmp_path / "seed_1"
    seed.mkdir()
    assert ingest._read_wall_time_secs(seed) is None
    (seed / "execution.json").write_text(json.dumps(
        {"schema": "camdl.execution/v1", "wall_time_seconds": 33.25}))
    assert ingest._read_wall_time_secs(seed) == 33.25
    # It reaches the summary as the ESS/second denominator.
    (seed / "mh_summary.json").write_text(json.dumps({"n_chains": 1, "rhat": {}, "ess": {}}))
    assert ingest.read_chain_summary(seed).wall_time_secs == 33.25
