"""Particle-health tests: the two samplers, and the readings that matter."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from camdl_watch import particles
from camdl_watch.state import ChainBuffer, RunState


class _Meta:
    def __init__(self, algo, particles_n=None):
        self.algorithm = algo
        self.algorithm_config = {"particles": particles_n} if particles_n else {}


def _run(algo, per_chain, n=200, particles_n=None):
    chains = {}
    for cid, cols in per_chain.items():
        b = ChainBuffer(cid=cid, path=Path(f"chain_{cid}/trace.tsv"))
        b.iters = np.arange(n, dtype=np.int64)
        for k, v in cols.items():
            b.aux[k] = v
        chains[cid] = b
    return RunState(meta=_Meta(algo, particles_n), chains=chains)


def test_pgas_reports_renewal_and_the_profile():
    n = 200
    prof = {f"renewal_b{i}": np.full(n, 0.05 + 0.1 * i) for i in range(10)}
    rs = _run("pgas", {1: {"trajectory_renewal": np.full(n, 0.4),
                           "as_accept": np.full(n, 0.02),
                           "as_opportunity": np.full(n, 90.0),
                           "as_proposed": np.full(n, 70.0), **prof}}, n, 4800)
    h = particles.particle_health(rs, 0)
    assert h.method == "pgas" and h.available
    assert h.particles == 4800
    assert len(h.renewal_profile) == 10
    assert h.renewal_profile[0] < h.renewal_profile[-1]
    assert {s.key for s in h.stats} >= {"trajectory_renewal", "as_accept", "as_opportunity"}


def test_a_frozen_path_is_flagged():
    n = 200
    rs = _run("pgas", {1: {"trajectory_renewal": np.full(n, 0.0005),
                           "as_opportunity": np.full(n, 90.0)}}, n)
    h = particles.particle_health(rs, 0)
    r = next(s for s in h.stats if s.key == "trajectory_renewal")
    assert r.severity == "warn"


def test_no_ancestry_draws_is_flagged_separately_from_a_bad_filter():
    # camdl gh#718: low renewal with as_opportunity near zero is a SCHEDULE
    # with nowhere to renew — more particles will not help. The two causes must
    # not collapse into one warning.
    n = 200
    rs = _run("pgas", {1: {"trajectory_renewal": np.full(n, 0.001),
                           "as_opportunity": np.full(n, 0.0)}}, n)
    h = particles.particle_health(rs, 0)
    opp = next(s for s in h.stats if s.key == "as_opportunity")
    assert opp.severity == "warn"


def test_per_chain_renewal_exposes_one_frozen_chain():
    n = 200
    rs = _run("pgas", {1: {"trajectory_renewal": np.full(n, 0.5)},
                       2: {"trajectory_renewal": np.full(n, 0.5)},
                       3: {"trajectory_renewal": np.full(n, 0.0)}}, n)
    h = particles.particle_health(rs, 0)
    assert h.renewal_by_chain[3] == 0.0
    assert h.renewal_by_chain[1] == 0.5


def test_pmmh_reports_the_longest_stall_not_just_acceptance():
    # A healthy average acceptance hiding one long stall is the PMMH failure
    # that a rate alone cannot show.
    n = 200
    acc = np.ones(n); acc[50:150] = 0.0   # 100-sweep stall, 50% acceptance
    rs = _run("pmmh", {1: {"accepted": acc}}, n)
    h = particles.particle_health(rs, 0)
    assert h.method == "pmmh"
    stall = next(s for s in h.stats if s.key == "longest_stall")
    assert stall.value == 100.0
    assert stall.severity == "warn"
    rate = next(s for s in h.stats if s.key == "accepted")
    assert rate.value == 0.5 and rate.severity == "ok"


def test_pmmh_gets_no_pgas_fields():
    n = 200
    rs = _run("pmmh", {1: {"accepted": np.ones(n)}}, n)
    h = particles.particle_health(rs, 0)
    assert h.renewal_profile == []
    assert h.renewal_by_chain == {}


def test_a_non_particle_sampler_says_so():
    rs = _run("nuts", {1: {"accepted": np.ones(200)}})
    h = particles.particle_health(rs, 0)
    assert not h.available
    assert "not a particle method" in (h.reason or "")
