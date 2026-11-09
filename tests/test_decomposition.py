"""Target-decomposition tests.

Synthetic chains with the terms constructed so the exact answer is known: the
whole point of the share is that it is an identity, so a test that only checked
"looks plausible" would miss the one property that matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path

from camdl_scope import decomposition
from camdl_scope.state import ChainBuffer, RunState


def _run(per_chain: dict[int, dict[str, np.ndarray]], n: int = 200) -> RunState:
    """A RunState whose chains carry the given aux columns.

    `meta` is unused by this module (it reads only `chains`), so the state is
    built directly rather than through a store fixture — these tests are about
    the arithmetic, not about discovery.
    """
    chains = {}
    for cid, cols in per_chain.items():
        buf = ChainBuffer(cid=cid, path=Path(f"chain_{cid}/trace.tsv"))
        buf.iters = np.arange(n, dtype=np.int64)
        for k, v in cols.items():
            buf.aux[k] = v
        chains[cid] = buf
    return RunState(meta=None, chains=chains)  # type: ignore[arg-type]


def _terms(obs: float, trans: float, init: float, prior: float, n: int = 200):
    """One chain's columns, at fixed per-term levels plus a little jitter."""
    rng = np.random.default_rng(abs(int(obs * 1000)) % 2**31)
    o = obs + rng.normal(0, 0.01, n)
    t = trans + rng.normal(0, 0.01, n)
    i = init + rng.normal(0, 0.01, n)
    complete = o + t + i
    return {
        "obs_ll": o,
        "transition_ll": t,
        "initial_state_ll": i,
        "log_complete_data_ll": complete,
        "log_posterior": complete + prior,
    }


def test_shares_sum_to_one(tmp_path):
    # The identity Var(T) = sum_i Cov(X_i, T) is the whole basis of the bars.
    rs = _run(
        {
            1: _terms(-100.0, -500.0, -10.0, -3.0),
            2: _terms(-104.0, -540.0, -12.0, -3.4),
            3: _terms(-101.0, -515.0, -11.0, -3.1),
            4: _terms(-108.0, -560.0, -9.0, -2.8),
        }
    )
    d = decomposition.decompose_target(rs, cutoff=0)
    assert d.available
    assert sum(t.share for t in d.terms) == pytest.approx(1.0, abs=1e-9)


def test_ranges_do_not_have_to_sum_to_the_total(tmp_path):
    # The reason shares are the bars and ranges only a column. If this ever
    # started holding, the warning in the UI copy would be wrong.
    rs = _run(
        {
            1: _terms(-100.0, -500.0, -10.0, -3.0),
            2: _terms(-110.0, -490.0, -10.0, -3.0),
            3: _terms(-105.0, -495.0, -10.0, -3.0),
        }
    )
    d = decomposition.decompose_target(rs, cutoff=0)
    parts = sum(t.range_nats for t in d.terms)
    # obs and transition move in opposite directions here, so the total's range
    # is strictly smaller than the sum of the parts'.
    assert parts > d.total_range


def test_the_dominant_term_is_identified(tmp_path):
    # Only the trajectory term varies across chains; it must take ~all the share.
    rs = _run(
        {
            1: _terms(-100.0, -500.0, -10.0, -3.0),
            2: _terms(-100.0, -560.0, -10.0, -3.0),
            3: _terms(-100.0, -530.0, -10.0, -3.0),
        }
    )
    d = decomposition.decompose_target(rs, cutoff=0)
    assert d.terms[0].key == "transition_ll"
    assert d.terms[0].share == pytest.approx(1.0, abs=0.01)


def test_a_term_pulling_chains_together_gets_a_negative_share(tmp_path):
    # A prior that is highest where the total is lowest opposes the separation.
    # A range cannot express this; the sign is the point of using shares.
    rs = _run(
        {
            1: _terms(-100.0, -500.0, -10.0, -1.0),
            2: _terms(-100.0, -560.0, -10.0, -9.0),
            3: _terms(-100.0, -530.0, -10.0, -5.0),
        }
    )
    d = decomposition.decompose_target(rs, cutoff=0)
    prior = next(t for t in d.terms if t.key == "prior")
    assert prior.share > 0  # here the prior REINFORCES; flip it below
    rs2 = _run(
        {
            1: _terms(-100.0, -500.0, -10.0, -9.0),
            2: _terms(-100.0, -560.0, -10.0, -1.0),
            3: _terms(-100.0, -530.0, -10.0, -5.0),
        }
    )
    d2 = decomposition.decompose_target(rs2, cutoff=0)
    prior2 = next(t for t in d2.terms if t.key == "prior")
    assert prior2.share < 0


def test_marginal_sampler_is_unavailable_with_a_reason(tmp_path):
    rs = _run(
        {
            1: {"log_likelihood": np.full(200, -100.0), "log_posterior": np.full(200, -103.0)},
            2: {"log_likelihood": np.full(200, -110.0), "log_posterior": np.full(200, -113.0)},
        }
    )
    d = decomposition.decompose_target(rs, cutoff=0)
    assert not d.available
    assert "complete-data" in (d.reason or "")


def test_single_chain_is_unavailable(tmp_path):
    rs = _run({1: _terms(-100.0, -500.0, -10.0, -3.0)})
    d = decomposition.decompose_target(rs, cutoff=0)
    assert not d.available
    assert "one" in (d.reason or "")


def test_nonfinite_term_refuses_rather_than_charting_garbage(tmp_path):
    # The wider store contains fits with -inf obs_ll and totals of 4e8; a bar
    # chart drawn from those is worse than no panel.
    bad = _terms(-100.0, -500.0, -10.0, -3.0)
    bad["obs_ll"] = np.full(200, -np.inf)
    bad["log_complete_data_ll"] = np.full(200, -np.inf)
    bad["log_posterior"] = np.full(200, -np.inf)
    rs = _run({1: _terms(-100.0, -500.0, -10.0, -3.0), 2: bad})
    d = decomposition.decompose_target(rs, cutoff=0)
    assert not d.available
    assert "non-finite" in (d.reason or "")


def test_initial_state_is_derived_when_the_column_is_absent(tmp_path):
    # Runs predating the column must still decompose, flagged as derived.
    def drop_init(t):
        t = dict(t)
        del t["initial_state_ll"]
        return t

    rs = _run(
        {
            1: drop_init(_terms(-100.0, -500.0, -10.0, -3.0)),
            2: drop_init(_terms(-104.0, -540.0, -12.0, -3.4)),
            3: drop_init(_terms(-101.0, -515.0, -11.0, -3.1)),
        }
    )
    d = decomposition.decompose_target(rs, cutoff=0)
    assert d.available
    assert d.initial_state_derived
    assert sum(t.share for t in d.terms) == pytest.approx(1.0, abs=1e-9)
    init = next(t for t in d.terms if t.key == "initial_state_ll")
    assert init.chain_means[0] == pytest.approx(-10.0, abs=0.05)


def test_identity_residual_is_reported(tmp_path):
    # A term camdl added that this module does not know about would otherwise
    # be silently absorbed into the implied prior.
    a = _terms(-100.0, -500.0, -10.0, -3.0)
    b = _terms(-104.0, -540.0, -12.0, -3.4)
    b["log_complete_data_ll"] = b["log_complete_data_ll"] - 7.0  # a missing term
    rs = _run({1: a, 2: b})
    d = decomposition.decompose_target(rs, cutoff=0)
    assert d.available
    assert d.residual_max > decomposition.RESIDUAL_TOL


# --- per-stream drill-down ---------------------------------------------------


def test_stream_shares_sum_to_one_and_rank_the_culprit(tmp_path):
    def with_streams(obs_a, obs_b, trans):
        t = _terms(obs_a + obs_b, trans, -10.0, -3.0)
        n = len(t["obs_ll"])
        t["obs_ll"] = np.full(n, obs_a + obs_b)
        t["obs_ll_cases"] = np.full(n, obs_a)
        t["obs_ll_deaths"] = np.full(n, obs_b)
        t["log_complete_data_ll"] = t["obs_ll"] + t["transition_ll"] + t["initial_state_ll"]
        t["log_posterior"] = t["log_complete_data_ll"] - 3.0
        return t

    # `cases` is the stream the chains disagree about; `deaths` is constant.
    rs = _run(
        {
            1: with_streams(-50.0, -30.0, -500.0),
            2: with_streams(-70.0, -30.0, -520.0),
            3: with_streams(-60.0, -30.0, -510.0),
        }
    )
    streams = decomposition.decompose_streams(rs, cutoff=0)
    assert [s.stream for s in streams][0] == "cases"
    assert sum(s.share for s in streams) == pytest.approx(1.0, abs=1e-9)
    assert next(s for s in streams if s.stream == "deaths").share == pytest.approx(
        0.0, abs=1e-9
    )


def test_no_per_stream_columns_gives_no_breakdown(tmp_path):
    rs = _run({1: _terms(-100.0, -500.0, -10.0, -3.0), 2: _terms(-104.0, -540.0, -12.0, -3.4)})
    assert decomposition.decompose_streams(rs, cutoff=0) == []
