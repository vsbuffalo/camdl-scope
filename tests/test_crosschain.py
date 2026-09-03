"""Cross-chain compatibility reader tests.

The artifact is not produced by any camdl on this machine yet, so these are the
only check on the contract: matrices are built by hand with a known block
structure and known row scales, so centring and ordering have right answers
rather than plausible ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from camdl_watch import crosschain


def _write(seed: Path, chains, M, **extra) -> Path:
    seed.mkdir(parents=True, exist_ok=True)
    body = {
        "chains": chains,
        "M": [[float(x) for x in row] for row in M],
        "terms": "transition + initial_state",
        "chain_numbering": "1-based, matching chain_N/",
        **extra,
    }
    (seed / crosschain.FILENAME).write_text(json.dumps(body))
    return seed


def _blocked(ids, block, within=-2.0, across=-80.0, row_scale=-50.0):
    """A matrix with two mutually-compatible blocks and a per-row offset.

    The row offset is the point: raw values are then incomparable across rows,
    so anything reading them uncentred gets the wrong answer.
    """
    k = len(ids)
    M = np.zeros((k, k))
    for i, ci in enumerate(ids):
        base = -1000 + row_scale * i
        for j, cj in enumerate(ids):
            M[i, j] = base + (0.0 if i == j else (within if block[ci] == block[cj] else across))
    return M


def test_centring_zeroes_the_diagonal_and_removes_the_row_scale(tmp_path):
    ids = [1, 2, 3, 4]
    block = {1: 0, 2: 0, 3: 1, 4: 1}
    c = crosschain.read_compat(_write(tmp_path, ids, _blocked(ids, block)))
    assert c is not None
    for i in range(4):
        assert c.centred[i][i] == 0.0
    # Every within-block off-diagonal is the same -2.0 regardless of row scale.
    assert c.centred[0][1] == -2.0
    assert c.centred[2][3] == -2.0
    assert c.centred[0][2] == -80.0


def test_clustering_recovers_blocks_that_chain_id_order_scatters(tmp_path):
    # Interleaved ids: {1,3,5} and {2,4,6}. Ordering by id would alternate.
    ids = [1, 2, 3, 4, 5, 6]
    block = {1: 0, 3: 0, 5: 0, 2: 1, 4: 1, 6: 1}
    c = crosschain.read_compat(_write(tmp_path, ids, _blocked(ids, block)))
    assert c is not None
    ordered = [c.chains[i] for i in c.order]
    halves = {frozenset(ordered[:3]), frozenset(ordered[3:])}
    assert halves == {frozenset({1, 3, 5}), frozenset({2, 4, 6})}, ordered


def test_a_positive_off_diagonal_is_flagged(tmp_path):
    # Chain 2 scores chain 1's path higher than its own: it never found a mode.
    ids = [1, 2]
    M = np.array([[-1000.0, -1080.0], [-1045.0, -1050.0]])
    c = crosschain.read_compat(_write(tmp_path, ids, M))
    assert c is not None
    assert c.centred[1][0] == 5.0
    assert c.has_positive_offdiagonal


def test_all_negative_off_diagonals_are_not_flagged(tmp_path):
    ids = [1, 2]
    M = np.array([[-1000.0, -1080.0], [-1090.0, -1050.0]])
    c = crosschain.read_compat(_write(tmp_path, ids, M))
    assert c is not None and not c.has_positive_offdiagonal


def test_chain_count_comes_from_the_artifact_not_the_launch(tmp_path):
    # Six chains launched, four survived seeding: the matrix is 4x4 and the
    # chain ids are non-contiguous. Reading the count anywhere else reports the
    # launch rather than the evidence.
    ids = [2, 3, 7, 11]
    block = {i: 0 for i in ids}
    c = crosschain.read_compat(_write(tmp_path, ids, _blocked(ids, block)))
    assert c is not None
    assert c.chains == [2, 3, 7, 11]
    assert len(c.centred) == 4


def test_scalars_and_terms_are_passed_through(tmp_path):
    ids = [1, 2]
    c = crosschain.read_compat(
        _write(
            tmp_path, ids, np.array([[-1.0, -3.0], [-4.0, -2.0]]),
            diagonal_dominance=35.06, asymmetry=84.08,
        )
    )
    assert c is not None
    assert c.diagonal_dominance == 35.06
    assert c.asymmetry == 84.08
    assert c.terms == "transition + initial_state"


def test_absent_artifact_is_none(tmp_path):
    assert crosschain.read_compat(tmp_path) is None


def test_malformed_matrix_is_none_rather_than_half_parsed(tmp_path):
    # A ragged matrix drawn as a grid would be confident and wrong.
    seed = tmp_path / "s"
    seed.mkdir()
    (seed / crosschain.FILENAME).write_text(
        json.dumps({"chains": [1, 2, 3], "M": [[0.0, 1.0], [2.0, 3.0]]})
    )
    assert crosschain.read_compat(seed) is None


def test_nonfinite_cells_are_counted_and_kept_out_of_the_numbers(tmp_path):
    ids = [1, 2]
    M = np.array([[-1000.0, float("-inf")], [-1090.0, -1050.0]])
    c = crosschain.read_compat(_write(tmp_path, ids, M))
    assert c is not None
    assert c.n_nonfinite == 1
    assert not np.isfinite(c.centred[0][1])
    assert c.centred[0][0] == 0.0  # the row's own diagonal still centres


def test_find_compat_locates_the_stage_seed_directory(tmp_path):
    ids = [1, 2]
    seed = tmp_path / "01-posterior-abcd" / "seed_1-ef01"
    _write(seed, ids, np.array([[-1.0, -3.0], [-4.0, -2.0]]))
    c = crosschain.find_compat(tmp_path)
    assert c is not None and c.chains == [1, 2]


def test_find_compat_is_none_for_a_run_without_one(tmp_path):
    (tmp_path / "01-posterior-abcd" / "seed_1-ef01").mkdir(parents=True)
    assert crosschain.find_compat(tmp_path) is None


def test_an_impossible_cell_still_clusters_but_an_unknown_one_does_not(tmp_path):
    """`-inf` is maximal incompatibility and can be ordered around; `NaN` is a
    missing score and must not be stood in for."""
    ids = [1, 2, 3, 4, 5, 6]
    block = {1: 0, 3: 0, 5: 0, 2: 1, 4: 1, 6: 1}

    M = _blocked(ids, block)
    M[0][1] = float("-inf")  # impossible, but the blocks are still readable
    c = crosschain.read_compat(_write(tmp_path / "inf", ids, M))
    assert c is not None
    halves = {frozenset([c.chains[i] for i in c.order][:3]),
              frozenset([c.chains[i] for i in c.order][3:])}
    assert halves == {frozenset({1, 3, 5}), frozenset({2, 4, 6})}

    M2 = _blocked(ids, block)
    M2[0][1] = float("nan")
    c2 = crosschain.read_compat(_write(tmp_path / "nan", ids, M2))
    assert c2 is not None
    assert c2.order == list(range(6))
