"""Cross-chain path compatibility — are the chains locked onto their own modes?

``<seed>/cross_chain_compat.json``, written beside ``diagnostics.json`` for
multi-chain PGAS stages, holds ``M[i][j] = log p(x_j | θ_i)``: chain *i*'s
parameters scoring chain *j*'s latent path. It answers something R̂ cannot.
R̂ says the chains disagree; this says whether each chain is welded to its own
(θ, x) pair — a strong diagonal — or whether the chains are exchangeable in the
augmented space and the separation lives in the marginal posterior, which shows
up as off-diagonals comparable to the diagonal.

Absent for PMMH, which stores no path to score.

Three things this module does to the raw matrix, each because the raw form
misleads:

* **Row-centres on the diagonal.** Raw log-densities are not comparable across
  rows: a different θ gives a different overall scale, so the biggest number in
  the matrix tells you which chain has the loosest process noise, not which
  pairing is compatible. ``M[i][j] − M[i][i]`` reads directly as "how many nats
  worse is chain *j*'s path under chain *i*'s parameters", and the diagonal
  becomes exactly zero by construction.
* **Keeps the sign.** A *positive* centred entry — another chain's path being
  more probable under my parameters than my own path is — means that chain has
  not found its own mode. It is rare and it is the alarming case, so it must
  not be flattened into "small" by a one-sided scale.
* **Orders by clustering, not by chain id.** Blocks of mutually compatible
  chains are the multimodality signal: two blocks is two modes, k singletons is
  k stuck chains. Chain id ordering scatters that structure across the grid.

The chain list is authoritative for the chain count. Chains refused at seeding
are omitted from both ``chains`` and ``M``, so a 10×10 matrix on a 16-chain
launch means six never started — reading the count from the fit config would
report the launch, not the evidence.

Pure reader; no dependency on ingest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

FILENAME = "cross_chain_compat.json"


@dataclass(frozen=True)
class CrossChainCompat:
    """One stage's cross-chain path-compatibility matrix, ready to draw."""

    #: Chain ids, 1-based as camdl names ``chain_N/``. Authoritative for the
    #: count — see the module docstring.
    chains: list[int]
    #: Raw ``log p(x_j | θ_i)``, in the artifact's own order.
    raw: list[list[float]]
    #: ``M[i][j] − M[i][i]``, in the artifact's own order. Zero on the diagonal.
    centred: list[list[float]]
    #: Display permutation over the artifact's indices, from clustering the
    #: symmetrised matrix. Identity when clustering is not possible.
    order: list[int]
    #: Which log-density terms the score sums, verbatim from the artifact.
    terms: str
    #: Which draw the (θ, x) pairs come from, verbatim. Load-bearing: the
    #: matrix is one sweep, not an average over the posterior, so every cell is
    #: a single realisation and small differences are not meaningful.
    draw: str
    diagonal_dominance: float | None
    asymmetry: float | None
    #: True when at least one centred off-diagonal is positive — another
    #: chain's path scoring better than the row's own.
    has_positive_offdiagonal: bool
    #: Non-finite cells, which are excluded from clustering and drawn blank
    #: rather than as a number the reader would trust.
    n_nonfinite: int


def _clustered_order(centred: np.ndarray) -> list[int]:
    """Leaf order from average-linkage clustering of the symmetrised matrix.

    The centred matrix is not symmetric — ``i`` scoring ``j``'s path is a
    different quantity from the reverse, and their *difference* is what
    ``asymmetry`` measures. For ordering we want one number per unordered pair,
    so the two directions are averaged and negated into a dissimilarity: 0 for
    a chain against itself, larger the less compatible the pair.

    ``-inf`` and ``NaN`` are treated differently, because they mean different
    things. A ``-inf`` cell is a path with *zero* probability under those
    parameters — maximally incompatible, and imputing it as the largest finite
    distance is what it already asserts. A ``NaN`` is a score that was not
    computed, and standing in a value there would invent a pairing the artifact
    does not claim, so any surviving ``NaN`` drops back to the identity order.
    Losing the clustering on one unknown cell is worth it; losing it on one
    *impossible* cell, which is common and informative, is not.
    """
    k = centred.shape[0]
    if k < 3:
        return list(range(k))
    d = -(centred + centred.T) / 2.0
    np.fill_diagonal(d, 0.0)
    if np.isnan(d).any():
        return list(range(k))
    if np.isinf(d).any():
        finite = d[np.isfinite(d)]
        if finite.size == 0:
            return list(range(k))
        d = np.where(np.isinf(d), finite.max(), d)
        np.fill_diagonal(d, 0.0)
    # Linkage needs a proper dissimilarity: non-negative and zero on the
    # diagonal. A positive centred entry makes d negative, so shift the whole
    # matrix up rather than clipping — clipping would merge "slightly better"
    # and "equally good" into one value and lose the very pairs worth seeing.
    lo = float(d.min())
    if lo < 0:
        d = d - lo
        np.fill_diagonal(d, 0.0)
    try:
        return [int(i) for i in leaves_list(linkage(squareform(d, checks=False), "average"))]
    except (ValueError, MemoryError):
        return list(range(k))


def read_compat(seed_dir: Path) -> CrossChainCompat | None:
    """Read the artifact from a stage seed directory.

    ``None`` when absent (any PMMH stage, or a PGAS stage predating the
    artifact) or malformed — a half-parsed matrix would be drawn as a
    confident-looking grid of wrong numbers.
    """
    path = Path(seed_dir) / FILENAME
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None

    chains = raw.get("chains")
    m = raw.get("M")
    if not isinstance(chains, list) or not isinstance(m, list) or not chains:
        return None
    try:
        chain_ids = [int(c) for c in chains]
    except (TypeError, ValueError):
        return None
    k = len(chain_ids)
    if len(m) != k or any(not isinstance(row, list) or len(row) != k for row in m):
        return None

    try:
        mat = np.array(m, dtype=float)
    except (TypeError, ValueError):
        return None
    n_nonfinite = int((~np.isfinite(mat)).sum())

    # Row-centre on the diagonal. A non-finite diagonal makes its whole row
    # meaningless rather than merely sparse, so the row goes to NaN entirely.
    diag = np.diag(mat).astype(float)
    centred = mat - diag[:, None]
    centred[~np.isfinite(diag), :] = np.nan
    np.fill_diagonal(centred, np.where(np.isfinite(diag), 0.0, np.nan))

    off = ~np.eye(k, dtype=bool)
    finite_off = centred[off][np.isfinite(centred[off])]
    has_pos = bool(finite_off.size and (finite_off > 0).any())

    def _num(key: str) -> float | None:
        v = raw.get(key)
        return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None

    return CrossChainCompat(
        chains=chain_ids,
        raw=[[float(x) for x in row] for row in mat],
        centred=[[float(x) for x in row] for row in centred],
        order=_clustered_order(centred),
        terms=str(raw.get("terms") or ""),
        draw=str(raw.get("draw") or ""),
        diagonal_dominance=_num("diagonal_dominance"),
        asymmetry=_num("asymmetry"),
        has_positive_offdiagonal=has_pos,
        n_nonfinite=n_nonfinite,
    )


def find_compat(run_dir: Path) -> CrossChainCompat | None:
    """The run's cross-chain matrix, from its posterior stage's seed directory.

    Searches the same shape camdl writes (`<run>/<NN>-<stage>/<seed>/`); the
    newest match wins, mirroring how the other stage artifacts are resolved.
    """
    run_dir = Path(run_dir)
    cands: list[Path] = []
    for pat in (f"*/*/{FILENAME}", f"*/{FILENAME}", FILENAME):
        try:
            cands = sorted(
                run_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True
            )
        except OSError:
            continue
        if cands:
            break
    for c in cands:
        got = read_compat(c.parent)
        if got is not None:
            return got
    return None
