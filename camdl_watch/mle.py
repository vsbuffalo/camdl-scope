"""MLE fits — point estimates from an optimizer, not posterior draws. Read-only,
parallel to :mod:`camdl_watch.profiles`.

An optimizer (``nl-sbplx``, ``if2``, …) runs from N restarts and writes, per
seed leaf::

    <segment>/<method dir>/seed_*/
        mle_params.toml           # θ̂ + [provenance] (log_likelihood)
        chain_evaluations.tsv     # one row per restart's optimum

``mle_params.toml`` is what marks the leaf as a point estimate;
:func:`camdl_watch.ingest._mle_seed_dir` finds it, since the method directory's
name says nothing reliable about what it holds.

There are no chains/marginals here — the fit is a single θ̂ plus the multi-start
results, which are the MLE analogue of R̂/ESS: did the optimizer reliably find
the mode, or did restarts scatter / fail?
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# camdl marks a failed/infeasible restart with a huge-negative sentinel (≈ −1e100).
FAILED_LOGLIK = -1e99


@dataclass(frozen=True)
class MleRestart:
    chain: int  # restart id (camdl calls the restarts "chains")
    loglik: float  # optimized log-likelihood (≤ FAILED_LOGLIK if the restart failed)
    status: str  # optimizer exit status, e.g. "xtol_reached"
    n_evals: int  # objective evaluations spent


@dataclass(frozen=True)
class MleParam:
    name: str
    value: float | None  # the MLE θ̂ for this coordinate
    # Spread of this coordinate across the *converged* restarts — a rough
    # identifiability signal (tight = well-determined; wide = sloppy/multimodal).
    restart_lo: float | None
    restart_hi: float | None


@dataclass(frozen=True)
class MleFit:
    loglik: float | None  # log-likelihood at the reported optimum
    n_restarts: int
    n_converged: int  # restarts that didn't hit the failure sentinel
    params: list[MleParam]  # in the model's estimated order
    restarts: list[MleRestart]  # every restart, best-first


def _finite(x: object) -> float | None:
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


#: The per-restart table, newest name first. camdl renamed
#: ``chain_results.tsv`` to ``chain_evaluations.tsv`` and changed its columns:
#: the NLopt-era ``status``/``n_evals`` gave way to IF2's ``se``/``ess_mean``/
#: ``ess_min``. Both names are in stores on disk, and reading only the old one
#: left a current MLE fit reporting zero restarts and no spread — a point
#: estimate with nothing to say about whether the optimizer found the mode.
_RESTART_TABLES = ("chain_evaluations.tsv", "chain_results.tsv")


def _restart_table(seed_dir: Path) -> Path | None:
    """The leaf's per-restart results table, whichever name it was written
    under, or ``None`` when the optimizer wrote none."""
    for name in _RESTART_TABLES:
        p = seed_dir / name
        if p.is_file():
            return p
    return None


def read_mle(seed_dir: Path, estimated: list[str]) -> MleFit | None:
    """The point estimate + multi-start results for one MLE seed leaf, or ``None``
    if ``mle_params.toml`` is missing/unreadable. ``estimated`` fixes the param
    order (and which top-level toml keys are coordinates vs metadata)."""
    try:
        obj = tomllib.loads((seed_dir / "mle_params.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    prov = obj.get("provenance") if isinstance(obj.get("provenance"), dict) else {}
    values = {
        k: float(v)
        for k, v in obj.items()
        if k != "provenance" and isinstance(v, (int, float))
    }

    restarts: list[MleRestart] = []
    spread: dict[str, tuple[float, float]] = {}
    csv = _restart_table(seed_dir)
    if csv is not None:
        try:
            # The current table opens with a `# <version> (<date>)` stamp, which
            # would otherwise be parsed as the header row.
            df = pl.read_csv(
                csv, separator="\t", comment_prefix="#", infer_schema_length=10000
            )
        except Exception:
            df = None
        if df is not None and "loglik" in df.columns:
            for row in df.iter_rows(named=True):
                restarts.append(
                    MleRestart(
                        chain=int(row.get("chain", 0) or 0),
                        loglik=_finite(row.get("loglik")) or FAILED_LOGLIK,
                        # NLopt reported an exit status and an evaluation
                        # count; IF2 reports neither, so both are optional.
                        status=str(row.get("status", "") or ""),
                        n_evals=int(row.get("n_evals", 0) or 0),
                    )
                )
            conv = df.filter(pl.col("loglik") > FAILED_LOGLIK)
            for p in estimated:
                if p in conv.columns and conv.height:
                    col = conv[p].drop_nulls()
                    if col.len():
                        spread[p] = (float(col.min()), float(col.max()))
    restarts.sort(key=lambda r: r.loglik, reverse=True)

    params = [
        MleParam(
            name=p,
            value=_finite(values.get(p)),
            restart_lo=spread.get(p, (None, None))[0],
            restart_hi=spread.get(p, (None, None))[1],
        )
        for p in estimated
    ]
    return MleFit(
        loglik=_finite(prov.get("log_likelihood")),
        n_restarts=len(restarts),
        n_converged=sum(1 for r in restarts if r.loglik > FAILED_LOGLIK),
        params=params,
        restarts=restarts,
    )
