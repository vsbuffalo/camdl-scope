"""MLE fits — point estimates from an optimization ('scout') stage, not posterior
draws. Read-only, parallel to :mod:`camdl_watch.profiles`.

A scout stage runs an optimizer (``nl-sbplx``, ``if2``, …) from N restarts and
writes, per seed leaf::

    <run>/NN-scout-<hash>/seed_*/mle_params.toml    # θ̂ + [provenance] (log_likelihood)
                                /chain_results.tsv    # one row per restart's optimum

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


def find_mle_seed(run_dir: Path) -> Path | None:
    """The lowest-numbered non-posterior stage seed carrying an ``mle_params.toml``
    (a scout/optimization stage), or ``None``. Posterior stages are surfaced by
    the trace path instead, so we skip them here."""
    for stage in sorted(run_dir.glob("[0-9]*-*-*")):
        if not stage.is_dir() or "-posterior-" in stage.name:
            continue
        for seed in sorted(stage.glob("seed_*")):
            if (seed / "mle_params.toml").is_file():
                return seed
    return None


def _finite(x: object) -> float | None:
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


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
    csv = seed_dir / "chain_results.tsv"
    if csv.is_file():
        try:
            df = pl.read_csv(csv, separator="\t", infer_schema_length=10000)
        except Exception:
            df = None
        if df is not None and "loglik" in df.columns:
            for row in df.iter_rows(named=True):
                restarts.append(
                    MleRestart(
                        chain=int(row.get("chain", 0) or 0),
                        loglik=_finite(row.get("loglik")) or FAILED_LOGLIK,
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
