"""MLE fits — point estimates from an optimizer, not posterior draws. Read-only,
parallel to :mod:`camdl_scope.profiles`.

An optimizer (``nl-sbplx``, ``if2``, …) runs from N starts and writes, per
seed leaf::

    <segment>/<method dir>/seed_*/
        mle_params.toml           # θ̂ + [provenance] (log_likelihood); written last
        chain_results.tsv         # one row per start's optimum (NLopt)
        chain_evaluations.tsv     # the same table under IF2's name and columns

``mle_params.toml`` marks a *finished* point estimate. An NLopt run creates
``chain_results.tsv`` with its header before any start runs and appends each
start's row as that start finishes, in completion order (camdl gh#1197); at the
end camdl replaces the file, by rename, with the same rows in chain order. So
while a run is going, the table is the only result on disk, and the best row in
it is a provisional estimate — the best of the starts that have finished, not
the fit's answer.

There are no chains/marginals here — the fit is a single θ̂ plus the multi-start
results, which are the MLE analogue of R̂/ESS: did the optimizer reliably find
the mode, or did starts scatter / fail?
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# camdl marks a failed/infeasible restart with a huge-negative sentinel (≈ −1e100).
FAILED_LOGLIK = -1e99

#: NLopt's exit status for a start that spent its whole ``max_evals`` budget.
#: Its optimum is where the budget ran out, not where the optimizer converged.
MAXEVAL_STATUS = "maxeval_reached"


@dataclass(frozen=True)
class MleRestart:
    chain: int  # restart id (camdl calls the restarts "chains")
    loglik: float  # optimized log-likelihood (≤ FAILED_LOGLIK if the restart failed)
    status: str  # optimizer exit status, e.g. "xtol_reached"; "" when not written
    n_evals: int  # objective evaluations spent; 0 when not written
    # This start's optimum, per estimated parameter (None: absent or non-finite).
    values: dict[str, float | None] = field(default_factory=dict)

    @property
    def hit_maxeval(self) -> bool:
        return self.status == MAXEVAL_STATUS


@dataclass(frozen=True)
class MleParam:
    name: str
    value: float | None  # the MLE θ̂ for this coordinate
    # Spread of this coordinate across the *converged* restarts — a rough
    # identifiability signal (tight = well-determined; wide = sloppy/multimodal).
    restart_lo: float | None
    restart_hi: float | None


@dataclass(frozen=True)
class TracePoint:
    eval: int  # the search's evaluation count at this row
    best_loglik: float | None  # best finite log-likelihood so far; None if none yet


@dataclass(frozen=True)
class SearchTrace:
    """One optimiser search as camdl's ``opt_trace.tsv`` records it, rows in
    the order written (camdl appends a row every 25 evaluations or 2 s, and one
    at the search's end)."""

    chain: int
    points: tuple[TracePoint, ...]
    #: The parameter values at the best point so far (last row).
    best_values: dict[str, float | None]
    #: The search's first evaluation was not finite: a bad start.
    started_nonfinite: bool

    @property
    def evals(self) -> int:
        return self.points[-1].eval if self.points else 0

    @property
    def best_loglik(self) -> float | None:
        return self.points[-1].best_loglik if self.points else None


@dataclass(frozen=True)
class MleFit:
    loglik: float | None  # log-likelihood at the reported optimum
    n_restarts: int
    n_converged: int  # restarts that didn't hit the failure sentinel
    params: list[MleParam]  # in the model's estimated order
    restarts: list[MleRestart]  # every restart, best-first
    # True until camdl writes ``mle_params.toml``: ``loglik`` and ``params`` are
    # then the best finished start's row, not the fit's reported optimum.
    provisional: bool = False
    # How many starts the run was configured with (run.json), when known. A
    # hint: ``starts = "single"`` collapses the run to one start.
    n_expected: int | None = None
    #: Each search's live trace (``opt_trace.tsv``), when camdl wrote one.
    searches: list[SearchTrace] = field(default_factory=list)
    #: While provisional, the estimate comes from a search still running (its
    #: best point so far beats every finished start), not a finished one.
    provisional_from_running: bool = False

    @property
    def n_maxeval(self) -> int:
        """Finished starts that stopped because their evaluation budget ran out."""
        return sum(1 for r in self.restarts if r.hit_maxeval)

    @property
    def all_maxeval(self) -> bool:
        """Every finished start stopped at ``max_evals`` — so the reported
        optimum is where the best start's budget ran out (camdl warns the same)."""
        return bool(self.restarts) and self.n_maxeval == len(self.restarts)


def _finite(x: object) -> float | None:
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int(x: object) -> int:
    try:
        return int(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


#: The per-restart table, newest name first. camdl renamed
#: ``chain_results.tsv`` to ``chain_evaluations.tsv`` for IF2 and changed its
#: columns: the NLopt ``status``/``n_evals`` gave way to IF2's ``se``/
#: ``ess_mean``/``ess_min``. Both are in stores on disk, and reading only one
#: left an MLE fit reporting zero restarts and no spread.
_RESTART_TABLES = ("chain_evaluations.tsv", "chain_results.tsv")


def _restart_table(seed_dir: Path) -> Path | None:
    """The leaf's per-restart results table, whichever name it was written
    under, or ``None`` when the optimizer wrote none."""
    for name in _RESTART_TABLES:
        p = seed_dir / name
        if p.is_file():
            return p
    return None


def read_restart_rows(path: Path, *, live: bool) -> list[dict[str, str]]:
    """The data rows of a restart table, as ``column -> raw text``.

    Read whole, by path, on every call: camdl replaces the live file with the
    sorted one by rename at the end, so a held handle would keep reading the
    unlinked file. ``#`` lines (IF2's version stamp) are skipped. With
    ``live=True`` a last line without its newline is dropped as a torn append;
    camdl writes each row in one call, so it is only ever a row not yet
    complete. A row whose field count disagrees with the header is skipped."""
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    lines = text.split("\n")
    if live and not text.endswith("\n"):
        lines.pop()
    lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    if not lines:
        return []
    header = lines[0].split("\t")
    out: list[dict[str, str]] = []
    for ln in lines[1:]:
        fields = ln.split("\t")
        if len(fields) == len(header):
            out.append(dict(zip(header, fields)))
    return out


def has_restart_rows(seed_dir: Path) -> bool:
    """Whether at least one start has finished and been written to the table."""
    p = _restart_table(seed_dir)
    return p is not None and bool(read_restart_rows(p, live=True))


def _restarts(seed_dir: Path, estimated: list[str], *, live: bool) -> list[MleRestart]:
    p = _restart_table(seed_dir)
    if p is None:
        return []
    out = [
        MleRestart(
            chain=_int(row.get("chain")),
            loglik=_finite(row.get("loglik")) or FAILED_LOGLIK,
            status=row.get("status", "") or "",
            n_evals=_int(row.get("n_evals")),
            values={name: _finite(row.get(name)) for name in estimated},
        )
        for row in read_restart_rows(p, live=live)
        if "loglik" in row
    ]
    # Best first; chain id breaks ties so the order does not depend on which
    # start happened to finish first.
    out.sort(key=lambda r: (-r.loglik, r.chain))
    return out


OPT_TRACE = "opt_trace.tsv"


def read_opt_trace(seed_dir: Path, estimated: list[str], *, live: bool) -> list[SearchTrace]:
    """Each search's trace from camdl's ``opt_trace.tsv`` (one file, a
    ``chain`` column), by chain. Read whole by path on every call, as the file
    grows; with ``live=True`` a torn last line is dropped (camdl writes each
    row in one append). Empty when camdl wrote no trace (older camdl, or a
    method that is not an NLopt search)."""
    p = seed_dir / OPT_TRACE
    if not p.is_file():
        return []
    by_chain: dict[int, list[dict[str, str]]] = {}
    for row in read_restart_rows(p, live=live):
        if "eval" in row and "chain" in row:
            by_chain.setdefault(_int(row["chain"]), []).append(row)
    out: list[SearchTrace] = []
    for chain, rows in sorted(by_chain.items()):
        last = rows[-1]
        out.append(SearchTrace(
            chain=chain,
            points=tuple(TracePoint(_int(r["eval"]), _finite(r.get("best_loglik"))) for r in rows),
            best_values={name: _finite(last.get(name)) for name in estimated},
            started_nonfinite=_finite(rows[0].get("loglik")) is None,
        ))
    return out


def read_mle(
    seed_dir: Path, estimated: list[str], n_expected: int | None = None
) -> MleFit | None:
    """The point estimate + multi-start results for one MLE seed leaf.

    Finished (``mle_params.toml`` present): θ̂ and logL are camdl's. Running or
    stopped before the end: the best finished start's row stands in, and the fit
    is marked ``provisional``; with no start finished yet it has no estimate and
    no restarts. ``None`` only when ``mle_params.toml`` exists but cannot be
    read. ``estimated`` fixes the param order."""
    mle_path = seed_dir / "mle_params.toml"
    provisional = not mle_path.is_file()
    restarts = _restarts(seed_dir, estimated, live=provisional)
    converged = [r for r in restarts if r.loglik > FAILED_LOGLIK]
    searches = read_opt_trace(seed_dir, estimated, live=provisional)
    from_running = False

    if provisional:
        best = converged[0] if converged else None
        values = dict(best.values) if best else {}
        loglik = best.loglik if best else None
        # A search still running may already be past every finished start.
        finished = {r.chain for r in restarts}
        running = [t for t in searches if t.chain not in finished and t.best_loglik is not None]
        lead = max(running, key=lambda t: t.best_loglik or -math.inf, default=None)
        if lead is not None and (loglik is None or (lead.best_loglik or -math.inf) > loglik):
            values, loglik, from_running = dict(lead.best_values), lead.best_loglik, True
    else:
        try:
            obj = tomllib.loads(mle_path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            return None
        prov = obj.get("provenance") if isinstance(obj.get("provenance"), dict) else {}
        values = {
            k: _finite(v)
            for k, v in obj.items()
            if k != "provenance" and isinstance(v, (int, float))
        }
        loglik = _finite(prov.get("log_likelihood"))

    spread: dict[str, tuple[float, float]] = {}
    for p in estimated:
        xs = [v for r in converged if (v := r.values.get(p)) is not None]
        if xs:
            spread[p] = (min(xs), max(xs))

    params = [
        MleParam(
            name=p,
            value=values.get(p),
            restart_lo=spread.get(p, (None, None))[0],
            restart_hi=spread.get(p, (None, None))[1],
        )
        for p in estimated
    ]
    return MleFit(
        loglik=loglik,
        n_restarts=len(restarts),
        n_converged=len(converged),
        params=params,
        restarts=restarts,
        provisional=provisional,
        n_expected=n_expected,
        searches=searches,
        provisional_from_running=from_running,
    )
