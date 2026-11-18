"""Profile-likelihood core — pure functions over the ``profiles/`` CAS tree.

``camdl profile`` writes a tree *separate* from ``fits/``::

    profiles/<base>/<point>/<stage>/<seed>/<start>/run.json

- ``base``  — the inference *problem* (label e.g. ``ctl_binom``). The base hash
  deliberately EXCLUDES the focal grid, so points from different grid
  invocations of the same problem co-locate under one base; we aggregate them.
- ``point`` — one focal grid cell. Its level label names the profiled params:
  1D is ``g=0.0300``; a 2D grid cell is ``g=0.0316__Cscale=0.2000`` (params
  joined by ``__``). The label rounds each value to 4 decimals (grid value
  ``0.23125`` is labelled ``0.2313``), so the coordinate is read exactly from a
  start's ``mle.toml`` ``[focal]`` table, and from the label only when that is
  unreadable. ``inputs.grid_point`` is only the grid *index*, not the value. A profile can be over one param (a
  curve) or two (a likelihood surface); higher-D is read but not rendered.
- ``stage`` — the optimizer method (``nlopt(sbplx)`` / ``if2`` / ``pmmh``).
- ``seed`` / ``start`` — one optimization restart each; ``inputs.best_loglik``
  is its optimized log-likelihood. The profile point's loglik is the MAX over
  restarts (the best optimum found), matching how ``camdl`` reports it.

A start's leaf appears only when that start finishes, so the tree alone cannot
say which grid values an invocation will run or whether it is still running.
camdl records both per invocation, outside every hash::

    profiles/<base>/invocations/<invocation_id>/invocation.json   # the grid, n_starts, seeds
    profiles/<base>/invocations/<invocation_id>/progress.json     # heartbeat, as fits write it

From those, a grid value with no finished start is shown as pending, each value
reports how many of its starts have finished, and a value whose starts are not
all in is *partial*: its log-likelihood is the max over the starts so far, so
it can only rise, and the MLE and interval are computed from complete values
only and marked provisional until the grid is complete. A profile written by a
camdl that kept no invocation record reads as before, with every point treated
as complete.

A profile curve is loglik vs the profiled value; the MLE is the argmax and the
95% CI is the set where loglik ≥ max − 1.92 (½·χ²₁,₀.₉₅), interpolated at the
threshold crossings — ``None`` on a side where the grid never drops below it
(the grid is too narrow to bracket that limit).
"""

from __future__ import annotations

import json
import math
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import progress_is_fresh, read_progress
from .state import RunProgress

# The profile-loglik drop for a 95% confidence set, ½·χ²(df, 0.95): df = number
# of profiled params. 1D → ½·3.8415 (a CI); 2D → ½·5.9915 (a joint region).
CI_DROP_95 = 1.9207
CI_DROP_95_BY_DIM = {1: 1.9207, 2: 2.9957}
CI_LEVEL = 0.95
# camdl writes a huge-negative sentinel (≈ −1e100) for a grid cell whose optimize
# failed / was infeasible. Those aren't real log-likelihoods, so the CI ignores
# them (a mix of real and sentinel points would otherwise interpolate nonsense).
SENTINEL_LOGLIK = -1e99

#: Where camdl keeps each invocation's record under a profile base.
INVOCATIONS_DIR = "invocations"
INVOCATION_SCHEMA = "camdl.profile-invocation/v1"


@dataclass(frozen=True)
class ProfilePoint:
    coords: tuple[float, ...]  # profiled value per param (1 for 1D, 2 for 2D)
    loglik: float  # best optimized log-likelihood over restarts at this cell
    n_starts: int  # restarts (leaves) that contributed
    # The conditional MLE of the OTHER (nuisance) params at this cell — the values
    # the optimizer settled on with the profiled coords held fixed (from the
    # winning restart's mle.toml [mle] table). Empty if unreadable.
    nuisance: dict[str, float]
    # Starts this cell is expected to have (n_starts × seeds of the invocations
    # that list it), or None when no invocation record names it.
    n_expected: int | None = None
    # Spread (max − min) of the finished starts' log-likelihoods, failed
    # starts excluded; None with fewer than two. A wide spread at a complete
    # cell means the optimizer did not find that cell's maximum reliably.
    loglik_spread: float | None = None

    @property
    def complete(self) -> bool:
        """Every expected start has finished (or nothing says what to expect)."""
        return self.n_expected is None or self.n_starts >= self.n_expected


@dataclass(frozen=True)
class PendingPoint:
    """A grid cell an invocation will run that has no finished start yet."""

    coords: tuple[float, ...]
    n_expected: int


@dataclass(frozen=True)
class ExpectedPoint:
    coords: tuple[float, ...]  # exact focal values, in the invocation's param order
    dir: str  # the <point> directory its leaves are written under


@dataclass(frozen=True)
class ProfileInvocation:
    """One ``camdl profile`` invocation's record and heartbeat."""

    invocation_id: str
    started_at: float | None
    label: str | None
    algorithm: str
    params: tuple[str, ...]
    points: tuple[ExpectedPoint, ...]
    n_starts: int
    seeds: tuple[int, ...]
    progress: RunProgress | None

    @property
    def starts_per_point(self) -> int:
        return self.n_starts * max(1, len(self.seeds))

    def status(self, now: float | None = None) -> str:
        """``running`` (fresh heartbeat) | ``stalled`` (a ``running`` heartbeat
        gone stale: killed partway) | ``done`` | ``failed`` | ``unknown`` (no
        heartbeat)."""
        prog = self.progress
        if prog is None:
            return "unknown"
        if prog.state == "running":
            return "running" if progress_is_fresh(prog, now) else "stalled"
        if prog.state in ("done", "failed"):
            return prog.state
        return "unknown"


@dataclass(frozen=True)
class ProfileCurve:
    base_id: str  # profiles/<base_id> dir name, e.g. ctl_binom-e7b69bc0
    label: str  # human label from the base level, e.g. ctl_binom
    params: tuple[str, ...]  # profiled param names, e.g. ('g',) or ('g', 'Cscale')
    method: str  # optimizer, e.g. nl-sbplx
    loglik_type: str  # e.g. ode_marginal
    points: list[ProfilePoint]  # 1D: ascending by value; ND: grid order
    # Grid cells an invocation will run with no finished start yet.
    pending: list[PendingPoint] = field(default_factory=list)
    invocations: list[ProfileInvocation] = field(default_factory=list)

    @property
    def ndim(self) -> int:
        return len(self.params)

    @property
    def ci_drop(self) -> float:
        return CI_DROP_95_BY_DIM.get(self.ndim, CI_DROP_95)

    @property
    def provisional(self) -> bool:
        """The grid is not complete: some cell is pending or partial, so the
        MLE and interval may still move."""
        return bool(self.pending) or any(not p.complete for p in self.points)

    @property
    def estimate_points(self) -> list[ProfilePoint]:
        """The cells the MLE and interval are computed from: the complete ones
        when there are any, since a partial cell's log-likelihood is only a
        lower bound; else every cell (and the result is provisional)."""
        done = [p for p in self.points if p.complete]
        return done or list(self.points)

    @property
    def mle(self) -> ProfilePoint | None:
        pts = self.estimate_points
        return max(pts, key=lambda p: p.loglik) if pts else None

    def status(self, now: float | None = None) -> str:
        """``running`` if any invocation's heartbeat is fresh, else the latest
        invocation's status; ``unknown`` when camdl kept no invocation record."""
        if not self.invocations:
            return "unknown"
        if any(inv.status(now) == "running" for inv in self.invocations):
            return "running"
        latest = max(self.invocations, key=lambda inv: inv.started_at or 0.0)
        return latest.status(now)

    def starts_progress(self) -> tuple[int, int] | None:
        """(finished, expected) starts over the cells an invocation names, or
        None when no invocation record says what to expect."""
        cells = [p for p in self.points if p.n_expected is not None]
        if not cells and not self.pending:
            return None
        done = sum(min(p.n_starts, p.n_expected or 0) for p in cells)
        expected = sum(p.n_expected or 0 for p in cells) + sum(p.n_expected for p in self.pending)
        return done, expected

    def ci_bounds_1d(self) -> tuple[float | None, float | None]:
        """1D-only: (lo, hi) where loglik crosses ``max − ci_drop``, linearly
        interpolated between the bracketing grid points. A side is ``None`` when
        the curve never falls below the threshold before the grid ends. Returns
        (None, None) for a profile that isn't one-dimensional."""
        if self.ndim != 1:
            return None, None
        pts = sorted(
            (p for p in self.estimate_points if p.loglik > SENTINEL_LOGLIK),
            key=lambda p: p.coords[0],
        )
        if len(pts) < 2:
            return None, None
        thresh = max(pts, key=lambda p: p.loglik).loglik - self.ci_drop
        mle_i = max(range(len(pts)), key=lambda i: pts[i].loglik)

        def cross(a: ProfilePoint, b: ProfilePoint) -> float:
            # Linear interpolation of the value where loglik == thresh on [a, b].
            if a.loglik == b.loglik:
                return a.coords[0]
            t = (thresh - a.loglik) / (b.loglik - a.loglik)
            return a.coords[0] + t * (b.coords[0] - a.coords[0])

        lo: float | None = None
        for i in range(mle_i, 0, -1):
            if pts[i - 1].loglik <= thresh <= pts[i].loglik:
                lo = cross(pts[i - 1], pts[i])
                break
        hi: float | None = None
        for i in range(mle_i, len(pts) - 1):
            if pts[i + 1].loglik <= thresh <= pts[i].loglik:
                hi = cross(pts[i + 1], pts[i])
                break
        return lo, hi


def profiles_root(store: Path) -> Path:
    """The ``profiles/`` tree — a sibling of the fit store (``results/fits`` →
    ``results/profiles``)."""
    return store.parent / "profiles"


def _load_json(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _level_label(rec: dict, name: str) -> str | None:
    for lv in rec.get("levels", []):
        if isinstance(lv, dict) and lv.get("name") == name:
            lbl = lv.get("label")
            return str(lbl) if lbl is not None else None
    return None


def _parse_coords(
    label: str | None, seg: str
) -> tuple[tuple[str, ...], tuple[float, ...]] | None:
    """(param names, values) from the point level label, N-D aware: a 1D point is
    ``g=0.0300`` and a 2D grid cell is ``g=0.0316__Cscale=0.2000`` (params joined
    by ``__``). The label is authoritative for the param *names* (the dir segment
    lower-cases them); fall back to the dir segment ``g_0.0316__cscale_0.2000-<hash>``
    only when the label is missing or malformed."""
    if label and "=" in label:
        params: list[str] = []
        values: list[float] = []
        for part in label.split("__"):
            k, _, v = part.partition("=")
            try:
                values.append(float(v))
            except ValueError:
                params = []
                break
            params.append(k.strip())
        if params and len(params) == len(values):
            return tuple(params), tuple(values)
    # Fallback: g_0.0316__cscale_0.2000-<hash>
    body = seg.rsplit("-", 1)[0]
    params, values = [], []
    for part in body.split("__"):
        k, _, v = part.partition("_")
        if not v:
            return None
        try:
            values.append(float(v))
        except ValueError:
            return None
        params.append(k)
    return (tuple(params), tuple(values)) if params else None


def _read_focal(leaf: Path, names: tuple[str, ...]) -> tuple[float, ...] | None:
    """The exact grid values of ``names`` from a start's ``mle.toml``
    ``[focal]`` table, or ``None`` if the table is unreadable or lacks one."""
    try:
        obj = tomllib.loads((leaf / "mle.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    focal = obj.get("focal")
    if not isinstance(focal, dict):
        return None
    out: list[float] = []
    for n in names:
        v = focal.get(n)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        out.append(float(v))
    return tuple(out)


def _read_nuisance(leaf: Path) -> dict[str, float]:
    """The conditional-MLE nuisance params from a restart leaf's ``mle.toml``
    ``[mle]`` table (the coords held fixed live under ``[focal]``, not here).
    Empty on any read/parse failure or if the leaf wrote no such table."""
    try:
        obj = tomllib.loads((leaf / "mle.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    mle = obj.get("mle")
    if not isinstance(mle, dict):
        return {}
    return {
        str(k): float(v) for k, v in mle.items() if isinstance(v, (int, float))
    }


def _read_invocation(d: Path) -> ProfileInvocation | None:
    """One invocation's record, or ``None`` if it is missing, unreadable or of
    a schema this reader does not know (which is skipped, not guessed at)."""
    rec = _load_json(d / "invocation.json")
    if rec is None or rec.get("schema") != INVOCATION_SCHEMA:
        return None
    focal = rec.get("focal")
    if not isinstance(focal, list) or not focal:
        return None
    params = tuple(str(f.get("name")) for f in focal if isinstance(f, dict) and f.get("name"))
    if len(params) != len(focal):
        return None
    points: list[ExpectedPoint] = []
    for pt in rec.get("points") or []:
        if not isinstance(pt, dict) or not isinstance(pt.get("dir"), str):
            continue
        coords = pt.get("coords") if isinstance(pt.get("coords"), dict) else {}
        vals: list[float] = []
        for n in params:
            v = coords.get(n)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                break
            vals.append(float(v))
        if len(vals) == len(params):
            points.append(ExpectedPoint(tuple(vals), pt["dir"]))
    n_starts = rec.get("n_starts")
    seeds = rec.get("seeds")
    started = rec.get("started_at")
    label = rec.get("label")
    return ProfileInvocation(
        invocation_id=str(rec.get("invocation_id") or d.name),
        started_at=float(started) if isinstance(started, (int, float)) else None,
        label=label if isinstance(label, str) else None,
        algorithm=str(rec.get("algorithm") or ""),
        params=params,
        points=tuple(points),
        n_starts=n_starts if isinstance(n_starts, int) and n_starts > 0 else 1,
        seeds=tuple(x for x in seeds if isinstance(x, int)) if isinstance(seeds, list) else (),
        progress=read_progress(d),
    )


def read_invocations(base_dir: Path) -> list[ProfileInvocation]:
    """Every readable invocation record under a profile base, oldest first."""
    root = base_dir / INVOCATIONS_DIR
    try:
        dirs = sorted(d for d in root.iterdir() if d.is_dir())
    except OSError:
        return []
    invs = [inv for d in dirs if (inv := _read_invocation(d)) is not None]
    invs.sort(key=lambda inv: inv.started_at or 0.0)
    return invs


def load_profile(store: Path, base_id: str) -> ProfileCurve | None:
    """The full profile (1D curve or 2D grid) for one base, or ``None`` if it has
    neither a readable finished point nor an invocation record naming one. A
    running profile returns the points computed so far plus the cells still
    pending, each with how many of its starts have finished."""
    base_dir = profiles_root(store) / base_id
    if not base_dir.is_dir():
        return None

    invocations = read_invocations(base_dir)
    # Starts expected per cell directory: the most any invocation asks of it.
    expected: dict[str, int] = {}
    expected_coords: dict[str, tuple[float, ...]] = {}
    for inv in invocations:
        for ep in inv.points:
            expected[ep.dir] = max(expected.get(ep.dir, 0), inv.starts_per_point)
            expected_coords.setdefault(ep.dir, ep.coords)

    label = base_id
    params: tuple[str, ...] = ()
    method = ""
    loglik_type = ""
    points: list[ProfilePoint] = []
    seen_dirs: set[str] = set()

    for point_dir in sorted(base_dir.iterdir()):
        if not point_dir.is_dir() or point_dir.name == INVOCATIONS_DIR:
            continue
        best = -math.inf
        n = 0
        finished: list[float] = []
        coords: tuple[float, ...] | None = None
        best_leaf: Path | None = None
        for run_json in point_dir.rglob("run.json"):
            rec = _load_json(run_json)
            if rec is None or rec.get("kind") != "profile_point":
                continue
            inp = rec.get("inputs") or {}
            if coords is None:
                parsed = _parse_coords(_level_label(rec, "point"), point_dir.name)
                if parsed is not None:
                    names, coords = parsed
                    coords = _read_focal(run_json.parent, names) or coords
                    if not params:
                        params = names
                base_lbl = _level_label(rec, "profile")
                if base_lbl:
                    label = base_lbl
                method = str(inp.get("method", method) or method)
                loglik_type = str(inp.get("loglik_type", loglik_type) or loglik_type)
            ll = inp.get("best_loglik")
            if isinstance(ll, (int, float)) and math.isfinite(ll):
                fll = float(ll)
                if fll > SENTINEL_LOGLIK:
                    finished.append(fll)
                if fll > best:  # track the argmax restart for its nuisance MLE
                    best = fll
                    best_leaf = run_json.parent
                n += 1
        if coords is not None and n > 0:
            seen_dirs.add(point_dir.name)
            points.append(ProfilePoint(
                coords=coords, loglik=best, n_starts=n,
                nuisance=_read_nuisance(best_leaf) if best_leaf else {},
                n_expected=expected.get(point_dir.name),
                loglik_spread=(max(finished) - min(finished)) if len(finished) >= 2 else None,
            ))

    pending = [
        PendingPoint(coords=expected_coords[d], n_expected=k)
        for d, k in expected.items() if d not in seen_dirs
    ]
    if not params and invocations:
        params = invocations[-1].params
    if not method and invocations:
        method = invocations[-1].algorithm
    if (not points and not pending) or not params:
        return None
    # 1D reads best as a sorted curve; a 2D grid stays in discovery order.
    if len(params) == 1:
        points.sort(key=lambda p: p.coords[0])
        pending.sort(key=lambda p: p.coords[0])
    return ProfileCurve(
        base_id=base_id, label=label, params=params,
        method=method, loglik_type=loglik_type, points=points,
        pending=pending, invocations=invocations,
    )


def discover_profiles(store: Path) -> list[ProfileCurve]:
    """Every readable profile under ``profiles/``, newest base first by mtime."""
    root = profiles_root(store)
    if not root.is_dir():
        return []
    bases = [d for d in root.iterdir() if d.is_dir()]
    bases.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    curves = [load_profile(store, d.name) for d in bases]
    return [c for c in curves if c is not None]
