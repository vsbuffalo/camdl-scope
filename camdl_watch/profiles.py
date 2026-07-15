"""Profile-likelihood core — pure functions over the ``profiles/`` CAS tree.

``camdl profile`` writes a tree *separate* from ``fits/``::

    profiles/<base>/<point>/<stage>/<seed>/<start>/run.json

- ``base``  — the inference *problem* (label e.g. ``ctl_binom``). The base hash
  deliberately EXCLUDES the focal grid, so points from different grid
  invocations of the same problem co-locate under one base; we aggregate them.
- ``point`` — one focal grid cell. Its level label is the authority for the
  coordinate: 1D is ``g=0.0300``; a 2D grid cell is ``g=0.0316__Cscale=0.2000``
  (params joined by ``__``). ``inputs.grid_point`` is only the grid *index*, not
  the value, so we never use it for the axis. A profile can be over one param (a
  curve) or two (a likelihood surface); higher-D is read but not rendered.
- ``stage`` — the optimizer method (``nlopt(sbplx)`` / ``if2`` / ``pmmh``).
- ``seed`` / ``start`` — one optimization restart each; ``inputs.best_loglik``
  is its optimized log-likelihood. The profile point's loglik is the MAX over
  restarts (the best optimum found), matching how ``camdl`` reports it.

A profile curve is loglik vs the profiled value; the MLE is the argmax and the
95% CI is the set where loglik ≥ max − 1.92 (½·χ²₁,₀.₉₅), interpolated at the
threshold crossings — ``None`` on a side where the grid never drops below it
(the grid is too narrow to bracket that limit).
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

# The profile-loglik drop for a 95% confidence set, ½·χ²(df, 0.95): df = number
# of profiled params. 1D → ½·3.8415 (a CI); 2D → ½·5.9915 (a joint region).
CI_DROP_95 = 1.9207
CI_DROP_95_BY_DIM = {1: 1.9207, 2: 2.9957}
CI_LEVEL = 0.95
# camdl writes a huge-negative sentinel (≈ −1e100) for a grid cell whose optimize
# failed / was infeasible. Those aren't real log-likelihoods, so the CI ignores
# them (a mix of real and sentinel points would otherwise interpolate nonsense).
SENTINEL_LOGLIK = -1e99


@dataclass(frozen=True)
class ProfilePoint:
    coords: tuple[float, ...]  # profiled value per param (1 for 1D, 2 for 2D)
    loglik: float  # best optimized log-likelihood over restarts at this cell
    n_starts: int  # restarts (leaves) that contributed
    # The conditional MLE of the OTHER (nuisance) params at this cell — the values
    # the optimizer settled on with the profiled coords held fixed (from the
    # winning restart's mle.toml [mle] table). Empty if unreadable.
    nuisance: dict[str, float]


@dataclass(frozen=True)
class ProfileCurve:
    base_id: str  # profiles/<base_id> dir name, e.g. ctl_binom-e7b69bc0
    label: str  # human label from the base level, e.g. ctl_binom
    params: tuple[str, ...]  # profiled param names, e.g. ('g',) or ('g', 'Cscale')
    method: str  # optimizer, e.g. nl-sbplx
    loglik_type: str  # e.g. ode_marginal
    points: list[ProfilePoint]  # 1D: ascending by value; ND: grid order

    @property
    def ndim(self) -> int:
        return len(self.params)

    @property
    def ci_drop(self) -> float:
        return CI_DROP_95_BY_DIM.get(self.ndim, CI_DROP_95)

    @property
    def mle(self) -> ProfilePoint:
        return max(self.points, key=lambda p: p.loglik)

    def ci_bounds_1d(self) -> tuple[float | None, float | None]:
        """1D-only: (lo, hi) where loglik crosses ``max − ci_drop``, linearly
        interpolated between the bracketing grid points. A side is ``None`` when
        the curve never falls below the threshold before the grid ends. Returns
        (None, None) for a profile that isn't one-dimensional."""
        if self.ndim != 1:
            return None, None
        pts = sorted(
            (p for p in self.points if p.loglik > SENTINEL_LOGLIK),
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


def load_profile(store: Path, base_id: str) -> ProfileCurve | None:
    """The full profile (1D curve or 2D grid) for one base, or ``None`` if it has
    no readable completed points. A running profile returns the points computed so
    far — the grid fills in as more land."""
    base_dir = profiles_root(store) / base_id
    if not base_dir.is_dir():
        return None

    label = base_id
    params: tuple[str, ...] = ()
    method = ""
    loglik_type = ""
    points: list[ProfilePoint] = []

    for point_dir in sorted(base_dir.iterdir()):
        if not point_dir.is_dir():
            continue
        best = -math.inf
        n = 0
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
                if fll > best:  # track the argmax restart for its nuisance MLE
                    best = fll
                    best_leaf = run_json.parent
                n += 1
        if coords is not None and n > 0:
            points.append(ProfilePoint(
                coords=coords, loglik=best, n_starts=n,
                nuisance=_read_nuisance(best_leaf) if best_leaf else {},
            ))

    if not points or not params:
        return None
    # 1D reads best as a sorted curve; a 2D grid stays in discovery order.
    if len(params) == 1:
        points.sort(key=lambda p: p.coords[0])
    return ProfileCurve(
        base_id=base_id, label=label, params=params,
        method=method, loglik_type=loglik_type, points=points,
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
