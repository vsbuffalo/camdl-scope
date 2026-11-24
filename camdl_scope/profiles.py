"""Profile likelihoods — pure functions over camdl's stored profile points.

``camdl profile`` stores each (grid cell, seed, start) as a ``profile_point``
result of its own (camdl run-store proposal, "A profile")::

    <build>/profile_point/<problem>/<focal values>/<method>/seed_<n>-start_<i>-<12 hex>/
        mle.toml          # [focal] the cell's exact values, [mle] the nuisance MLE
        provenance.json   # best_loglik, method, loglik_type
        identity.json  run.json

A point is reused by any later command that asks for it, so the stored object
is a point and a *profile* is one command: the points that command asked for.
That is how camdl itself renders a profile table, one per command. A finished
command's invocation record lists every point it asked for, computed or
reused; while it runs, its points are those whose ``run.json`` names it as
their invocation, and the grid it will run is the ``--sweep`` it was given
(``given.args.sweep``), expanded as camdl's ``Grid::expand`` expands it.

From those, a grid value with no finished start is shown as pending, each value
reports how many of its starts have finished, and a value whose starts are not
all in is *partial*: its log-likelihood is the max over the starts so far, so
it can only rise, and the MLE and interval are computed from complete values
only and marked provisional until the grid is complete.

A profile curve is loglik vs the profiled value; the MLE is the argmax and the
95% CI is the set where loglik ≥ max − 1.92 (½·χ²₁,₀.₉₅), interpolated at the
threshold crossings — ``None`` on a side where the grid never drops below it
(the grid is too narrow to bracket that limit).
"""

from __future__ import annotations

import itertools
import json
import math
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import store

# The profile-loglik drop for a 95% confidence set, ½·χ²(df, 0.95): df = number
# of profiled params. 1D → ½·3.8415 (a CI); 2D → ½·5.9915 (a joint region).
CI_DROP_95 = 1.9207
CI_DROP_95_BY_DIM = {1: 1.9207, 2: 2.9957}
CI_LEVEL = 0.95
# camdl writes a huge-negative sentinel (≈ −1e100) for a grid cell whose optimize
# failed / was infeasible. Those aren't real log-likelihoods, so the CI ignores
# them (a mix of real and sentinel points would otherwise interpolate nonsense).
SENTINEL_LOGLIK = -1e99

#: A profile point's kind, and the depth of its folder below the kind's
#: directory (problem, focal values, method, then the point).
POINT_KIND = "profile_point"
POINT_DEPTH = 4
#: A running command's progress file is rewritten every few seconds; this many
#: seconds without one reads as stopped, for an owner on another host.
PROGRESS_STALE_S = 30.0


@dataclass(frozen=True)
class ProfilePoint:
    coords: tuple[float, ...]  # profiled value per param (1 for 1D, 2 for 2D)
    loglik: float  # best optimized log-likelihood over restarts at this cell
    n_starts: int  # restarts (leaves) that contributed
    # The conditional MLE of the OTHER (nuisance) params at this cell — the values
    # the optimizer settled on with the profiled coords held fixed (from the
    # winning restart's mle.toml [mle] table). Empty if unreadable.
    nuisance: dict[str, float]
    # Starts this cell is expected to have (n_starts × seeds of the command),
    # or None when the cell is not on the command's grid.
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
    """A grid cell the command will run that has no finished start yet."""

    coords: tuple[float, ...]
    n_expected: int


@dataclass(frozen=True)
class ProfileCurve:
    #: The command's invocation sequence number, as a string: what the API and
    #: a ``?profile=`` link name it by.
    base_id: str
    label: str  # the command's --label, else the model's name
    params: tuple[str, ...]  # profiled param names, e.g. ('g',) or ('g', 'Cscale')
    method: str  # optimizer, e.g. nl-sbplx
    loglik_type: str  # e.g. ode_marginal
    points: list[ProfilePoint]  # 1D: ascending by value; ND: grid order
    # Grid cells the command will run with no finished start yet.
    pending: list[PendingPoint] = field(default_factory=list)
    #: ``running`` | ``stalled`` (the command stopped without ending: killed
    #: partway) | ``done`` | ``failed`` (some point failed).
    run_status: str = "done"
    #: When the running command last reported progress (unix seconds).
    heartbeat_at: float | None = None

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
        only, since a partial cell's log-likelihood is a lower bound. An
        interval needs two of them; none is drawn from partial cells."""
        return [p for p in self.points if p.complete]

    @property
    def mle(self) -> ProfilePoint | None:
        """The best complete cell; while none is complete, the best cell so
        far (a provisional answer, flagged by :attr:`provisional`)."""
        pts = self.estimate_points or self.points
        return max(pts, key=lambda p: p.loglik) if pts else None

    def status(self) -> str:
        return self.run_status

    def starts_progress(self) -> tuple[int, int] | None:
        """(finished, expected) starts over the command's grid, or None when
        no cell is on it."""
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


def _read_toml(path: Path) -> dict:
    try:
        obj = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _floats(table: object) -> dict[str, float]:
    if not isinstance(table, dict):
        return {}
    return {
        str(k): float(v) for k, v in table.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


@dataclass(frozen=True)
class _Start:
    """One finished start at one grid cell, from its ``profile_point`` folder."""

    focal: dict[str, float]  # the cell's exact values ([focal])
    loglik: float
    nuisance: dict[str, float]  # the conditional MLE of the other params ([mle])
    method: str
    loglik_type: str


def _read_start(folder: Path) -> _Start | None:
    mle = _read_toml(folder / "mle.toml")
    focal = _floats(mle.get("focal"))
    try:
        prov = json.loads((folder / "provenance.json").read_text())
    except (OSError, json.JSONDecodeError):
        prov = {}
    ll = prov.get("best_loglik", mle.get("final_loglik"))
    if not focal or isinstance(ll, bool) or not isinstance(ll, (int, float)) or not math.isfinite(ll):
        return None
    return _Start(
        focal=focal, loglik=float(ll), nuisance=_floats(mle.get("mle")),
        method=str(prov.get("method") or ""), loglik_type=str(prov.get("loglik_type") or ""),
    )


def expand_grid(spec: object) -> list[float]:
    """One ``--sweep`` grid as camdl recorded it in ``given`` (``{"list": [...]}``,
    ``{"linear": {min, max, n}}`` or ``{"log10": {min, max, n}}``), expanded as
    camdl's ``Grid::expand`` does (``cli/src/args/types.rs``): end points
    inclusive, and ``n = 1`` is ``[min]``."""
    if not isinstance(spec, dict) or len(spec) != 1:
        return []
    (kind, body), = spec.items()
    if kind == "list":
        return [float(v) for v in body] if isinstance(body, list) else []
    if kind not in ("linear", "log10") or not isinstance(body, dict):
        return []
    lo, hi, n = float(body["min"]), float(body["max"]), int(body["n"])
    if n == 1:
        return [lo]
    if kind == "linear":
        return [lo + (hi - lo) * (i / (n - 1)) for i in range(n)]
    llo, lhi = math.log10(lo), math.log10(hi)
    return [10.0 ** (llo + (lhi - llo) * (i / (n - 1))) for i in range(n)]


def _cell_key(coords: tuple[float, ...]) -> tuple[float, ...]:
    """Coordinates rounded to 12 significant digits: two commands can reach one
    grid value by different arithmetic (``0.3`` listed, ``0.30000000000000004``
    from a linear grid), which camdl stores as two cells and a curve shows as
    one."""
    return tuple(float(f"{c:.12g}") for c in coords)


@dataclass(frozen=True)
class _Command:
    """What a ``camdl profile`` command asked for, from its ``given`` arguments."""

    params: tuple[str, ...]
    grid: list[tuple[float, ...]]
    starts_per_cell: int
    model: str

    @classmethod
    def of(cls, inv: store.Invocation) -> "_Command | None":
        args = inv.given.get("args")
        if not isinstance(args, dict):
            return None
        sweeps = [s for s in args.get("sweep") or [] if isinstance(s, dict) and s.get("name")]
        if not sweeps:
            return None
        params = tuple(str(s["name"]) for s in sweeps)
        axes = [expand_grid(s.get("grid")) for s in sweeps]
        seeds = args.get("seeds")
        n_seeds = len(seeds) if isinstance(seeds, list) and seeds else 1
        n_starts = args.get("n_starts")
        n_starts = n_starts if isinstance(n_starts, int) and n_starts > 0 else 1
        model = str(args.get("model") or "")
        return cls(
            params=params,
            grid=list(itertools.product(*axes)),
            starts_per_cell=n_starts * n_seeds,
            model=Path(model).stem if model else "",
        )


def _status(
    inv: store.Invocation, liveness: store.Liveness, beat: float | None, now: float,
) -> str:
    if inv.status == "ended":
        return "failed" if any(r.outcome == "failed" for r in inv.results) else "done"
    if inv.status == "interrupted":
        return "stalled"
    alive = liveness.alive(inv.owner)
    if alive is None:
        alive = beat is not None and now - beat <= PROGRESS_STALE_S
    return "running" if alive else "stalled"


@dataclass(frozen=True)
class _StoredPoints:
    """Every stored profile point, read once per request: its folder by run id,
    and which invocation claimed it (``run.json`` ``invocation``)."""

    by_id: dict[str, Path]
    by_invocation: dict[int, list[Path]]

    @classmethod
    def read(cls, root: Path) -> "_StoredPoints":
        by_id: dict[str, Path] = {}
        by_inv: dict[int, list[Path]] = {}
        for kd in store.kind_dirs(root, POINT_KIND):
            for folder in store.walk(kd, POINT_DEPTH):
                ident = store.read_identity(folder)
                rec = store.read_run_record(folder)
                if ident is None or rec is None or rec.status != "completed":
                    continue
                by_id[ident.run_id] = folder
                if rec.invocation is not None:
                    by_inv.setdefault(rec.invocation, []).append(folder)
        return cls(by_id, by_inv)

    def of(self, inv: store.Invocation) -> list[Path]:
        """The points a command asked for: those its record lists, and those
        it computed (which a running command has not listed yet)."""
        listed = [self.by_id[r.run_id] for r in inv.results
                  if r.kind == POINT_KIND and r.run_id in self.by_id]
        return list(dict.fromkeys([*listed, *self.by_invocation.get(inv.seq, [])]))


def _curve(
    root: Path, inv: store.Invocation, points: _StoredPoints, liveness: store.Liveness, now: float,
) -> ProfileCurve | None:
    cmd = _Command.of(inv)
    if cmd is None:
        return None
    by_cell: dict[tuple[float, ...], list[_Start]] = {}
    for folder in points.of(inv):
        st = _read_start(folder)
        if st is None or any(p not in st.focal for p in cmd.params):
            continue
        coords = tuple(st.focal[p] for p in cmd.params)
        by_cell.setdefault(_cell_key(coords), []).append(st)
    grid = {_cell_key(c): c for c in cmd.grid}
    out: list[ProfilePoint] = []
    for key, starts in by_cell.items():
        best = max(starts, key=lambda s: s.loglik)
        finished = [s.loglik for s in starts if s.loglik > SENTINEL_LOGLIK]
        out.append(ProfilePoint(
            coords=grid.get(key, tuple(best.focal[p] for p in cmd.params)),
            loglik=best.loglik, n_starts=len(starts), nuisance=best.nuisance,
            n_expected=cmd.starts_per_cell if key in grid else None,
            loglik_spread=(max(finished) - min(finished)) if len(finished) >= 2 else None,
        ))
    pending = [PendingPoint(c, cmd.starts_per_cell) for k, c in grid.items() if k not in by_cell]
    if not out and not pending:
        return None
    if len(cmd.params) == 1:
        out.sort(key=lambda p: p.coords[0])
        pending.sort(key=lambda p: p.coords[0])
    prog = store.invocation_progress(root, inv.seq) if inv.status == "running" else None
    beat = prog.at if prog else None
    any_start = next((s for ss in by_cell.values() for s in ss), None)
    args = inv.given.get("args") if isinstance(inv.given.get("args"), dict) else {}
    return ProfileCurve(
        base_id=str(inv.seq),
        label=inv.label or cmd.model,
        params=cmd.params,
        method=any_start.method if any_start else str(args.get("algorithm") or ""),
        loglik_type=any_start.loglik_type if any_start else "",
        points=out, pending=pending,
        run_status=_status(inv, liveness, beat, now),
        heartbeat_at=beat,
    )


def _profile_invocations(root: Path) -> list[store.Invocation]:
    return [inv for inv in store.read_invocations(root) if inv.command == "profile"]


def discover_profiles(root: Path) -> list[ProfileCurve]:
    """Every profile command in the store, newest first."""
    if not store.is_store(root):
        return []
    points = _StoredPoints.read(root)
    liveness = store.Liveness.read(root)
    now = time.time()
    curves = [_curve(root, inv, points, liveness, now) for inv in reversed(_profile_invocations(root))]
    return [c for c in curves if c is not None]


def load_profile(root: Path, base_id: str) -> ProfileCurve | None:
    """The profile of the command whose sequence number is ``base_id``."""
    if not store.is_store(root):
        return None
    inv = next((i for i in _profile_invocations(root) if str(i.seq) == base_id), None)
    if inv is None:
        return None
    return _curve(root, inv, _StoredPoints.read(root), store.Liveness.read(root), time.time())
