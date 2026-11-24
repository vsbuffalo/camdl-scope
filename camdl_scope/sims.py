"""Forward-simulation runs — ``camdl simulate`` results in the run store.

Each simulation cell is a ``sim`` result (camdl run-store proposal, "On disk")::

    <build>/sim/<model>/<settings>/<params>/<scenario>/seed_<n>-<12 hex>/
        identity.json  # the cell: its model, settings, params.values, scenario, seed
        run.json       # status, and the invocation that claimed it
        traj.tsv       # WIDE trajectory: one row per time t, a column per
                       # (state|flow × stratum), fully expanded

A *sim* here is one ``camdl simulate`` command, and its members are the cells
it asked for: the ones its finished record lists, and while it runs, those whose
``run.json`` names it. A sweep's members differ in their parameter values, and
each is labelled by the values that vary. The store records no role for a
``traj.tsv`` column, so roles follow camdl's naming: the time columns, the
``flow_`` columns, and every other column a state (:func:`resolve_roles`); a
compartment *total* sums the columns that share a compartment's name (see
:func:`compartment_bases`).

Pure readers over the store — no dependency on the fit ingest. Time is dated via
the model's calendar (:func:`read_sim_model`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from . import store

#: A simulation cell's kind, and the depth of its folder below the kind's
#: directory (model, settings, params, scenario, then the cell).
SIM_KIND = "sim"
SIM_DEPTH = 5
#: The time columns of a wide trajectory.
TIME_COLUMNS = ("t", "t_start", "t_stop", "time")


@dataclass(frozen=True)
class SimMember:
    """One sweep point of a sim: the (config, params, scenario, seed) leaf that
    carries a ``traj.tsv``. ``label`` is the human sweep-point tag (the swept
    param value(s) when they can be read, e.g. ``het=0.5``)."""

    label: str
    scenario: str
    seed: str
    traj_path: Path
    params: dict[str, str] = field(default_factory=dict)  # the cell's parameter values


@dataclass(frozen=True)
class SimMeta:
    """One ``camdl simulate`` command and its members. ``sim_id`` is the
    command's invocation sequence number; ``updated_at`` is the newest member
    trajectory mtime."""

    sim_id: str
    model: str
    #: ``running`` | ``stalled`` | ``done`` | ``failed`` (some cell failed).
    status: str
    members: list[SimMember] = field(default_factory=list)
    updated_at: float = 0.0
    #: The compiled model's folder (``source.camdl``, ``model.ir.json``).
    model_dir: Path | None = None


def _sweep_labels(param_sets: list[dict[str, str]], fallbacks: list[str]) -> list[str]:
    """Label each member by the param(s) that vary across the sweep (e.g.
    ``het=0.5``); fall back to the params-level tag when nothing varies or no
    params were recorded."""
    keys = set().union(*[set(p) for p in param_sets]) if param_sets else set()
    varying = sorted(k for k in keys if len({p.get(k) for p in param_sets}) > 1)
    # Only a systematic sweep (one or two varied params) reads as a legend; a
    # many-param ensemble (a PPC over posterior draws) falls back to its tag.
    if not (1 <= len(varying) <= 2):
        return list(fallbacks)
    return [
        ", ".join(f"{k}={p[k]}" for k in varying) if all(k in p for k in varying)
        else fallbacks[i]
        for i, p in enumerate(param_sets)
    ]


@dataclass(frozen=True)
class _Cell:
    folder: Path
    params: dict[str, str]
    scenario: str
    seed: str
    model_run_id: str


def _read_cell(folder: Path) -> _Cell | None:
    ident = store.read_identity(folder)
    if ident is None or ident.kind != SIM_KIND or not (folder / "traj.tsv").is_file():
        return None
    body = ident.body
    values = (body.get("params") or {}).get("values") or {}
    names = (body.get("scenario") or {}).get("names") or []
    seed = (body.get("seed") or {}).get("seed")
    model = body.get("model") or {}
    return _Cell(
        folder=folder,
        params={str(k): f"{v:g}" if isinstance(v, (int, float)) else str(v) for k, v in values.items()},
        scenario=",".join(str(n) for n in names),
        seed="" if seed is None else str(seed),
        model_run_id=str(model.get("run_id") or ""),
    )


def _sim_status(inv: store.Invocation, liveness: store.Liveness) -> str:
    if inv.status == "ended":
        return "failed" if any(r.outcome == "failed" for r in inv.results) else "done"
    if inv.status == "interrupted":
        return "stalled"
    return "stalled" if liveness.alive(inv.owner) is False else "running"


def discover_sims(root: Path) -> list[SimMeta]:
    """Every ``camdl simulate`` command in the store with a finished cell,
    newest first."""
    if not store.is_store(root):
        return []
    by_id: dict[str, Path] = {}
    by_inv: dict[int, list[Path]] = {}
    for kd in store.kind_dirs(root, SIM_KIND):
        for folder in store.walk(kd, SIM_DEPTH):
            ident = store.read_identity(folder)
            rec = store.read_run_record(folder)
            if ident is None or rec is None or rec.status != "completed":
                continue
            by_id[ident.run_id] = folder
            if rec.invocation is not None:
                by_inv.setdefault(rec.invocation, []).append(folder)
    models = store.folders_by_id(root, "compiled_model", 1)
    liveness = store.Liveness.read(root)
    sims: list[SimMeta] = []
    for inv in store.read_invocations(root):
        if inv.command != "simulate":
            continue
        listed = [by_id[r.run_id] for r in inv.results if r.kind == SIM_KIND and r.run_id in by_id]
        folders = list(dict.fromkeys([*listed, *by_inv.get(inv.seq, [])]))
        cells = [c for f in folders if (c := _read_cell(f)) is not None]
        if not cells:
            continue
        labels = _sweep_labels([c.params for c in cells], [c.folder.parent.parent.name for c in cells])
        members = [
            SimMember(label=labels[i], scenario=c.scenario, seed=c.seed,
                      traj_path=c.folder / "traj.tsv", params=c.params)
            for i, c in enumerate(cells)
        ]
        newest = max(((c.folder / "traj.tsv").stat().st_mtime for c in cells), default=0.0)
        model_dir = models.get(cells[0].model_run_id)
        args = inv.given.get("args") if isinstance(inv.given.get("args"), dict) else {}
        model = str(args.get("model") or "")
        sims.append(SimMeta(
            sim_id=str(inv.seq), model=Path(model).stem if model else "",
            status=_sim_status(inv, liveness), members=members, updated_at=newest,
            model_dir=model_dir,
        ))
    sims.sort(key=lambda s: s.updated_at, reverse=True)
    return sims


@dataclass(frozen=True)
class SimModel:
    """Model facts read from a sim's source ``.camdl`` (via provenance): the time
    calendar (dates) and the exact compartment names (so column totals group on
    the real states, not a heuristic). Any field is None/empty when unreadable."""

    origin: str | None = None
    time_unit: str = "days"
    states: list[str] = field(default_factory=list)


_ORIGIN_RE = re.compile(r'origin\s*=\s*date\(\s*"([^"]+)"\s*\)')
_TIME_UNIT_RE = re.compile(r"time_unit\s*=\s*'?([A-Za-z_]+)")
_COMPARTMENTS_RE = re.compile(r"compartments\s*\{([^}]*)\}")


def read_sim_model(sim: SimMeta) -> SimModel:
    """Parse the sim's model source (the compiled model's ``source.camdl``) for
    its origin/time_unit and compartment list. Read-only, self-contained (no
    ``camdl`` subprocess); an empty :class:`SimModel` when it cannot be read."""
    if sim.model_dir is None:
        return SimModel()
    try:
        text = (sim.model_dir / "source.camdl").read_text()
    except OSError:
        return SimModel()
    text = re.sub(r"#.*", "", text)  # strip comments before matching
    om = _ORIGIN_RE.search(text)
    tm = _TIME_UNIT_RE.search(text)
    cm = _COMPARTMENTS_RE.search(text)
    states = (
        [s.strip() for s in cm.group(1).split(",") if s.strip()] if cm else []
    )
    return SimModel(
        origin=om.group(1) if om else None,
        time_unit=(tm.group(1) if tm else "days"),
        states=states,
    )


def _read_header(path: Path) -> list[str]:
    try:
        return pl.read_csv(path, separator="\t", n_rows=0).columns
    except (OSError, pl.exceptions.PolarsError):
        return []


def resolve_roles(member: SimMember) -> dict[str, str]:
    """Column → role for a member's trajectory, by camdl's naming: a time
    column (``t``, ``t_start``, ``t_stop``), a ``flow_`` column, or a state.
    Columns are the same across a sim's members (only params sweep), so callers
    resolve once."""
    roles: dict[str, str] = {}
    for c in _read_header(member.traj_path):
        if c in TIME_COLUMNS:
            roles[c] = "time"
        elif c.startswith("flow_"):
            roles[c] = "flow"
        else:
            roles[c] = "state"
    return roles


def _cols_by_role(roles: dict[str, str], role: str) -> list[str]:
    return [c for c, r in roles.items() if r == role]


def _time_col(roles: dict[str, str]) -> str | None:
    ts = _cols_by_role(roles, "time")
    return ts[0] if ts else None


def _token_suffix_len(a: str, b: str) -> int:
    """Length (in underscore tokens) of the shared trailing token run of a, b."""
    at, bt = a.split("_"), b.split("_")
    i = 0
    while i < min(len(at), len(bt)) and at[-1 - i] == bt[-1 - i]:
        i += 1
    return i


def compartment_bases(
    columns: list[str], states: list[str] | None = None
) -> dict[str, str]:
    """Map each wide column to its compartment base name. With the model's exact
    ``states`` (from :func:`read_sim_model`), a column's compartment is the
    longest state that prefixes it (``S_naive_kwaru_…`` → ``S_naive``) — exact.
    Without them, fall back to a model-free heuristic: the stratum is the longest
    token-suffix a column shares with a column whose leading token differs, and
    the compartment is what remains (this folds ``S_naive``/``S_immune`` into the
    ``S`` total — a per-stratum breakdown then needs the model schema)."""
    if states:
        by_len = sorted(states, key=len, reverse=True)
        return {
            c: next(
                (s for s in by_len if c == s or c.startswith(s + "_")),
                c.split("_", 1)[0],
            )
            for c in columns
        }
    first = {c: c.split("_", 1)[0] for c in columns}
    bases: dict[str, str] = {}
    for c in columns:
        ct = c.split("_")
        best = 0
        for d in columns:
            if first[d] == first[c]:
                continue
            l = _token_suffix_len(c, d)
            if l > best:
                best = l
            if best >= len(ct) - 1:
                break
        bases[c] = "_".join(ct[: len(ct) - best]) if 0 < best < len(ct) else c
    return bases


@dataclass(frozen=True)
class SimSeries:
    """One member's trajectory of a compartment total: aligned ``time`` + ``value``
    (summed over the compartment's strata)."""

    member: str
    scenario: str
    time: list[float]
    value: list[float]


def available_series(
    roles: dict[str, str], role: str = "state", states: list[str] | None = None
) -> list[str]:
    """The compartment (``state``) base names present, in first-seen order —
    resolved column roles in, compartment totals out. ``states`` (exact model
    compartments) makes the grouping exact instead of heuristic."""
    cols = _cols_by_role(roles, role)
    if not cols:
        return []
    bases = compartment_bases(cols, states)
    seen: list[str] = []
    for c in cols:
        b = bases[c]
        if b not in seen:
            seen.append(b)
    return seen


def read_member_total(
    member: SimMember,
    base: str,
    roles: dict[str, str],
    role: str = "state",
    states: list[str] | None = None,
) -> SimSeries | None:
    """A member's trajectory for compartment ``base``: the time column plus the
    row-wise sum of every ``role`` column whose base name is ``base``. None when
    the trajectory or the columns are missing."""
    tcol = _time_col(roles)
    if tcol is None:
        return None
    cols = _cols_by_role(roles, role)
    bases = compartment_bases(cols, states)
    group = [c for c in cols if bases.get(c) == base]
    if not group:
        return None
    try:
        df = pl.read_csv(member.traj_path, separator="\t", columns=[tcol, *group])
    except (OSError, pl.exceptions.PolarsError):
        return None
    total = df.select(pl.sum_horizontal(group).alias("v"))["v"]
    return SimSeries(
        member=member.label,
        scenario=member.scenario,
        time=df[tcol].cast(pl.Float64).to_list(),
        value=total.cast(pl.Float64).to_list(),
    )
