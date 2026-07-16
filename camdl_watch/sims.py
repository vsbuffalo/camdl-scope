"""Forward-simulation runs — the ``sims/`` CAS tree.

A ``camdl simulate`` run is a sibling kind of the fit store, laid out as::

    sims/<model>-<hash>/<config>-<hash>/<params>-<hash>/<scenario>-<hash>/<seed>-<hash>/
        run.json      # kind="sim", output_schema, levels, status
        traj.tsv      # WIDE trajectory: one row per time t, a column per
                      # (state|flow × stratum), fully expanded

The ``<params>`` level (``base-*``) is the **sweep**: several param points under
one model form the overlay members. ``output_schema`` tags each ``traj.tsv``
column with a role (``time`` / ``state`` / ``flow``); the columns carry no
per-column stratum breakdown, so a compartment *total* is obtained by summing
the columns that share a compartment base name (see :func:`compartment_bases`).

Pure readers over the tree — no dependency on the fit ingest, no cycle. Time is
dated via the fit-level calendar (:func:`camdl_watch.predictive.read_calendar`),
which is the same origin across a project's runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

SIMS_DIRNAME = "sims"


def sims_root(store: Path) -> Path:
    """The ``sims/`` tree — a sibling of the fit store (``results/fits`` →
    ``results/sims``); ``store/sims`` when the store isn't the ``fits`` leaf."""
    store = Path(store)
    return (store.parent if store.name == "fits" else store) / SIMS_DIRNAME


@dataclass(frozen=True)
class SimMember:
    """One sweep point of a sim: the (config, params, scenario, seed) leaf that
    carries a ``traj.tsv``. ``label`` is the human sweep-point tag (the swept
    param value(s) when they can be read, e.g. ``het=0.5``)."""

    label: str
    scenario: str
    seed: str
    traj_path: Path
    schema: dict  # output_schema for traj.tsv: {"role":..., "columns":[...]}
    params: dict[str, str] = field(default_factory=dict)  # from provenance argv


@dataclass(frozen=True)
class SimMeta:
    """A discoverable sim (the top ``sims/<model>-<hash>`` dir) and its sweep
    members. ``updated_at`` is the newest member trajectory mtime."""

    sim_id: str
    sim_dir: Path
    model: str
    status: str
    members: list[SimMember] = field(default_factory=list)
    updated_at: float = 0.0
    source_path: str = ""  # provenance model source, relative to the project root


def _leaf_run_jsons(sim_dir: Path) -> list[Path]:
    """Every ``run.json`` under a sim (one per leaf), sorted by path for a stable
    member order."""
    return sorted(sim_dir.rglob("run.json"))


def _member_levels(levels: list[dict]) -> tuple[str, str, str]:
    """(params, scenario, seed) labels from a run.json ``levels`` list — the CAS
    hierarchy [model, config, params, scenario, seed]."""
    by_name = {lv.get("name"): str(lv.get("label") or "") for lv in levels}
    return by_name.get("params", ""), by_name.get("scenario", ""), by_name.get("seed", "")


def _member_params(provenance: dict) -> dict[str, str]:
    """The ``--param k=v`` pairs from a sim's provenance ``argv`` — the actual
    parameter point simulated (what the sweep varies)."""
    argv = provenance.get("argv") or []
    out: dict[str, str] = {}
    for i, a in enumerate(argv):
        if a == "--param" and i + 1 < len(argv):
            k, _, v = str(argv[i + 1]).partition("=")
            if k:
                out[k] = v
    return out


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


def discover_sims(store: Path) -> list[SimMeta]:
    """Every simulation run under ``sims/``, newest first. Each top model dir is
    one sim; its leaves are the sweep members. Best-effort: a malformed leaf is
    skipped, never fatal."""
    root = sims_root(store)
    if not root.is_dir():
        return []
    sims: list[SimMeta] = []
    for sim_dir in sorted(root.iterdir()):
        if not sim_dir.is_dir():
            continue
        # Collect leaves, then label members by the swept param(s).
        raw: list[dict] = []
        newest = 0.0
        status = "unknown"
        source_path = ""
        for rj_path in _leaf_run_jsons(sim_dir):
            leaf = rj_path.parent
            traj = leaf / "traj.tsv"
            if not traj.is_file():
                continue
            try:
                rj = json.loads(rj_path.read_text())
            except (OSError, ValueError):
                continue
            prov = rj.get("provenance") or {}
            params, scenario, seed = _member_levels(rj.get("levels") or [])
            raw.append({
                "schema": (rj.get("output_schema") or {}).get("traj.tsv") or {},
                "scenario": scenario, "seed": seed, "traj": traj,
                "fallback": params or leaf.parent.parent.name,
                "params": _member_params(prov),
            })
            status = str(rj.get("status") or status)
            srcs = prov.get("source_paths") or []
            if srcs and not source_path:
                source_path = str(srcs[0])
            try:
                newest = max(newest, traj.stat().st_mtime)
            except OSError:
                pass
        if not raw:
            continue
        labels = _sweep_labels([r["params"] for r in raw], [r["fallback"] for r in raw])
        members = [
            SimMember(
                label=labels[i], scenario=r["scenario"], seed=r["seed"],
                traj_path=r["traj"], schema=r["schema"], params=r["params"],
            )
            for i, r in enumerate(raw)
        ]
        sims.append(
            SimMeta(
                sim_id=sim_dir.name, sim_dir=sim_dir,
                model=sim_dir.name.rsplit("-", 1)[0], status=status,
                members=members, updated_at=newest, source_path=source_path,
            )
        )
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


def read_sim_model(store: Path, sim: SimMeta) -> SimModel:
    """Parse the sim's source model (``provenance.source_paths``, resolved under
    the project root) for its origin/time_unit and compartment list. Read-only,
    self-contained (no ``camdl`` subprocess); returns an empty :class:`SimModel`
    when the source can't be resolved or read."""
    if not sim.source_path:
        return SimModel()
    root = store.parent.parent if store.name == "fits" else store.parent
    src = root / sim.source_path
    if not src.is_file():
        return SimModel()
    try:
        text = src.read_text()
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


def _infer_roles(columns: list[str]) -> dict[str, str]:
    """Column → role when a run carries no ``output_schema`` (older sims): the
    first column (``t``/``time``/``sweep``/``step``) is time, ``flow_*`` columns
    are flows, everything else is a state. Matches camdl's wide-trajectory naming."""
    roles: dict[str, str] = {}
    for i, c in enumerate(columns):
        if i == 0 or c in ("t", "time", "sweep", "step"):
            roles[c] = "time"
        elif c.startswith("flow_"):
            roles[c] = "flow"
        else:
            roles[c] = "state"
    return roles


def _read_header(path: Path) -> list[str]:
    try:
        return pl.read_csv(path, separator="\t", n_rows=0).columns
    except (OSError, pl.exceptions.PolarsError):
        return []


def resolve_roles(member: SimMember) -> dict[str, str]:
    """Column → role for a member's trajectory: from ``output_schema`` when the
    run declared one, else inferred from the ``traj.tsv`` header. Columns are the
    same across a sim's members (only params sweep), so callers resolve once."""
    cols = member.schema.get("columns")
    if cols:
        return {c["name"]: c.get("role", "state") for c in cols}
    return _infer_roles(_read_header(member.traj_path))


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
