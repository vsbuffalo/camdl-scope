"""Build camdl run stores for tests, in the layout camdl writes.

Every test fixture goes through :class:`StoreKit`, so the records the readers
see — ``store.json``, ``identity.json``, ``run.json``, the invocation records
and the owners' beat files — have one shape, the one camdl's run store writes
(``docs/dev/proposals/2026-11-20-run-store-api.md``, "On disk"). Folder names
follow camdl's pattern (``<label>-<12 hex>``); the hex is a counter, since the
readers never derive anything from it.

A ``running`` result is owned by this test process (alive) unless the test
asks for a dead owner, which is a process id that has exited, with its beat
file left behind as a killed camdl leaves it.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

BUILD = "0.1.0_test0000-000000000000"


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    return repr(v)


def method_toml(method: dict) -> str:
    """A ``method.toml`` holding ``[method]`` and any sub-tables."""
    flat = {k: v for k, v in method.items() if not isinstance(v, dict)}
    out = ["[method]", *(f"{k} = {_toml_value(v)}" for k, v in flat.items())]
    for k, v in method.items():
        if isinstance(v, dict):
            out += ["", f"[method.{k}]", *(f"{kk} = {_toml_value(vv)}" for kk, vv in v.items())]
    return "\n".join(out) + "\n"


def events_jsonl(events: list[dict]) -> str:
    """An ``events.jsonl`` as the store writes it: each of camdl's event
    records wrapped in a numbered line."""
    return "".join(
        json.dumps({"format": 1, "n": i + 1, "at": "2026-11-24T00:00:00.000Z", "event": e}) + "\n"
        for i, e in enumerate(events)
    )


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@dataclass
class StoreKit:
    root: Path
    build: str = BUILD
    _hex: int = 0
    _seq: int = 0
    _owners: dict[bool, dict] = field(default_factory=dict)

    @classmethod
    def create(cls, root: Path) -> "StoreKit":
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "store.json", {"format": 1, "top_level": "by_build"})
        for d in (".invocations", ".index", ".owners", ".staging", ".quarantine"):
            (root / d).mkdir(exist_ok=True)
        return cls(root)

    # -- ids and owners -------------------------------------------------------

    def _id(self) -> str:
        self._hex += 1
        return f"{self._hex:064x}"

    def owner(self, *, alive: bool = True) -> dict:
        """The owner of running results: this process, or one that has exited.
        Either has a beat file; only a clean end removes it."""
        if alive not in self._owners:
            pid = os.getpid() if alive else _dead_pid()
            o = {"host": socket.gethostname(), "pid": pid, "started": f"2026-11-24T00:00:0{int(alive)}.000Z"}
            _write_json(self.root / ".owners" / f"{o['host']}-{pid}.json", {"format": 1, **o, "beat": 1})
            self._owners[alive] = o
        return self._owners[alive]

    # -- records --------------------------------------------------------------

    def _result(
        self, folder: Path, kind: str, body: dict, *, status: str = "completed",
        alive: bool = True, invocation: int | None = None, given: dict | None = None,
        parent: tuple[str, str] | None = None, failure: str | None = None,
    ) -> str:
        run_id = self._id()
        folder.mkdir(parents=True, exist_ok=True)
        _write_json(folder / "identity.json", {
            "format": 1, "kind": kind, "run_id": run_id, "hash_version": 9,
            "ambient": {"build": "0" * 32, "env": {}}, "identity": body,
            "parent": {"relation": "inside", "kind": parent[0], "run_id": parent[1]} if parent else None,
        })
        _write_json(folder / "run.json", {
            "format": 1, "status": status,
            "owner": self.owner(alive=alive), "invocation": invocation,
            "started": "2026-11-24T00:00:01.000Z",
            "finished": None if status == "running" else "2026-11-24T00:00:02.000Z",
            "given": given or {}, "execution": {}, "files": {}, "deps": [], "lineage": [],
            "reads": [], "recorded": {},
            "failure": {"operation": "run", "reason": failure} if status == "failed" else None,
        })
        return run_id

    def invocation(
        self, argv: list[str], results: list[dict], *, label: str | None = None,
        status: str = "ended", given: dict | None = None, alive: bool = True,
    ) -> int:
        """An invocation record; ``results`` are ``{kind, run_id, labels,
        outcome}`` entries (outcome defaults to ``computed``)."""
        self._seq += 1
        _write_json(self.root / ".invocations" / f"{self._seq}.json", {
            "format": 1, "seq": self._seq, "record": "invocation", "status": status,
            "owner": self.owner(alive=alive), "started": "2026-11-24T00:00:00.000Z",
            "ended": None if status == "running" else "2026-11-24T00:00:03.000Z",
            "label": label, "argv": ["camdl", *argv], "given": given or {}, "config_files": [],
            "results": [{"outcome": "computed", "labels": {}, "failure": None, **r} for r in results],
            "counts": {"completed": len(results), "failed": 0}, "progress": None,
        })
        return self._seq

    def next_seq(self) -> int:
        return self._seq + 1

    # -- kinds ----------------------------------------------------------------

    def compiled_model(self, name: str = "sir", *, source: str = "", docs: dict | None = None) -> str:
        folder = self.root / self.build / "compiled_model" / f"{name}-{self._hex + 1:012x}"
        rid = self._result(folder, "compiled_model", {"name": name})
        (folder / "source.camdl").write_text(source)
        (folder / "model.ir.json").write_text("{}")
        _write_json(folder / "docs.json", docs or {})
        return rid

    def problem(
        self, model: str = "sir", *, source: str = "", docs: dict | None = None,
        name: str | None = None,
    ) -> Path:
        """A FitProblem holding ``model.camdl``, and its compiled model. ``name``
        fixes the folder's name (default ``<model>-<12 hex>``)."""
        model_id = self.compiled_model(model, source=source, docs=docs)
        folder = self.root / self.build / "fit" / (name or f"{model}-{self._hex + 1:012x}")
        self._result(folder, "fit", {"model": {"kind": "compiled_model", "run_id": model_id}})
        (folder / "model.camdl").write_text(source)
        (folder / "problem.toml").write_text('[model]\ncamdl = "model.camdl"\n')
        return folder

    def fit_run(
        self, problem: Path, *, algorithm: str = "mh", seed: int = 1,
        status: str = "completed", alive: bool = True, label: str | None = None,
        method: dict | None = None, estimated: list[str] | None = None,
        meta: dict | None = None, fit_toml: str = "", config: str = "fit.toml",
        method_dir: str | None = None, seed_dir: str | None = None,
        failure: str | None = None,
    ) -> Path:
        """A FitRun inside ``problem``, with its records, ``method.toml``,
        ``fit.toml`` and (unless ``meta`` is ``None`` while running)
        ``fit.meta.json``, and the invocation that made it."""
        method = {"algorithm": algorithm, "backend": "ode", "chains": 4, **(method or {})}
        mdir = problem / "runs" / (method_dir or f"{algorithm}-{self._hex + 1:012x}")
        leaf = mdir / (seed_dir or f"seed_{seed}-{self._hex + 2:012x}")
        problem_id = json.loads((problem / "identity.json").read_text())["run_id"]
        seq = self.next_seq()
        rid = self._result(
            leaf, "fit_run",
            {"chain": {"method": {"algorithm": algorithm, "settings": json.dumps(method)},
                       "seed": {"seed": seed}, "problem": {"kind": "fit", "run_id": problem_id}},
             "length": 0, "resumed_from": None},
            status=status, alive=alive, invocation=seq,
            given={"args": {"config": config}, "config": {"path": config, "sha256": "0" * 64, "keys": {}}},
            parent=("fit", problem_id), failure=failure,
        )
        (leaf / "method.toml").write_text(method_toml(method))
        (leaf / "fit.toml").write_text(fit_toml)
        if status != "running" or meta is not None:
            _write_json(leaf / "fit.meta.json", {
                "schema_version": 3, "estimated": estimated if estimated is not None else ["beta", "gamma"],
                **(meta or {}),
            })
        model = problem.name.rsplit("-", 1)[0]
        self.invocation(
            ["fit", "run", config, "--seed", str(seed)],
            [{"kind": "fit", "run_id": problem_id, "labels": {"model": model}, "outcome": "reused"},
             {"kind": "fit_run", "run_id": rid, "labels": {"method": algorithm, "seed": seed},
              "outcome": "planned" if status == "running" else ("failed" if status == "failed" else "computed")}],
            label=label, status="running" if status == "running" else "ended", alive=alive,
        )
        return leaf

    def beat(self, leaf: Path, *, state: str = "running", step: int = 100, total: int = 1000,
             phase: str = "sampling", age: float = 0.0) -> None:
        """A run's heartbeat, as camdl wraps it in ``progress.json``."""
        body: object = (
            {"running": {"phase": phase, "step": step, "total": total}} if state == "running" else state
        )
        _write_json(leaf / "progress.json", {
            "format": 1, "at": "2026-11-24T00:00:05.000Z",
            "snapshot": {"updated_at": int(time.time() - age), "pid": os.getpid(), "state": body},
        })

    def finish(self, leaf: Path) -> None:
        """Mark a running result completed, as camdl seals it at the end."""
        rec = json.loads((leaf / "run.json").read_text())
        rec["status"] = "completed"
        rec["finished"] = "2026-11-24T00:00:09.000Z"
        _write_json(leaf / "run.json", rec)

    def prediction(
        self, leaf: Path, mode: str = "free_forward", *, options: dict | None = None,
        name: str | None = None,
    ) -> Path:
        """A completed prediction inside fit run ``leaf``; the caller writes its
        tables (``predictive/``, ``observed/``, ``quantities-model/``)."""
        run_id = json.loads((leaf / "identity.json").read_text())["run_id"]
        folder = leaf / "predict" / (name or f"{mode}-{self._hex + 1:012x}")
        self._result(folder, "prediction", {"options": {"mode": mode}}, parent=("fit_run", run_id))
        _write_json(folder / "predictive.json", {"schema": "camdl.predictive/v4",
                                                 "options": {"mode": mode, **(options or {})}})
        return folder

    def profile(
        self, sweep: list[dict], points: list[dict], *, n_starts: int = 1,
        status: str = "ended", alive: bool = True, label: str | None = None,
        model: str = "sir", algorithm: str = "nl-sbplx", seeds: list[int] | None = None,
        progress: dict | None = None,
    ) -> int:
        """A ``camdl profile`` command and its finished points.

        ``sweep`` is ``given.args.sweep`` (``[{"name", "grid": {...}}]``);
        each of ``points`` is ``{"focal": {...}, "loglik": x, "mle": {...}}``,
        written as one start's folder. A running command's record lists none
        of them, as camdl's does."""
        seq = self.next_seq()
        results = []
        for i, pt in enumerate(points):
            focal = pt["focal"]
            cell = "__".join(f"{k}_{v:.4f}" for k, v in focal.items())
            folder = (self.root / self.build / "profile_point" / f"{model}-{'a' * 12}"
                      / f"{cell}-{self._hex + 1:012x}" / f"{algorithm}-{'b' * 12}"
                      / f"seed_{pt.get('seed', 1)}-start_{i}-{self._hex + 2:012x}")
            rid = self._result(folder, "profile_point", {"focal": focal}, invocation=seq)
            body = [f"final_loglik = {pt['loglik']!r}", "", "[focal]",
                    *(f"{k} = {v!r}" for k, v in focal.items()), "", "[mle]",
                    *(f"{k} = {v!r}" for k, v in (pt.get("mle") or {}).items())]
            (folder / "mle.toml").write_text("\n".join(body) + "\n")
            _write_json(folder / "provenance.json", {
                "best_loglik": pt["loglik"], "method": algorithm, "loglik_type": "ode_marginal"})
            results.append({"kind": "profile_point", "run_id": rid, "labels": {"start": i}})
        self.invocation(
            ["profile", f"{model}.camdl"], [] if status == "running" else results,
            label=label, status=status, alive=alive,
            given={"args": {"model": f"{model}.camdl", "sweep": sweep, "n_starts": n_starts,
                            "seeds": seeds, "algorithm": algorithm}},
        )
        if progress is not None:
            _write_json(self.root / ".invocations" / f"{seq}.progress.json",
                        {"format": 1, "at": progress.pop("at", "2026-11-24T00:00:05.000Z"),
                         "snapshot": progress})
        return seq

    def sim(
        self, cells: list[dict], *, model: str = "sir", source: str = "",
        status: str = "ended", alive: bool = True,
    ) -> int:
        """A ``camdl simulate`` command; each of ``cells`` is ``{"params": {...},
        "traj": "<tsv text>", "scenario": [...], "seed": n}``."""
        model_id = self.compiled_model(model, source=source)
        seq = self.next_seq()
        results = []
        for c in cells:
            folder = (self.root / self.build / "sim" / f"{model}-{'c' * 12}" / f"ode-{'d' * 12}"
                      / f"p{self._hex + 1}-{'e' * 12}" / f"_none_-{'f' * 12}"
                      / f"seed_{c.get('seed', 1)}-{self._hex + 1:012x}")
            rid = self._result(folder, "sim", {
                "model": {"kind": "compiled_model", "run_id": model_id},
                "params": {"values": c["params"]}, "scenario": {"names": c.get("scenario", [])},
                "seed": {"seed": c.get("seed", 1)},
            }, invocation=seq)
            (folder / "traj.tsv").write_text(c["traj"])
            results.append({"kind": "sim", "run_id": rid})
        self.invocation(["simulate", f"{model}.camdl"], [] if status == "running" else results,
                        status=status, alive=alive, given={"args": {"model": f"{model}.camdl"}})
        return seq
