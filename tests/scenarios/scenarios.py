"""A scenario store made by running camdl, for checking how the dashboard shows
the common cases.

The hand-written golden store (``tests/fixtures``) encodes our reading of
camdl's output, so it cannot catch camdl drifting from that reading. This store
is written by the real ``camdl`` binary from the committed inputs in
``inputs/`` (a closed SIR model, a negative-binomial variant, and 21 points of
case data simulated from the SIR model at ``inputs/truth.toml`` with
``camdl simulate sir.camdl --params truth.toml --backend ode --dt 1 --seed 1
--obs-only``), so regenerating it with a newer camdl shows what changed.

Three subcommands, each a make target:

* ``generate`` (``make scenarios``): run camdl into ``build/scenarios/`` and
  derive the mid-run states (below); writes ``manifest.json``.
* ``snapshot`` (``make scenario-snapshots``): the JSON every endpoint returns
  for every view, with run-to-run noise removed, into ``snapshots/``
  (committed). ``--check`` compares instead of writing; ``tests/test_scenarios.py``
  runs it when the store exists.
* ``gallery`` (``make gallery``): screenshot every view in light and dark with
  headless Chrome, into ``build/scenarios/gallery/index.html``.

Mid-run states are derived, not captured: a live run cannot be frozen
reliably. A finished run is truncated (finished starts or grid values removed,
the end-of-run files deleted) and given a ``running`` heartbeat in camdl's
shape. The file shapes are camdl's; the truncation is ours, and the manifest
marks those scenarios ``derived``. A running heartbeat is re-stamped to the
current time (:func:`touch_live`) before anything reads it, since a heartbeat
older than 30 s reads as stalled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SCENARIOS = Path(__file__).resolve().parent
INPUTS = SCENARIOS / "inputs"
SNAPSHOTS = SCENARIOS / "snapshots"
REPO = SCENARIOS.parents[1]
DEFAULT_OUT = REPO / "build" / "scenarios"
CAMDL = os.environ.get("CAMDL_BIN", "camdl")
#: An old heartbeat time: a stalled run's last sign of life.
STALE_AT = 1_000_000_000


# ---------------------------------------------------------------------------
# What to run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fit:
    config: str  # file stem in inputs/; also the run folder's label
    kind: str  # "optimizer" | "mcmc" | "score"
    note: str
    #: Derived mid-run state: (finished starts kept, heartbeat "fresh"|"stale").
    truncate: tuple[int, str] | None = None
    #: A derived fit copies a finished config; its own seed makes it a distinct
    #: run (camdl reuses a stored result for an identical config and seed).
    seed: int = 1


@dataclass(frozen=True)
class Profile:
    name: str
    config: str
    sweeps: tuple[tuple[str, str], ...]
    n_starts: int
    note: str
    #: Derived mid-run state: (grid values dropped, heartbeat "fresh"|"stale").
    truncate: tuple[int, str] | None = None


FITS: tuple[Fit, ...] = (
    Fit("opt_agree", "optimizer", "every start converges to one optimum"),
    Fit("opt_bobyqa", "optimizer", "the same problem under nl-bobyqa"),
    Fit("opt_ridge", "optimizer", "starts agree in log-likelihood but spread along the beta/N0 ridge"),
    Fit("opt_capped", "optimizer", "every start stops at a 120-evaluation budget, scattered"),
    Fit("opt_running", "optimizer", "derived: 2 of 6 starts finished, live heartbeat", (2, "fresh"), seed=2),
    Fit("opt_stalled", "optimizer", "derived: 3 of 6 starts finished, heartbeat stopped", (3, "stale"), seed=3),
    Fit("mh_converged", "mcmc", "four MH chains, camdl's verdict converged"),
    Fit("mh_unconverged", "mcmc", "four short MH chains, camdl's verdict not converged"),
    Fit("pf_pois", "score", "particle-filter score of the Poisson model"),
    Fit("pf_nb", "score", "particle-filter score of the negative-binomial model"),
)

PROFILES: tuple[Profile, ...] = (
    Profile("gamma_1d", "prof", (("gamma", "lin(0.2,0.45,6)"),), 3, "finished 1D profile"),
    Profile("gamma_beta_2d", "prof3", (("gamma", "lin(0.26,0.39,5)"), ("beta", "lin(0.84,1.0,5)")), 2,
            "finished 2D profile, N0 profiled out"),
    Profile("n0_1d", "prof3", (("N0", "lin(6000,18000,5)"),), 2,
            "finished 1D profile of N0: flat, because beta absorbs it"),
    Profile("beta_1d_running", "prof", (("beta", "lin(0.8,1.05,6)"),), 3,
            "derived: 2 grid values not started, one partial, live heartbeat", (2, "fresh")),
    Profile("gamma_1d_stalled", "prof", (("gamma", "lin(0.24,0.40,5)"),), 3,
            "derived: 2 grid values not started, one partial, heartbeat stopped", (2, "stale")),
    Profile("gamma_beta_2d_running", "prof3", (("gamma", "lin(0.27,0.36,4)"), ("beta", "lin(0.86,0.98,4)")), 2,
            "derived: 2D profile with 2 grid cells not started, one partial, live heartbeat", (2, "fresh")),
)

COMPARE = ("pf_pois", "pf_nb")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def _camdl(args: list[str], cwd: Path) -> None:
    env = {**os.environ, "CAMDL_SKIP_VERSION_CHECK": "1", "CAMDL_IR_CACHE_DIR": str(cwd / ".ir")}
    proc = subprocess.run([CAMDL, *args], cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"camdl {' '.join(args)} failed:\n{proc.stderr[-3000:]}")


def _heartbeat(path: Path, state: object, fresh: bool) -> None:
    path.write_text(json.dumps(
        {"updated_at": int(time.time()) if fresh else STALE_AT, "pid": 0, "state": state}, indent=2
    ) + "\n")


def _truncate_fit(run_dir: Path, keep: int, beat: str) -> Path:
    """Cut a finished optimizer leaf back to ``keep`` finished starts, as it
    looked mid-run: the start table, run.json, the event log without its
    end-of-run report, and a running heartbeat."""
    [leaf] = run_dir.glob("*/seed_*")
    rows = (leaf / "chain_results.tsv").read_text().splitlines()
    (leaf / "chain_results.tsv").write_text("\n".join(rows[: 1 + keep]) + "\n")
    done = {r.split("\t")[0] for r in rows[1 : 1 + keep]}
    trace = leaf / "opt_trace.tsv"
    if trace.is_file():
        # A finished search keeps its whole trace; an unfinished one, the first
        # 40% of its rows (it was cut off mid-climb).
        head, *body = trace.read_text().splitlines()
        by: dict[str, list[str]] = {}
        for ln in body:
            by.setdefault(ln.split("\t")[0], []).append(ln)
        kept = [ln for c, lns in by.items() for ln in (lns if c in done else lns[: max(1, len(lns) * 2 // 5)])]
        trace.write_text("\n".join([head, *kept]) + "\n")
    events = [ln for ln in (leaf / "events.jsonl").read_text().splitlines() if "method_report" not in ln]
    (leaf / "events.jsonl").write_text("\n".join(events) + ("\n" if events else ""))
    for f in leaf.iterdir():
        if f.name not in {"run.json", "events.jsonl", "chain_results.tsv", "opt_trace.tsv"}:
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    state = {"running": {"phase": "optimizing", "step": 1400, "total": 5000}}
    _heartbeat(leaf / "progress.json", state, beat == "fresh")
    return leaf / "progress.json"


def _truncate_profile(base: Path, drop: int, beat: str) -> Path:
    """Cut a finished profile back to a mid-run state: the last ``drop`` grid
    values have no finished start, the first keeps one start, the rest are
    complete; the invocation's heartbeat says running."""
    [inv] = base.glob("invocations/*/invocation.json")
    record = json.loads(inv.read_text())
    dirs = [p["dir"] for p in record["points"]]
    for d in dirs[len(dirs) - drop :]:
        shutil.rmtree(base / d, ignore_errors=True)
    starts = sorted((base / dirs[0]).glob("*/*/*"))
    for s in starts[1:]:
        shutil.rmtree(s)
    left = sum(1 for _ in base.glob("*/*/*/*/run.json"))
    state = {"running": {"phase": "profiling", "step": left, "total": record["total_jobs"]}}
    _heartbeat(inv.parent / "progress.json", state, beat == "fresh")
    return inv.parent / "progress.json"


def generate(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in INPUTS.iterdir():
        if f.is_file() and not f.name.startswith("."):
            shutil.copy2(f, out / f.name)
    fits_root = out / "results" / "fits"
    prof_root = out / "results" / "profiles"
    live: list[str] = []
    fits_m: dict[str, dict[str, object]] = {}
    profs_m: dict[str, dict[str, object]] = {}

    for fit in FITS:
        print(f"fit {fit.config}", flush=True)
        _camdl(["fit", "run", f"{fit.config}.toml", "--seed", str(fit.seed), "--parallel", "4", "--progress", "none"], out)
        [run_dir] = fits_root.glob(f"{fit.config}-*")
        entry = {"run_id": run_dir.name, "kind": fit.kind, "note": fit.note, "derived": fit.truncate is not None}
        if fit.truncate:
            hb = _truncate_fit(run_dir, *fit.truncate)
            if fit.truncate[1] == "fresh":
                live.append(str(hb.relative_to(out)))
        fits_m[fit.config] = entry

    for prof in PROFILES:
        print(f"profile {prof.name}", flush=True)
        before = set(prof_root.glob("*")) if prof_root.exists() else set()
        sweeps = [a for name, grid in prof.sweeps for a in ("--sweep", f"{name}={grid}")]
        _camdl(["profile", "sir.camdl", "--fit", f"{prof.config}.toml", *sweeps, "--algorithm", "nl-sbplx",
                "--backend", "ode", "--n-starts", str(prof.n_starts), "--parallel", "4", "--no-progress"], out)
        [base] = set(prof_root.glob("*")) - before
        if prof.truncate:
            hb = _truncate_profile(base, *prof.truncate)
            if prof.truncate[1] == "fresh":
                live.append(str(hb.relative_to(out)))
        profs_m[prof.name] = {
            "base_id": base.name, "note": prof.note, "derived": prof.truncate is not None}

    manifest = {"camdl": _version(), "fits": fits_m, "profiles": profs_m, "live_heartbeats": live}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")


def _version() -> str:
    proc = subprocess.run([CAMDL, "--version"], capture_output=True, text=True,
                          env={**os.environ, "CAMDL_SKIP_VERSION_CHECK": "1"})
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "unknown"


def load_manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text())


def touch_live(out: Path) -> None:
    """Re-stamp every derived running heartbeat to now, so it reads as live."""
    for rel in load_manifest(out)["live_heartbeats"]:
        p = out / rel
        hb = json.loads(p.read_text())
        hb["updated_at"] = int(time.time())
        p.write_text(json.dumps(hb, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Views: what the gallery shows and the snapshots record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class View:
    name: str
    query: str  # the SPA's URL query
    api: tuple[str, ...] = field(default=())  # endpoints recorded for it


def views(out: Path) -> list[View]:
    m = load_manifest(out)
    vs: list[View] = []
    for cfg, f in m["fits"].items():
        rid = f["run_id"]
        if f["kind"] == "optimizer":
            vs.append(View(f"{cfg}-estimate", f"run={rid}&tab=estimate", (f"/api/runs/{rid}/mle",)))
            vs.append(View(f"{cfg}-starts", f"run={rid}&tab=starts"))
        elif f["kind"] == "mcmc":
            vs.append(View(f"{cfg}-posterior", f"run={rid}&tab=posterior", (f"/api/runs/{rid}/posterior",)))
            vs.append(View(f"{cfg}-diagnostics", f"run={rid}&tab=diagnostics", (f"/api/runs/{rid}/diagnostics",)))
    for name, p in m["profiles"].items():
        vs.append(View(f"profile-{name}", f"profile={p['base_id']}", (f"/api/profiles/{p['base_id']}",)))
    ids = [m["fits"][c]["run_id"] for c in COMPARE]
    vs.append(View("compare", "compare=" + ",".join(ids),
                   ("/api/compare/candidates", "/api/compare?" + "&".join(f"runs={i}" for i in ids))))
    vs.append(View("run-list", "", ("/api/runs", "/api/profiles")))
    return vs


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

#: Keys whose values change between runs of the same camdl on the same inputs
#: (clocks, process ids, timings): replaced with a placeholder.
_VOLATILE = re.compile(
    r"(mtime|_at$|^at$|elapsed|wall|seconds|duration|^pid$|per_s|per_sec|timestamp|started|age)", re.I
)


def _scrub(x: object, store: str) -> object:
    if isinstance(x, dict):
        return {k: ("<volatile>" if _VOLATILE.search(k) and v is not None else _scrub(v, store))
                for k, v in sorted(x.items())}
    if isinstance(x, list):
        return [_scrub(v, store) for v in x]
    if isinstance(x, str):
        return x.replace(store, "<store>")
    if isinstance(x, float):
        # Six significant digits: below that, threads finishing in a different
        # order move the last bits of a sum.
        return float(f"{x:.6g}") if math.isfinite(x) else str(x)
    return x


def _record(out: Path) -> dict[str, object]:
    os.environ["CAMDL_SCOPE_STORE"] = str(out / "results" / "fits")
    from fastapi.testclient import TestClient

    from camdl_scope.api.app import app

    touch_live(out)
    client = TestClient(app)
    got: dict[str, object] = {}
    for v in views(out):
        for ep in v.api:
            r = client.get(ep)
            got[ep] = {"status": r.status_code, "body": _scrub(r.json(), str(out))}
    return got


def _snapshot_name(ep: str, aliases: dict[str, str]) -> str:
    """A file name for an endpoint's snapshot that names scenarios, not hashes,
    so a change in camdl's hashing does not rename every file."""
    for ident, alias in aliases.items():
        ep = ep.replace(ident, alias)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ep.strip("/")) + ".json"


def _aliases(out: Path) -> dict[str, str]:
    m = load_manifest(out)
    return {f["run_id"]: c for c, f in m["fits"].items()} | {
        p["base_id"]: n for n, p in m["profiles"].items()}


def snapshot(out: Path, check: bool) -> int:
    got = _record(out)
    aliases = _aliases(out)
    bad: list[str] = []
    SNAPSHOTS.mkdir(exist_ok=True)
    if not check:
        for old in SNAPSHOTS.glob("*.json"):
            old.unlink()
    for ep, body in got.items():
        p = SNAPSHOTS / _snapshot_name(ep, aliases)
        text = json.dumps(body, indent=1, ensure_ascii=False) + "\n"
        if check:
            if not p.exists() or p.read_text() != text:
                bad.append(ep)
        else:
            p.write_text(text)
    if check and bad:
        print("API output differs from the committed snapshot for:\n  " + "\n  ".join(bad))
        print("If camdl's change is expected, run `make scenario-snapshots` and review the diff.")
        return 1
    print(f"{'checked' if check else 'wrote'} {len(got)} snapshots")
    return 0


# ---------------------------------------------------------------------------
# gallery
# ---------------------------------------------------------------------------

CHROME = os.environ.get("CHROME_BIN", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def gallery(out: Path) -> None:
    dist = REPO / "web" / "dist" / "index.html"
    if not dist.exists():
        sys.exit("web/dist is missing: run `make build` first")
    port = _free_port()
    env = {**os.environ, "CAMDL_SCOPE_STORE": str(out / "results" / "fits")}
    server = subprocess.Popen([sys.executable, "-m", "camdl_scope", "--port", str(port), "--host", "127.0.0.1"],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shots = out / "gallery"
    shutil.rmtree(shots, ignore_errors=True)
    shots.mkdir(parents=True)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.2)
        touch_live(out)
        vs = views(out)
        for v in vs:
            for theme, scheme in (("light", 1), ("dark", 0)):
                png = shots / f"{v.name}-{theme}.png"
                subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                                f"--user-data-dir={shots / '.chrome' / theme}", "--window-size=1200,1500",
                                "--virtual-time-budget=8000", f"--blink-settings=preferredColorScheme={scheme}",
                                f"--screenshot={png}", f"http://127.0.0.1:{port}/?{v.query}"],
                               capture_output=True, check=True)
            print(f"shot {v.name}", flush=True)
    finally:
        server.terminate()
    shutil.rmtree(shots / ".chrome", ignore_errors=True)
    m = load_manifest(out)
    notes = {f"{c}-": f["note"] for c, f in m["fits"].items()} | {
        f"profile-{n}": p["note"] for n, p in m["profiles"].items()}
    rows = []
    for v in vs:
        note = next((t for k, t in notes.items() if v.name.startswith(k)), "")
        rows.append(f"<h2>{v.name}</h2><p>{note}</p><div class=row>"
                    f"<img src='{v.name}-light.png'><img src='{v.name}-dark.png'></div>")
    (shots / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>camdl-scope scenarios</title>"
        "<style>body{font:14px system-ui;margin:24px;background:#888}"
        ".row{display:flex;gap:12px}img{width:49%;border:1px solid #444}h2{margin:28px 0 4px}</style>"
        f"<h1>camdl-scope scenarios</h1><p>{m['camdl']}</p>" + "".join(rows))
    print(f"wrote {shots / 'index.html'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["generate", "snapshot", "gallery"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="snapshot: compare instead of writing")
    a = ap.parse_args()
    if a.command == "generate":
        generate(a.out)
    elif a.command == "snapshot":
        return snapshot(a.out, a.check)
    else:
        gallery(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
