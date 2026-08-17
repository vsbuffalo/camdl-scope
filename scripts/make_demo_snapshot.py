"""Bake a static snapshot of the camdl-scope API for a public demo.

The viewer is a live client-server app (FastAPI reads the CAS and computes R̂/ESS,
quantiles, etc. on the fly). A static web host can't run that backend, so this
freezes the API responses for a *curated* set of runs into a tree of JSON files
that the SPA (built with ``VITE_DEMO=1``) reads directly — the CAS never leaves
this machine.

Reproducible: drives the real app via Starlette's TestClient (no running server),
so re-running against the same store writes byte-stable files. Query-parameterised
endpoints (warm-up %, chain subset, sim window) are captured at their DEFAULTS
only — the demo build disables those live controls.

Usage::

    CAMDL_WATCH_STORE=/path/to/results/fits \
        uv run python -m scripts.make_demo_snapshot --out demo-snapshot \
        --posterior 4 --mle 1 --sims 3

Writes ``<out>/snap/api/...json`` mirroring the request paths (query strings
dropped), plus ``<out>/manifest.json`` listing what was captured.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from starlette.testclient import TestClient


# A single static file this big would stall a web demo. Heavily-stratified
# quantity-series / predictive ribbons (a band point per scenario×time×stratum)
# blow past it — those are skipped and recorded, never silently truncated.
MAX_FILE_BYTES = 3_000_000


def _serialize(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _write(out: Path, api_path: str, payload) -> None:
    """Write ``payload`` at ``<out>/snap/<api_path>.json`` (api_path has no query)."""
    dst = out / "snap" / (api_path.lstrip("/") + ".json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_serialize(payload))


def _write_capped(out: Path, api_path: str, payload, dropped: list[dict]) -> bool:
    """Write only if under the size cap; else record the drop. Returns written?."""
    body = _serialize(payload)
    if len(body.encode("utf-8")) > MAX_FILE_BYTES:
        dropped.append({"path": api_path, "bytes": len(body.encode("utf-8"))})
        return False
    dst = out / "snap" / (api_path.lstrip("/") + ".json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(body)
    return True


def _get(client: TestClient, path: str):
    r = client.get(path)
    return r.json() if r.status_code == 200 else None


def _camdl_render(run_id: str, fmt: str) -> dict | None:
    """Generate the equations (``--format json``) or flow-graph (``--format
    graph``) JSON for a run from its archived model source. Both are pure
    functions of the model (no sampling) — cheap — so runs that predate the
    archived artifact still get an equations/diagram view in the demo. ``None``
    if the model source or the ``camdl`` binary is unavailable."""
    if shutil.which("camdl") is None:
        return None
    model = Path(os.environ["CAMDL_WATCH_STORE"]) / run_id / "model.camdl.original"
    if not model.is_file():
        return None
    try:
        proc = subprocess.run(
            ["camdl", "render", str(model), "--format", fmt],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        obj = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _capture_run(client: TestClient, out: Path, run_id: str, dropped: list[dict]) -> dict:
    """Snapshot every default-state endpoint for one run; return a capture note."""
    rid = run_id
    detail = _get(client, f"/api/runs/{rid}")
    if detail is None:
        return {"run_id": rid, "ok": False}
    model_name = detail.get("model")
    kind = detail.get("fit_kind", "posterior")
    captured = ["detail"]

    if kind == "mle":
        mle = _get(client, f"/api/runs/{rid}/mle")
        if mle is not None:
            _write(out, f"api/runs/{rid}/mle", mle)
            captured.append("mle")
    else:
        for ep in ("posterior", "draws", "traces", "diagnostics"):
            payload = _get(client, f"/api/runs/{rid}/{ep}")
            if payload is not None:
                _write(out, f"api/runs/{rid}/{ep}", payload)
                captured.append(ep)

    src = _get(client, f"/api/runs/{rid}/source")
    if src is not None:
        _write(out, f"api/runs/{rid}/source", src)
        captured.append("source")

    # Equations (model-render) + flow graph (model-graph): use the archived
    # artifact when present, else GENERATE it from the model source (cheap), and
    # flip the corresponding `has_*` flag so the demo's tab/toggle lights up.
    for ep, flag, fmt in (
        ("model-render", "has_model_render", "json"),
        ("model-graph", "has_model_graph", "graph"),
    ):
        payload = _get(client, f"/api/runs/{rid}/{ep}")
        generated = False
        if payload is None:
            payload = _camdl_render(rid, fmt)
            generated = payload is not None
            if generated and model_name:
                payload["model"] = model_name  # render's title is the filename
        if payload is not None:
            _write(out, f"api/runs/{rid}/{ep}", payload)
            detail[flag] = True
            captured.append(ep + ("*" if generated else ""))

    # Predictive: one file per available stream (path param, snapshot each; capped).
    for stream in detail.get("available_streams", []) or []:
        payload = _get(client, f"/api/runs/{rid}/predictive/{stream}")
        if payload is not None and _write_capped(out, f"api/runs/{rid}/predictive/{stream}", payload, dropped):
            captured.append(f"predictive/{stream}")

    # Quantities: the scalars table (always small) + each series quantity (capped —
    # a village×age×time ribbon can be enormous).
    scalars = _get(client, f"/api/runs/{rid}/quantity-scalars")
    if scalars is not None:
        _write(out, f"api/runs/{rid}/quantity-scalars", scalars)
        captured.append("quantity-scalars")
    for q in detail.get("available_quantities", []) or []:
        if q.get("shape") == "series":
            payload = _get(client, f"/api/runs/{rid}/quantity-series/{q['name']}")
            if payload is not None and _write_capped(
                out, f"api/runs/{rid}/quantity-series/{q['name']}", payload, dropped
            ):
                captured.append(f"quantity-series/{q['name']}")

    # Detail is written LAST so the generated-artifact flag patches are included.
    _write(out, f"api/runs/{rid}", detail)
    return {"run_id": rid, "ok": True, "kind": kind, "captured": captured}


def _stem(run_id: str) -> str:
    """Model stem = run-dir name without the trailing content hash."""
    return run_id.rsplit("-", 1)[0] if "-" in run_id else run_id


def _curate(
    client: TestClient, n_posterior: int, n_mle: int, include: list[str]
) -> list[str]:
    """Pick a representative, model-DIVERSE run set. ``include`` run ids are pinned
    first (validated); the rest fill from posterior fits that carry the flow
    diagram and a predictive stream, one per distinct model stem (richest by
    quantity count), then a few MLE fits."""
    runs = _get(client, "/api/runs") or []
    valid = {r["run_id"] for r in runs}
    pinned = [rid for rid in include if rid in valid]
    missing = [rid for rid in include if rid not in valid]
    if missing:
        print(f"WARNING: --include run(s) not found, skipped: {missing}")
    seen_stem = {_stem(rid) for rid in pinned}

    posterior, mle = [], []
    for r in runs:
        rid = r["run_id"]
        if rid in pinned:
            continue
        if r.get("fit_kind") == "mle":
            mle.append(rid)
            continue
        d = _get(client, f"/api/runs/{rid}") or {}
        if d.get("has_model_graph") and (d.get("available_streams") or []):
            posterior.append((rid, len(d.get("available_quantities") or [])))
    posterior.sort(key=lambda t: t[1], reverse=True)  # richest first
    picked_post: list[str] = list(pinned)
    for rid, _q in posterior:
        if len(picked_post) >= n_posterior:
            break
        s = _stem(rid)
        if s in seen_stem:  # one run per distinct model stem, for variety
            continue
        seen_stem.add(s)
        picked_post.append(rid)
    return picked_post + mle[:n_mle]


def _curate_all_models(client: TestClient, include: list[str]) -> list[str]:
    """One representative fit per DISTINCT model stem — the widest demo. For each
    stem, keep the best-populated fit (predictive + quantities + equations +
    diagram), pinned includes first."""
    runs = _get(client, "/api/runs") or []
    valid = {r["run_id"] for r in runs}
    pinned = [rid for rid in include if rid in valid]
    seen = {_stem(rid) for rid in pinned}
    best: dict[str, tuple[float, str]] = {}
    for r in runs:
        rid = r["run_id"]
        s = _stem(rid)
        if s in seen:
            continue
        d = _get(client, f"/api/runs/{rid}") or {}
        score = (
            3 * bool(d.get("available_streams"))
            + len(d.get("available_quantities") or [])
            + bool(d.get("has_model_render"))
            + bool(d.get("has_model_graph"))
        )
        if s not in best or score > best[s][0]:
            best[s] = (score, rid)
    return pinned + [rid for _s, (_sc, rid) in best.items()]


def build(
    store: Path, out: Path, n_posterior: int, n_mle: int, n_sims: int,
    include: list[str] | None = None, all_models: bool = False,
) -> dict:
    os.environ["CAMDL_WATCH_STORE"] = str(store)
    # Discovery re-runs per request and shells out to `camdl list` for native
    # labels each time — fine for a live server, O(N) subprocesses here. The demo
    # uses derived labels, so skip it: turns a multi-minute bake into seconds.
    from camdl_watch import ingest as _ingest
    _ingest._native_labels = lambda _store: {}
    from camdl_watch.api.app import app  # import AFTER env is set

    client = TestClient(app)
    dropped: list[dict] = []

    run_ids = (
        _curate_all_models(client, include or [])
        if all_models
        else _curate(client, n_posterior, n_mle, include or [])
    )
    notes = [_capture_run(client, out, rid, dropped) for rid in run_ids]
    kept = {n["run_id"] for n in notes if n["ok"]}

    # Filtered run list — the demo dropdown shows only the curated runs.
    all_runs = _get(client, "/api/runs") or []
    _write(out, "api/runs", [r for r in all_runs if r["run_id"] in kept])
    _write(out, "api/health", {"status": "ok", "store": "demo", "runs": len(kept)})

    # Profiles (all) — the Profile workspace.
    profiles = _get(client, "/api/profiles") or []
    _write(out, "api/profiles", profiles)
    for p in profiles:
        payload = _get(client, f"/api/profiles/{p['base_id']}")
        if payload is not None:
            _write(out, f"api/profiles/{p['base_id']}", payload)

    # Sims (a few) — the Sims workspace, default compartment/window.
    sims = _get(client, "/api/sims") or []
    picked_sims = sims[:n_sims]
    _write(out, "api/sims", picked_sims)
    for s in picked_sims:
        payload = _get(client, f"/api/sims/{s['sim_id']}/series")
        if payload is not None:
            _write_capped(out, f"api/sims/{s['sim_id']}/series", payload, dropped)
        # also the rendered model math for the sim, when present
        for ep in ("model-render", "model-graph"):
            mp = _get(client, f"/api/runs/{s['sim_id']}/{ep}")
            if mp is not None:
                _write(out, f"api/runs/{s['sim_id']}/{ep}", mp)

    manifest = {
        "runs": notes,
        "n_profiles": len(profiles),
        "n_sims": len(picked_sims),
        "dropped_oversize": dropped,  # files skipped for exceeding MAX_FILE_BYTES
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=Path(os.environ.get("CAMDL_WATCH_STORE", "results/fits")))
    ap.add_argument("--out", type=Path, default=Path("demo-snapshot"))
    ap.add_argument("--posterior", type=int, default=4)
    ap.add_argument("--mle", type=int, default=1)
    ap.add_argument("--sims", type=int, default=3)
    ap.add_argument("--include", default="",
                    help="comma-separated run ids to pin into the demo")
    ap.add_argument("--all-models", action="store_true",
                    help="one representative fit per distinct model stem (widest demo)")
    args = ap.parse_args()
    include = [s.strip() for s in args.include.split(",") if s.strip()]
    m = build(args.store, args.out, args.posterior, args.mle, args.sims,
              include, args.all_models)
    ok = [n for n in m["runs"] if n["ok"]]
    print(f"snapshot → {args.out}: {len(ok)} runs, {m['n_profiles']} profiles, {m['n_sims']} sims")
    for n in ok:
        print(f"  {n['run_id']} ({n['kind']}): {len(n['captured'])} files")
    if m["dropped_oversize"]:
        print(f"dropped {len(m['dropped_oversize'])} oversize files (> {MAX_FILE_BYTES // 1_000_000} MB):")
        for d in m["dropped_oversize"]:
            print(f"  {d['path']}  ({d['bytes'] // 1_000_000} MB)")


if __name__ == "__main__":
    main()
