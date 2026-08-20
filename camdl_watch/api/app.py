"""FastAPI application: the typed seam between the Python core and the browser.

Routes live under ``/api/*``; the built frontend (``web/dist``) is mounted at
``/`` when present, so a single ``camdl-watch`` process serves both. The store
to read is taken from ``CAMDL_WATCH_STORE`` (set by the CLI / env), else
``results/fits`` under the working directory — matching the v1 app's contract.

This module must import cleanly even while the rest of the core is mid-edit:
the run store is only touched inside request handlers (lazily), never at import.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_log = logging.getLogger("camdl_watch")

# Store resolution mirrors the v1 app: an explicit override wins, else the
# conventional ``results/fits`` under the directory camdl-watch is launched from.
_DEFAULT_STORE = Path.cwd() / "results" / "fits"

# Where the built SPA can live, in resolution order:
#   1. inside the installed package (``camdl_watch/_web``) — put there by the
#      wheel build hook, the only layout a non-editable install ever sees;
#   2. ``web/dist`` at the repo root — a source checkout / editable install,
#      where `make build` writes.
# Checking the package first means an installed copy never depends on a path
# outside itself (``parents[2]`` in site-packages resolves to a directory that
# cannot contain the bundle).
_PACKAGED_WEB = Path(__file__).resolve().parents[1] / "_web"
_CHECKOUT_WEB = Path(__file__).resolve().parents[2] / "web" / "dist"


def _web_candidates() -> tuple[Path, ...]:
    """Where to look for the bundle, in order. ``CAMDL_WATCH_WEB_DIST`` is an
    explicit override (for a packager serving a bundle from elsewhere) and wins
    *exclusively*: falling back after an override would quietly serve a
    different UI than the one asked for, hiding the typo."""
    override = os.environ.get("CAMDL_WATCH_WEB_DIST")
    if override:
        return (Path(override),)
    return (_PACKAGED_WEB, _CHECKOUT_WEB)


def _find_web_dist() -> Path | None:
    """The built frontend to serve, or ``None`` when this install has no
    bundle (an API-only wheel, or a checkout where the frontend was never
    built)."""
    for p in _web_candidates():
        if (p / "index.html").is_file():
            return p
    return None


_WEB_DIST = _find_web_dist()

#: Shown at ``/`` when there is no bundle, so the failure explains itself
#: instead of surfacing as a bare 404 (or as "the server hangs").
_NO_BUNDLE_MESSAGE = (
    "The camdl-watch API is running, but this installation has no browser UI "
    "bundle, so there is nothing to serve here."
)


def current_store() -> Path:
    """The fit store to read, resolved *fresh* on every call from
    ``CAMDL_WATCH_STORE`` (the CLI/env override), else the conventional
    ``results/fits``. Reading the env each call — rather than freezing it at
    import — lets the CLI and the tests repoint the store after this module is
    imported."""
    return Path(os.environ.get("CAMDL_WATCH_STORE", str(_DEFAULT_STORE)))


app = FastAPI(title="camdl-watch", version="2.0.0-dev")


@app.middleware("http")
async def _no_cache_html(request, call_next):
    """Serve the SPA shell (index.html) no-cache so a rebuilt frontend shows up
    on a normal refresh — it references content-hashed assets, which stay
    cacheable. Without this the browser pins a stale index.html and never sees
    new builds."""
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/api/health")
def health() -> dict:
    """Liveness + a cheap store summary, for the frontend's plumbing check.

    Run discovery is best-effort: a malformed/absent store must never 500 the
    health probe, so discovery failures degrade to ``runs: 0`` rather than
    propagating (the broad guard is deliberate and scoped to this probe)."""
    store = current_store()
    runs = 0
    try:
        from .. import ingest

        runs = len(ingest.discover_runs(store, include_warming=True))
    except Exception:  # health must stay green regardless of store state
        runs = 0
    return {"status": "ok", "store": str(store), "runs": runs}


# Typed read-only API under /api. Mounted before the SPA static block so the
# /api routes win; the catch-all "/" mount must stay last.
from .routes import router  # noqa: E402  (import here: see ordering note above)

app.include_router(router)


# Serve the built SPA at the root when this install has one. A missing bundle is
# normal in `make dev` (Vite serves the frontend and proxies /api here), but for
# an installed `camdl-watch` it means the UI is simply absent — which used to
# surface as an unexplained 404 at `/`. Say so, in the log and at `/`.
if _WEB_DIST is not None:
    app.mount("/", StaticFiles(directory=str(_WEB_DIST), html=True), name="web")
else:
    _searched = ", ".join(str(p) for p in _web_candidates())
    _log.warning(
        "camdl-watch: no frontend bundle found (searched: %s) — serving the API "
        "only; / will explain how to get the UI.",
        _searched,
    )

    @app.get("/", include_in_schema=False)
    def _no_bundle() -> HTMLResponse:
        """Explain the missing UI rather than 404-ing at the root."""
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8>"
            "<title>camdl-watch — API only</title>"
            '<body style="font:14px/1.6 system-ui,sans-serif;max-width:46rem;'
            'margin:3rem auto;padding:0 1rem;color:#171717">'
            f"<h1 style='font-size:1.1rem'>camdl-watch is running (API only)</h1>"
            f"<p>{_NO_BUNDLE_MESSAGE}</p>"
            "<p>The JSON API works: try "
            "<a href='/api/health'><code>/api/health</code></a>.</p>"
            "<p>To get the browser UI, reinstall from a source checkout with the "
            "frontend built:</p>"
            "<pre style='background:#f5f5f5;padding:.75rem;border-radius:4px;"
            "overflow-x:auto'>git clone https://github.com/vsbuffalo/camdl-scope\n"
            "cd camdl-scope/web &amp;&amp; npm install &amp;&amp; npm run build\n"
            "uv tool install --force --editable ..</pre>"
            "<p style='color:#737373'>Installing from git builds the bundle "
            "automatically when Node.js is available; a wheel built with "
            "<code>CAMDL_WATCH_SKIP_WEB_BUILD=1</code> has no UI by design.</p>"
            "</body>",
            status_code=503,
        )
