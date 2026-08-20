"""camdl-watch CLI — launch the browser results viewer.

A thin ``defopt`` wrapper over :func:`uvicorn.run`. ``--store`` maps to the
``CAMDL_WATCH_STORE`` env var (the app's public override), set before the app is
imported so it reads the chosen store. The server hosts the JSON API under
``/api`` and, when ``web/dist`` has been built, the React frontend at ``/``.

For frontend development, run the API with reload and the Vite dev server
side by side via ``make dev`` instead of this launcher.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import defopt


def main(
    *,
    store: Optional[Path] = None,
    host: str = "127.0.0.1",
    port: int = 8800,
) -> None:
    """Launch the camdl-watch results viewer.

    Serves the JSON API (and the built frontend, if present) for a camdl fit
    store. Bind ``--host 0.0.0.0`` to reach it from a phone over LAN / Tailscale.

    :param store: camdl fit store to read (the directory of run dirs).
        Defaults to ``results/fits`` under the current working directory.
    :param host: Network interface to bind.
    :param port: TCP port to serve on.
    """
    if store is not None:
        os.environ["CAMDL_WATCH_STORE"] = str(store)

    # Announce BEFORE the imports: pulling in the app (arviz → xarray → pandas →
    # scipy) takes seconds, and on a cold cache tens of seconds, during which
    # uvicorn has printed nothing. Silence for that long reads as a hang, so the
    # URL goes to stdout first — uvicorn's own banner goes to stderr later.
    resolved = Path(os.environ.get("CAMDL_WATCH_STORE", "results/fits")).resolve()
    shown_host = "localhost" if host in ("0.0.0.0", "::") else host
    print(
        f"camdl-watch: serving {resolved} on http://{shown_host}:{port}"
        f"{' (all interfaces)' if host in ('0.0.0.0', '::') else ''}\n"
        "camdl-watch: starting … (first launch takes a few seconds)",
        flush=True,
    )
    if not resolved.is_dir():
        print(
            f"camdl-watch: note — {resolved} does not exist yet; "
            "run from your camdl project root or pass --store.",
            flush=True,
        )

    # Import after setting the env var: the app reads CAMDL_WATCH_STORE at import.
    import uvicorn

    uvicorn.run("camdl_watch.api.app:app", host=host, port=port, reload=False)


def cli() -> None:
    """Console-script entry point."""
    defopt.run(main)
