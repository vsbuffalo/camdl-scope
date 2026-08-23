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
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

import defopt


def _port_holder(port: int) -> str | None:
    """One line describing what already listens on ``port``, via ``lsof``, or
    ``None`` when it cannot be determined. Best-effort: naming the holder turns
    "it served the wrong fits" into "that tmux pane is already serving"."""
    exe = shutil.which("lsof")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    return lines[1] if len(lines) > 1 else None


def _port_is_free(host: str, port: int) -> bool:
    """Whether ``(host, port)`` can be bound right now.

    A plain bind, deliberately without ``SO_REUSEADDR``: a listening socket
    still refuses the address, which is exactly the condition we want to
    detect. Racy in principle — uvicorn binds a moment later — but the race
    loses nothing, since uvicorn's own failure remains the backstop."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.bind((host, port))
        return True
    except OSError:
        return False


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

    # Refuse a held port instead of announcing one we will not get. uvicorn
    # fails to bind AND EXITS 0, so the announcement below would name a URL
    # served by whatever else is listening — a different store, silently, with
    # a success exit code that a script or an agent will believe.
    if not _port_is_free(host, port):
        holder = _port_holder(port)
        print(
            f"camdl-watch: port {port} is already in use — refusing to start.\n"
            + (f"camdl-watch: held by  {holder}\n" if holder else "")
            + f"camdl-watch: pick another port (--port {port + 1}), or stop that "
            "process first.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
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
