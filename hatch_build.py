"""Wheel build hook: ship the built SPA *inside* the package.

The frontend is a Vite bundle under ``web/dist``, which is gitignored — a build
artifact, not source. Without this hook the wheel contains only ``camdl_watch``,
so every non-editable install (``uv tool install git+…``, ``uvx --from``,
``uv add --dev``) yields a server that answers the API but 404s at ``/``: the
app looks for the bundle relative to a repo root that does not exist in
site-packages, and there is nothing to find there anyway.

So the hook copies the bundle to ``camdl_watch/_web`` in the wheel, next to the
code that serves it, and builds it first when it is missing (an sdist / fresh
clone has ``web/src`` but no ``web/dist``). Building requires npm; when npm is
absent the build FAILS with an actionable message rather than quietly producing
the bundle-less wheel that caused this in the first place.

Escape hatch: set ``CAMDL_WATCH_SKIP_WEB_BUILD=1`` to build an API-only wheel
deliberately (CI that only runs the Python suite). The server then says so at
startup instead of 404-ing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

#: Where the bundle lands inside the wheel, relative to the package root.
PACKAGED_WEB = "camdl_watch/_web"


class WebBundleBuildHook(BuildHookInterface):
    """Put ``web/dist`` into the wheel at :data:`PACKAGED_WEB`."""

    PLUGIN_NAME = "camdl-watch-web"

    def initialize(self, version: str, build_data: dict) -> None:
        if os.environ.get("CAMDL_WATCH_SKIP_WEB_BUILD"):
            self.app.display_warning(
                "CAMDL_WATCH_SKIP_WEB_BUILD set — building an API-only wheel "
                "with no frontend bundle."
            )
            return

        root = Path(self.root)
        dist = root / "web" / "dist"
        if not (dist / "index.html").is_file():
            self._build_frontend(root)
        if not (dist / "index.html").is_file():
            raise RuntimeError(
                f"frontend bundle missing after build: {dist}/index.html not found"
            )

        # force_include copies at wheel-assembly time, so the bundle needs no
        # entry in `packages` and stays out of the source tree.
        build_data["force_include"][str(dist)] = PACKAGED_WEB
        build_data["artifacts"].append(f"/{PACKAGED_WEB}")

    def _build_frontend(self, root: Path) -> None:
        web = root / "web"
        if not (web / "package.json").is_file():
            raise RuntimeError(
                f"cannot build the frontend: {web}/package.json is missing. "
                "The sdist/checkout appears incomplete."
            )
        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError(
                "npm is required to build the camdl-watch frontend bundle "
                "and was not found on PATH.\n"
                "  Fix: install Node.js (https://nodejs.org), then reinstall.\n"
                "  Or:  set CAMDL_WATCH_SKIP_WEB_BUILD=1 to build an API-only "
                "wheel (no browser UI)."
            )
        self.app.display_info("camdl-watch: building the frontend bundle …")
        # `npm ci` when the lockfile is present (reproducible), else `npm install`.
        install = ["ci"] if (web / "package-lock.json").is_file() else ["install"]
        subprocess.run([npm, *install], cwd=web, check=True)
        subprocess.run([npm, "run", "build"], cwd=web, check=True)
