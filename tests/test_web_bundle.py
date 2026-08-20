"""The frontend bundle is found (and its absence explained).

Regression cover for camdl-scope#3: every non-editable install shipped a wheel
with no bundle *and* resolved the bundle path relative to a repo root that does
not exist in site-packages, so `camdl-watch` answered `/` with a bare 404 — read
by the user as "the server hangs". Two independent defects, so two kinds of test
here: the packaged layout must be found, and a missing bundle must announce
itself rather than 404.

The wheel-packaging half is asserted structurally (the build hook declares the
force-include) rather than by running a full `uv build`, which is far too slow
for the unit suite.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_finds_bundle_in_the_installed_package_layout(tmp_path, monkeypatch):
    """`camdl_watch/_web/index.html` — the only layout a wheel install has."""
    import camdl_watch.api.app as app_mod

    packaged = tmp_path / "_web"
    packaged.mkdir()
    (packaged / "index.html").write_text("<!doctype html>")
    monkeypatch.setattr(app_mod, "_PACKAGED_WEB", packaged)
    monkeypatch.setattr(app_mod, "_CHECKOUT_WEB", tmp_path / "web" / "dist")
    assert app_mod._find_web_dist() == packaged


def test_prefers_the_packaged_bundle_over_the_checkout(tmp_path, monkeypatch):
    """An installed copy must not depend on a path outside itself."""
    import camdl_watch.api.app as app_mod

    packaged = tmp_path / "_web"
    checkout = tmp_path / "web" / "dist"
    for d in (packaged, checkout):
        d.mkdir(parents=True)
        (d / "index.html").write_text("<!doctype html>")
    monkeypatch.setattr(app_mod, "_PACKAGED_WEB", packaged)
    monkeypatch.setattr(app_mod, "_CHECKOUT_WEB", checkout)
    assert app_mod._find_web_dist() == packaged


def test_a_directory_without_index_html_is_not_a_bundle(tmp_path, monkeypatch):
    """An empty/partial `dist` must not count — that is how the 404 looked."""
    import camdl_watch.api.app as app_mod

    empty = tmp_path / "_web"
    empty.mkdir()
    monkeypatch.setattr(app_mod, "_PACKAGED_WEB", empty)
    monkeypatch.setattr(app_mod, "_CHECKOUT_WEB", tmp_path / "absent")
    assert app_mod._find_web_dist() is None


def test_explicit_override_wins_and_does_not_fall_back(tmp_path, monkeypatch):
    """`CAMDL_WATCH_WEB_DIST` is exclusive: a bad override must fail visibly
    rather than silently serving the bundle it was meant to replace."""
    import camdl_watch.api.app as app_mod

    real = tmp_path / "_web"
    real.mkdir()
    (real / "index.html").write_text("<!doctype html>")
    monkeypatch.setattr(app_mod, "_PACKAGED_WEB", real)
    monkeypatch.setenv("CAMDL_WATCH_WEB_DIST", str(tmp_path / "typo"))
    assert app_mod._find_web_dist() is None


def test_missing_bundle_explains_itself_at_root(tmp_path, monkeypatch):
    """No bundle: `/` must say so (503 + guidance), never a bare 404."""
    from fastapi.testclient import TestClient

    import camdl_watch.api.app as app_mod

    monkeypatch.setenv("CAMDL_WATCH_STORE", str(tmp_path))
    monkeypatch.setenv("CAMDL_WATCH_WEB_DIST", str(tmp_path / "no-bundle-here"))
    # The static mount is decided at import, so the module must be re-executed;
    # the env override survives that (a patched global would not).
    fresh = importlib.reload(app_mod)
    try:
        assert fresh._WEB_DIST is None
        client = TestClient(fresh.app)
        r = client.get("/")
        assert r.status_code == 503
        assert "api only" in r.text.lower()
        # The API stays available — that distinction is the whole point.
        assert client.get("/api/health").status_code == 200
    finally:
        monkeypatch.delenv("CAMDL_WATCH_WEB_DIST", raising=False)
        importlib.reload(app_mod)  # leave the module as the suite expects


def test_build_hook_packages_the_bundle_into_the_wheel():
    """The hook must force-include `web/dist` at the packaged location, and
    pyproject must register it — either half missing ships a UI-less wheel."""
    root = Path(__file__).resolve().parents[1]
    hook_src = (root / "hatch_build.py").read_text()
    assert 'PACKAGED_WEB = "camdl_watch/_web"' in hook_src
    assert 'build_data["force_include"]' in hook_src

    pyproject = (root / "pyproject.toml").read_text()
    assert "[tool.hatch.build.targets.wheel.hooks.custom]" in pyproject
    assert 'path = "hatch_build.py"' in pyproject


@pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1"])
def test_cli_announces_the_url_before_importing_the_app(capsys, tmp_path, host):
    """The URL must reach stdout before the multi-second import, or the wait
    reads as a hang (camdl-scope#3). Patch uvicorn so nothing actually binds."""
    from camdl_watch import cli

    calls = {}
    monkey = SimpleNamespace(run=lambda *a, **k: calls.update(ran=True))
    saved = sys.modules.get("uvicorn")
    sys.modules["uvicorn"] = monkey
    try:
        cli.main(store=tmp_path, host=host, port=8899)
    finally:
        if saved is not None:
            sys.modules["uvicorn"] = saved
        else:
            del sys.modules["uvicorn"]
    out = capsys.readouterr().out
    assert "camdl-watch: serving" in out
    assert "8899" in out
    assert str(tmp_path) in out
    assert calls.get("ran") is True
