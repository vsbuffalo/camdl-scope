"""The dashboard's API output on the camdl-made scenario store matches the
committed snapshots (``tests/scenarios/snapshots``).

Skipped unless ``make scenarios`` has built ``build/scenarios`` and ``camdl`` is
on PATH (the Compare endpoint shells out to it). A failure after a camdl update
is the point: review what changed, then ``make scenario-snapshots`` and commit
the diff.
"""

from __future__ import annotations

import shutil

import pytest

from tests.scenarios import scenarios


@pytest.mark.skipif(
    not (scenarios.DEFAULT_OUT / "manifest.json").exists(),
    reason="no scenario store: run `make scenarios`",
)
@pytest.mark.skipif(shutil.which(scenarios.CAMDL) is None, reason="camdl not on PATH")
def test_api_matches_scenario_snapshots(monkeypatch):
    monkeypatch.setenv("CAMDL_SCOPE_STORE", str(scenarios.DEFAULT_OUT / "results" / "fits"))
    assert scenarios.snapshot(scenarios.DEFAULT_OUT, check=True) == 0
