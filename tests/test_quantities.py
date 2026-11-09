"""Generated-quantities reader tests. No ``camdl fit predict`` quantities output
exists in the reference store, so these use synthetic fixtures written to a tmp
dir, matching camdl's on-disk layout (``quantity_output.rs``):

* ``quantities.json`` — a manifest, one entry per (quantity × scenario), each
  tagged with a ``scenario`` (omitted for the scenario-less simulate path).
* ``quantities/<name>.tsv`` — a banded TSV with a leading ``scenario`` column.

The reader is manifest-driven: it dedupes to *logical* quantities, collects the
scenario set, and ignores stale orphan TSVs the manifest doesn't list.
"""

from __future__ import annotations

import json
from pathlib import Path

from camdl_scope import quantities


def _write_manifest(run_dir: Path, entries: list[dict]) -> None:
    (run_dir / "quantities.json").write_text(
        json.dumps({"schema": "camdl.quantities/v1", "quantities": entries})
    )


def _entry(name: str, shape: str, scenario: str | None = None, **kw) -> dict:
    e = {
        "name": name, "shape": shape, "source": kw.get("source", "state"),
        "index_dims": kw.get("index_dims", []), "reduce": kw.get("reduce"),
        "unit": None, "censoring": kw.get("censoring"),
    }
    if scenario is not None:
        e["scenario"] = scenario
    return e


def test_read_manifest_dedupes_to_logical_quantities_and_collects_scenarios(tmp_path):
    # Two scenarios × three logical quantities = six denormalized entries.
    entries = []
    for sc in ("baseline", "no_sia"):
        entries.append(_entry("prevalence", "series", sc))
        entries.append(_entry("peak_prev", "scalar", sc, reduce="max"))
        entries.append(
            _entry("onset", "scalar", sc, reduce="time_of_max",
                   censoring={"kind": "right", "conditional_quantiles": True})
        )
    _write_manifest(tmp_path, entries)

    m = quantities.read_manifest(tmp_path)
    assert [q.name for q in m.quantities] == ["prevalence", "peak_prev", "onset"]
    assert m.scenarios == ["baseline", "no_sia"]
    by_name = {q.name: q for q in m.quantities}
    assert by_name["prevalence"].shape == "series"
    assert by_name["peak_prev"].reduce == "max"
    assert by_name["onset"].censorable is True
    assert by_name["peak_prev"].censorable is False


def test_read_manifest_scenarioless_has_empty_scenarios(tmp_path):
    _write_manifest(tmp_path, [_entry("attack_rate", "scalar", reduce="final")])
    m = quantities.read_manifest(tmp_path)
    assert m.scenarios == []
    assert [q.name for q in m.quantities] == ["attack_rate"]


def test_read_manifest_absent_is_empty(tmp_path):
    m = quantities.read_manifest(tmp_path)
    assert m.quantities == [] and m.scenarios == []


def test_read_quantity_reads_listed_tsv_else_none(tmp_path):
    qdir = tmp_path / "quantities"
    qdir.mkdir()
    (qdir / "peak_prev.tsv").write_text(
        "scenario\tn_draws\tq05\tq25\tq50\tq75\tq95\n"
        "baseline\t2000\t0.01\t0.04\t0.08\t0.10\t0.13\n"
    )
    df = quantities.read_quantity(tmp_path, "peak_prev")
    assert df is not None
    assert df["q50"][0] == 0.08
    assert df["scenario"][0] == "baseline"
    # A name not on disk (e.g. a stale orphan the manifest no longer lists).
    assert quantities.read_quantity(tmp_path, "ghost") is None
