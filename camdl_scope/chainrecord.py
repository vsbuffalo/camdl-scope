"""What camdl persisted about each chain of a method leaf.

Two records, both computed once by camdl at the end of the run and read back
here (camdl run spec 10.6); this module holds no threshold and compares
nothing:

* ``frozen_chains``: the chains that took no step after burn-in, a list of
  chain numbers under ``<algorithm>_summary.json`` (and under
  ``method_result.diagnostics`` of the ``method_report`` in ``events.jsonl``).
  ``[]`` means every chain moved; *absent* means the leaf was written before
  camdl recorded the field, which is not the same thing and is returned as
  ``None``.
* ``chain_loglik_agreement``: each chain's mean log-likelihood against the
  pooled mean of the others, with camdl's Z statistic, its Bonferroni cutoff and
  its verdict per chain (``flagged``), or the reason a chain was not assessed.
  A sampler leaf records it in ``<algorithm>_summary.json`` (and as
  ``method_result.diagnostics.chain_logliks`` of the method report); an IF2 leaf
  records it in ``fit_state.toml``. ``None`` means the leaf does not record one.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .verdict import parse_number, read_events


@dataclass(frozen=True)
class ChainLoglik:
    """One chain of camdl's log-likelihood comparison. ``status`` is
    ``assessed`` (then ``diff`` in nats, the chain minus the others, with its
    standard error, Z, the cutoff Z is held against and camdl's ``flagged``) or
    ``not_assessed`` (then ``reason``, and ``ess`` when the reason is too few
    effective draws). A non-finite number stays the float it names."""

    chain: int
    mean: float | None
    status: str
    diff: float | None = None
    se: float | None = None
    z: float | None = None
    cutoff: float | None = None
    flagged: bool | None = None
    reason: str | None = None
    ess: float | None = None


@dataclass(frozen=True)
class ChainLogliks:
    """The whole record. ``column`` names what each chain's mean is a mean of
    (``log_likelihood``, ``obs_ll``, ``clean_loglik``); ``column_absent`` is
    camdl's statement that traces were read and none carried it."""

    column: str
    column_absent: bool
    family_alpha: float | None
    min_ess: float | None
    chains: tuple[ChainLoglik, ...] = field(default_factory=tuple)


def _chain_loglik(raw: Any) -> ChainLoglik:
    if not isinstance(raw, dict) or not isinstance(raw.get("chain"), int):
        raise ValueError(f"a chain_loglik row is an object with an integer 'chain', got {raw!r}")
    out = raw.get("outcome")
    if not isinstance(out, dict) or not isinstance(out.get("status"), str):
        raise ValueError(f"chain {raw['chain']}: outcome needs a string 'status', got {out!r}")
    status = out["status"]
    mean = parse_number(raw.get("mean"))
    if status == "assessed":
        flagged = out.get("flagged")
        return ChainLoglik(
            chain=raw["chain"], mean=mean, status=status,
            diff=parse_number(out.get("diff")), se=parse_number(out.get("se")),
            z=parse_number(out.get("z")), cutoff=parse_number(out.get("cutoff")),
            flagged=flagged if isinstance(flagged, bool) else None,
        )
    if status == "not_assessed":
        reason = out.get("reason")
        return ChainLoglik(
            chain=raw["chain"], mean=mean, status=status,
            reason=reason if isinstance(reason, str) else None,
            ess=parse_number(out.get("ess")),
        )
    raise ValueError(f"chain {raw['chain']}: unknown comparison status {status!r}")


def parse_chain_logliks(raw: Any) -> ChainLogliks | None:
    """The record as camdl writes it (``None`` or absent: not recorded).
    Raises ``ValueError`` on a shape this reader does not know, so a changed
    grammar fails visibly rather than as a quietly empty table."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("chains"), list):
        raise ValueError(f"chain log-likelihood record needs a 'chains' list, got {raw!r}")
    return ChainLogliks(
        column=str(raw.get("column", "")),
        column_absent=raw.get("column_absent") is True,
        family_alpha=parse_number(raw.get("family_alpha")),
        min_ess=parse_number(raw.get("min_ess")),
        chains=tuple(_chain_loglik(c) for c in raw["chains"]),
    )


def _json(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _summary(leaf: Path) -> dict | None:
    for p in sorted(leaf.glob("*_summary.json")):
        if (obj := _json(p)) is not None:
            return obj
    return None


def _report_diagnostics(leaf: Path) -> dict:
    """``method_result.diagnostics`` of the leaf's ``method_report``, or ``{}``
    (no events log, a running or failed leaf, a method with no such block)."""
    try:
        ev = read_events(leaf)
    except (ValueError, OSError):
        return {}
    rep = ev.method_report if ev is not None else None
    report = rep.get("report") if isinstance(rep, dict) else None
    mr = report.get("method_result") if isinstance(report, dict) else None
    diag = mr.get("diagnostics") if isinstance(mr, dict) else None
    return diag if isinstance(diag, dict) else {}


def read_frozen_chains(leaf: Path) -> list[int] | None:
    """The chains camdl recorded as taking no step after burn-in, ascending;
    ``[]`` when it recorded that every chain moved; ``None`` when the leaf does
    not record the field."""
    for src in (_summary(leaf) or {}, _report_diagnostics(leaf)):
        raw = src.get("frozen_chains")
        if isinstance(raw, list) and all(isinstance(c, int) for c in raw):
            return sorted(raw)
    return None


def read_chain_logliks(leaf: Path) -> ChainLogliks | None:
    """camdl's per-chain log-likelihood comparison for the leaf, or ``None``
    when it is not recorded there. Sources, in the order camdl documents them:
    the sampler summary, the method report, and (IF2) ``fit_state.toml``.
    Raises ``ValueError`` on a record this reader cannot parse."""
    raw = (_summary(leaf) or {}).get("chain_loglik_agreement")
    if raw is None:
        raw = _report_diagnostics(leaf).get("chain_logliks")
    if raw is None:
        try:
            raw = tomllib.loads((leaf / "fit_state.toml").read_text()).get(
                "chain_loglik_agreement")
        except (OSError, tomllib.TOMLDecodeError):
            raw = None
    return parse_chain_logliks(raw)
