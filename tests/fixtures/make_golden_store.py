"""Generate a deterministic golden fit store for viewer development and tests.

The runs in a real camdl store predate the ``docs``/``schema`` sidecar, so they
carry no parameter documentation. This builds one synthetic-but-faithful
posterior run that exercises the *full* sidecar the v2 viewer is designed
around:

  * ``docs`` — symbols (β/σ/γ/ρ), prose, and ``@ref`` citations, plus a base
    ``k_raw`` block whose estimated coordinates are expanded per patch
    (``k_raw_Bo``/``k_raw_Bombali``), to exercise base-name doc resolution;
  * ``schema`` — a ``cases`` stream indexed by a ``patch`` dimension with named
    levels;
  * a two-chain posterior ``trace.tsv`` so the run is discoverable and has
    draws to summarize;
  * ``events.jsonl`` (``camdl.events/v1``, run spec 10.11) with camdl's end-of-run
    ``method_report``, so the viewer reads a verdict camdl persisted. It is
    written to the spec's shape by this generator, not captured from a camdl
    run: the verdict it carries is a fixed ``converged`` whose ``max_rhat`` is
    the largest R-hat of the synthetic summary beside it;
  * ``predictive``/``observed`` TSVs for the ``cases`` stream, in two
    free-forward predictions stored inside the run (``predict/<mode>-<12 hex>/``),
    the second with a longer horizon, so the predict picker has two to choose
    between.

Deterministic: a fixed RNG seed and fixed content hashes, so re-running writes
byte-identical files. Run via ``make fixture`` or ``python -m
tests.fixtures.make_golden_store --out <dir>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tests.storekit import BUILD, StoreKit, events_jsonl

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = _REPO_ROOT / "tests" / "fixtures" / "golden-store"

#: The fit problem's folder name, which is the run's entry id (it holds one run).
RUN_DIR = "seir_patch_demo-a1b2c3d4e5f6"
#: The run's method folder and its own folder inside ``<problem>/runs/``.
POSTERIOR_DIR = "pgas-b2c3d4e5f6a7"
SEED_DIR = "seed_1-c3d4e5f6a7b8"

# (param, posterior mean, sd, chain-offset) — the offset gives R̂ a touch above 1.
PARAMS: list[tuple[str, float, float, float]] = [
    ("beta", 0.55, 0.040, 0.010),
    ("sigma", 0.19, 0.020, 0.004),
    ("gamma", 0.14, 0.015, 0.003),
    ("rho", 0.32, 0.030, 0.006),
    ("k_raw_Bo", 0.20, 0.450, 0.050),
    ("k_raw_Bombali", -0.35, 0.500, 0.060),
]
PARAM_NAMES = [p[0] for p in PARAMS]

N_SWEEPS = 600
N_CHAINS = 2

DOCS = {
    "parameters": {
        "beta": {
            "text": "per-capita transmission rate (contact rate × per-contact "
            "transmission probability)",
            "symbol": "β",
            "ref": "Anderson & May 1991",
        },
        "sigma": {
            "text": "rate of progression from exposed to infectious; the mean "
            "latent period is 1/σ",
            "symbol": "σ",
        },
        "gamma": {
            "text": "recovery rate; the mean infectious period is 1/γ",
            "symbol": "γ",
            "ref": "Keeling & Rohani 2008",
        },
        "rho": {"text": "case reporting probability", "symbol": "ρ"},
        "k_raw": {
            "text": "patch-level transmission deviation (non-centered "
            "parameterization)",
            "symbol": "k",
        },
    },
    "dimensions": {"patch": {"text": "spatial patch (administrative district)"}},
    "observations": {"cases": {"text": "reported case counts", "symbol": "y"}},
}

SCHEMA = {
    "dimensions": {"patch": {"levels": ["Bo", "Bombali"]}},
    "streams": [
        {
            "name": "cases",
            "index_dims": ["patch"],
            "value_column": "cases",
            "value_kind": "count",
            "likelihood": "neg_binomial",
        }
    ],
}

# Compartmental flow graph (model.graph.json) — a patch SEIRS consistent with the
# fixture's states/params, exercising every FlowDiagram branch: a forward chain,
# a waning return arc (R→S), a birth inflow chip, an all-compartment iterator
# edge (death), and a mean-field read-pool coupling.
MODEL_GRAPH = {
    "model": "seir_patch_demo",
    "nodes": [
        {"id": "S", "label": "S"},
        {"id": "E", "label": "E"},
        {"id": "I", "label": "I"},
        {"id": "R", "label": "R"},
    ],
    "plates": [{"name": "patch", "levels": ["Bo", "Bombali"]}],
    "edges": [
        {"id": "infection", "from": "S", "to": "E",
         "rate": "\\frac{\\beta\\,S\\,I_{p}}{N_{p}}", "advances": None, "reads_pool": True},
        {"id": "progression", "from": "E", "to": "I",
         "rate": "\\sigma\\,E", "advances": None, "reads_pool": False},
        {"id": "recovery", "from": "I", "to": "R",
         "rate": "\\gamma\\,I", "advances": None, "reads_pool": False},
        {"id": "waning", "from": "R", "to": "S",
         "rate": "\\omega\\,R", "advances": None, "reads_pool": False},
        {"id": "birth", "from": None, "to": "S",
         "rate": "\\delta\\,N_{p}", "advances": None, "reads_pool": True},
        {"id": "death", "from": "c", "to": None,
         "rate": "\\delta\\,c", "advances": None, "reads_pool": False},
    ],
    "couplings": [
        {"edge": "infection", "aggregate": "I_patch", "over": []},
        {"edge": "infection", "aggregate": "N_patch", "over": []},
        {"edge": "birth", "aggregate": "N_patch", "over": []},
    ],
}

FIT_META = {
    "model_identity": "demo0000000000000000000000000000",
    "fit_toml_hash": "demo0000",
    "data_hashes": {"cases": "demo0001"},
    "estimated": PARAM_NAMES,
    "fixed": {"N0": 1_000_000.0},
    "resolved_priors": [{"param": p, "source": "fit_toml"} for p in PARAM_NAMES],
    "parameters_provenance": {},
    "schema": SCHEMA,
}

FIT_TOML = """\
# Golden demo fit config (synthetic). Mirrors the shape camdl writes.
[method]
algorithm = "pgas"
backend = "chain_binomial"
sweeps = 600
burn_in = 300

[estimate.beta]
bounds = [0.0, 2.0]
[estimate.beta.prior.log_normal]
mu = -0.6
sigma = 0.4

[estimate.sigma]
bounds = [0.0, 1.0]
[estimate.sigma.prior.log_normal]
mu = -1.7
sigma = 0.3

[estimate.gamma]
bounds = [0.0, 1.0]
[estimate.gamma.prior.log_normal]
mu = -2.0
sigma = 0.3

[estimate.rho]
bounds = [0.0, 1.0]
[estimate.rho.prior.beta]
alpha = 3.0
beta = 6.0

[estimate.k_raw_Bo]
bounds = [-5.0, 5.0]
[estimate.k_raw_Bo.prior.normal]
mu = 0.0
sigma = 1.0

[estimate.k_raw_Bombali]
bounds = [-5.0, 5.0]
[estimate.k_raw_Bombali.prior.normal]
mu = 0.0
sigma = 1.0
"""


def _trace_for_chain(rng: np.random.Generator, chain: int) -> list[list[float]]:
    """Build one chain's rows: sweep, params…, log_likelihood, log_posterior."""
    sweeps = np.arange(N_SWEEPS)
    cols: dict[str, np.ndarray] = {}
    for name, mean, sd, offset in PARAMS:
        center = mean + offset * chain
        draws = rng.normal(center, sd, N_SWEEPS)
        if not name.startswith("k_raw"):  # rate/probability params stay positive
            draws = np.abs(draws)
        cols[name] = draws
    # A warm-up climb into a noisy plateau, so traces and ll look like traces.
    climb = -300.0 * np.exp(-sweeps / 90.0)
    ll = -1200.0 + climb + rng.normal(0.0, 18.0, N_SWEEPS)
    lp = ll - 12.0 + rng.normal(0.0, 4.0, N_SWEEPS)
    rows = []
    for i in range(N_SWEEPS):
        row = [float(sweeps[i])]
        row += [float(cols[name][i]) for name in PARAM_NAMES]
        row += [float(ll[i]), float(lp[i])]
        rows.append(row)
    return rows


def _write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join(_fmt(v) for v in r))
    path.write_text("\n".join(lines) + "\n")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _summary(rng: np.random.Generator) -> dict:
    rhat = {p: round(1.0 + 0.02 * abs(off) / sd, 3) for p, _m, sd, off in PARAMS}
    ess = {p: round(float(rng.uniform(450, 950)), 0) for p in PARAM_NAMES}
    epc = {p: [round(e * 0.48, 0), round(e * 0.52, 0)] for p, e in ess.items()}
    return {
        # camdl tags each summary with the contract its numbers follow (gh#728).
        "schema": "camdl.fit-summary/v2",
        "stage": "posterior",
        "n_chains": N_CHAINS,
        "rhat": rhat,
        "ess": ess,
        "ess_per_chain": epc,
        "acceptance_rates": [[0.31, 0.29], [0.30, 0.28]],
    }


#: The option-keyed prediction's folder and the options it records.
DEFAULT_PREDICT = "free_forward-d4e5f6a7d4e5"
KEYED_PREDICT = "free_forward-e5f6a7b8c9d0"
KEYED_OPTIONS = {"horizon": 30, "n_draws": 200}

#: The sampler settings camdl resolved for the run (``run_inputs.json``).
ALGORITHM = {"algorithm": "pgas", "backend": "chain_binomial",
             "chains": N_CHAINS, "particles": 256, "sweeps": N_SWEEPS}


def leaf_path(store: Path) -> Path:
    """The golden run's own folder in a store built by :func:`build`."""
    return store / BUILD / "fit" / RUN_DIR / "runs" / POSTERIOR_DIR / SEED_DIR


def _events(summary: dict) -> str:
    """The leaf's ``events.jsonl``: ``begin``, one ``progress`` per chain, then
    the ``method_report`` camdl appends after the leaf commits. Written to the
    shape of run spec 10.11; the verdict is the one a camdl that read this
    summary's R-hat would record, fixed here rather than recomputed."""
    rhat = summary["rhat"]
    worst = max(rhat, key=rhat.get)
    base = {"schema": "camdl.events/v1"}
    lines = [
        {**base, "seq": 0, "elapsed": 0.0, "event": "begin", "method": "pgas",
         "header": {"config": "fit.toml", "method": "pgas", "model": "seir_patch_demo",
                    "estimate": PARAM_NAMES, "fixed": [], "starts": "from_prior",
                    "output": "seir_patch_demo-a1b2c3d4"}},
        *[{**base, "seq": 1 + c, "elapsed": 1.0, "event": "progress", "chain": c + 1,
           "step": N_SWEEPS, "of": N_SWEEPS, "loglik": None, "accept": None}
          for c in range(N_CHAINS)],
        {**base, "seq": 1 + N_CHAINS, "elapsed": 8.0, "event": "method_report",
         "outcome": "completed",
         "report": {"name": "pgas", "method": "pgas", "n_chains": N_CHAINS,
                    "checks": [], "findings": [],
                    "verdict": {"kind": "converged", "max_rhat": rhat[worst],
                                "param": worst}}},
    ]
    return events_jsonl(lines)


def _predictive_observed(
    rng: np.random.Generator, pdir: Path, options: dict, t_end: int
) -> None:
    times = list(range(0, t_end, 2))
    patches = SCHEMA["dimensions"]["patch"]["levels"]
    # camdl.predictive/v3 columns (run spec §2.2.2), less the per-row channels.
    phead = ["time", "patch", "horizon", "treatment", "fit_rhat_max", "fit_ess_min",
             "n_draws", "q05", "q25", "q50", "q75", "q95"]
    ohead = ["time", "patch", "value"]
    prows, orows = [], []
    for patch_i, patch in enumerate(patches):
        base = 40.0 + 25.0 * patch_i
        for t in times:
            mid = base + 18.0 * np.sin(t / 4.0) + 4.0 * patch_i
            spread = 0.25 * mid
            q = [mid - 1.6 * spread, mid - 0.7 * spread, mid,
                 mid + 0.7 * spread, mid + 1.6 * spread]
            q = [max(0.0, x) for x in q]
            prows.append([t, patch, 0, "none", 1.01, 480, 600, *q])
            obs = max(0.0, mid + rng.normal(0.0, spread * 0.5))
            orows.append([t, patch, round(obs)])
    _write_tsv(pdir / "predictive" / "cases.tsv", phead, prows)
    _write_tsv(pdir / "observed" / "cases.tsv", ohead, orows)
    for name in ("predictive.json", "observed.json"):
        (pdir / name).write_text(json.dumps(
            {"schema": "camdl.predictive/v4", "options": {"mode": "free_forward", **options}},
            indent=2) + "\n")


def build(out: Path) -> Path:
    """Write the golden store under ``out`` and return the run's own folder."""
    rng = np.random.default_rng(0)
    kit = StoreKit.create(out)
    problem = kit.problem("seir_patch_demo", name=RUN_DIR, docs=DOCS)
    (problem / "model.graph.json").write_text(
        json.dumps(MODEL_GRAPH, indent=2, ensure_ascii=False) + "\n")
    seed_dir = kit.fit_run(
        problem, algorithm="pgas", seed=1, method_dir=POSTERIOR_DIR, seed_dir=SEED_DIR,
        method={"backend": "chain_binomial", "chains": N_CHAINS, "sweeps": N_SWEEPS, "burn_in": 300},
        estimated=PARAM_NAMES, meta=FIT_META, fit_toml=FIT_TOML, config="seir_patch_demo.toml",
    )

    header = ["sweep", *PARAM_NAMES, "log_likelihood", "log_posterior"]
    for c in range(N_CHAINS):
        rows = _trace_for_chain(rng, c)
        _write_tsv(seed_dir / f"chain_{c}" / "trace.tsv", header, rows)

    kit.beat(seed_dir, state="done")
    (seed_dir / "run_inputs.json").write_text(json.dumps({"algorithm": ALGORITHM}, indent=2) + "\n")
    summary = _summary(rng)
    (seed_dir / "pgas_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (seed_dir / "events.jsonl").write_text(_events(summary))
    _predictive_observed(rng, kit.prediction(seed_dir, name=DEFAULT_PREDICT), {}, 20)
    _predictive_observed(rng, kit.prediction(seed_dir, name=KEYED_PREDICT, options=KEYED_OPTIONS),
                         KEYED_OPTIONS, 30)
    return seed_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                    help=f"output store dir (default: {_DEFAULT_OUT})")
    args = ap.parse_args()
    run_dir = build(args.out)
    print(f"wrote golden store: {run_dir}")


if __name__ == "__main__":
    main()
