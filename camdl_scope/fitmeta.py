"""Where a fit's model and config files are, from ``fit.meta.json``.

camdl records the two source paths under different names and bases by schema:

* version 2 (``schema_version: 2``): ``source_model`` and ``source_config``,
  each a path relative to the *fit folder* that holds ``fit.meta.json``, so
  ``<fit folder>/<source_model>`` names the file from any working directory.
* version 1 (no ``schema_version`` key): ``model_path`` and ``fit_toml_path``,
  relative to the directory ``fit run`` was started from, which for a store at
  ``<root>/results/fits`` is the project root (or absolute).

Any other version is refused by name. The archived copies in the fit folder
(``model.camdl.original``, ``model.ir.json``, ``fit.toml.original``) are
preferred by the readers; this module only resolves the *live* source.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .verdict import UnsupportedSchema

FITMETA_VERSIONS_KNOWN = (1, 2)


def schema_version(meta: dict[str, Any]) -> int:
    """The record's schema version: 1 when the key is absent. Refuses another."""
    v = meta.get("schema_version", 1)
    if v not in FITMETA_VERSIONS_KNOWN or isinstance(v, bool):
        raise UnsupportedSchema(
            f"fit.meta.json schema_version {v!r} is not supported; camdl-scope reads "
            f"versions {', '.join(map(str, FITMETA_VERSIONS_KNOWN))}"
        )
    return v


def project_root(store: Path) -> Path:
    """The directory a version 1 record's relative paths resolve against: the
    store is ``<root>/results/fits``, so two levels up."""
    return store.parent.parent if store.name == "fits" else store.parent


def model_ref(meta: dict[str, Any]) -> str:
    """The recorded model path as written (for naming), or ``""`` when none or
    when the record's version is not one this reader knows."""
    try:
        key = "source_model" if schema_version(meta) == 2 else "model_path"
    except UnsupportedSchema:
        return ""
    return str(meta.get(key) or "")


def config_ref(meta: dict[str, Any]) -> str:
    """The recorded producing-config path as written, or ``""``."""
    try:
        key = "source_config" if schema_version(meta) == 2 else "fit_toml_path"
    except UnsupportedSchema:
        return ""
    return str(meta.get(key) or "")


def source_model_path(meta: dict[str, Any], fit_folder: Path, store: Path) -> Path | None:
    """The live model file the record names, resolved by the rule of its
    version; ``None`` when none is recorded. Raises :class:`UnsupportedSchema`
    for an unknown version. The file may no longer exist."""
    ref = model_ref(meta) if schema_version(meta) in FITMETA_VERSIONS_KNOWN else ""
    if not ref:
        return None
    p = Path(ref)
    if p.is_absolute():
        return p
    if schema_version(meta) == 2:
        return Path(os.path.normpath(fit_folder.resolve() / p))
    return project_root(store) / p
