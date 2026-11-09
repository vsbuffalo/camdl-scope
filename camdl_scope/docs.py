"""Model documentation — the ``#'`` declaration doc comments camdl surfaces.

A camdl model can document its declarations with ``#'`` doc comments, the
compiler folds them into a ``docs`` dictionary on the IR envelope, and each fit
carries a faithful projection of that dictionary in its ``fit.meta.json``
sidecar (``docs`` key, omitted when the model documents nothing).

The dictionary is keyed by *category* — ``parameters | compartments |
transitions | observations | dimensions`` — and within each by base
declaration name. Every entry is a :class:`DocBlock` (``text`` / ``symbol`` /
``ref``); the JSON spelling of the reference is ``"ref"`` (we expose it as
``reference``). These are read-only labels: a downstream consumer joins an
output column name (a posterior-draw parameter, a trajectory compartment, a
predict stream) against this index to render a human label and symbol.

This module is intentionally dependency-free (stdlib only) and knows nothing of
the run store — :mod:`camdl_scope.ingest` reads the meta and populates these;
nothing here imports ingest, so there is no cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The five doc categories, in the compiler's serialization order.
DOC_CATEGORIES: tuple[str, ...] = (
    "parameters",
    "compartments",
    "transitions",
    "observations",
    "dimensions",
)


def _as_str(v: object) -> str | None:
    """A doc field is a string or absent; coerce anything else to ``None`` so a
    malformed sidecar never injects a non-string into a label slot."""
    return v if isinstance(v, str) else None


@dataclass(frozen=True)
class DocBlock:
    """One declaration's ``#'`` doc: free ``text``, a display ``symbol``
    (``@symbol``, e.g. ``"β"``), and a literature ``reference`` (``@ref``; JSON
    key ``"ref"``). Each field is independently optional — an undocumented
    declaration has no block at all, and a partially documented one carries only
    the fields it set."""

    text: str | None = None
    symbol: str | None = None
    reference: str | None = None

    def is_empty(self) -> bool:
        return self.text is None and self.symbol is None and self.reference is None


@dataclass(frozen=True)
class ModelDocs:
    """The model's ``#'`` documentation dictionary: one ``{decl_name: DocBlock}``
    map per category. Empty by default, so a fit whose model documents nothing
    (or which predates the ``docs`` sidecar field) yields an empty envelope
    rather than ``None`` — callers can always join against it unconditionally."""

    parameters: dict[str, DocBlock] = field(default_factory=dict)
    compartments: dict[str, DocBlock] = field(default_factory=dict)
    transitions: dict[str, DocBlock] = field(default_factory=dict)
    observations: dict[str, DocBlock] = field(default_factory=dict)
    dimensions: dict[str, DocBlock] = field(default_factory=dict)
    quantities: dict[str, DocBlock] = field(default_factory=dict)

    @classmethod
    def from_meta(cls, meta: dict) -> "ModelDocs":
        """Read the ``docs`` block off a parsed ``fit.meta.json``.

        Tolerant of every absence: no ``docs`` key, a ``null`` value, a missing
        category, or a non-object entry all degrade to empty rather than raise.
        """
        docs = meta.get("docs")
        if not isinstance(docs, dict):
            return cls()

        def category(name: str) -> dict[str, DocBlock]:
            block = docs.get(name)
            if not isinstance(block, dict):
                return {}
            out: dict[str, DocBlock] = {}
            for decl, entry in block.items():
                if not isinstance(entry, dict):
                    continue
                out[str(decl)] = DocBlock(
                    text=_as_str(entry.get("text")),
                    symbol=_as_str(entry.get("symbol")),
                    reference=_as_str(entry.get("ref")),
                )
            return out

        return cls(
            parameters=category("parameters"),
            compartments=category("compartments"),
            transitions=category("transitions"),
            observations=category("observations"),
            dimensions=category("dimensions"),
            quantities=category("quantities"),
        )

    def is_empty(self) -> bool:
        return not (
            self.parameters
            or self.compartments
            or self.transitions
            or self.observations
            or self.dimensions
            or self.quantities
        )

    def for_quantity(self, name: str) -> DocBlock | None:
        """Resolve a generated-quantity name to its :class:`DocBlock` (exact
        match — quantities aren't index-expanded like params). ``None`` if
        undocumented."""
        return self.quantities.get(name)

    def for_param(self, coord: str) -> DocBlock | None:
        """Resolve an estimated *coordinate* to its parameter :class:`DocBlock`.

        Exact match first; else the longest base-name prefix, mirroring the
        ``<base>_<Level>`` expansion of indexed model params that
        :func:`camdl_scope.ingest._resolve_ir_for_param` resolves (``k_raw_Bo``
        inherits the doc of ``k_raw``). Returns ``None`` if nothing matches.
        """
        exact = self.parameters.get(coord)
        if exact is not None:
            return exact
        best: str | None = None
        for base in self.parameters:
            if coord.startswith(base + "_"):
                if best is None or len(base) > len(best):
                    best = base
        return self.parameters[best] if best is not None else None
