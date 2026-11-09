"""Parameter grouping — pure, dependency-free.

Hierarchical fits expand an indexed model parameter (``k_raw[patch]``) into one
estimated coordinate per stratum (``k_raw_Bo``, ``k_raw_Kenema``, …). That can be
14+ leaves, which turns the pair plot into a 22x22 monster and clutters the trace
grid and diagnostics table.

This module detects those indexed *families* so the UI can collapse them by
default and let the user toggle whole families or individual leaves on/off.

A **family** is a base name ``B`` such that two or more estimated coordinates
match ``B_<stratum>``. The base is the longest such prefix, so ``k_raw_Bo`` and
``k_raw_Western_Area_Urban`` both group under ``k_raw`` (not ``k`` or ``k_raw_Western``).
A coordinate that is the *only* member of its base (e.g. ``mu_k`` -> base ``mu``)
is a **scalar**, not a family.

The default pair-plot selection shows scalars and hyperparameters (everything
that is not an indexed leaf) and hides the leaves — but falls back to the
parameters themselves when there are no scalars (a fully-indexed fit), and is
capped so a wide fit opens on a legible grid rather than an N² wall.
"""

from __future__ import annotations

from dataclasses import dataclass

# Cap the pair plot's default parameter count: N params → N²/2 scatter panels, so
# past ~8 the grid is a wall. The default opens legible; the ⚙ control adds more.
DEFAULT_PAIR_CAP = 8


@dataclass(frozen=True)
class ParamGroups:
    """The result of grouping a flat estimated-parameter list.

    * ``scalars`` — non-indexed coordinates, in original order.
    * ``families`` — ``{base: [member, ...]}`` for indexed families (>=2 members),
      members in original order, insertion-ordered by first appearance.
    """

    scalars: list[str]
    families: dict[str, list[str]]

    @property
    def family_names(self) -> list[str]:
        return list(self.families)

    def default_selection(self, cap: int = DEFAULT_PAIR_CAP) -> list[str]:
        """The pair plot's initial parameter set: scalars + hyperparameters
        (indexed-family leaves hidden), in original order.

        Two guards so the pair plot is neither empty nor a wall:
          * a *fully-indexed* fit has no scalars, so that set is empty — fall back
            to the parameters themselves (better an all-leaf grid than nothing);
          * cap the result at ``cap`` so a wide fit opens on a legible grid, not
            an N² monster. The user adds the rest via the ⚙ params control.
        """
        leaves = {m for members in self.families.values() for m in members}
        primary = [p for p in self._original_order() if p not in leaves]
        chosen = primary if len(primary) >= 2 else self._original_order()
        return chosen[:cap]

    def all_params(self) -> list[str]:
        return self._original_order()

    def _original_order(self) -> list[str]:
        # Reconstructed below by group_params via the stored order.
        return list(self._order)

    # populated by group_params; kept out of the public signature.
    _order: tuple[str, ...] = ()


def group_params(params: list[str]) -> ParamGroups:
    """Partition ``params`` into scalars and indexed families.

    Algorithm: for every candidate base ``B`` (each ``"<head>_<tail>"`` split of
    a name), count how many params match ``B_<something>``. A param is assigned
    to the valid base (>=2 members) with the *most* members, ties broken by the
    *shortest* base. The most-members rule groups all ``k_raw_<patch>`` under
    ``k_raw`` even when a stratum name itself contains underscores
    (``k_raw_Western_Area_Rural`` would otherwise spuriously split into a
    2-member ``k_raw_Western_Area`` sub-family under a longest-base rule).
    """
    order = tuple(params)
    pset = list(params)

    # All candidate bases: every prefix ending just before an underscore.
    # For "k_raw_Bo": candidates "k", "k_raw". For "mu_k": candidate "mu".
    def candidate_bases(name: str) -> list[str]:
        bases: list[str] = []
        idx = name.find("_")
        while idx != -1:
            bases.append(name[:idx])
            idx = name.find("_", idx + 1)
        return bases

    # Count members per candidate base across all params.
    base_members: dict[str, list[str]] = {}
    for p in pset:
        for b in candidate_bases(p):
            base_members.setdefault(b, [])
    for p in pset:
        for b in candidate_bases(p):
            # `p` is a member of base `b` iff p == b + "_" + suffix (suffix nonempty)
            if p.startswith(b + "_") and len(p) > len(b) + 1:
                base_members[b].append(p)

    # Keep only bases with >=2 members.
    valid_bases = {b: ms for b, ms in base_members.items() if len(ms) >= 2}

    # Assign each param to the valid base with the MOST members; ties broken by
    # the LONGER base. "Most members" stops an underscore-containing stratum
    # from splitting off a spurious sub-family (``k_raw`` has 14 members, the
    # accidental ``k_raw_Western_Area`` only 2). The longer-base tie-break keeps
    # the specific common prefix when counts match (``k_raw`` over bare ``k``
    # when every member is ``k_raw_*``).
    assigned: dict[str, str] = {}
    for p in pset:
        best: str | None = None
        best_key: tuple[int, int] | None = None
        for b in valid_bases:
            if p.startswith(b + "_") and len(p) > len(b) + 1:
                key = (len(valid_bases[b]), len(b))  # more members, then longer base
                if best_key is None or key > best_key:
                    best, best_key = b, key
        if best is not None:
            assigned[p] = best

    # Build families (insertion-ordered by first member appearance) and scalars.
    families: dict[str, list[str]] = {}
    scalars: list[str] = []
    for p in pset:
        base = assigned.get(p)
        if base is None:
            scalars.append(p)
        else:
            families.setdefault(base, []).append(p)

    # A family must still have >=2 *assigned* members (longest-base assignment
    # can in principle thin a base); demote singletons back to scalars.
    demoted = [b for b, ms in families.items() if len(ms) < 2]
    for b in demoted:
        scalars.extend(families.pop(b))
    # Restore original order for scalars.
    scalars = [p for p in pset if p in set(scalars)]

    pg = ParamGroups(scalars=scalars, families=families)
    object.__setattr__(pg, "_order", order)
    return pg
