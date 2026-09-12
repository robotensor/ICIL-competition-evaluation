"""Unit lists for a field whose benchmark lives in another repository.

`pools/units.py` derives units from a pool: it knows what a LIBERO initial state is and what a
drawing board's angle ranges are, because those benchmarks ship here. A plugged benchmark's units
are not knowable that way - only the benchmark knows what one of its units *is* - so the ABI puts
`derive_units` on the plugin and this module calls it.

Two properties the orchestrator keeps for itself, because they are the competition's and not the
benchmark's:

- **The identity of a unit.** `unit_id` is `<skill code>-<index>`, the same shape every field
  uses, so a record reads the same whichever benchmark produced it and a benchmark cannot collide
  with another benchmark's ids.
- **The seed material.** Derivation must be reproducible by a third party holding the published
  record, so it is a pure function of the duel id and the skill - never of a clock, a global RNG,
  or anything the benchmark chooses for itself.

What the benchmark returns beyond that is passed through untouched. The orchestrator does not
know what a scene seed means and does not need to.
"""

from __future__ import annotations

from typing import Any

from ..ids import unit_id, unit_seed
from ..simulators import for_skill

#: Keys the orchestrator sets on every unit. A benchmark that returns one of them has it
#: overwritten rather than honoured: these are the competition's to decide.
RESERVED = ("unit_id", "skill", "index", "seed")


class DerivationError(RuntimeError):
    """A benchmark did not return the units its field asked for."""


def plugin_units(spec: Any, track: str, duel_id: str, size: str | None = None) -> list[dict]:
    """Every unit of one duel, from the benchmarks behind that field's skills.

    Skills in spec order, units in the order the benchmark returned them, so the list is a pure
    function of `(spec, track, duel_id, size)` exactly as the pool-derived one is.
    """
    out: list[dict[str, Any]] = []
    for skill in spec.skills(track):
        sim = for_skill(spec, skill)
        benchmark = getattr(sim, "benchmark", None)
        if benchmark is None:
            raise DerivationError(
                f"{skill}: {sim.name} ships in this repository and derives its units from a pool"
            )
        count = spec.units_per_skill(track, size)
        suite = spec.skill(skill).get("tasks", {}).get("suite", "v1")
        try:
            derived = benchmark.derive_units(
                seed_material=f"{duel_id}|{skill}", count=count, suite=suite
            )
        except Exception as exc:  # noqa: BLE001 - named, because a bare traceback here is useless
            raise DerivationError(f"{skill}: {sim.name} could not derive its units: {exc}") from exc

        derived = list(derived or [])
        if len(derived) != count:
            raise DerivationError(
                f"{skill}: {sim.name} returned {len(derived)} units, not the {count} "
                f"{track}/{size or spec.default_size(track)} asks for"
            )
        code = spec.skill_code(skill)
        for index, unit in enumerate(derived):
            if not isinstance(unit, dict):
                raise DerivationError(f"{skill}: unit {index} is {type(unit).__name__}, not a dict")
            out.append(
                {
                    **{k: v for k, v in unit.items() if k not in RESERVED},
                    "unit_id": unit_id(code, len(out)),
                    "skill": skill,
                    "index": index,
                    "seed": unit_seed(duel_id, skill, index),
                }
            )
    return out


def derives_its_own_units(spec: Any, track: str) -> bool:
    """Whether this field's skills come from plugged benchmarks rather than a pool."""
    skills = spec.skills(track)
    return bool(skills) and all(
        getattr(for_skill(spec, skill), "benchmark", None) is not None for skill in skills
    )
