"""Deterministic unit lists.

A unit is one paired episode: skill, task, a numbered reset of the task's
scene, a prompt generated for the unit and a seed. Both sides of a duel run the
same list. The list is a pure function of the catalogue and the duel id, so a
third party holding the catalogue can regenerate it from the published record.

A skill's scored units are spread evenly over its eligible tasks, in the
catalogue's order, shuffled by the duel id; after them come the units of the
skill's unscored diagnostics, spread over the diagnostic tasks of each. A
LIBERO unit's instance numbers the reset its scored scene starts from; a
drawing unit's is a board angle and a pen start derived from the task id and
instance index, in the ranges BPP's `DrawEnv` samples from. The prompt never
starts from the scored state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..ids import unit_seed
from ..rng import HashRng
from ..simulators import for_skill
from ..spec import Spec
from .schema import Pool, PoolTask


@dataclass
class Unit:
    unit_id: str
    skill: str
    index: int
    task: str
    task_label: str
    instance: int
    demo: str
    seed: int
    instance_params: dict[str, Any]
    max_steps: int
    bddl: str | None
    init: str | None
    goal: list[list[str]] = field(default_factory=list)
    steps: list[list[str]] = field(default_factory=list)
    #: the one change applied to the scored scene: {"kind": <skills.<skill>.changes entry>, ...}
    change: dict[str, Any] = field(default_factory=lambda: {"kind": "none"})
    #: the task the unit was first derived for, when its prompt could not be generated
    substituted_from: str | None = None
    #: units of an unscored diagnostic never enter a score
    diagnostic: bool = False
    #: sha256 of the generated prompt once it exists
    prompt_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Unit:
        return cls(**d)


def _spread(n: int, entries: list[str], rng: HashRng) -> list[str]:
    """n picks over `entries`, no entry used more than ceil(n/len) times, order shuffled."""
    if not entries:
        return []
    order = rng.shuffled(entries)
    cap = math.ceil(n / len(order))
    picks: list[str] = []
    counts = dict.fromkeys(order, 0)
    i = 0
    while len(picks) < n:
        e = order[i % len(order)]
        if counts[e] < cap:
            picks.append(e)
            counts[e] += 1
        i += 1
    return picks


# ---------------------------------------------------------------- derivation
def diagnostic_units(spec: Spec, size: str | None, diag: dict[str, Any]) -> int:
    """`units_per_duel` at the default duel size, scaled with the size; at least one when any."""
    n = int(diag["units_per_duel"])
    if n == 0:
        return 0
    return max(1, round(n * spec.units_per_skill(size) / spec.units_per_skill(None)))


def derive_units(pool: Pool, spec: Spec, duel: str, size: str | None = None) -> list[Unit]:
    per_skill = spec.units_per_skill(size)
    out: list[Unit] = []
    for skill in spec.skills:
        rng = HashRng(duel, skill)
        entries = pool.eligible(skill)
        if per_skill and not entries:
            raise ValueError(f"catalogue has no eligible tasks for {skill}")
        index = 0
        for entry in _spread(per_skill, entries, rng):
            out.append(_unit(pool, spec, duel, skill, index, entry, rng))
            index += 1
        for name, diag in spec.diagnostics.items():
            if diag["skill"] != skill:
                continue
            n = diagnostic_units(spec, size, diag)
            tasks = [t for t in pool.diagnostic(skill) if t.startswith(f"{diag['group']}/")]
            if n and not tasks:
                raise ValueError(f"catalogue has no tasks for diagnostic {name}")
            for entry in _spread(n, tasks, rng):
                out.append(_unit(pool, spec, duel, skill, index, entry, rng))
                index += 1
    return out


def _unit(pool: Pool, spec: Spec, duel: str, skill: str, index: int, entry: str, rng: HashRng):
    task = pool.tasks[entry]
    seed = unit_seed(duel, skill, index)
    return for_skill(spec, skill).make_unit(spec, skill, index, task, seed, rng)


def max_steps_of(task: PoolTask, spec: Spec, skill: str) -> int:
    return min(task.max_steps, spec.max_steps(skill)) if task.max_steps else spec.max_steps(skill)
