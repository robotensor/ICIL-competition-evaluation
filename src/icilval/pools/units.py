"""Deterministic unit lists.

A unit is one paired episode: skill, task, initial-state index, prompt
demonstration and seed. Both sides of a duel run the same list. The list is a
pure function of the pool and the duel id, so a third party holding the pool
can regenerate it from the published record.

A skill's units are spread evenly over its eligible tasks, in the pool's order,
shuffled by the duel id. A LIBERO task's initial state is one of its benchmark
init states; a drawing task's is a board angle and a pen start derived from
the task id and instance index, in the ranges BPP's `DrawEnv` samples from.
The prompt demonstration never starts from the scored initial state.
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
def derive_units(pool: Pool, spec: Spec, duel: str, size: str | None = None) -> list[Unit]:
    per_skill = spec.units_per_skill(size)
    out: list[Unit] = []
    for skill in spec.skills:
        rng = HashRng(duel, skill)
        entries = pool.eligible(skill)
        if per_skill and not entries:
            raise ValueError(f"pool has no eligible tasks for {skill}")
        for index, entry in enumerate(_spread(per_skill, entries, rng)):
            out.append(_unit(pool, spec, duel, skill, index, entry, rng))
    return out


def _unit(pool: Pool, spec: Spec, duel: str, skill: str, index: int, entry: str, rng: HashRng):
    task = pool.tasks[entry]
    seed = unit_seed(duel, skill, index)
    return for_skill(spec, skill).make_unit(spec, skill, index, task, seed, rng)


def max_steps_of(task: PoolTask, spec: Spec, skill: str) -> int:
    return min(task.max_steps, spec.max_steps(skill)) if task.max_steps else spec.max_steps(skill)
