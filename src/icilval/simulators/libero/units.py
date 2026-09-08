"""A LIBERO unit's instance: one of the task's benchmark initial states."""

from __future__ import annotations

from ...ids import unit_id
from ...pools.schema import PoolTask
from ...pools.units import Unit, max_steps_of
from ...rng import HashRng
from ...spec import Spec


def libero_unit(spec: Spec, skill: str, index: int, task: PoolTask, seed: int, rng: HashRng):
    """One of the task's benchmark initial states, and a demonstration that did not start there."""
    valid = task.valid_instances
    if not valid:
        raise ValueError(f"{task.task_id} has no valid initial states")
    instance = valid[rng.below(len(valid))]
    candidates = [d for d in task.demos if task.demo_init_index.get(d) != instance] or list(
        task.demos
    )
    if not candidates:
        raise ValueError(f"task {task.task_id} has no demonstrations")
    demo = candidates[rng.below(len(candidates))]
    return Unit(
        unit_id=unit_id(spec.skill_code(skill), index),
        skill=skill,
        index=index,
        task=task.task_id,
        task_label=task.label,
        instance=instance,
        demo=demo,
        seed=seed,
        instance_params={},
        max_steps=max_steps_of(task, spec, skill),
        bddl=task.bddl,
        init=task.init,
        goal=task.goal,
        steps=task.steps,
    )
