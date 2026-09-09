"""A LIBERO unit: a catalogue task, a numbered reset of its scene, a prompt generated for it."""

from __future__ import annotations

from ...ids import unit_id
from ...pools.schema import PoolTask
from ...pools.units import Unit, max_steps_of
from ...rng import HashRng
from ...spec import Spec
from .changes import sample_change


def libero_unit(spec: Spec, skill: str, index: int, task: PoolTask, seed: int, rng: HashRng):
    """One of the task's `n_init` numbered resets and one change from the skill's menu; the
    prompt is generated for the unit later."""
    valid = task.valid_instances
    if not valid:
        raise ValueError(f"{task.task_id} has no instances")
    instance = valid[rng.below(len(valid))]
    uid = unit_id(spec.skill_code(skill), index)
    change = sample_change(spec, skill, task.steps, rng)
    return Unit(
        unit_id=uid,
        skill=skill,
        index=index,
        task=task.task_id,
        task_label=task.label,
        instance=instance,
        demo=f"generated/{uid}",
        seed=seed,
        instance_params={},
        max_steps=max_steps_of(task, spec, skill),
        bddl=task.bddl,
        init=task.init,
        goal=task.goal,
        steps=task.steps,
        change=change,
    )
