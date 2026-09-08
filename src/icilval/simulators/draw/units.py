"""A drawing unit's instance: a board angle and a pen start derived from the ids."""

from __future__ import annotations

from typing import Any

from ...ids import unit_id
from ...pools.schema import PoolTask
from ...pools.units import Unit, max_steps_of
from ...rng import HashRng
from ...spec import Spec


def draw_instance(task_id: str, instance: int, env: dict[str, Any]) -> dict[str, Any]:
    """The board angle and cursor start of one drawing instance: a pure function of the ids."""
    rng = HashRng("draw-instance", task_id, instance)
    lo, hi = (float(x) for x in env["board_angle_range_rad"])
    c_lo, c_hi = (int(x) for x in env["cursor_start_range_px"])
    return {
        "angle_rad": round(rng.uniform(lo, hi), 6),
        "cursor_px": [c_lo + rng.below(c_hi - c_lo + 1), c_lo + rng.below(c_hi - c_lo + 1)],
    }


def draw_unit(spec: Spec, skill: str, index: int, task: PoolTask, seed: int, rng: HashRng):
    """A board angle and pen start derived from the ids. A generated task's prompt is generated
    for the unit later and carries the task's primitive family; a diagnostic task is prompted
    with one of its stored demonstrations."""
    env = spec.env(skill)
    valid = task.valid_instances
    if not valid:
        raise ValueError(f"{task.task_id} has no instances")
    instance = valid[rng.below(len(valid))]
    state = draw_instance(task.task_id, instance, env)
    uid = unit_id(spec.skill_code(skill), index)
    params: dict[str, Any] = {"angle_rad": state["angle_rad"], "cursor_px": state["cursor_px"]}
    if task.diagnostic:
        if not task.demos:
            raise ValueError(f"task {task.task_id} has no demonstrations")
        demo = task.demos[rng.below(len(task.demos))]
        demo_angle = float(task.meta.get("demo_angles", {}).get(demo, 0.0))
        params["demo_angle_rad"] = round(demo_angle, 6)
    else:
        demo = f"generated/{uid}"
        params["family"] = str(task.meta.get("family", ""))
    return Unit(
        unit_id=uid,
        skill=skill,
        index=index,
        task=task.task_id,
        task_label=task.label,
        instance=instance,
        demo=demo,
        seed=seed,
        instance_params=params,
        max_steps=max_steps_of(task, spec, skill),
        bddl=None,
        init=None,
        goal=[],
        steps=[],
        diagnostic=task.diagnostic,
    )
