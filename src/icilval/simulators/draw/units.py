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


def sample_draw_change(spec: Spec, skill: str, rng: HashRng) -> dict[str, Any]:
    """One entry of the skill's menu (`board_angle` or `pen_start`), uniform. Which quantity the
    scored board keeps from the demonstration follows from the kind once the prompt exists."""
    menu = sorted(spec.changes(skill))
    return {"kind": menu[rng.below(len(menu))]}


def draw_unit(spec: Spec, skill: str, index: int, task: PoolTask, seed: int, rng: HashRng):
    """A board angle and pen start derived from the ids, and one change from the menu. A
    generated task's prompt is generated for the unit later and carries the task's primitive
    family; a diagnostic task is prompted with one of its stored demonstrations."""
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
    change = {"kind": "none"} if task.diagnostic else sample_draw_change(spec, skill, rng)
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
        change=change,
        diagnostic=task.diagnostic,
    )


def finalize_draw_unit(unit: dict[str, Any], demo: dict[str, Any], spec: Spec) -> dict[str, Any]:
    """Once the prompt exists, the scored board keeps one of the demonstration's two quantities
    by the unit's change kind: `board_angle` turns the board to an angle at least `min_delta_rad`
    from the demonstration's (redrawn from the unit id until it is) and starts the pen where the
    demonstration's did; `pen_start` keeps the demonstration's angle and the sampled pen start."""
    skill = unit["skill"]
    env = spec.env(skill)
    params = dict(unit.get("instance_params") or {})
    kind = str((unit.get("change") or {}).get("kind", "none"))
    demo_angle = round(float(demo["boundary_angle"]), 6)
    start = [int(round(float(demo["agent_pos"][0][0]))), int(round(float(demo["agent_pos"][0][1])))]
    params["demo_angle_rad"] = demo_angle
    change = dict(unit.get("change") or {"kind": "none"})
    if kind == "board_angle":
        min_delta = float(spec.changes(skill)["board_angle"]["min_delta_rad"])
        lo, hi = (float(x) for x in env["board_angle_range_rad"])
        angle = float(params["angle_rad"])
        rng = HashRng("draw-angle", unit["unit_id"])
        for _ in range(1000):
            if abs(angle - demo_angle) >= min_delta:
                break
            angle = round(rng.uniform(lo, hi), 6)
        params["angle_rad"] = angle
        params["cursor_px"] = start
        change.update({"angle_rad": angle, "delta_rad": round(abs(angle - demo_angle), 6)})
    elif kind == "pen_start":
        params["angle_rad"] = demo_angle
        change.update({"cursor_px": list(params["cursor_px"])})
    return {**unit, "instance_params": params, "change": change}
