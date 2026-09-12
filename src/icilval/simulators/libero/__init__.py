"""LIBERO (MuJoCo, robosuite): BPP's pick-and-place and goal-chain domains."""

from __future__ import annotations

from typing import Any

from .. import Simulator, register


def _make_policy(model_dir: Any, arch_dir: Any, spec: Any, skill: str, device: str = "cuda"):
    from .policy import BPPPolicy

    return BPPPolicy(model_dir, arch_dir, spec, skill, device=device)


def _run_units(ctx, skill, policy, pool, units, spec, media_dir, record_video) -> None:
    from .run import run_units

    run_units(ctx, skill, policy, pool, units, spec, media_dir, record_video)


def _build_stage(pool, spec, src, skill, **kw) -> None:
    from .pool import stage_libero_gen

    stage_libero_gen(pool, spec, src, skill, **kw)


def _make_unit(spec, skill, index, task, seed, rng):
    from .units import libero_unit

    return libero_unit(spec, skill, index, task, seed, rng)


def _demo_frames(demo: dict[str, Any]) -> list[Any]:
    from .demos import demo_frames

    return demo_frames(demo)


def _validate_skill(skill: str, doc: dict[str, Any]) -> list[str]:
    env = doc.get("environment") or {}
    errors = []
    for key in ("cameras", "camera_resolution", "controller", "control_freq", "action_dim"):
        if key not in env:
            errors.append(f"skills.{skill}.environment.{key}")
    return errors


def _verify_pool_task(pool: Any, task: Any) -> list[str]:
    """A LIBERO task's BDDL must parse: `pools.build.verify_pool` only checks the file is there,
    and a BDDL that is present but unreadable fails the duel instead of the pool build."""
    if not task.bddl:
        return []
    from .bddl import load

    try:
        load(pool.path(task.bddl))
    except Exception as exc:  # noqa: BLE001
        return [f"bddl unparsable: {exc}"]
    return []


#: What each LIBERO demonstration array carries. `init_state` is deliberately in no channel: it
#: is the scene the demonstration started from, which the pool builder reads and no policy sees.
DEMO_CHANNELS = {
    "video": ("agentview", "eye_in_hand"),
    "proprio": ("ee_pos", "ee_ori", "gripper"),
    "actions": ("actions",),
}


SIMULATOR = register(
    Simulator(
        name="libero",
        make_policy=_make_policy,
        run_units=_run_units,
        build_stage=_build_stage,
        make_unit=_make_unit,
        demo_frames=_demo_frames,
        validate_skill=_validate_skill,
        verify_pool_task=_verify_pool_task,
        demo_channels=DEMO_CHANNELS,
    )
)
