"""LIBERO (MuJoCo, robosuite): BPP's pick-and-place and goal-chain domains."""

from __future__ import annotations

from typing import Any

from .. import Simulator, register


def _make_policy(model_dir: Any, arch_dir: Any, spec: Any, skill: str, device: str = "cuda"):
    from ...model.bpp import BPPPolicy

    return BPPPolicy(model_dir, arch_dir, spec, skill, device=device)


def _run_units(ctx, skill, policy, pool, units, spec, media_dir, record_video) -> None:
    from .run import run_units

    run_units(ctx, skill, policy, pool, units, spec, media_dir, record_video)


def _build_stage(pool, spec, src, skill, **kw) -> None:
    from ...pools.build import stage_libero_gen

    stage_libero_gen(pool, spec, src, skill, **kw)


def _make_unit(spec, skill, index, task, seed, rng):
    from ...pools.units import libero_unit

    return libero_unit(spec, skill, index, task, seed, rng)


def _demo_frames(demo: dict[str, Any]) -> list[Any]:
    from ...pools.demos import demo_frames

    return demo_frames(demo)


def _validate_skill(skill: str, doc: dict[str, Any]) -> list[str]:
    env = doc.get("environment") or {}
    errors = []
    for key in ("cameras", "camera_resolution", "controller", "control_freq", "action_dim"):
        if key not in env:
            errors.append(f"skills.{skill}.environment.{key}")
    return errors


SIMULATOR = register(
    Simulator(
        name="libero",
        make_policy=_make_policy,
        run_units=_run_units,
        build_stage=_build_stage,
        make_unit=_make_unit,
        demo_frames=_demo_frames,
        validate_skill=_validate_skill,
    )
)
