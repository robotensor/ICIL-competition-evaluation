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


def _generate_prompt(unit, task, pool, spec, seeds, out_npz, ctx) -> dict[str, Any]:
    from .generate import generate_prompt

    res = generate_prompt(
        task,
        pool,
        spec,
        unit["skill"],
        seeds,
        out_npz,
        bpp_root=ctx.bpp_root,
        work_dir=out_npz.parent / "work" / unit["unit_id"],
        demo_id=unit["demo"],
    )
    return res.as_dict()


def _finalize_unit(unit: dict[str, Any], demo: dict[str, Any], spec: Any) -> dict[str, Any]:
    return dict(unit)  # the scored reset is the unit's instance whatever the prompt shows


SIMULATOR = register(
    Simulator(
        name="libero",
        make_policy=_make_policy,
        run_units=_run_units,
        build_stage=_build_stage,
        make_unit=_make_unit,
        demo_frames=_demo_frames,
        validate_skill=_validate_skill,
        generate_prompt=_generate_prompt,
        finalize_unit=_finalize_unit,
    )
)
