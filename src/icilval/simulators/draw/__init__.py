"""DrawAnything-Sim (pygame/pymunk): BPP's drawing domain."""

from __future__ import annotations

from typing import Any

from .. import Simulator, register


def _make_policy(model_dir: Any, arch_dir: Any, spec: Any, skill: str, device: str = "cuda"):
    from .policy import DrawPolicy

    return DrawPolicy(model_dir, arch_dir, spec, skill, device=device)


def _run_units(ctx, skill, policy, pool, units, spec, media_dir, record_video) -> None:
    from .run import run_units

    run_units(ctx, skill, policy, pool, units, spec, media_dir, record_video)


def _build_stage(pool, spec, src, skill, *, limit=None, fetch_missing=False, **_ignored) -> None:
    from .pool import stage_draw

    stage_draw(pool, spec, src, skill=skill, limit=limit, fetch_missing=fetch_missing)


def _make_unit(spec, skill, index, task, seed, rng):
    from .units import draw_unit

    return draw_unit(spec, skill, index, task, seed, rng)


def _demo_frames(demo: dict[str, Any]) -> list[Any]:
    from .demos import draw_demo_frames

    return draw_demo_frames(demo)


def _validate_skill(skill: str, doc: dict[str, Any]) -> list[str]:
    env = doc.get("environment") or {}
    errors = []
    for key in ("board_angle_range_rad", "cursor_start_range_px"):
        rng = env.get(key)
        ok = (
            isinstance(rng, list)
            and len(rng) == 2
            and all(isinstance(x, (int, float)) for x in rng)
            and rng[0] < rng[1]
        )
        if not ok:
            errors.append(f"skills.{skill}.environment.{key} range")
    success = doc.get("success") or {}
    if not (isinstance(success.get("threshold"), (int, float)) and success["threshold"] > 0):
        errors.append(f"skills.{skill}.success.threshold>0")
    return errors


def _generate_prompt(unit, task, pool, spec, seeds, out_npz, ctx) -> dict[str, Any]:
    from .generate import generate_drawing_prompt

    family = str(unit["instance_params"].get("family") or task.meta.get("family", ""))
    return generate_drawing_prompt(spec, unit["skill"], family, seeds, out_npz, unit["demo"])


def _finalize_unit(unit: dict[str, Any], demo: dict[str, Any], spec: Any) -> dict[str, Any]:
    from .units import finalize_draw_unit

    return finalize_draw_unit(unit, demo, spec)


SIMULATOR = register(
    Simulator(
        name="draw",
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
