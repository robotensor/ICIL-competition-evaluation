"""DrawAnything-Sim (pygame/pymunk): BPP's drawing domain."""

from __future__ import annotations

from typing import Any

from ...model.architectures import register as register_architecture
from .. import Simulator, register

#: The architecture in `arch/` this simulator's policy implements.
ARCHITECTURE = "bpp_draw_v1"


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


#: What each drawing demonstration array carries. `boundary_angle` and `drawing` are the target
#: the episode is scored against, not something the demonstrator observed, so they belong to no
#: channel and reach no policy.
DEMO_CHANNELS = {
    "video": ("image",),
    "proprio": ("agent_pos", "pen_down"),
    "actions": ("actions",),
}


register_architecture(ARCHITECTURE, _make_policy)

SIMULATOR = register(
    Simulator(
        name="draw",
        run_units=_run_units,
        build_stage=_build_stage,
        make_unit=_make_unit,
        demo_frames=_demo_frames,
        validate_skill=_validate_skill,
        demo_channels=DEMO_CHANNELS,
    )
)
