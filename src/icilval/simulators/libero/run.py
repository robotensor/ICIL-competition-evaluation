"""Run a skill's units on LIBERO: one environment per task scene (and joint-noise setting), units grouped by it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...pools.demos import load_demo
from ...pools.schema import Pool
from ...spec import Spec
from ...video import VideoWriter


def run_units(
    ctx: Any,
    skill: str,
    policy: Any,
    pool: Pool,
    units: list[dict[str, Any]],
    spec: Spec,
    media_dir: Path,
    record_video: bool,
) -> None:
    from .changes import init_noise_of
    from .env import LiberoEnv, load_init_states
    from .episode import run_episode

    video_cfg = spec.media["video"]
    fps = int(spec.env(skill)["control_freq"])
    env: LiberoEnv | None = None
    env_key: tuple[str, float | None] | None = None
    init_cache: dict[str, np.ndarray] = {}

    def key_of(unit: dict[str, Any]) -> tuple[str, float | None]:
        # a scene, built with the joint noise the unit's change asks for (if any)
        return (str(unit["bddl"]), init_noise_of(unit.get("change")))

    try:
        # keep env switches rare: run units grouped by scene and noise, in unit order within
        order = sorted(
            range(len(units)), key=lambda i: (key_of(units[i])[0], str(key_of(units[i])[1]), i)
        )
        for i in order:
            unit = units[i]
            if ctx.out_of_time():
                ctx.finish(unit, ctx.timed_out(unit), None)
                continue
            key = key_of(unit)
            if env is None or env_key != key:
                if env is not None:
                    env.close()
                env = LiberoEnv(
                    pool.path(unit["bddl"]), spec, skill=skill, init_noise_magnitude=key[1]
                )
                env_key = key
            init_state = None
            if unit.get("init"):
                if unit["init"] not in init_cache:
                    init_cache[unit["init"]] = load_init_states(pool.path(unit["init"]))
                init_state = init_cache[unit["init"]][int(unit["instance"])]
            demo = load_demo(pool.path("demos") / f"{unit['demo']}.npz")
            clip = media_dir / f"{unit['unit_id']}.mp4"
            writer = VideoWriter(clip, fps, video_cfg) if record_video else None
            try:
                res = run_episode(
                    env, policy, unit, init_state, demo, spec, video=writer, executor=ctx.executor
                )
            finally:
                ctx.close_writer(writer, unit)
            ctx.finish(unit, ctx.record(unit, res, clip), res)
    finally:
        if env is not None:
            env.close()
