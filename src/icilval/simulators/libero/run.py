"""Run a skill's units on LIBERO: one environment per task scene, units grouped by scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...pools.demos import load_demo
from ...pools.schema import Pool
from ...sim.video import VideoWriter
from ...spec import Spec


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
    from ...sim.episode import run_episode
    from ...sim.libero_env import LiberoEnv, load_init_states

    video_cfg = spec.media["video"]
    fps = int(spec.env(skill)["control_freq"])
    env: LiberoEnv | None = None
    env_key: str | None = None
    init_cache: dict[str, np.ndarray] = {}
    try:
        # keep env switches rare: run units grouped by task scene, in unit order within a group
        order = sorted(range(len(units)), key=lambda i: (units[i]["bddl"], i))
        for i in order:
            unit = units[i]
            if ctx.out_of_time():
                ctx.finish(unit, ctx.timed_out(unit), None)
                continue
            key = unit["bddl"]
            if env is None or env_key != key:
                if env is not None:
                    env.close()
                env = LiberoEnv(pool.path(unit["bddl"]), spec, skill=skill)
                env_key = key
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
