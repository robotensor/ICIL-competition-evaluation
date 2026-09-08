"""Run a skill's units on the drawing board: one board for the whole skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    from .env import DrawBoard
    from .episode import run_draw_episode

    video_cfg = spec.media["video"]
    fps = int(spec.env(skill)["control_freq"])
    board = DrawBoard(spec, skill)
    try:
        for unit in units:
            if ctx.out_of_time():
                ctx.finish(unit, ctx.timed_out(unit), None)
                continue
            demo = load_demo(pool.path("demos") / f"{unit['demo']}.npz")
            clip = media_dir / f"{unit['unit_id']}.mp4"
            writer = VideoWriter(clip, fps, video_cfg) if record_video else None
            try:
                res = run_draw_episode(
                    board, policy, unit, demo, spec, video=writer, executor=ctx.executor
                )
            finally:
                ctx.close_writer(writer, unit)
            ctx.finish(unit, ctx.record(unit, res, clip), res)
    finally:
        board.close()
