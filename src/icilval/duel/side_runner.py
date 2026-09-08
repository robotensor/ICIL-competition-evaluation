"""Run one side of a duel over a unit list. Resumable; runs where the simulators and torch live.

Units are grouped by skill (spec order). For each skill the side's checkpoint for that skill
is loaded from `<model_dir>/<skill>`, its units run on its simulator, and the model is
unloaded before the next skill's is loaded.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..canon import sha256_file
from ..model.bpp import make_policy
from ..pools.demos import load_demo
from ..pools.schema import Pool
from ..sim.video import VideoWriter
from ..spec import Spec

log = logging.getLogger(__name__)


def read_results(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        out[rec["unit_id"]] = rec
    return out


def run_side(
    *,
    side: str,
    model_dir: Path,
    arch_dir: Path,
    pool: Pool,
    units: list[dict[str, Any]],
    spec: Spec,
    out_dir: Path,
    device: str = "cuda",
    on_unit: Callable[[dict[str, Any]], None] | None = None,
    record_video: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    media_dir.mkdir(exist_ok=True)
    results_path = out_dir / "units.jsonl"
    done = read_results(results_path)
    t_start = time.monotonic()
    side_wall = float(spec.budgets["side_wall_seconds"])
    summary: dict[str, Any] = {
        "side": side,
        "units": len(units),
        "ran": 0,
        "resumed": len(done),
        "void": 0,
        "success": 0,
        "errors": 0,
        "load_seconds": {},
        "skills": {},
    }

    class _Ctx:
        def __init__(self) -> None:
            self.t_start = t_start
            self.side_wall = side_wall
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def out_of_time(self) -> bool:
            return time.monotonic() - self.t_start > self.side_wall

        def finish(self, unit: dict[str, Any], rec: dict[str, Any], res: Any | None) -> None:
            _append(results_path, rec)
            done[rec["unit_id"]] = rec
            skill_summary = summary["skills"].setdefault(
                unit["skill"], {"ran": 0, "void": 0, "success": 0}
            )
            summary["ran"] += 1
            skill_summary["ran"] += 1
            summary["void"] += int(bool(rec.get("void")))
            skill_summary["void"] += int(bool(rec.get("void")))
            summary["success"] += int(bool(rec.get("success")))
            skill_summary["success"] += int(bool(rec.get("success")))
            if res is not None:
                summary["errors"] += res.model_errors
                log.info(
                    "%s %s: success=%s metric=%s steps=%d wall=%.1fs%s",
                    side,
                    unit["unit_id"],
                    res.success,
                    res.metric,
                    res.steps,
                    res.wall_s,
                    f" error={res.error}" if res.error else "",
                )
            if on_unit is not None:
                on_unit(rec)

    ctx = _Ctx()
    try:
        for skill in spec.skills:
            todo = [u for u in units if u["skill"] == skill and u["unit_id"] not in done]
            if not todo:
                continue
            policy = make_policy(model_dir / skill, arch_dir, spec, skill, device=device)
            policy.load()
            summary["load_seconds"][skill] = round(policy.load_seconds, 1)
            try:
                if spec.simulator(skill) == "draw":
                    _run_draw(ctx, skill, policy, pool, todo, spec, media_dir, record_video)
                else:
                    _run_libero(ctx, skill, policy, pool, todo, spec, media_dir, record_video)
            finally:
                policy.unload()
    finally:
        ctx.executor.shutdown(wait=False)
    summary["wall_seconds"] = round(time.monotonic() - t_start, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _record(unit: dict[str, Any], res: Any, clip: Path, out_dir: Path, record_video: bool) -> dict:
    rec = {
        "unit_id": unit["unit_id"],
        "skill": unit["skill"],
        "success": None if res.void else bool(res.success),
        "progress": res.progress,
        "metric": res.metric,
        "first_step_done_at": res.first_step_done_at,
        "steps": res.steps,
        "model_errors": res.model_errors,
        "wall_s": round(res.wall_s, 2),
        "error": res.error,
        "void": res.void,
        "prompt_steps": res.prompt_steps,
        "prompt_chunks": res.prompt_chunks,
        "instance_applied": res.instance_applied,
        "video": None,
        "video_sha256": None,
    }
    if record_video and clip.exists() and clip.stat().st_size > 0 and not res.void:
        rec["video"] = str(clip.relative_to(out_dir))
        rec["video_sha256"] = sha256_file(clip)
    return rec


def _timed_out(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit_id": unit["unit_id"],
        "skill": unit["skill"],
        "void": True,
        "error": "side wall time exceeded",
        "success": None,
        "metric": None,
    }


def _run_libero(
    ctx: Any,
    skill: str,
    policy: Any,
    pool: Pool,
    units: list[dict[str, Any]],
    spec: Spec,
    media_dir: Path,
    record_video: bool,
) -> None:
    from ..sim.episode import run_episode
    from ..sim.libero_env import LiberoEnv, load_init_states

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
                ctx.finish(unit, _timed_out(unit), None)
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
                _close_writer(writer, unit)
            ctx.finish(unit, _record(unit, res, clip, media_dir.parent, record_video), res)
    finally:
        if env is not None:
            env.close()


def _run_draw(
    ctx: Any,
    skill: str,
    policy: Any,
    pool: Pool,
    units: list[dict[str, Any]],
    spec: Spec,
    media_dir: Path,
    record_video: bool,
) -> None:
    from ..sim.draw_env import DrawBoard
    from ..sim.draw_episode import run_draw_episode

    video_cfg = spec.media["video"]
    fps = int(spec.env(skill)["control_freq"])
    board = DrawBoard(spec, skill)
    try:
        for unit in units:
            if ctx.out_of_time():
                ctx.finish(unit, _timed_out(unit), None)
                continue
            demo = load_demo(pool.path("demos") / f"{unit['demo']}.npz")
            clip = media_dir / f"{unit['unit_id']}.mp4"
            writer = VideoWriter(clip, fps, video_cfg) if record_video else None
            try:
                res = run_draw_episode(
                    board, policy, unit, demo, spec, video=writer, executor=ctx.executor
                )
            finally:
                _close_writer(writer, unit)
            ctx.finish(unit, _record(unit, res, clip, media_dir.parent, record_video), res)
    finally:
        board.close()


def _close_writer(writer: VideoWriter | None, unit: dict[str, Any]) -> None:
    if writer is None:
        return
    try:
        writer.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("video encode failed for %s: %s", unit["unit_id"], exc)


def _append(path: Path, rec: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
