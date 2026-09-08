"""One scored drawing episode: set the board to the unit's angle and cursor, show the target,
prompt the policy with one demonstration, run action chunks until the step cap or BPP's idle
stop, and keep the best Chamfer distance seen. Success is `best <= success.threshold`.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections import deque
from typing import Any

import numpy as np

from ..spec import Spec
from .draw_env import DrawBoard
from .episode import EpisodeResult
from .video import VideoWriter

log = logging.getLogger(__name__)


def run_draw_episode(
    board: DrawBoard,
    policy: Any,
    unit: dict[str, Any],
    demo: dict[str, Any],
    spec: Spec,
    *,
    video: VideoWriter | None = None,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> EpisodeResult:
    skill = unit["skill"]
    budgets = spec.budgets
    env_cfg = spec.env(skill)
    exec_h = int(env_cfg["exec_horizon"])
    idle_steps = int(env_cfg["idle_stop_steps"])
    idle_px = float(env_cfg["idle_stop_px"])
    threshold = float(spec.success(skill)["threshold"])
    max_steps = int(unit.get("max_steps") or spec.max_steps(skill))
    soft_t = float(budgets["act_soft_timeout_s"])
    hard_t = float(budgets["act_hard_timeout_s"])
    max_errors = int(budgets["max_model_errors_per_episode"])
    unit_wall = float(budgets["unit_wall_seconds"])
    result = EpisodeResult()
    t0 = time.monotonic()
    own_executor = executor is None
    executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        p = unit["instance_params"]
        policy.seed(int(unit["seed"]))
        obs = board.reset(
            int(unit["seed"]),
            angle=float(p["angle_rad"]),
            cursor=(int(p["cursor_px"][0]), int(p["cursor_px"][1])),
            target=np.asarray(demo["drawing"], dtype=bool),
            target_angle=float(demo["boundary_angle"]),
        )
        result.instance_applied = {
            "angle_rad": float(p["angle_rad"]),
            "cursor_px": [int(p["cursor_px"][0]), int(p["cursor_px"][1])],
        }
        info = policy.set_prompt(demo)
        result.prompt_steps, result.prompt_chunks = info.steps, info.chunks
        history: deque = deque(maxlen=int(env_cfg["obs_history"]))
        history.append(obs)
        best = board.chamfer()
        if video is not None:
            video.write(board.render())
            result.video_frames += 1
        steps = 0
        idle_for = 0
        idle_origin: np.ndarray | None = None
        done = False
        while not done and steps < max_steps:
            if time.monotonic() - t0 > unit_wall:
                result.error = "unit wall time exceeded"
                break
            fut = executor.submit(policy.act, list(history))
            hold = np.tile(
                np.array([*history[-1]["agent_pos"], 0.0], dtype=np.float64), (exec_h, 1)
            )
            try:
                t_act = time.monotonic()
                actions = fut.result(timeout=hard_t)
                if time.monotonic() - t_act > soft_t:
                    log.info("slow act: %.2fs", time.monotonic() - t_act)
            except concurrent.futures.TimeoutError:
                result.model_errors += 1
                actions = hold
                log.warning("act timed out (%d)", result.model_errors)
            except Exception as exc:  # noqa: BLE001 - model failures are scored as holds
                result.model_errors += 1
                actions = hold
                log.warning("act failed (%d): %s", result.model_errors, exc)
            if result.model_errors > max_errors:
                result.error = "too many model errors"
                break
            actions = np.asarray(actions, dtype=np.float64)[:exec_h]
            # BPP's idle stop: the predicted pen positions stay within idle_px of where the
            # current idle run began for idle_steps steps -> the drawing is finished
            if idle_origin is None:
                idle_origin = actions[0, :2].copy()
            moved = np.linalg.norm(actions[:, :2] - idle_origin, axis=1) >= idle_px
            if moved.any():
                idle_for = 0
                idle_origin = None
            else:
                idle_for += len(actions)
            for a in actions:
                obs, chamfer = board.step(a)
                history.append(obs)
                steps += 1
                best = min(best, chamfer)
                if video is not None:
                    video.write(board.render())
                    result.video_frames += 1
                if steps >= max_steps:
                    break
            if idle_for >= idle_steps:
                done = True
        result.steps = steps
        result.metric = round(float(best), 4)
        result.success = bool(np.isfinite(best) and best <= threshold)
        result.progress = None
    except Exception as exc:  # noqa: BLE001 - infrastructure failure voids the unit
        result.void = True
        result.error = f"{type(exc).__name__}: {exc}"[:300]
        log.exception("draw episode failed")
    finally:
        if own_executor:
            executor.shutdown(wait=False)
        result.wall_s = time.monotonic() - t0
    return result
