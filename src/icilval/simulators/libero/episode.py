"""One scored LIBERO episode: restore the initial state, prompt the policy with one
demonstration, run action chunks until success or the step cap, record video. `EpisodeResult` is shared with the drawing episode (`draw_episode.py`).
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections import deque
from typing import Any

import numpy as np

from ...spec import Spec
from ...video import VideoWriter
from ..result import EpisodeResult
from .env import LiberoEnv

log = logging.getLogger(__name__)

OPEN_GRIPPER = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)


def _progress(status: list[bool], initially: list[bool]) -> float | None:
    n_goal = len(status)
    if n_goal < 2:
        return None
    base = sum(initially)
    denom = n_goal - base
    if denom <= 0:
        return None
    return max(0.0, (sum(status) - base) / denom)


def run_episode(
    env: LiberoEnv,
    policy: Any,
    unit: dict[str, Any],
    init_state: np.ndarray,
    demo: dict[str, Any],
    spec: Spec,
    *,
    video: VideoWriter | None = None,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> EpisodeResult:
    budgets = spec.budgets
    env_cfg = spec.env(unit["skill"])
    exec_h = int(env_cfg["exec_horizon"])
    warmup = int(env_cfg["warmup_open_gripper_steps"])
    max_steps = int(unit.get("max_steps") or spec.max_steps(unit["skill"]))
    soft_t = float(budgets["act_soft_timeout_s"])
    hard_t = float(budgets["act_hard_timeout_s"])
    max_errors = int(budgets["max_model_errors_per_episode"])
    unit_wall = float(budgets["unit_wall_seconds"])
    result = EpisodeResult()
    t0 = time.monotonic()
    own_executor = executor is None
    executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        policy.seed(int(unit["seed"]))
        obs = env.reset(int(unit["seed"]), init_state)
        result.instance_applied = {"init_state_index": int(unit["instance"])}
        # prompt
        info = policy.set_prompt(demo)
        result.prompt_steps, result.prompt_chunks = info.steps, info.chunks
        # warm-up: open gripper, no motion (as BPP does)
        for _ in range(warmup):
            obs, _, _ = env.step(OPEN_GRIPPER)
        initially = env.goal_status()
        history: deque = deque(maxlen=int(env_cfg["obs_history"]))
        history.append(obs)
        last_gripper = -1.0
        best_progress = _progress(initially, initially)
        steps = 0
        done = False
        while not done and steps < max_steps:
            if time.monotonic() - t0 > unit_wall:
                result.error = "unit wall time exceeded"
                break
            fut = executor.submit(policy.act, list(history))
            try:
                t_act = time.monotonic()
                actions = fut.result(timeout=hard_t)
                if time.monotonic() - t_act > soft_t:
                    log.info("slow act: %.2fs", time.monotonic() - t_act)
            except concurrent.futures.TimeoutError:
                result.model_errors += 1
                actions = np.tile(np.array([0, 0, 0, 0, 0, 0, last_gripper]), (exec_h, 1))
                log.warning("act timed out (%d)", result.model_errors)
            except Exception as exc:  # noqa: BLE001 - model failures are scored as holds
                result.model_errors += 1
                actions = np.tile(np.array([0, 0, 0, 0, 0, 0, last_gripper]), (exec_h, 1))
                log.warning("act failed (%d): %s", result.model_errors, exc)
            if result.model_errors > max_errors:
                result.error = "too many model errors"
                break
            for a in np.asarray(actions, dtype=np.float64)[:exec_h]:
                obs, _, _ = env.step(a)
                last_gripper = float(a[6])
                history.append(obs)
                steps += 1
                if video is not None:
                    video.write(env.render())
                    result.video_frames += 1
                status = env.goal_status()
                prog = _progress(status, initially)
                if prog is not None:
                    best_progress = prog if best_progress is None else max(best_progress, prog)
                    if result.first_step_done_at is None and sum(status) > sum(initially):
                        result.first_step_done_at = steps
                if env.success():
                    result.success = True
                    done = True
                    break
                if steps >= max_steps:
                    break
        result.steps = steps
        result.progress = 1.0 if result.success else best_progress
    except Exception as exc:  # noqa: BLE001 - infrastructure failure voids the unit
        result.void = True
        result.error = f"{type(exc).__name__}: {exc}"[:300]
        log.exception("episode failed")
    finally:
        if own_executor:
            executor.shutdown(wait=False)
        result.wall_s = time.monotonic() - t0
    return result
