"""LIBERO environment for one task: build from BDDL, restore an initial state, step, observe, render.

Conventions match BPP's rollout: OSC_POSE delta control at 20 Hz, 128 px cameras,
images flipped to upright. Simulator imports are function-local.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import Spec

log = logging.getLogger(__name__)

RESET_ATTEMPTS = 20


def load_init_states(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["states"], dtype=np.float64)


def save_init_states(path: str | Path, states: np.ndarray, source_sha256: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, states=np.asarray(states, dtype=np.float64), source_sha256=np.array(source_sha256)
    )


class LiberoEnv:
    def __init__(
        self,
        bddl_path: str | Path,
        spec: Spec,
        *,
        skill: str,
        render_size: int | None = None,
        gpu_id: int = -1,
    ):
        from libero.libero.envs import OffScreenRenderEnv

        env_cfg = spec.env(skill)
        self.skill = skill
        self.bddl_path = str(bddl_path)
        self.camera_res = int(env_cfg["camera_resolution"])
        self.render_size = int(render_size or spec.media["video"]["resolution"])
        self.render_camera = str(spec.media["video"]["camera"])
        self.env = OffScreenRenderEnv(
            bddl_file_name=self.bddl_path,
            robots=["Panda"],
            controller=str(env_cfg["controller"]),
            control_freq=int(env_cfg["control_freq"]),
            camera_names=list(env_cfg["cameras"]),
            camera_heights=self.camera_res,
            camera_widths=self.camera_res,
            horizon=10_000_000,
            ignore_done=True,
            hard_reset=False,
            render_gpu_device_id=gpu_id,
        )
        self.raw = self.env.env  # the robosuite/LIBERO problem instance
        self.language = self.env.language_instruction
        self.goal_state = list(self.raw.parsed_problem["goal_state"])

    # ---------------------------------------------------------------- lifecycle
    def reset(self, seed: int, init_state: np.ndarray | None) -> dict[str, Any]:
        from robosuite.utils.errors import RandomizationError

        self.env.seed(int(seed))
        last: Exception | None = None
        for _ in range(RESET_ATTEMPTS):
            try:
                raw = self.raw.reset()
                break
            except RandomizationError as exc:  # placement sampler failed; try again
                last = exc
        else:
            raise RuntimeError(f"env reset failed {RESET_ATTEMPTS} times: {last}")
        if init_state is not None:
            raw = self.env.set_init_state(np.asarray(init_state, dtype=np.float64))
        return self.observe(raw)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001 - best effort
            pass

    # ---------------------------------------------------------------- stepping
    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool]:
        raw, reward, done, _ = self.env.step(np.asarray(action, dtype=np.float64))
        return self.observe(raw), float(reward), bool(done)

    def observe(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "agentview": np.ascontiguousarray(raw["agentview_image"][::-1]).astype(np.uint8),
            "eye_in_hand": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1]).astype(
                np.uint8
            ),
            "ee_pos": np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
            "ee_quat": np.asarray(raw["robot0_eef_quat"], dtype=np.float32),
            "gripper": np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
        }

    def success(self) -> bool:
        return bool(self.env.check_success())

    def goal_status(self) -> list[bool]:
        return [bool(self.raw._eval_predicate(state)) for state in self.goal_state]

    def render(self) -> np.ndarray:
        frame = self.env.sim.render(
            camera_name=self.render_camera, width=self.render_size, height=self.render_size
        )
        return np.ascontiguousarray(frame[::-1]).astype(np.uint8)
