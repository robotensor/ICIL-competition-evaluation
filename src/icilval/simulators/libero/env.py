"""LIBERO environment for one task: build from BDDL, restore an initial state, step, observe, render.

Conventions match BPP's rollout: OSC_POSE delta control at 20 Hz, 128 px cameras,
images flipped to upright. Simulator imports are function-local.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...spec import Spec

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
        init_noise_magnitude: float | None = None,
    ):
        from libero.libero.envs import OffScreenRenderEnv

        env_cfg = spec.env(skill)
        self.skill = skill
        self.bddl_path = str(bddl_path)
        self.camera_res = int(env_cfg["camera_resolution"])
        self.render_size = int(render_size or spec.media["video"]["resolution"])
        self.render_camera = str(spec.media["video"]["camera"])
        self.init_noise_magnitude = init_noise_magnitude
        kwargs: dict[str, Any] = {}
        if init_noise_magnitude is not None:
            # robosuite's joint initialization noise; LIBERO's default is 0.02 gaussian
            kwargs["initialization_noise"] = {
                "magnitude": float(init_noise_magnitude),
                "type": "gaussian",
            }
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
            **kwargs,
        )
        self.raw = self.env.env  # the robosuite/LIBERO problem instance
        self.language = self.env.language_instruction
        self.goal_state = list(self.raw.parsed_problem["goal_state"])
        self._model_id: int | None = None
        self._model_snapshot: dict[str, np.ndarray] = {}
        self._snapshot_model()

    # ---------------------------------------------------------------- lifecycle
    def reset(self, seed: int, init_state: np.ndarray | None) -> dict[str, Any]:
        from robosuite.utils.errors import RandomizationError

        self._restore_model()  # a previous unit's change never leaks into this one
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

    def current_observation(self) -> dict[str, Any]:
        """A fresh observation of the scene as it is now (after a change moved something)."""
        return self.observe(self.raw._get_observations(force_update=True))

    def success(self) -> bool:
        return bool(self.env.check_success())

    def goal_status(self) -> list[bool]:
        return [bool(self.raw._eval_predicate(state)) for state in self.goal_state]

    def render(self) -> np.ndarray:
        frame = self.env.sim.render(
            camera_name=self.render_camera, width=self.render_size, height=self.render_size
        )
        return np.ascontiguousarray(frame[::-1]).astype(np.uint8)

    # ---------------------------------------------------------------- introspection
    @property
    def sim(self) -> Any:
        return self.env.sim

    @property
    def mj_model(self) -> Any:
        return self.env.sim.model._model

    def movable_objects(self) -> list[str]:
        return list(self.raw.objects_dict.keys())

    def object_joint(self, name: str) -> str:
        obj = self.raw.objects_dict.get(name) or self.raw.fixtures_dict.get(name)
        if obj is None:
            raise KeyError(name)
        joints = list(getattr(obj, "joints", []) or [])
        if not joints:
            raise ValueError(f"{name} has no free joint")
        return joints[0]

    def object_position(self, name: str) -> np.ndarray:
        qpos = np.asarray(
            self.env.sim.data.get_joint_qpos(self.object_joint(name)), dtype=np.float64
        ).reshape(-1)
        if qpos.shape != (7,):
            raise ValueError(f"{name} is not a free-joint object")
        return qpos[:3]

    def camera_id(self, name: str) -> int:
        return int(self.env.sim.model.camera_name2id(name))

    def forward(self) -> None:
        self.env.sim.forward()

    def settle(self, n: int = 20) -> None:
        for _ in range(n):
            self.env.sim.step()
        self.env.sim.forward()

    def sim_state(self) -> np.ndarray:
        return np.asarray(self.env.get_sim_state(), dtype=np.float64)

    def set_sim_state(self, state: np.ndarray) -> None:
        self.env.set_init_state(np.asarray(state, dtype=np.float64))

    # ---------------------------------------------------------------- model snapshot
    MODEL_FIELDS = (
        "light_active",
        "light_pos",
        "light_dir",
        "light_diffuse",
        "light_ambient",
        "light_specular",
        "cam_pos",
        "cam_quat",
    )

    def _snapshot_model(self) -> None:
        """The lights and cameras as the model was built; `reset` puts them back."""
        m = self.mj_model
        self._model_id = id(m)
        snap = {name: np.array(getattr(m, name), copy=True) for name in self.MODEL_FIELDS}
        snap["headlight_diffuse"] = np.array(m.vis.headlight.diffuse, copy=True)
        snap["headlight_ambient"] = np.array(m.vis.headlight.ambient, copy=True)
        self._model_snapshot = snap

    def _restore_model(self) -> None:
        m = self.mj_model
        if id(m) != self._model_id:  # the model was rebuilt: it is pristine
            self._snapshot_model()
            return
        for name in self.MODEL_FIELDS:
            getattr(m, name)[:] = self._model_snapshot[name]
        m.vis.headlight.diffuse[:] = self._model_snapshot["headlight_diffuse"]
        m.vis.headlight.ambient[:] = self._model_snapshot["headlight_ambient"]
        self.forward()  # light_xpos/xdir and camera frames are recomputed by mj_forward
