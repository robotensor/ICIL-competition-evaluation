"""BPP inference for one unit: instantiate the allow-listed architecture, load weights strictly,
prompt with one demonstration, predict action chunks. Runs where behavior_prompting is
importable. `BPPPolicy` is the LIBERO (pick-and-place) policy; `draw.DrawPolicy` shares the
loading and tensor plumbing and differs in observation keys and action shape.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import Spec
from .prompt import PromptInfo, build_prompt
from .rotations import actions_10_to_7, quat_xyzw_to_rotation_6d

log = logging.getLogger(__name__)


class PolicyBase:
    """Load one skill's checkpoint from `<model_dir>` against `arch/<architecture>`."""

    def __init__(
        self,
        model_dir: str | Path,
        arch_dir: str | Path,
        spec: Spec,
        skill: str,
        device: str = "cuda",
    ):
        self.model_dir = Path(model_dir)
        self.arch_dir = Path(arch_dir)
        self.spec = spec
        self.skill = skill
        self.device = device
        env = spec.env(skill)
        self.architecture = spec.architecture(skill)
        self.image_size = int(env["policy_image_resolution"])
        self.obs_history = int(env["obs_history"])
        self.exec_horizon = int(env["exec_horizon"])
        self.action_horizon = int(env["action_horizon"])
        self.chunk_n = int(env["prompt_actions_per_chunk"])
        self.pad_end = str(env.get("prompt_pad_end", "zeros"))
        self.policy: Any = None
        self.max_chunks: int | None = None
        self.load_seconds = 0.0

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        import hydra
        import torch
        from safetensors.torch import load_file

        t0 = time.monotonic()
        template = json.loads((self.arch_dir / f"{self.architecture}.cfg.json").read_text())
        policy = hydra.utils.instantiate(template["model"])
        state = load_file(str(self.model_dir / "model.safetensors"))
        policy.load_state_dict(state, strict=True)
        policy.eval()
        policy.to(self.device)
        self.policy = policy
        try:
            self.max_chunks = int(policy.obs_encoder.prompt_pos_emb.shape[1])
        except AttributeError:
            self.max_chunks = None
        self.load_seconds = time.monotonic() - t0
        log.info(
            "loaded %s (%s) in %.1fs (max prompt chunks %s)",
            self.model_dir,
            self.architecture,
            self.load_seconds,
            self.max_chunks,
        )
        torch.cuda.synchronize() if self.device.startswith("cuda") else None

    def unload(self) -> None:
        import torch

        self.policy = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- tensors
    def _images(self, frames: np.ndarray) -> Any:
        """(N,H,W,3) uint8 upright -> (1,N,3,S,S) float in [0,1], bilinear like BPP."""
        import torch
        import torch.nn.functional as F

        x = (
            torch.from_numpy(np.ascontiguousarray(frames))
            .to(self.device)
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )
        if x.shape[-1] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        return x.unsqueeze(0)

    def _lowdim(self, arr: np.ndarray) -> Any:
        import torch

        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(self.device).unsqueeze(0)

    def _mask(self, mask: np.ndarray) -> Any:
        import torch

        return torch.from_numpy(np.asarray(mask, dtype=bool)).to(self.device).unsqueeze(0)

    def _history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not history:
            raise ValueError("empty observation history")
        obs = list(history)[-self.obs_history :]
        while len(obs) < self.obs_history:
            obs.insert(0, obs[0])
        return obs

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset(action_exec_horizon=self.exec_horizon)

    @staticmethod
    def seed(seed: int) -> None:
        import torch

        torch.manual_seed(seed)

    # ---------------------------------------------------------------- to implement
    def set_prompt(self, demo: dict[str, Any]) -> PromptInfo:
        raise NotImplementedError

    def act(self, history: list[dict[str, Any]]) -> np.ndarray:
        raise NotImplementedError


class BPPPolicy(PolicyBase):
    """LIBERO: two 128 px cameras, end-effector pose, gripper; 7-d delta actions."""

    def set_prompt(self, demo: dict[str, Any]) -> PromptInfo:
        prompt, info = build_prompt(demo, self.chunk_n, self.max_chunks, self.pad_end)
        prompt_dict = {
            "obs": {
                "agentview_rgb": self._images(prompt["obs"]["agentview_rgb"]),
                "eye_in_hand_rgb": self._images(prompt["obs"]["eye_in_hand_rgb"]),
                "ee_pos": self._lowdim(prompt["obs"]["ee_pos"]),
                "ee_ori": self._lowdim(prompt["obs"]["ee_ori"]),
                "gripper_states": self._lowdim(prompt["obs"]["gripper_states"]),
            },
            "action": self._lowdim(prompt["action"]),
            "metadata": {"mask": self._mask(prompt["mask"])},
        }
        self.policy.reset(action_exec_horizon=self.exec_horizon)
        self.policy.prompt(prompt_dict)
        return info

    def act(self, history: list[dict[str, Any]]) -> np.ndarray:
        """history: last observations (oldest first); returns (exec_horizon, 7) actions."""
        import torch

        obs = self._history(history)
        obs_dict = {
            "agentview_rgb": self._images(np.stack([o["agentview"] for o in obs])),
            "eye_in_hand_rgb": self._images(np.stack([o["eye_in_hand"] for o in obs])),
            "ee_pos": self._lowdim(np.stack([o["ee_pos"] for o in obs])),
            "ee_ori": self._lowdim(quat_xyzw_to_rotation_6d(np.stack([o["ee_quat"] for o in obs]))),
            "gripper_states": self._lowdim(np.stack([o["gripper"] for o in obs])),
        }
        with torch.inference_mode():
            out = self.policy.predict_action(obs_dict)
        a10 = out["action"][0].detach().float().cpu().numpy()
        return actions_10_to_7(a10)[: self.exec_horizon]
