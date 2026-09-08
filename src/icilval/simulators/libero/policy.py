"""BPP inference on LIBERO: two 128 px cameras, end-effector pose, gripper; 7-d delta actions."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...model.policy import PolicyBase
from ...model.prompt import PromptInfo
from .prompt import build_prompt
from .rotations import actions_10_to_7, quat_xyzw_to_rotation_6d


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
