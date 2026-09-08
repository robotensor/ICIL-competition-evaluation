"""BPP inference on the drawing board: one 224 px image, pen position and pen state in;
(x, y, pen_down) action chunks out. Loading and tensor plumbing are `bpp.PolicyBase`'s."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...model.policy import PolicyBase
from ...model.prompt import PromptInfo
from .prompt import build_draw_prompt


class DrawPolicy(PolicyBase):
    def _board_images(self, frames: np.ndarray) -> Any:
        """(N,3,S,S) float in [0,1] as the board renders it -> (1,N,3,S,S) on the device."""
        import torch

        x = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32)).to(self.device)
        return x.unsqueeze(0)

    def set_prompt(self, demo: dict[str, Any]) -> PromptInfo:
        prompt, info = build_draw_prompt(demo, self.chunk_n, self.max_chunks, self.pad_end)
        prompt_dict = {
            "obs": {
                "image": self._images(prompt["obs"]["image"]),
                "agent_pos": self._lowdim(prompt["obs"]["agent_pos"]),
                "pen_down": self._lowdim(prompt["obs"]["pen_down"]),
            },
            "action": self._lowdim(prompt["action"]),
            "metadata": {"mask": self._mask(prompt["mask"])},
        }
        self.policy.reset(action_exec_horizon=self.exec_horizon)
        self.policy.prompt(prompt_dict)
        return info

    def act(self, history: list[dict[str, Any]]) -> np.ndarray:
        """history: last board observations (oldest first); returns (exec_horizon, 3) actions."""
        import torch

        obs = self._history(history)
        obs_dict = {
            "image": self._board_images(np.stack([o["image"] for o in obs])),
            "agent_pos": self._lowdim(np.stack([o["agent_pos"] for o in obs])),
            "pen_down": self._lowdim(np.stack([o["pen_down"] for o in obs])),
        }
        with torch.inference_mode():
            out = self.policy.predict_action(obs_dict)
        actions = out["action"][0].detach().float().cpu().numpy()
        return np.asarray(actions, dtype=np.float64)[: self.exec_horizon]
