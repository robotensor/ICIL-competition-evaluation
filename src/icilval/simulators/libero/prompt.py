"""A LIBERO demonstration -> BPP's behavior prompt (cameras, pose, gripper; 10-d actions)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...model.prompt import PromptInfo, chunk_actions
from .rotations import actions_7_to_10, axis_angle_to_rotation_6d


def build_prompt(
    demo: dict[str, Any], chunk_n: int, max_chunks: int | None = None, pad_end: str = "zeros"
) -> tuple[dict[str, Any], PromptInfo]:
    """LIBERO: two cameras, end-effector pose (rot6d), gripper; 7-d actions widened to 10."""
    a10 = actions_7_to_10(np.asarray(demo["actions"], dtype=np.float32))
    chunks, idx, info = chunk_actions(a10, chunk_n, pad_end, max_chunks)
    prompt = {
        "obs": {
            "agentview_rgb": np.asarray(demo["agentview"], dtype=np.uint8)[idx],
            "eye_in_hand_rgb": np.asarray(demo["eye_in_hand"], dtype=np.uint8)[idx],
            "ee_pos": np.asarray(demo["ee_pos"], dtype=np.float32)[idx],
            "ee_ori": axis_angle_to_rotation_6d(
                np.asarray(demo["ee_ori"], dtype=np.float64)[idx]
            ).astype(np.float32),
            "gripper_states": np.asarray(demo["gripper"], dtype=np.float32)[idx],
        },
        "action": chunks,
        "mask": np.zeros((info.chunks,), dtype=bool),
    }
    return prompt, info
