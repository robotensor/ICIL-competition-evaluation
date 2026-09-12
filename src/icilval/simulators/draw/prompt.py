"""A drawing demonstration -> BPP's behavior prompt (board image, pen position and state)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...model.prompt import PromptInfo, chunk_actions


def build_draw_prompt(
    demo: dict[str, Any], chunk_n: int, max_chunks: int | None = None, pad_end: str = "no"
) -> tuple[dict[str, Any], PromptInfo]:
    """DrawAnything: one board image, pen position and pen state; 3-d actions (x, y, pen_down)."""
    chunks, idx, info = chunk_actions(
        np.asarray(demo["actions"], np.float32), chunk_n, pad_end, max_chunks
    )
    prompt = {
        "obs": {
            "image": np.asarray(demo["image"], dtype=np.uint8)[idx],
            "agent_pos": np.asarray(demo["agent_pos"], dtype=np.float32)[idx],
            "pen_down": np.asarray(demo["pen_down"], dtype=np.float32).reshape(-1, 1)[idx],
        },
        "action": chunks,
        "mask": np.zeros((info.chunks,), dtype=bool),
    }
    return prompt, info
