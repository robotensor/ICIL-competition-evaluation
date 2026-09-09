"""Drawing demonstrations: npz -> clip frames, and the mask of what a board holds.

image (T,224,224,3) uint8   the board as the policy sees it
agent_pos (T,2) pen_down (T,1) actions (T,3) float32
boundary_angle ()  float64  the board angle the demonstration was drawn at
drawing (512,512) bool      the strokes the demonstration left, in canvas pixels
"""

from __future__ import annotations

from typing import Any

import numpy as np

# BPP's own tolerance for reading a (lossy) drawing image back into a stroke mask.
STROKE_TOLERANCE = 50


def stroke_mask(drawing_image: np.ndarray, tolerance: int = STROKE_TOLERANCE) -> np.ndarray:
    """Blue pen pixels of a (H,W,3) uint8 drawing image, read the way BPP reads its targets."""
    img = np.asarray(drawing_image)
    return (
        (img[:, :, 0] <= tolerance)
        & (img[:, :, 1] <= tolerance)
        & (img[:, :, 2] >= 255 - tolerance)
    )


def draw_demo_frames(demo: dict[str, Any], stride: int = 1) -> list[np.ndarray]:
    return [
        np.asarray(demo["image"][t], np.uint8) for t in range(0, demo["image"].shape[0], stride)
    ]
