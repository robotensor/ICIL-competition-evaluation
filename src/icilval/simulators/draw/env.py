"""The DrawAnything-Sim board for one unit: BPP's `DrawEnv` behind the same small surface the
LIBERO environment has (reset, step, observe, render), with the board angle, the cursor start
and the target strokes set from the unit record rather than from the environment's own RNG.

The score is BPP's: the symmetric Chamfer distance, in canvas pixels, between the target
strokes (the demonstration's, turned upright) and the strokes drawn so far (turned upright
by the unit's board angle). Simulator imports are function-local.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ...spec import Spec

PEN_BLUE = (0, 0, 255)


def strokes_to_image(mask: np.ndarray) -> np.ndarray:
    """(H,W) bool strokes -> the (H,W,3) uint8 image DrawEnv.set_target_drawing reads."""
    img = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    img[np.asarray(mask, dtype=bool)] = PEN_BLUE
    return img


class DrawBoard:
    def __init__(self, spec: Spec, skill: str):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        from behavior_prompting.train_network.env.draw.draw_env import DrawEnv

        env_cfg = spec.env(skill)
        self.window = int(env_cfg["window_size"])
        self.obs_size = int(env_cfg["policy_image_resolution"])
        self.render_size = int(env_cfg["render_resolution"])
        self.raw = DrawEnv(
            boundary_angle=0.0,
            render_size=self.obs_size,
            render_cache_size=self.render_size,
            overlay_reward=False,
            overlay_action_cross=False,
            overlay_target_drawing=True,
            render_mode="rgb_array",
        )
        if self.raw.window_size != self.window or self.raw.board_length != int(
            env_cfg["board_length"]
        ):
            raise RuntimeError("the vendored DrawEnv geometry differs from spec.json")
        self.angle = 0.0

    # ---------------------------------------------------------------- lifecycle
    def reset(
        self,
        seed: int,
        *,
        angle: float,
        cursor: tuple[int, int],
        target: np.ndarray,
        target_angle: float,
    ) -> dict[str, Any]:
        self.raw.seed(int(seed))
        self.raw.boundary_angle = float(angle)
        self.raw.randomize_boundary_angle = False
        self.raw.reset_to_state = np.array([float(cursor[0]), float(cursor[1]), 0.0])
        self.raw.set_target_drawing(strokes_to_image(target), float(target_angle))
        self.angle = float(angle)
        return self.observe(self.raw.reset(no_rotation=False))

    def close(self) -> None:
        try:
            self.raw.close()
        except Exception:  # noqa: BLE001 - best effort
            pass

    # ---------------------------------------------------------------- stepping
    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float]:
        """Apply one (x, y, pen_down) action; returns the observation and the Chamfer distance."""
        obs, reward, _, _ = self.raw.step(np.asarray(action, dtype=np.float64))
        return self.observe(obs), float(-reward)

    def observe(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": np.asarray(raw["image"], dtype=np.float32),  # (3,S,S) in [0,1]
            "agent_pos": np.asarray(raw["agent_pos"], dtype=np.float32),
            "pen_down": np.asarray(raw["pen_down"], dtype=np.float32).reshape(1),
        }

    def chamfer(self) -> float:
        return float(-self.raw.compute_reward())

    def render(self) -> np.ndarray:
        """The board with the target strokes overlaid, at the clip resolution."""
        frame = self.raw.render_cache
        if frame is None:
            self.raw._get_obs()
            frame = self.raw.render_cache
        return np.ascontiguousarray(frame).astype(np.uint8)

    def strokes(self) -> np.ndarray:
        """What has been drawn so far, as a (window, window) bool mask in canvas pixels."""
        from ..pools.demos import stroke_mask

        return stroke_mask(self.raw.get_drawing_image())
