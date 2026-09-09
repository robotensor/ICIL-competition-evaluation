"""What one scored episode reports, whatever the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeResult:
    success: bool = False
    progress: float | None = None
    first_step_done_at: int | None = None
    metric: float | None = None
    steps: int = 0
    model_errors: int = 0
    wall_s: float = 0.0
    error: str | None = None
    void: bool = False
    prompt_steps: int = 0
    prompt_chunks: int = 0
    instance_applied: dict[str, Any] = field(default_factory=dict)
    change_applied: dict[str, Any] = field(default_factory=dict)
    video_frames: int = 0
