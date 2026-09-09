"""Demonstrations: one npz per demo under `demos/<task_id>/<demo>.npz`, read back for prompts and
clips. Each simulator's `demos.py` writes its own arrays (see `simulators/<sim>/demos.py`);
every file carries a `meta` JSON string: source file, demo key, index in file, steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import Spec


@dataclass
class DemoMeta:
    demo_id: str
    path: str
    steps: int
    source: str
    source_key: str
    source_index: int
    init_sha256: str = ""
    boundary_angle: float | None = None


def select_indices(n_available: int, k: int) -> list[int]:
    """k evenly spread indices over [0, n). Deterministic, keeps the first demo."""
    if k >= n_available:
        return list(range(n_available))
    return sorted({int(round(i * (n_available - 1) / max(1, k - 1))) for i in range(k)})


def sorted_demo_keys(keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k)


GENERATED_PREFIX = "generated/"


def prompt_path(pool: Any, unit: dict[str, Any], assets_dir: str | Path | None) -> Path:
    """The npz a unit is prompted with: a generated prompt in the duel's assets directory
    (`<assets>/<unit_id>.npz`), else one of the catalogue's stored demonstrations."""
    demo = str(unit["demo"])
    if demo.startswith(GENERATED_PREFIX):
        if assets_dir is None:
            raise ValueError(f"{unit['unit_id']}: generated prompt but no assets directory")
        return Path(assets_dir) / f"{unit['unit_id']}.npz"
    return pool.path("demos") / f"{demo}.npz"


# ---------------------------------------------------------------- reading back
def load_demo(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def render_demo(path: str | Path, out_mp4: str | Path, spec: Spec, skill: str) -> str:
    """A demonstration clip at the skill's control rate: both LIBERO cameras side by side,
    or the drawing board as the policy saw it."""
    from ..simulators import for_skill
    from ..video import encode_frames

    demo = load_demo(path)
    fps = int(spec.env(skill)["control_freq"])
    frames = for_skill(spec, skill).demo_frames(demo)
    return encode_frames(frames, out_mp4, fps, spec.media["video"])
