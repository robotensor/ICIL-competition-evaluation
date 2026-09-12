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


# ---------------------------------------------------------------- reading back
def load_demo_raw(path: str | Path) -> dict[str, Any]:
    """Every array in the file.

    Build-time and rendering only. A duel must not call this: what a policy may see depends on
    the field it is being scored in, which is what `load_demo_for` applies.
    """
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def demo_path(pool: Any, demo_id: str) -> Path:
    return Path(pool.path("demos")) / f"{demo_id}.npz"


def load_demo_for(pool: Any, spec: Spec, track: str, unit: dict[str, Any]) -> dict[str, Any]:
    """The demonstration this unit's field allows, and nothing else.

    The only loader a duel may call. What the field withholds is never put in the mapping, so a
    benchmark's episode loop cannot reach it however it is written.
    """
    from .. import demoview, simulators

    view = demoview.view_for(spec, track)
    channels = simulators.for_skill(spec, unit["skill"]).demo_channels
    raw = load_demo_raw(demo_path(pool, unit["demo"]))
    handed = demoview.apply(raw, channels, view)
    leaked = demoview.check(handed, channels, view)
    if leaked:  # unreachable unless `apply` and `check` disagree; loud rather than silent
        raise ValueError(f"{unit['demo']}: {'; '.join(leaked)}")
    return handed


def render_demo(path: str | Path, out_mp4: str | Path, spec: Spec, skill: str) -> str:
    """A demonstration clip at the skill's control rate: both LIBERO cameras side by side,
    or the drawing board as the policy saw it."""
    from ..simulators import for_skill
    from ..video import encode_frames

    demo = load_demo_raw(path)
    fps = int(spec.env(skill)["control_freq"])
    frames = for_skill(spec, skill).demo_frames(demo)
    return encode_frames(frames, out_mp4, fps, spec.media["video"])
