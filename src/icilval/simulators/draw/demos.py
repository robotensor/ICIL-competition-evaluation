"""Drawing demonstrations: BPP's zarr replay buffers -> npz, and npz -> clip frames.

image (T,224,224,3) uint8   the board as the policy sees it
agent_pos (T,2) pen_down (T,1) actions (T,3) float32
boundary_angle ()  float64  the board angle the demonstration was drawn at
drawing (512,512) bool      the strokes the demonstration left, in canvas pixels
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...pools.demos import DemoMeta, select_indices

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


def open_replay_buffer(path: str | Path) -> Any:
    """BPP's zarr replay buffer (images are JPEG-XL chunks; the codec has to be registered)."""
    import zarr

    try:
        import imagecodecs.numcodecs

        imagecodecs.numcodecs.register_codecs()
    except ImportError:  # pragma: no cover - only lossless buffers open without it
        pass
    return zarr.open(str(path), mode="r")


def draw_task_names(root: Any) -> list[str]:
    return sorted(set(str(n) for n in root["meta"]["task_names"][:]))


def import_draw_task(
    root: Any,
    task_name: str,
    out_dir: str | Path,
    task_id: str,
    k: int,
    *,
    source: str,
    demo_prefix: str = "demo",
) -> list[DemoMeta]:
    """k of a drawing task's episodes from an open replay buffer into npz demos."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_g = root["meta"]
    names = np.asarray(meta_g["task_names"][:]).astype(str)
    data_ends = np.asarray(meta_g["task_data_ends"][:], dtype=np.int64)
    label_ends = np.asarray(meta_g["task_labels_ends"][:], dtype=np.int64)
    episodes = [int(i) for i in np.where(names == task_name)[0]]
    if not episodes:
        raise KeyError(task_name)
    data, labels = root["data"], root["labels"]
    metas: list[DemoMeta] = []
    for n, pick in enumerate(select_indices(len(episodes), k)):
        ep = episodes[pick]
        d0, d1 = (int(data_ends[ep - 1]) if ep else 0), int(data_ends[ep])
        l0, l1 = (int(label_ends[ep - 1]) if ep else 0), int(label_ends[ep])
        angle = float(np.asarray(labels["boundary_angle"][l1 - 1]).reshape(-1)[0])
        drawing = stroke_mask(np.asarray(labels["drawing_image"][l1 - 1]))
        arrays = {
            "image": np.asarray(data["image"][d0:d1], dtype=np.uint8),
            "agent_pos": np.asarray(data["agent_pos"][d0:d1], dtype=np.float32),
            "pen_down": np.asarray(data["pen_down"][d0:d1], dtype=np.float32).reshape(-1, 1),
            "actions": np.asarray(data["action"][d0:d1], dtype=np.float32),
            "boundary_angle": np.asarray(angle, dtype=np.float64),
            "drawing": drawing,
        }
        demo_id = f"{task_id}/{demo_prefix}_{n:02d}"
        path = out_dir / f"{demo_prefix}_{n:02d}.npz"
        meta = {
            "demo_id": demo_id,
            "source": source,
            "source_key": task_name,
            "source_index": ep,
            "steps": int(arrays["actions"].shape[0]),
            "boundary_angle": angle,
            "stroke_pixels": int(drawing.sum()),
            "label_span": [l0, l1],
        }
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)
        metas.append(
            DemoMeta(
                demo_id=demo_id,
                path=str(path),
                steps=meta["steps"],
                source=source,
                source_key=task_name,
                source_index=ep,
                boundary_angle=angle,
            )
        )
    return metas


def draw_demo_frames(demo: dict[str, Any], stride: int = 1) -> list[np.ndarray]:
    return [
        np.asarray(demo["image"][t], np.uint8) for t in range(0, demo["image"].shape[0], stride)
    ]
