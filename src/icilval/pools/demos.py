"""Demonstrations: benchmark files -> one npz per demo, and back into prompts and clips.

LIBERO (`demos/<task_id>/<demo>.npz`, from hdf5):
  agentview   (T,128,128,3) uint8, upright (hdf5 stores the OpenGL orientation; we flip once here)
  eye_in_hand (T,128,128,3) uint8, upright
  ee_pos (T,3) ee_ori (T,3 axis-angle) gripper (T,2) joints (T,7) actions (T,7) float32
  init_state (D,) float64   the simulator state the demo started from

DrawAnything (`demos/<task_id>/<demo>.npz`, from BPP's zarr replay buffers):
  image (T,224,224,3) uint8   the board as the policy sees it
  agent_pos (T,2) pen_down (T,1) actions (T,3) float32
  boundary_angle ()  float64  the board angle the demonstration was drawn at
  drawing (512,512) bool      the strokes the demonstration left, in canvas pixels

meta: source file, demo key, index in file, steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import Spec

# BPP's own tolerance for reading a (lossy) drawing image back into a stroke mask.
STROKE_TOLERANCE = 50


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


def _sorted_demo_keys(keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k)


# ---------------------------------------------------------------- LIBERO
def import_hdf5(
    src: str | Path, out_dir: str | Path, task_id: str, k: int, *, demo_prefix: str = "demo"
) -> list[DemoMeta]:
    import h5py

    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas: list[DemoMeta] = []
    with h5py.File(src, "r") as h:
        data = h["data"]
        keys = _sorted_demo_keys(list(data.keys()))
        for n, idx in enumerate(select_indices(len(keys), k)):
            key = keys[idx]
            g = data[key]
            obs = g["obs"]
            demo_id = f"{task_id}/{demo_prefix}_{n:02d}"
            path = out_dir / f"{demo_prefix}_{n:02d}.npz"
            init_state = (
                np.asarray(g.attrs["init_state"], dtype=np.float64)
                if "init_state" in g.attrs
                else np.asarray(g["states"][0], dtype=np.float64)
            )
            arrays = {
                "agentview": np.asarray(obs["agentview_rgb"], dtype=np.uint8)[:, ::-1].copy(),
                "eye_in_hand": np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)[:, ::-1].copy(),
                "ee_pos": np.asarray(obs["ee_pos"], dtype=np.float32),
                "ee_ori": np.asarray(obs["ee_ori"], dtype=np.float32),
                "gripper": np.asarray(obs["gripper_states"], dtype=np.float32),
                "joints": np.asarray(obs["joint_states"], dtype=np.float32),
                "actions": np.asarray(g["actions"], dtype=np.float32),
                "init_state": init_state,
            }
            meta = {
                "demo_id": demo_id,
                "source": src.name,
                "source_key": key,
                "source_index": idx,
                "image_convention": "upright",
                "steps": int(arrays["actions"].shape[0]),
            }
            np.savez_compressed(path, meta=json.dumps(meta), **arrays)
            metas.append(
                DemoMeta(
                    demo_id=demo_id,
                    path=str(path),
                    steps=meta["steps"],
                    source=src.name,
                    source_key=key,
                    source_index=idx,
                )
            )
    return metas


# ---------------------------------------------------------------- DrawAnything
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


# ---------------------------------------------------------------- reading back
def load_demo(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def demo_frames(demo: dict[str, Any], upscale_factor: int = 2, stride: int = 1) -> list[np.ndarray]:
    from ..sim.video import side_by_side, upscale

    frames = []
    for t in range(0, demo["agentview"].shape[0], stride):
        a = upscale(demo["agentview"][t], upscale_factor)
        b = upscale(demo["eye_in_hand"][t], upscale_factor)
        frames.append(side_by_side(a, b))
    return frames


def draw_demo_frames(demo: dict[str, Any], stride: int = 1) -> list[np.ndarray]:
    return [
        np.asarray(demo["image"][t], np.uint8) for t in range(0, demo["image"].shape[0], stride)
    ]


def render_demo(path: str | Path, out_mp4: str | Path, spec: Spec, skill: str) -> str:
    """A demonstration clip at the skill's control rate: both LIBERO cameras side by side,
    or the drawing board as the policy saw it."""
    from ..sim.video import encode_frames
    from ..simulators import for_skill

    demo = load_demo(path)
    fps = int(spec.env(skill)["control_freq"])
    frames = for_skill(spec, skill).demo_frames(demo)
    return encode_frames(frames, out_mp4, fps, spec.media["video"])
