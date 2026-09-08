"""LIBERO demonstrations: hdf5 -> npz, and npz -> clip frames.

agentview   (T,128,128,3) uint8, upright (hdf5 stores the OpenGL orientation; we flip once here)
eye_in_hand (T,128,128,3) uint8, upright
ee_pos (T,3) ee_ori (T,3 axis-angle) gripper (T,2) joints (T,7) actions (T,7) float32
init_state (D,) float64   the simulator state the demo started from
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...pools.demos import DemoMeta, select_indices, sorted_demo_keys


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
        keys = sorted_demo_keys(list(data.keys()))
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


def demo_frames(demo: dict[str, Any], upscale_factor: int = 2, stride: int = 1) -> list[np.ndarray]:
    from ...video import side_by_side, upscale

    frames = []
    for t in range(0, demo["agentview"].shape[0], stride):
        a = upscale(demo["agentview"][t], upscale_factor)
        b = upscale(demo["eye_in_hand"][t], upscale_factor)
        frames.append(side_by_side(a, b))
    return frames
