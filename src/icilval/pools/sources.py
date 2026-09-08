"""Where raw inputs live and how they become pool files. Pickles are read only here, at build time."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sim.libero_env import save_init_states

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path(os.environ.get("ICILVAL_CACHE", Path.home() / ".cache" / "icilval"))

# BPP's DrawAnything-Sim dataset (austinpatel/drawanything_sim): the human-drawn evaluation set,
# unpacked, and the procedural training set, read straight from its zip (zarr's ZipStore).
DRAW_HANDMADE = "eval_handmade.zarr"
DRAW_PROCEDURAL = "procedural_2000_10.zarr.zip"

# the Hugging Face dataset each LIBERO-Gen raw directory mirrors (bddl_files/, init_files/,
# demonstration_data/<view>/<task>_demo.hdf5)
HUB_DATASETS = {
    "gen_spatial_combination": "austinpatel/libero_gen_spatial_combination_hdf5",
    "gen_goal_chain": "austinpatel/libero_gen_goal_chain_hdf5",
}

# which raw directories a build stage reads
STAGE_SOURCES = {
    "pick_and_place": ("gen_spatial_combination", "bpp_root"),
    "draw": ("drawanything",),
}


@dataclass
class Sources:
    gen_spatial_combination: Path  # raw/libero_gen_spatial_combination
    gen_goal_chain: Path  # raw/libero_gen_goal_chain
    drawanything: Path  # raw/drawanything_sim (eval_handmade.zarr, unpacked)
    bpp_root: Path  # vendor/behavior_prompting

    @classmethod
    def default(cls, repo_root: Path, cache: Path = DEFAULT_CACHE) -> Sources:
        raw = cache / "raw"
        return cls(
            gen_spatial_combination=raw / "libero_gen_spatial_combination",
            gen_goal_chain=raw / "libero_gen_goal_chain",
            drawanything=raw / "drawanything_sim",
            bpp_root=repo_root / "vendor" / "behavior_prompting",
        )

    @property
    def draw_handmade(self) -> Path:
        return self.drawanything / DRAW_HANDMADE

    @property
    def draw_procedural(self) -> Path:
        return self.drawanything / DRAW_PROCEDURAL

    def check(self, names: tuple[str, ...]) -> list[str]:
        missing = []
        for name in names:
            p = getattr(self, name)
            if not p.exists():
                missing.append(f"{name}: {p}")
        return missing


def hub_files(key: str, prefix: str) -> list[str]:
    """Paths under `prefix` in the Hugging Face dataset mirrored by raw directory `key`."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(HUB_DATASETS[key], repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix))


def fetch(root: Path, key: str, rel: str) -> Path:
    """`root/rel`, downloaded from the dataset mirrored at `root` if it is not there yet."""
    path = root / rel
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    log.info("fetch %s/%s", HUB_DATASETS[key], rel)
    hf_hub_download(HUB_DATASETS[key], rel, repo_type="dataset", local_dir=str(root))
    return path


def evict(path: Path) -> None:
    """Delete a fetched demonstration file once its demos are in the pool (they are 0.5-1.3 GB each)."""
    if path.exists():
        path.unlink()
        log.info("evicted %s", path.name)


def load_pruned_init(path: str | Path) -> np.ndarray:
    """LIBERO's `.pruned_init` is a pickled numpy array (torch.save). Build-time only."""
    import torch

    states = torch.load(str(path), weights_only=False)
    return np.asarray(states, dtype=np.float64)


def convert_init(src: Path, dst: Path) -> int:
    from ..canon import sha256_file

    states = load_pruned_init(src)
    save_init_states(dst, states, sha256_file(src))
    return int(states.shape[0])


def copy_bddl(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
