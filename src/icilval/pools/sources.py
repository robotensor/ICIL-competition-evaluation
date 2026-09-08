"""Where raw inputs live and how they become pool files. Pickles are read only here, at build time."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sim.libero_env import save_init_states

DEFAULT_CACHE = Path(os.environ.get("ICILVAL_CACHE", Path.home() / ".cache" / "icilval"))

DRAW_HANDMADE = "eval_handmade.zarr"

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

    def check(self, names: tuple[str, ...]) -> list[str]:
        missing = []
        for name in names:
            p = getattr(self, name)
            if not p.exists():
                missing.append(f"{name}: {p}")
        return missing


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
