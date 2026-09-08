"""Where raw inputs live and how they become pool files. Pickles are read only here, at build time."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sim.libero_env import save_init_states

DEFAULT_CACHE = Path(os.environ.get("ICILVAL_CACHE", Path.home() / ".cache" / "icilval"))

LIBERO_SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")
SUITE_MAX_STEPS = {
    "libero_spatial": 300,
    "libero_goal": 400,
    "libero_object": 300,
    "libero_10": 550,
}

# libero_goal tasks that LIBERO-Gen's first-step view reproduces verbatim; not novel pairings.
LIBERO_GOAL_ORIGINALS = {
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "turn_on_the_stove",
}

DRAW_HANDMADE = "eval_handmade.zarr"

LIBERO_SOURCES = (
    "libero_root",
    "libero_datasets",
    "gen_goal_chain",
    "gen_spatial_combination",
)
DRAW_SOURCES = ("drawanything",)


@dataclass
class Sources:
    libero_root: Path  # .../deps/LIBERO/libero/libero  (bddl_files/, init_files/)
    libero_datasets: Path  # raw/libero_datasets/<suite>/<task>_demo.hdf5
    gen_goal_chain: Path  # raw/libero_gen_goal_chain
    gen_spatial_combination: Path  # raw/libero_gen_spatial_combination
    drawanything: Path  # raw/drawanything_sim (eval_handmade.zarr, unpacked)
    bpp_root: Path  # vendor/behavior_prompting

    @classmethod
    def default(cls, repo_root: Path, cache: Path = DEFAULT_CACHE) -> Sources:
        raw = cache / "raw"
        bpp = repo_root / "vendor" / "behavior_prompting"
        return cls(
            libero_root=bpp / "deps" / "LIBERO" / "libero" / "libero",
            libero_datasets=raw / "libero_datasets",
            gen_goal_chain=raw / "libero_gen_goal_chain",
            gen_spatial_combination=raw / "libero_gen_spatial_combination",
            drawanything=raw / "drawanything_sim",
            bpp_root=bpp,
        )

    @property
    def draw_handmade(self) -> Path:
        return self.drawanything / DRAW_HANDMADE

    def check(self, names: tuple[str, ...] = LIBERO_SOURCES + DRAW_SOURCES) -> list[str]:
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
