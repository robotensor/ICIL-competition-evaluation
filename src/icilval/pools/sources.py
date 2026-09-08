"""Where raw inputs live and how they become pool files. Pickles are read only here, at build time.

A skill's tasks come from one Hugging Face dataset (`spec.json` `skills.<skill>.tasks.dataset`),
mirrored under `raw/<name>` where `<name>` is the repo name without an `_hdf5` suffix:
`austinpatel/libero_gen_goal_chain_hdf5` -> `raw/libero_gen_goal_chain`. Files are fetched on
demand when a build asks for it, and the demonstration files can be evicted once imported.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path(os.environ.get("ICILVAL_CACHE", Path.home() / ".cache" / "icilval"))


def raw_dir_name(dataset: str) -> str:
    name = dataset.split("/", 1)[1]
    return name[: -len("_hdf5")] if name.endswith("_hdf5") else name


@dataclass
class Sources:
    raw: Path  # ~/.cache/icilval/raw
    bpp_root: Path  # vendor/behavior_prompting

    @classmethod
    def default(cls, repo_root: Path, cache: Path = DEFAULT_CACHE) -> Sources:
        return cls(raw=cache / "raw", bpp_root=repo_root / "vendor" / "behavior_prompting")

    def dataset_root(self, dataset: str) -> Path:
        return self.raw / raw_dir_name(dataset)

    def missing(self, datasets: tuple[str, ...]) -> list[str]:
        return [
            f"{d}: {self.dataset_root(d)}" for d in datasets if not self.dataset_root(d).exists()
        ]


def hub_files(dataset: str, prefix: str) -> list[str]:
    """Paths under `prefix` in a Hugging Face dataset."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(dataset, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix))


def fetch(root: Path, dataset: str, rel: str) -> Path:
    """`root/rel`, downloaded from `dataset` if it is not there yet."""
    path = root / rel
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    log.info("fetch %s/%s", dataset, rel)
    hf_hub_download(dataset, rel, repo_type="dataset", local_dir=str(root))
    return path


def evict(path: Path) -> None:
    """Delete a fetched demonstration file once its demos are in the pool (0.5-1.5 GB each)."""
    if path.exists():
        path.unlink()
        log.info("evicted %s", path.name)
