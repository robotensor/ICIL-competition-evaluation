"""Where raw inputs live and how they become catalogue files.

A skill's task definitions come from one Hugging Face dataset (`spec.json`
`skills.<skill>.tasks.dataset`), mirrored under `raw/<name>` where `<name>` is the repo name
without an `_hdf5` suffix: `austinpatel/libero_gen_spatial_combination_hdf5` ->
`raw/libero_gen_spatial_combination`. Files are fetched on demand when a build asks for it. The
human teleoperation files a skill's generated demonstrations lift their grasps from
(`tasks.grasp_sources`) are recorded by sha256 from the hub's tree listing and fetched by hash
only where a prompt is generated.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ..canon import sha256_file

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


def hub_file_hashes(dataset: str, prefix: str) -> dict[str, str]:
    """File name -> sha256 of every LFS file under `prefix` in a dataset, from the hub's tree
    listing; nothing is downloaded."""
    from huggingface_hub import HfApi

    out: dict[str, str] = {}
    entries = HfApi().list_repo_tree(
        dataset, path_in_repo=prefix.rstrip("/"), repo_type="dataset", expand=True
    )
    for entry in entries:
        lfs = getattr(entry, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if sha:
            out[Path(str(entry.path)).name] = str(sha)
    return dict(sorted(out.items()))


def grasp_source_hashes(src: Sources, dataset: str, prefix: str, online: bool) -> dict[str, str]:
    """The grasp-source hashes: from the hub when online, else from the files already in the
    raw cache under `raw/<dataset name>/<prefix>`."""
    if online:
        return hub_file_hashes(dataset, prefix)
    local = src.dataset_root(dataset) / prefix.rstrip("/")
    files = sorted(local.glob("*.hdf5")) if local.exists() else []
    if not files:
        raise FileNotFoundError(f"no grasp-source files under {local}; build with --fetch")
    return {f.name: sha256_file(f) for f in files}


def fetch_by_hash(root: Path, dataset: str, rel: str, sha256: str) -> Path:
    """`root/rel`, downloaded if missing, and refused unless its sha256 is the recorded one."""
    path = fetch(root, dataset, rel)
    got = sha256_file(path)
    if got != sha256:
        raise ValueError(f"{path}: sha256 {got[:12]} is not the catalogue's {sha256[:12]}")
    return path
