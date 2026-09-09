"""Generate a LIBERO prompt demonstration for a unit with BPP's own generator.

BPP's `TaskDemonstrationGenerator` (vendored, `scripts/libero/generate_demonstrations.py`) lifts
the grasp from the human teleoperation demonstration the task's metadata names, scripts the
transport and the place with randomized waypoints and records the simulator states; BPP's
`create_dataset` then replays those states to render both cameras. This module drives it for one
prompt at a time: seeded attempts until one succeeds, then one npz in the catalogue's demo format.

Everything here runs in the BPP environment, headless, under a validator-owned LIBERO config
(`libero_config`) that points `datasets` at the raw cache's grasp-source files and every other
path at the vendored checkout; `~/.libero` is never read. LIBERO reads its config when it is
imported, so a process that generates calls `libero_config()` before anything imports `libero`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...canon import sha256_file
from ...pools.schema import Pool, PoolTask
from ...pools.sources import Sources, fetch_by_hash
from ...spec import Spec
from .demos import import_hdf5

log = logging.getLogger(__name__)

SCRIPTS_REL = "behavior_prompting/train_network/scripts/libero"
BDDL_FILES_REL = "behavior_prompting/train_network/env/libero/bddl_files"
INIT_FILES_REL = "behavior_prompting/train_network/env/libero/init_files"
LIBERO_REL = "deps/LIBERO/libero/libero"
CONFIG_ENV = "LIBERO_CONFIG_PATH"
HEADLESS_ENV = {"PYNPUT_BACKEND": "dummy", "MUJOCO_GL": "egl"}


# ---------------------------------------------------------------- the LIBERO config
def libero_config_doc(bpp_root: Path, datasets: Path) -> dict[str, str]:
    """LIBERO's five paths: the vendored checkout for everything but `datasets`."""
    return {
        "assets": str(bpp_root / LIBERO_REL / "assets"),
        "bddl_files": str(bpp_root / BDDL_FILES_REL),
        "benchmark_root": str(bpp_root / LIBERO_REL),
        "datasets": str(datasets),
        "init_states": str(bpp_root / INIT_FILES_REL),
    }


def libero_config(config_dir: Path, bpp_root: Path, datasets: Path) -> Path:
    """Write the validator's LIBERO config to `config_dir` and point LIBERO at it. Must run in a
    process that has not imported `libero` yet; also sets the headless variables."""
    import yaml

    if "libero" in sys.modules and os.environ.get(CONFIG_ENV) != str(config_dir):
        raise RuntimeError(
            "libero is already imported with another config; generate in a fresh process"
        )
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "config.yaml"
    path.write_text(yaml.safe_dump(libero_config_doc(bpp_root, datasets), sort_keys=True))
    os.environ[CONFIG_ENV] = str(config_dir)
    for key, value in HEADLESS_ENV.items():
        os.environ.setdefault(key, value)
    return path


# ---------------------------------------------------------------- grasp sources
def grasp_sources(pool: Pool, skill: str, spec: Spec) -> dict[str, Any]:
    """The catalogue's record of the skill's grasp-source files (dataset, prefix, name -> sha)."""
    dataset = str(spec.tasks(skill)["dataset"])
    for entry in pool.sources.values():
        if (
            isinstance(entry, dict)
            and entry.get("dataset") == dataset
            and entry.get("grasp_sources")
        ):
            return dict(entry["grasp_sources"])
    raise ValueError(f"catalogue records no grasp sources for {skill}")


def ensure_grasp_sources(pool: Pool, skill: str, spec: Spec, src: Sources) -> Path:
    """Every grasp-source file the catalogue records, fetched by hash into the raw cache;
    returns the `datasets` root LIBERO must read (the parent of the `<prefix>` directory)."""
    entry = grasp_sources(pool, skill, spec)
    dataset, prefix = str(entry["dataset"]), str(entry["prefix"])
    root = src.dataset_root(dataset)
    for name, sha in sorted(entry["files"].items()):
        fetch_by_hash(root, dataset, f"{prefix}{name}", str(sha))
    return root


# ---------------------------------------------------------------- BPP's generator
def bpp_generator_module(bpp_root: Path) -> Any:
    """BPP's `generate_demonstrations` module, importable only from its scripts directory."""
    scripts = str(bpp_root / SCRIPTS_REL)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import generate_demonstrations as gd  # noqa: PLC0415 - BPP script, not a package

    return gd


@dataclass
class Attempt:
    seed: int
    success: bool
    failed_stage: str | None = None


@dataclass
class GenerationResult:
    success: bool
    attempts: list[Attempt] = field(default_factory=list)
    npz: Path | None = None
    steps: int = 0
    sha256: str | None = None

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "attempts": [
                {"seed": a.seed, "success": a.success, "failed_stage": a.failed_stage}
                for a in self.attempts
            ],
            "steps": self.steps,
            "sha256": self.sha256,
        }


def task_name(task: PoolTask) -> str:
    return task.task_id.split("/", 1)[1]


def own_grasp_sources(meta: dict[str, Any], name: str) -> dict[str, Any] | None:
    """BPP's metadata names no grasp source for a task LIBERO itself ships (BPP prompted those
    with their human demonstrations), and the generator refuses every grasp step that has none.
    Such a task lifts its grasps from its own teleoperation file, which is one of the
    catalogue's grasp sources; a task with a source of its own is left alone."""
    if meta.get("grasps_from") or meta.get("actions_from"):
        return None
    objects = [
        s.split(" ", 1)[1] for s in meta.get("execution_steps", []) if s.startswith("Grasp ")
    ]
    if not objects:
        return None
    return {obj: {"task": name, "object": obj} for obj in objects}


def vendored_bddl(task: PoolTask, pool: Pool, bpp_root: Path) -> Path:
    """The task's BDDL in the vendored checkout's layout, which BPP's generator reads its metadata
    next to; it must be the catalogue's file byte for byte."""
    path = bpp_root / BDDL_FILES_REL / str(task.meta["source_split"]) / f"{task_name(task)}.bddl"
    if not path.exists():
        raise FileNotFoundError(f"{path} (vendored BPP checkout)")
    if path.read_bytes() != pool.path(task.bddl).read_bytes():
        raise ValueError(f"{task.task_id}: the vendored BDDL differs from the catalogue's")
    return path


class PromptGenerator:
    """BPP's generator for one task: an environment plus the grasp poses read from the human
    demonstrations, reused across seeded attempts; one prompt comes out of `finalize`."""

    def __init__(
        self,
        task: PoolTask,
        pool: Pool,
        spec: Spec,
        skill: str,
        *,
        bpp_root: Path,
        work_dir: Path,
        quiet: bool = True,
    ):
        self.task = task
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.work_dir / "generator.log"
        bddl = vendored_bddl(task, pool, bpp_root)
        gd = bpp_generator_module(bpp_root)
        name = task_name(task)
        load_metadata = gd.load_metadata_for_task
        self.own_grasps = False

        def load_with_own_grasps(bddl_file_path, **kwargs):
            meta = load_metadata(bddl_file_path, **kwargs)
            own = own_grasp_sources(meta, name)
            if own:
                meta["grasps_from"] = own
                self.own_grasps = True
            return meta

        gd.load_metadata_for_task = load_with_own_grasps
        try:
            with self._quiet(quiet):
                self.gen = gd.TaskDemonstrationGenerator(
                    str(bddl),
                    resolution=int(spec.env(skill)["camera_resolution"]),
                    suppress_print=quiet,
                    run_dir=str(self.work_dir),
                    bddl_path=str(bpp_root / BDDL_FILES_REL),
                )
        finally:
            gd.load_metadata_for_task = load_metadata
        self.quiet = quiet
        self._stage: str | None = None
        original = self.gen.complete_stage

        def complete_stage(from_stage, to_stage, demo_idx=0):
            self._stage = str(to_stage)
            return original(from_stage, to_stage, demo_idx=demo_idx)

        self.gen.complete_stage = complete_stage

    @contextlib.contextmanager
    def _quiet(self, quiet: bool):
        if not quiet:
            yield
            return
        with open(self.log_path, "a", encoding="utf-8") as fh:
            with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
                yield

    def attempt(self, seed: int) -> Attempt:
        """One seeded attempt: BPP resets the scene with the seed and records it; the seed also
        picks which human grasp pose is replayed."""
        self._stage = None
        with self._quiet(self.quiet):
            ok = bool(self.gen.generate_single_demo(int(seed)))
        return Attempt(int(seed), ok, None if ok else self._stage)

    def finalize(self, out_npz: Path, demo_id: str, extra_meta: dict[str, Any]) -> tuple[Path, int]:
        """Gather the successful attempt, render both cameras, write one npz. Closes the env."""
        with self._quiet(self.quiet):
            h5, _ = self.gen.finalize(generate_video=False, run_dir=str(self.work_dir))
        out_npz = Path(out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.work_dir / "npz"
        metas = import_hdf5(h5, tmp_dir, demo_id.rsplit("/", 1)[0], 1)
        if len(metas) != 1:
            raise RuntimeError(f"{h5}: expected one demonstration, found {len(metas)}")
        with np.load(metas[0].path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        meta.update({"demo_id": demo_id, "source": "generated", **extra_meta})
        if self.own_grasps:
            meta["grasps_from"] = "own"
        np.savez_compressed(out_npz, meta=json.dumps(meta), **arrays)
        return out_npz, int(meta["steps"])

    def close(self, keep_work_dir: bool = False) -> None:
        with contextlib.suppress(Exception):
            self.gen.env.close()
        if not keep_work_dir:
            shutil.rmtree(self.work_dir, ignore_errors=True)


def generate_prompt(
    task: PoolTask,
    pool: Pool,
    spec: Spec,
    skill: str,
    seeds: Sequence[int],
    out_npz: Path,
    *,
    bpp_root: Path,
    work_dir: Path,
    demo_id: str,
    quiet: bool = True,
    keep_work_dir: bool = False,
) -> GenerationResult:
    """Seeded attempts, in order, until one succeeds; then the prompt npz and its sha256."""
    result = GenerationResult(success=False)
    gen = PromptGenerator(
        task, pool, spec, skill, bpp_root=bpp_root, work_dir=work_dir, quiet=quiet
    )
    try:
        for seed in seeds:
            attempt = gen.attempt(int(seed))
            result.attempts.append(attempt)
            log.info(
                "%s seed %d: %s",
                task.task_id,
                seed,
                "ok" if attempt.success else f"failed at {attempt.failed_stage}",
            )
            if attempt.success:
                extra = {
                    "task": task.task_id,
                    "seed": int(seed),
                    "attempts": result.n_attempts,
                    "behavior_prompting_commit": str(spec.env(skill)["behavior_prompting_commit"]),
                }
                path, steps = gen.finalize(out_npz, demo_id, extra)
                result.success, result.npz, result.steps = True, path, steps
                result.sha256 = sha256_file(path)
                break
    finally:
        gen.close(keep_work_dir=keep_work_dir)
    return result
