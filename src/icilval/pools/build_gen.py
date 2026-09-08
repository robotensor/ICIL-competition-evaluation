"""Held-out task generation through BPP's LIBERO-Gen scripts, driven by our affordance table.

`icilval pools generate` runs, inside the BPP checkout:
  gen_extra_libero_envs.py --extra-envs-config-path affordance.yaml --splits <split> --suffix <suffix>
  generate_demonstrations.py --suffix <suffix> --include-splits <views…> --n-demos N --run-dir <dir>
and `import_generated` copies the run directory's bddl/init/hdf5 into a pool as
`generated/<task>` object-swap tasks of the pick-and-place skill (the filter in `build.py`
drops anything that is not a single grasp-then-place). Generation takes hours of CPU; the
public LIBERO-Gen splits populate the pool without it.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from ..spec import Spec
from .build import LIBERO_SKILL, _gen_task, _swap_from_goal
from .schema import Pool
from .validate import goal_from_bddl

log = logging.getLogger(__name__)


def scripts_dir(bpp_root: Path) -> Path:
    return bpp_root / "behavior_prompting" / "train_network" / "scripts" / "libero"


def generate(
    bpp_root: Path,
    affordance_yaml: Path,
    *,
    splits: list[str],
    views: list[str],
    suffix: str,
    run_dir: Path,
    n_demos: int,
    workers: int,
    python: str = "python",
    dry_run: bool = False,
) -> list[list[str]]:
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    cwd = scripts_dir(bpp_root)
    cmds = [
        [
            python,
            "gen_extra_libero_envs.py",
            "--extra-envs-config-path",
            str(affordance_yaml),
            "--splits",
            *splits,
            "--suffix",
            suffix,
            "--num-workers-init-states",
            str(workers),
        ],
        [
            python,
            "generate_demonstrations.py",
            "--suffix",
            suffix,
            "--include-splits",
            *views,
            "--n-demos",
            str(n_demos),
            "--num-workers",
            str(workers),
            "--run-dir",
            str(run_dir),
            "--skip-status",
        ],
    ]
    for cmd in cmds:
        log.info("run: %s (cwd %s)", " ".join(cmd), cwd)
        if not dry_run:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
    return cmds


def import_generated(
    pool: Pool,
    spec: Spec,
    run_dir: Path,
    views: list[str],
    *,
    skill: str = LIBERO_SKILL,
    max_steps: int | None = None,
    validate: bool = True,
) -> list[str]:
    """A run dir holds bddl_files/<view>/, init_files/<view>/ and <view>/<task>_demo.hdf5 (or demonstration_data/)."""
    imported: list[str] = []
    for view in views:
        bddl_dir = run_dir / "bddl_files" / view
        for bddl in sorted(bddl_dir.glob("*.bddl")):
            name = bddl.stem
            task_id = f"generated/{name}"
            if task_id in pool.tasks:
                continue
            root = (
                run_dir
                if (run_dir / "demonstration_data" / view).exists()
                else _shim_root(run_dir, view)
            )
            goal = goal_from_bddl(bddl)
            meta: dict[str, Any] = {
                "swap": _swap_from_goal(goal),
                "source_split": view,
                "generated": True,
            }
            t = _gen_task(
                pool,
                spec,
                root,
                view,
                name,
                task_id,
                skill,
                "object_swap",
                max_steps or spec.max_steps(skill),
                meta,
                validate,
            )
            if t:
                imported.append(task_id)
    pool.save()
    return imported


def _shim_root(run_dir: Path, view: str) -> Path:
    """BPP writes demos as <run_dir>/<view>/<task>_demo.hdf5; expose them under demonstration_data/."""
    link = run_dir / "demonstration_data"
    link.mkdir(exist_ok=True)
    target = link / view
    if not target.exists() and (run_dir / view).exists():
        target.symlink_to(run_dir / view)
    return run_dir
