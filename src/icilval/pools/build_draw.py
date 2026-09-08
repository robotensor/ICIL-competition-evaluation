"""The drawing skill's pool stage: BPP's public DrawAnything-Sim sets, and organizer-generated
procedural drawings through BPP's own generator.

`icilval pools build --stage draw` imports `eval_handmade.zarr` (50 human drawings, 5
demonstrations each) and `procedural_2000_10.zarr.zip` (2000 procedural drawings, 10 each, read
from the zip in place); both are public and the pool treats them alike. `icilval pools
generate-draw` runs, inside the BPP checkout:
  scripts/draw/procedural_generate_drawings.py -o <run_dir> --num-tasks N --demos-per-task K
      --base-seed <organizer seed> --no-visualize --no-group-demos --no-vis-grouped-demos
and imports every `<run_dir>/*.zarr` it wrote as `drawanything_generated/<task>`; those are
never published as training data.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from ..spec import Spec
from .demos import draw_task_names, import_draw_task, open_replay_buffer
from .schema import Pool, PoolTask
from .sources import Sources, fetch

log = logging.getLogger(__name__)

GENERATED_GROUP = "drawanything_generated"
SYMBOLS = {
    "!": "bang",
    "#": "hash",
    "$": "dollar",
    "%": "percent",
    "&": "amp",
    "*": "star",
    "+": "plus",
    ">": "gt",
    "?": "qmark",
    "@": "at",
}


def sanitize_task_name(name: str) -> str:
    """BPP's `draw <name>` -> a path-safe id; case is kept the way BPP's own dataset names do."""
    body = name[len("draw ") :] if name.startswith("draw ") else name
    if body.upper() == body and body.lower() != body:
        suffix = "_upper"
    elif body.lower() == body and body.upper() != body:
        suffix = "_lower"
    else:
        suffix = ""
    out = []
    for ch in body:
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        elif ch in SYMBOLS:
            out.append(SYMBOLS[ch])
        else:
            out.append("_")
    return "draw_" + "".join(out) + suffix


def import_draw_buffer(
    pool: Pool,
    spec: Spec,
    root: Any,
    *,
    skill: str,
    group: str,
    source: str,
    provenance: dict[str, Any],
    limit: int | None = None,
    task_names: list[str] | None = None,
) -> list[str]:
    k = int(spec.pools["demos_per_task"])
    n_init = int(spec.pools["init_states_per_task"])
    imported: list[str] = []
    for name in (task_names or draw_task_names(root))[:limit]:
        task_id = f"{group}/{sanitize_task_name(name)}"
        if task_id in pool.tasks:
            continue
        out_dir = pool.path("demos") / task_id
        metas = import_draw_task(root, name, out_dir, task_id, k, source=source)
        pool.tasks[task_id] = PoolTask(
            task_id=task_id,
            skill=skill,
            kind="drawing",
            suite=group,
            language=name,
            n_init=n_init,
            demos=[m.demo_id for m in metas],
            max_steps=spec.max_steps(skill),
            provenance={**provenance, "task": name},
            meta={
                "demo_angles": {m.demo_id: round(float(m.boundary_angle or 0.0), 6) for m in metas},
                "demo_steps": {m.demo_id: m.steps for m in metas},
            },
        )
        imported.append(task_id)
        log.info("draw %s: %d demos", task_id, len(metas))
    pool.save()
    return imported


def stage_draw(
    pool: Pool,
    spec: Spec,
    src: Sources,
    *,
    skill: str,
    limit: int | None = None,
    fetch_missing: bool = False,
) -> list[str]:
    """Every replay-buffer file `spec.json` lists for the skill, each under its own group;
    `limit` applies per file. A `.zarr.zip` is read in place through zarr's ZipStore."""
    tasks_cfg = spec.tasks(skill)
    dataset = str(tasks_cfg["dataset"])
    root = src.dataset_root(dataset)
    got: list[str] = []
    for group, filename in tasks_cfg["files"].items():
        path = root / filename
        if not path.exists() and fetch_missing and filename.endswith(".zip"):
            path = fetch(root, dataset, filename)
        if not path.exists():
            log.warning("draw: %s missing, skipping %s", path, group)
            continue
        buffer = open_replay_buffer(path)
        got += import_draw_buffer(
            pool,
            spec,
            buffer,
            skill=skill,
            group=group,
            source=path.name,
            provenance={"source": f"DrawAnything-Sim {filename} (public)", "dataset": dataset},
            limit=limit,
        )
        pool.sources.setdefault(root.name, {})[group] = str(path)
        pool.save()
    return got


# ---------------------------------------------------------------- generated drawings
def scripts_dir(bpp_root: Path) -> Path:
    return bpp_root / "behavior_prompting" / "train_network" / "scripts" / "draw"


def generate_draw(
    bpp_root: Path,
    run_dir: Path,
    *,
    n_tasks: int,
    demos_per_task: int,
    base_seed: int,
    workers: int,
    python: str = "python",
    dry_run: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
    cmd = [
        python,
        "procedural_generate_drawings.py",
        "-o",
        str(run_dir),
        "--num-tasks",
        str(n_tasks),
        "--demos-per-task",
        str(demos_per_task),
        "--base-seed",
        str(base_seed),
        "--max-workers",
        str(workers),
        "--no-visualize",
        "--no-group-demos",
        "--no-vis-grouped-demos",
        *(extra_args or []),
    ]
    log.info("run: %s (cwd %s)", " ".join(cmd), scripts_dir(bpp_root))
    if not dry_run:
        subprocess.run(cmd, cwd=scripts_dir(bpp_root), env=env, check=True)
    return cmd


def import_generated_draw(
    pool: Pool,
    spec: Spec,
    run_dir: Path,
    *,
    skill: str,
    limit: int | None = None,
) -> list[str]:
    imported: list[str] = []
    for zarr_dir in sorted(p for p in run_dir.glob("*.zarr") if p.is_dir())[:limit]:
        root = open_replay_buffer(zarr_dir)
        imported += import_draw_buffer(
            pool,
            spec,
            root,
            skill=skill,
            group=GENERATED_GROUP,
            source=zarr_dir.name,
            provenance={"source": "organizer-generated (BPP procedural_generate_drawings)"},
        )
    return imported
