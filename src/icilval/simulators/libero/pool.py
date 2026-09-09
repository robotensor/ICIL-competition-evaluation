"""The catalogue stage of a LIBERO-Gen skill: every task of the views `spec.json` names that
BPP generated demonstrations for, as a task definition - its BDDL as the release ships it, BPP's
task metadata (which human demonstration each grasp is lifted from, read from the vendored BPP
checkout at the pinned commit) and its goal - plus the sha256 of the human teleoperation files
those grasps come from. No demonstration or initial-state file is imported: a prompt is
generated per unit and the scored scene is a numbered reset of the task's BDDL."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ...pools.schema import Pool, PoolTask
from ...pools.sources import Sources, fetch, grasp_source_hashes, hub_files
from ...spec import Spec
from . import bddl as B
from . import validate as V

log = logging.getLogger(__name__)

METADATA_REL = "behavior_prompting/train_network/env/libero/bddl_files/{view}/task_metadata.yaml"
METADATA_KEYS = (
    "execution_steps",
    "grasps_from",
    "actions_from",
    "actions_from_steps",
    "is_existing_task",
)


def copy_bddl(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def load_task_metadata(bpp_root: Path, view: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """BPP's `task_metadata.yaml` of a view: the base split its grasps come from, and task name
    -> metadata (execution steps, grasps_from / actions_from)."""
    import yaml

    doc = yaml.safe_load((bpp_root / METADATA_REL.format(view=view)).read_text())
    return str(doc["base_split_name"]), dict(doc["tasks"])


def execution_steps(meta: dict[str, Any]) -> list[list[str]]:
    """`"Grasp akita_black_bowl_2"` -> `["Grasp", "akita_black_bowl_2"]`, per step."""
    return [str(step).split(" ") for step in meta.get("execution_steps", [])]


def task_meta(name: str, metadata: dict[str, dict[str, Any]], base_split: str, view: str) -> dict:
    meta = {k: metadata[name][k] for k in METADATA_KEYS if k in metadata[name]}
    meta["base_split"] = base_split
    meta["source_split"] = view
    return meta


def task_names(root: Path, dataset: str, view: str, online: bool, limit: int | None) -> list[str]:
    """The tasks BPP generated demonstrations for: the hub's `demonstration_data/<view>/` when
    online, else the local `bddl_files/<view>/`."""
    if online:
        names = [
            Path(f).name[: -len("_demo.hdf5")]
            for f in hub_files(dataset, f"demonstration_data/{view}/")
            if f.endswith("_demo.hdf5")
        ]
    else:
        names = sorted(p.stem for p in (root / "bddl_files" / view).glob("*.bddl"))
    return names[:limit]


def stage_libero_gen(
    pool: Pool,
    spec: Spec,
    src: Sources,
    skill: str,
    *,
    limit: int | None = None,
    validate: bool = True,
    fetch_missing: bool = False,
    **_ignored: Any,
) -> None:
    """Every task of the skill's LIBERO-Gen views that has demonstrations, as a definition."""
    tasks_cfg = spec.tasks(skill)
    dataset = str(tasks_cfg["dataset"])
    root = src.dataset_root(dataset)
    group = root.name
    n_init = int(spec.catalogue["init_states_per_task"])
    commit = str(spec.env(skill)["behavior_prompting_commit"])
    for view in tasks_cfg["views"]:
        base_split, metadata = load_task_metadata(src.bpp_root, view)
        for name in task_names(root, dataset, view, fetch_missing, limit):
            task_id = f"{group}/{name}"
            if task_id in pool.tasks:
                continue
            if name not in metadata:
                log.warning("skip %s: not in BPP's task metadata for %s", task_id, view)
                continue
            rel = f"bddl_files/{view}/{name}.bddl"
            bddl = fetch(root, dataset, rel) if fetch_missing else root / rel
            if not bddl.exists():
                log.warning("skip %s: missing %s", task_id, rel)
                continue
            bddl_rel = f"bddl/{group}/{name}.bddl"
            copy_bddl(bddl, pool.path(bddl_rel))
            if validate and not V.generates_from_reset(pool.path(bddl_rel), spec, skill):
                log.warning(
                    "skip %s: goal already satisfied at reset or scene does not build", name
                )
                continue
            tree = B.load(pool.path(bddl_rel))
            meta = task_meta(name, metadata, base_split, view)
            pool.tasks[task_id] = PoolTask(
                task_id=task_id,
                skill=skill,
                kind=str(tasks_cfg["kind"]),
                suite=view,
                bddl=bddl_rel,
                language=B.language(tree),
                n_init=n_init,
                goal=B.goal_predicates(tree),
                max_steps=spec.max_steps(skill),
                steps=execution_steps(meta),
                provenance={
                    "source": "LIBERO-Gen (public)",
                    "split": view,
                    "bddl": rel,
                    "task_metadata": {"behavior_prompting_commit": commit, "view": view},
                },
                meta=meta,
            )
            log.info("%s %s: %d steps", skill, task_id, len(meta.get("execution_steps", [])))
        pool.save()
    entry: dict[str, Any] = {
        "dataset": dataset,
        "root": str(root),
        "views": list(tasks_cfg["views"]),
    }
    grasp = tasks_cfg.get("grasp_sources") or {}
    if grasp:
        entry["grasp_sources"] = {
            "dataset": str(grasp["dataset"]),
            "prefix": str(grasp["prefix"]),
            "files": grasp_source_hashes(
                src, str(grasp["dataset"]), str(grasp["prefix"]), fetch_missing
            ),
        }
    pool.sources[group] = entry
    pool.save()
