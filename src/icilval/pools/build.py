"""`icilval pools build`: turn raw benchmark assets into a content-addressed pool.

One stage per skill, named by the skill id (each idempotent, each appends to pool.json), then
`finalize`. Where a skill's tasks come from is data: `spec.json` `skills.<skill>.tasks` names
the Hugging Face dataset and either the LIBERO-Gen views (bddl, init states, hdf5 demos per
task) or the DrawAnything-Sim replay-buffer files to import. The pool records the view or file
a task came from in `meta.source_split` / its group and treats them alike.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..sim import bddl as B
from ..spec import Spec
from . import validate as V
from .demos import import_hdf5, load_demo
from .schema import POOL_SCHEMA, Pool, PoolTask
from .sources import Sources, convert_init, copy_bddl, evict, fetch, hub_files

log = logging.getLogger(__name__)


def open_pool(out: Path, spec: Spec, version: str) -> Pool:
    if (out / "pool.json").exists():
        pool = Pool.load(out)
        pool.pool_id = None
        return pool
    return Pool(
        schema=POOL_SCHEMA,
        pool_version=version,
        spec_version=spec.version,
        sources={},
        tasks={},
        skills={s: {"eligible": []} for s in spec.skills},
        root=out,
    )


def _task_from_bddl(
    task_id: str,
    skill: str,
    kind: str,
    suite: str,
    bddl_rel: str,
    init_rel: str,
    n_init: int,
    demos: list[str],
    max_steps: int,
    pool: Pool,
    **extra: Any,
) -> PoolTask:
    tree = B.load(pool.path(bddl_rel))
    goal = B.goal_predicates(tree)
    task = PoolTask(
        task_id=task_id,
        skill=skill,
        kind=kind,
        suite=suite,
        bddl=bddl_rel,
        language=B.language(tree),
        init=init_rel,
        n_init=n_init,
        goal=goal,
        demos=demos,
        max_steps=max_steps,
        steps=goal,
        **extra,
    )
    pool.tasks[task_id] = task
    return task


def _import_demos(h5: Path, pool: Pool, task_id: str, k: int) -> list[str]:
    out_dir = pool.path("demos") / task_id
    if out_dir.exists() and len(list(out_dir.glob("demo_*.npz"))) >= min(k, 1):
        return sorted(f"{task_id}/{p.stem}" for p in out_dir.glob("demo_*.npz"))
    metas = import_hdf5(h5, out_dir, task_id, k)
    return [m.demo_id for m in metas]


def _gen_task(
    pool: Pool,
    spec: Spec,
    split_root: Path,
    split: str,
    name: str,
    task_id: str,
    skill: str,
    kind: str,
    max_steps: int,
    meta: dict[str, Any],
    validate: bool,
    *,
    source_key: str | None = None,
    evict_demos: bool = False,
) -> PoolTask | None:
    """One LIBERO-Gen task from `split_root` (bddl_files/, init_files/, demonstration_data/).

    With `source_key` the three files are fetched from the hub when missing; with
    `evict_demos` the demonstration file is deleted once its demos are in the pool.
    """
    rels = (
        f"bddl_files/{split}/{name}.bddl",
        f"init_files/{split}/{name}.pruned_init",
        f"demonstration_data/{split}/{name}_demo.hdf5",
    )
    if source_key is not None:
        bddl, init, h5 = (fetch(split_root, source_key, r) for r in rels)
    else:
        bddl, init, h5 = (split_root / r for r in rels)
    if not (bddl.exists() and init.exists() and h5.exists()):
        log.warning("skip %s: missing bddl/init/demos", task_id)
        return None
    group = task_id.split("/", 1)[0]
    bddl_rel, init_rel = f"bddl/{group}/{name}.bddl", f"init/{group}/{name}.npz"
    copy_bddl(bddl, pool.path(bddl_rel))
    n_init = convert_init(init, pool.path(init_rel))
    demos = _import_demos(h5, pool, task_id, int(spec.pools["demos_per_task"]))
    if evict_demos:
        evict(h5)
    init_index = demo_init_indices(pool, demos, V.load_init_states(pool.path(init_rel)))
    task = _task_from_bddl(
        task_id,
        skill,
        kind,
        split,
        bddl_rel,
        init_rel,
        n_init,
        demos,
        max_steps,
        pool,
        provenance={"source": "LIBERO-Gen (public)", "split": split, "demos": h5.name},
        meta=meta,
        demo_init_index=init_index,
    )
    if validate:
        env = V.build_task_env(pool.path(bddl_rel), spec, skill)
        if env is None:
            del pool.tasks[task_id]
            return None
        task.instances = V.valid_instances(env, V.load_init_states(pool.path(init_rel)))
        env.close()
    return task


def demo_init_indices(
    pool: Pool, demos: list[str], states: np.ndarray, atol: float = 1e-6
) -> dict[str, int | None]:
    """Which of the task's initial states each demonstration started from (None: none of them).

    Unit derivation never prompts with a demonstration that starts from the scored state.
    """
    out: dict[str, int | None] = {}
    for demo_id in demos:
        start = load_demo(pool.path("demos") / f"{demo_id}.npz")["init_state"]
        hit = None
        if states.ndim == 2 and states.shape[1] == start.shape[0]:
            close = np.all(np.abs(states - start[None, :]) <= atol, axis=1)
            idx = np.flatnonzero(close)
            hit = int(idx[0]) if idx.size else None
        out[demo_id] = hit
    return out


def _swap_from_goal(goal: list[list[str]]) -> dict[str, Any]:
    """What the task grasps and where it goes, read off its single On/In goal."""
    for p in goal:
        if len(p) == 3 and p[0].lower() in ("on", "in"):
            return {
                "operator": "place_in" if p[0].lower() == "in" else "place_on",
                "object": p[1],
                "target": p[2],
            }
    return {}


def stage_libero_gen(
    pool: Pool,
    spec: Spec,
    src: Sources,
    skill: str,
    *,
    limit: int | None = None,
    validate: bool = True,
    fetch_missing: bool = False,
    evict_demos: bool = False,
) -> None:
    """Every task of the skill's LIBERO-Gen views that has demonstrations.

    The task list is the hub's `demonstration_data/<view>/` when fetching (only tasks that
    have demonstrations), else the local `bddl_files/<view>/`.
    """
    tasks_cfg = spec.tasks(skill)
    dataset = str(tasks_cfg["dataset"])
    root = src.dataset_root(dataset)
    group = root.name
    key = dataset if fetch_missing else None
    for view in tasks_cfg["views"]:
        if fetch_missing:
            names = [
                Path(f).name[: -len("_demo.hdf5")]
                for f in hub_files(dataset, f"demonstration_data/{view}/")
                if f.endswith("_demo.hdf5")
            ][:limit]
        else:
            names = sorted(p.stem for p in (root / "bddl_files" / view).glob("*.bddl"))[:limit]
        for name in names:
            task_id = f"{group}/{name}"
            if task_id in pool.tasks:
                continue
            bddl = root / "bddl_files" / view / f"{name}.bddl"
            if key is not None:
                bddl = fetch(root, key, f"bddl_files/{view}/{name}.bddl")
            goal = V.goal_from_bddl(bddl)
            t = _gen_task(
                pool,
                spec,
                root,
                view,
                name,
                task_id,
                skill,
                str(tasks_cfg["kind"]),
                spec.max_steps(skill),
                {"swap": _swap_from_goal(goal), "source_split": view},
                validate,
                source_key=key,
                evict_demos=evict_demos,
            )
            if t:
                log.info("%s %s: %d inits, %d demos", skill, task_id, t.n_init, len(t.demos))
        pool.save()
    pool.sources[group] = {"dataset": dataset, "root": str(root), "views": list(tasks_cfg["views"])}
    pool.save()


def stage_skill(
    pool: Pool,
    spec: Spec,
    src: Sources,
    skill: str,
    *,
    limit: int | None = None,
    validate: bool = True,
    fetch_missing: bool = False,
    evict_demos: bool = False,
) -> None:
    """The pool stage of one skill, from its simulator."""
    from ..simulators import for_skill

    for_skill(spec, skill).build_stage(
        pool,
        spec,
        src,
        skill,
        limit=limit,
        validate=validate,
        fetch_missing=fetch_missing,
        evict_demos=evict_demos,
    )


# ---------------------------------------------------------------- finalize
def eligible_tasks(pool: Pool, skill: str) -> list[str]:
    """Tasks of a skill a unit can be drawn for: at least one usable initial state and one demo."""
    return sorted(
        t.task_id for t in pool.tasks.values() if t.skill == skill and t.valid_instances and t.demos
    )


def finalize(pool: Pool, spec: Spec) -> dict[str, int]:
    pool.skills = {skill: {"eligible": eligible_tasks(pool, skill)} for skill in spec.skills}
    pool.spec_version = spec.version
    pool.seal()
    pool.save()
    return {s: len(e["eligible"]) for s, e in pool.skills.items()}


def verify_pool(root: Path, spec: Spec | None = None) -> list[str]:
    errors: list[str] = []
    pool = Pool.load(root)
    for tid, t in pool.tasks.items():
        for rel in (t.bddl, t.init):
            if rel and not pool.path(rel).exists():
                errors.append(f"{tid}: missing {rel}")
        for d in t.demos:
            if not (pool.path("demos") / f"{d}.npz").exists():
                errors.append(f"{tid}: missing demo {d}")
        if t.bddl:
            try:
                B.load(pool.path(t.bddl))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{tid}: bddl unparsable: {exc}")
        if spec is not None and t.skill not in spec.skills:
            errors.append(f"{tid}: unknown skill {t.skill}")
    skills = list(spec.skills) if spec is not None else list(pool.skills)
    for skill in skills:
        if not pool.eligible(skill):
            errors.append(f"{skill}: nothing eligible")
        for e in pool.eligible(skill):
            if e not in pool.tasks:
                errors.append(f"{skill}: unknown task {e}")
            elif pool.tasks[e].skill != skill:
                errors.append(f"{skill}: task {e} belongs to {pool.tasks[e].skill}")
    return errors


def summary(pool: Pool) -> dict[str, Any]:
    kinds: dict[str, dict[str, int]] = {}
    for t in pool.tasks.values():
        kinds.setdefault(t.skill, {})
        kinds[t.skill][t.kind] = kinds[t.skill].get(t.kind, 0) + 1
    return {
        "pool_id": pool.pool_id,
        "tasks": {s: len(pool.tasks_of(s)) for s in pool.skills} | {"total": len(pool.tasks)},
        "kinds": kinds,
        "eligible": {s: len(e["eligible"]) for s, e in pool.skills.items()},
        "demos": sum(len(t.demos) for t in pool.tasks.values()),
    }


__all__ = [
    "open_pool",
    "stage_skill",
    "stage_libero_gen",
    "eligible_tasks",
    "finalize",
    "verify_pool",
    "summary",
]
