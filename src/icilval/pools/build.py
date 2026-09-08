"""`icilval pools build`: turn raw benchmark assets into a content-addressed pool.

Stages (each idempotent, each appends to pool.json):
  pick_and_place  LIBERO-Gen Combination, both views: bddl, init npz, k demos per task
  draw            DrawAnything-Sim: the human-drawn evaluation set (see build_draw)
  finalize        the eligible tasks per skill, pool_id

The pick-and-place tasks are exactly the ones BPP generated demonstrations for; the pool does
not distinguish the view a task came from beyond recording it in `meta.source_split`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..sim import bddl as B
from ..spec import Spec
from . import validate as V
from .demos import import_hdf5
from .schema import POOL_SCHEMA, Pool, PoolTask
from .sources import Sources, convert_init, copy_bddl

log = logging.getLogger(__name__)

LIBERO_SKILL = "pick_and_place"
DRAW_SKILL = "draw_anything"
COMBINATION_GROUP = "libero_gen_spatial_combination"
COMBINATION_VIEWS = (
    "libero_spatial_selected_combinations_view",
    "libero_spatial_selected_combinations_inverse_view",
)


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
) -> PoolTask | None:
    bddl = split_root / "bddl_files" / split / f"{name}.bddl"
    init = split_root / "init_files" / split / f"{name}.pruned_init"
    h5 = split_root / "demonstration_data" / split / f"{name}_demo.hdf5"
    if not (bddl.exists() and init.exists() and h5.exists()):
        log.warning("skip %s: missing bddl/init/demos", task_id)
        return None
    group = task_id.split("/", 1)[0]
    bddl_rel, init_rel = f"bddl/{group}/{name}.bddl", f"init/{group}/{name}.npz"
    copy_bddl(bddl, pool.path(bddl_rel))
    n_init = convert_init(init, pool.path(init_rel))
    demos = _import_demos(h5, pool, task_id, int(spec.pools["demos_per_task"]))
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
    )
    if validate:
        env = V.build_task_env(pool.path(bddl_rel), spec, skill)
        if env is None:
            del pool.tasks[task_id]
            return None
        task.instances = V.valid_instances(env, V.load_init_states(pool.path(init_rel)))
        env.close()
    return task


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


def stage_pick_and_place(
    pool: Pool,
    spec: Spec,
    src: Sources,
    *,
    skill: str = LIBERO_SKILL,
    limit: int | None = None,
    validate: bool = True,
) -> None:
    """Every LIBERO-Gen Combination task with demonstrations, from both of BPP's views."""
    root = src.gen_spatial_combination
    for view in COMBINATION_VIEWS:
        names = sorted(p.stem for p in (root / "bddl_files" / view).glob("*.bddl"))[:limit]
        for name in names:
            task_id = f"{COMBINATION_GROUP}/{name}"
            if task_id in pool.tasks:
                continue
            goal = V.goal_from_bddl(root / "bddl_files" / view / f"{name}.bddl")
            t = _gen_task(
                pool,
                spec,
                root,
                view,
                name,
                task_id,
                skill,
                "combination",
                spec.max_steps(skill),
                {"swap": _swap_from_goal(goal), "source_split": view},
                validate,
            )
            if t:
                log.info("%s %s: %d inits, %d demos", skill, task_id, t.n_init, len(t.demos))
        pool.save()
    pool.sources["libero_gen_spatial_combination"] = {
        "root": str(root),
        "views": list(COMBINATION_VIEWS),
    }
    pool.save()


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
    "stage_pick_and_place",
    "eligible_tasks",
    "finalize",
    "verify_pool",
    "summary",
]
