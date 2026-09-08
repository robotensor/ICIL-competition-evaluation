"""`icilval pools build`: turn raw benchmark assets into a content-addressed pool.

One stage per skill, named by the skill id (each idempotent, each appends to pool.json), then
`finalize`. Where a skill's tasks come from is data: `spec.json` `skills.<skill>.tasks` names
the Hugging Face dataset and what to import from it; how is the skill's simulator's
`build_stage` (`simulators/<sim>/pool.py`). The pool records the view or file a task came from
and treats them alike.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..spec import Spec
from .schema import POOL_SCHEMA, Pool
from .sources import Sources

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
    "eligible_tasks",
    "finalize",
    "verify_pool",
    "summary",
]
