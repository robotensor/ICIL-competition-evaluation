"""`icilval catalogue build`: turn task definitions into a content-addressed catalogue.

One stage per skill, named by the skill id (each idempotent, each appends to catalogue.json),
then `finalize`. Where a skill's tasks come from is data: `spec.json` `skills.<skill>.tasks`
names the Hugging Face dataset and what to import from it; how is the skill's simulator's
`build_stage` (`simulators/<sim>/pool.py`). A stage also imports the tasks of the skill's
unscored diagnostics (`spec.json` `diagnostics`), which are the only tasks holding stored
demonstrations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..spec import Spec
from .schema import CATALOGUE_FILE, POOL_SCHEMA, Pool
from .sources import Sources

log = logging.getLogger(__name__)


def open_pool(out: Path, spec: Spec, version: str) -> Pool:
    if (out / CATALOGUE_FILE).exists():
        pool = Pool.load(out)
        pool.pool_id = None
        return pool
    return Pool(
        schema=POOL_SCHEMA,
        pool_version=version,
        spec_version=spec.version,
        sources={},
        tasks={},
        skills={s: {"eligible": [], "diagnostic": []} for s in spec.skills},
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
) -> None:
    """The catalogue stage of one skill, from its simulator."""
    from ..simulators import for_skill

    for_skill(spec, skill).build_stage(
        pool, spec, src, skill, limit=limit, validate=validate, fetch_missing=fetch_missing
    )


# ---------------------------------------------------------------- finalize
def eligible_tasks(pool: Pool, skill: str) -> list[str]:
    """Tasks of a skill a scored unit can be drawn for: a prompt can be generated for them."""
    return sorted(
        t.task_id
        for t in pool.tasks.values()
        if t.skill == skill and not t.diagnostic and t.valid_instances
    )


def diagnostic_tasks(pool: Pool, skill: str) -> list[str]:
    """Tasks of a skill's unscored diagnostics: prompted with one of their stored demonstrations."""
    return sorted(
        t.task_id
        for t in pool.tasks.values()
        if t.skill == skill and t.diagnostic and t.valid_instances and t.demos
    )


def finalize(pool: Pool, spec: Spec) -> dict[str, dict[str, int]]:
    pool.skills = {
        skill: {
            "eligible": eligible_tasks(pool, skill),
            "diagnostic": diagnostic_tasks(pool, skill),
        }
        for skill in spec.skills
    }
    pool.spec_version = spec.version
    pool.seal()
    pool.save()
    return {
        s: {"eligible": len(e["eligible"]), "diagnostic": len(e["diagnostic"])}
        for s, e in pool.skills.items()
    }


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
        if t.diagnostic and not t.demos:
            errors.append(f"{tid}: diagnostic task without demonstrations")
        if spec is not None and t.skill not in spec.skills:
            errors.append(f"{tid}: unknown skill {t.skill}")
    skills = list(spec.skills) if spec is not None else list(pool.skills)
    for skill in skills:
        if not pool.eligible(skill):
            errors.append(f"{skill}: nothing eligible")
        for e in pool.eligible(skill) + pool.diagnostic(skill):
            if e not in pool.tasks:
                errors.append(f"{skill}: unknown task {e}")
            elif pool.tasks[e].skill != skill:
                errors.append(f"{skill}: task {e} belongs to {pool.tasks[e].skill}")
        for e in pool.eligible(skill):
            if e in pool.tasks and pool.tasks[e].diagnostic:
                errors.append(f"{skill}: diagnostic task {e} listed as eligible")
        if spec is not None and spec.tasks(skill).get("grasp_sources"):
            recorded = any(
                isinstance(src, dict) and (src.get("grasp_sources") or {}).get("files")
                for src in pool.sources.values()
            )
            if not recorded:
                errors.append(f"{skill}: grasp source hashes not recorded")
    if spec is not None:
        for name, diag in spec.diagnostics.items():
            if int(diag["units_per_duel"]) and not [
                t for t in pool.diagnostic(str(diag["skill"])) if t.startswith(f"{diag['group']}/")
            ]:
                errors.append(f"diagnostic {name}: no tasks in group {diag['group']}")
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
        "eligible": {s: len(e.get("eligible", [])) for s, e in pool.skills.items()},
        "diagnostic": {s: len(e.get("diagnostic", [])) for s, e in pool.skills.items()},
        "demos": sum(len(t.demos) for t in pool.tasks.values()),
    }


__all__ = [
    "open_pool",
    "stage_skill",
    "eligible_tasks",
    "diagnostic_tasks",
    "finalize",
    "verify_pool",
    "summary",
]
