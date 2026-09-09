"""The drawing skill's catalogue stage.

One generated-drawing task per primitive family (`skills.draw_anything.generation.families`): a
unit of such a task gets its target and its demonstration generated for it. The skill imports
nothing, so it names no dataset.
"""

from __future__ import annotations

import logging

from ...pools.schema import Pool, PoolTask
from ...pools.sources import Sources
from ...spec import Spec

log = logging.getLogger(__name__)

GENERATED_GROUP = "drawanything_generated"


def family_task_id(family: str) -> str:
    return f"{GENERATED_GROUP}/{family}"


def stage_families(pool: Pool, spec: Spec, skill: str) -> list[str]:
    """One task per primitive family; its units' targets and demonstrations are generated."""
    families = {
        k: v
        for k, v in (spec.skill_generation(skill).get("families") or {}).items()
        if not k.startswith("_")
    }
    n_init = int(spec.catalogue["init_states_per_task"])
    added: list[str] = []
    for family, cfg in families.items():
        task_id = family_task_id(family)
        if task_id in pool.tasks:
            continue
        pool.tasks[task_id] = PoolTask(
            task_id=task_id,
            skill=skill,
            kind="drawing",
            suite=GENERATED_GROUP,
            language=f"draw a generated {family} drawing",
            n_init=n_init,
            max_steps=spec.max_steps(skill),
            provenance={"source": "generated at duel time", "family": family},
            meta={"family": family, **{k: v for k, v in cfg.items() if not k.startswith("_")}},
        )
        added.append(task_id)
        log.info("draw %s: generated family", task_id)
    pool.save()
    return added


def stage_draw(
    pool: Pool,
    spec: Spec,
    src: Sources,
    *,
    skill: str,
    limit: int | None = None,
    fetch_missing: bool = False,
) -> list[str]:
    """One task per primitive family. Nothing is downloaded: a drawing task is a family and a
    table of initial states, and every prompt is generated when a duel runs."""
    return stage_families(pool, spec, skill)
