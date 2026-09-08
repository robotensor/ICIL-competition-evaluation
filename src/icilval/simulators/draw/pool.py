"""The drawing skill's catalogue stage.

One generated-drawing task per primitive family (`skills.draw_anything.generation.families`): a
unit of such a task gets its target and its demonstration generated for it. Beside them, for each
diagnostic of the skill (`spec.json` `diagnostics`), BPP's human-drawn set imported with its stored
demonstrations and flagged diagnostic; those are the only stored demonstrations in a catalogue.
"""

from __future__ import annotations

import logging
from typing import Any

from ...pools.schema import Pool, PoolTask
from ...pools.sources import Sources, fetch
from ...spec import Spec
from .demos import draw_task_names, import_draw_task, open_replay_buffer

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
            demos=[],
            max_steps=spec.max_steps(skill),
            provenance={"source": "generated at duel time", "family": family},
            meta={"family": family, **{k: v for k, v in cfg.items() if not k.startswith("_")}},
        )
        added.append(task_id)
        log.info("draw %s: generated family", task_id)
    pool.save()
    return added


def import_draw_buffer(
    pool: Pool,
    spec: Spec,
    root: Any,
    *,
    skill: str,
    group: str,
    source: str,
    provenance: dict[str, Any],
    demos_per_task: int,
    limit: int | None = None,
    task_names: list[str] | None = None,
    diagnostic: bool = False,
) -> list[str]:
    n_init = int(spec.catalogue["init_states_per_task"])
    imported: list[str] = []
    for name in (task_names or draw_task_names(root))[:limit]:
        task_id = f"{group}/{sanitize_task_name(name)}"
        if task_id in pool.tasks:
            continue
        out_dir = pool.path("demos") / task_id
        metas = import_draw_task(root, name, out_dir, task_id, demos_per_task, source=source)
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
            diagnostic=diagnostic,
        )
        imported.append(task_id)
        log.info("draw %s: %d demos%s", task_id, len(metas), " (diagnostic)" if diagnostic else "")
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
    """The family tasks, then every diagnostic of the skill from its replay-buffer file
    (`skills.<skill>.tasks.files[<group>]`); `limit` applies per file."""
    got = stage_families(pool, spec, skill)
    tasks_cfg = spec.tasks(skill)
    dataset = str(tasks_cfg["dataset"])
    root = src.dataset_root(dataset)
    for name, diag in spec.diagnostics.items():
        if diag["skill"] != skill:
            continue
        group = str(diag["group"])
        filename = str(tasks_cfg["files"][group])
        path = root / filename
        if not path.exists() and fetch_missing and filename.endswith(".zip"):
            path = fetch(root, dataset, filename)
        if not path.exists():
            log.warning("draw: %s missing, skipping diagnostic %s", path, name)
            continue
        buffer = open_replay_buffer(path)
        got += import_draw_buffer(
            pool,
            spec,
            buffer,
            skill=skill,
            group=group,
            source=path.name,
            provenance={
                "source": f"DrawAnything-Sim {filename} (public)",
                "dataset": dataset,
                "diagnostic": name,
            },
            demos_per_task=int(diag["demos_per_task"]),
            limit=limit,
            diagnostic=True,
        )
        pool.sources.setdefault(root.name, {})[group] = str(path)
        pool.save()
    return got
