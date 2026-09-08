"""`icilval pools upgrade`: a schema-2 pool (perturbation groups, variants) -> schema 3.

Every task and its files are copied unchanged, so nothing is re-simulated; the variants and
the per-group eligibility are dropped, and a task's object-swap description moves from
`perturbation` into `meta.swap`. Finalize afterwards to seal the new pool id.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from ..spec import Spec
from .schema import POOL_SCHEMA, Pool, PoolTask

log = logging.getLogger(__name__)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def upgrade_pool(old_root: Path, out: Path, spec: Spec) -> Pool:
    doc: dict[str, Any] = json.loads((old_root / "pool.json").read_text())
    if int(doc.get("schema", 1)) != 2:
        raise ValueError(f"{old_root}: expected a schema-2 pool, got {doc.get('schema')}")
    pool = Pool(
        schema=POOL_SCHEMA,
        pool_version=str(spec.pools["version"]),
        spec_version=spec.version,
        sources={
            **doc.get("sources", {}),
            "upgraded_from": {
                "pool_id": doc.get("pool_id"),
                "pool_version": doc.get("pool_version"),
            },
        },
        tasks={},
        skills={s: {"eligible": []} for s in spec.skills},
        root=out,
    )
    out.mkdir(parents=True, exist_ok=True)
    for tid, t in sorted(doc["tasks"].items()):
        if t["skill"] not in spec.skills:
            log.info("drop %s: skill %s is not in the spec", tid, t["skill"])
            continue
        meta = dict(t.get("meta", {}))
        swap = {k: v for k, v in t.get("perturbation", {}).items() if k != "kind"}
        if swap:
            meta["swap"] = {k: v for k, v in swap.items() if k != "source_split"}
            if "source_split" in swap:
                meta["source_split"] = swap["source_split"]
        pool.tasks[tid] = PoolTask(
            task_id=tid,
            skill=t["skill"],
            kind=t["kind"],
            suite=t["suite"],
            bddl=t.get("bddl"),
            language=t["language"],
            init=t.get("init"),
            n_init=int(t["n_init"]),
            goal=t.get("goal", []),
            steps=t.get("steps", []),
            demos=list(t["demos"]),
            max_steps=min(int(t["max_steps"]), spec.max_steps(t["skill"])),
            demo_init_index=dict(t.get("demo_init_index", {})),
            provenance=dict(t.get("provenance", {})),
            instances=t.get("instances"),
            meta=meta,
        )
        for rel in (t.get("bddl"), t.get("init")):
            if rel:
                _copy(old_root / rel, out / rel)
        for d in t["demos"]:
            _copy(old_root / "demos" / f"{d}.npz", out / "demos" / f"{d}.npz")
    log.info(
        "kept %d of %d tasks; dropped %d variants",
        len(pool.tasks),
        len(doc["tasks"]),
        len(doc.get("variants", {})),
    )
    pool.pool_id = None
    pool.save()
    return pool
