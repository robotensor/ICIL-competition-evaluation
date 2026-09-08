"""The pool manifest: everything a duel draws from, content-addressed.

A pool is a directory holding `pool.json` plus, per skill, the files its
simulator needs: `bddl/`, `init/` and `demos/` for LIBERO, `demos/` for the
drawing board. Tasks are the things a prompt demonstration exists for; a unit
is one task, one of its initial states, one of its demonstrations and a seed.
Eligibility is one list of task ids per skill, sealed into the pool id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..canon import canonical_sha256

POOL_SCHEMA = 3


@dataclass
class PoolTask:
    task_id: str
    skill: str
    kind: str  # base | object_swap | drawing
    suite: str
    language: str
    n_init: int
    demos: list[str]
    max_steps: int
    bddl: str | None = None
    init: str | None = None
    goal: list[list[str]] = field(default_factory=list)
    steps: list[list[str]] = field(default_factory=list)
    demo_init_index: dict[str, int | None] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    instances: list[int] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_instances(self) -> list[int]:
        return list(self.instances) if self.instances is not None else list(range(self.n_init))

    @property
    def label(self) -> str:
        return self.language.strip().rstrip(".").capitalize() if self.language else self.task_id


@dataclass
class Pool:
    schema: int
    pool_version: str
    spec_version: int
    sources: dict[str, Any]
    tasks: dict[str, PoolTask]
    skills: dict[str, dict[str, list[str]]]  # skill -> {"eligible": [task_id, ...]}
    pool_id: str | None = None
    root: Path | None = None

    # -- (de)serialisation
    @classmethod
    def from_dict(cls, d: dict[str, Any], root: Path | None = None) -> Pool:
        schema = int(d.get("schema", POOL_SCHEMA))
        if schema != POOL_SCHEMA:
            raise ValueError(
                f"pool schema {schema} is not {POOL_SCHEMA}; run `icilval pools upgrade`"
            )
        tasks = {
            k: PoolTask(task_id=k, **{kk: vv for kk, vv in v.items() if kk != "task_id"})
            for k, v in d.get("tasks", {}).items()
        }
        return cls(
            schema=schema,
            pool_version=str(d["pool_version"]),
            spec_version=int(d["spec_version"]),
            sources=dict(d.get("sources", {})),
            tasks=tasks,
            skills={
                s: {"eligible": list(e.get("eligible", []))} for s, e in d.get("skills", {}).items()
            },
            pool_id=d.get("pool_id"),
            root=root,
        )

    def to_dict(self, with_id: bool = True) -> dict[str, Any]:
        def task_dict(t: PoolTask) -> dict[str, Any]:
            return {
                "skill": t.skill,
                "kind": t.kind,
                "suite": t.suite,
                "bddl": t.bddl,
                "language": t.language,
                "init": t.init,
                "n_init": t.n_init,
                "goal": t.goal,
                "steps": t.steps,
                "demos": t.demos,
                "demo_init_index": t.demo_init_index,
                "max_steps": t.max_steps,
                "provenance": t.provenance,
                "instances": t.instances,
                "meta": t.meta,
            }

        d: dict[str, Any] = {
            "schema": self.schema,
            "pool_version": self.pool_version,
            "spec_version": self.spec_version,
            "sources": self.sources,
            "tasks": {k: task_dict(t) for k, t in sorted(self.tasks.items())},
            "skills": {
                s: {"eligible": sorted(e["eligible"])} for s, e in sorted(self.skills.items())
            },
        }
        if with_id:
            d["pool_id"] = self.pool_id
        return d

    def compute_id(self) -> str:
        return canonical_sha256(self.to_dict(with_id=False))

    def seal(self) -> str:
        self.pool_id = self.compute_id()
        return self.pool_id

    @classmethod
    def load(cls, root: str | Path) -> Pool:
        root = Path(root)
        doc = json.loads((root / "pool.json").read_text())
        pool = cls.from_dict(doc, root=root)
        if pool.pool_id and pool.pool_id != pool.compute_id():
            raise ValueError(f"{root}/pool.json: pool_id does not match its content")
        return pool

    def save(self, root: str | Path | None = None) -> Path:
        root = Path(root or self.root)
        root.mkdir(parents=True, exist_ok=True)
        if not self.pool_id:
            self.seal()
        path = root / "pool.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        self.root = root
        return path

    # -- lookups
    def eligible(self, skill: str) -> list[str]:
        return sorted(self.skills.get(skill, {}).get("eligible", []))

    def tasks_of(self, skill: str) -> list[PoolTask]:
        return [t for _, t in sorted(self.tasks.items()) if t.skill == skill]

    def path(self, rel: str) -> Path:
        if self.root is None:
            raise ValueError("pool has no root directory")
        return self.root / rel
