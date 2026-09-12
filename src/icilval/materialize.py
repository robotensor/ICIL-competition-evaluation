"""Fixing a duel's prompts before either side runs.

Two fields want two different things of a prompt, and the difference is not cosmetic.

A field whose demonstration starts somewhere other than the state it scores can publish its
prompts **up front**: knowing the pool tells an entrant what the tasks look like, not what to do
in the episode it is scored on. That is the `pool` source, and it is what the sensorimotor field
has always done.

A field that scores the very scene it demonstrates cannot. There the demonstration *is* the
answer for that scene, so a pool published in advance is close to a published answer key. That
field uses the `materialized` source: the organizer holds the seeds, the orchestrator produces
each duel's prompts on the validator host before either side runs, and they are published **with
the event**. Verification becomes immediate-after rather than before - anyone can still check the
bytes by hash and rebuild the scene from the published seed, they just cannot train on them
first.

Either way both sides see identical bytes, because the prompts are fixed before either side
starts. Reproducibility is verification by hash, not bit-exact regeneration: a third party checks
what was published rather than being asked to reproduce an expert's plan on another GPU stack.

Nothing here runs in the model container.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canon import sha256_file
from .spec import Spec

log = logging.getLogger(__name__)

#: Where a field's prompts come from.
FROM_POOL = "pool"
FROM_MATERIALIZE = "materialized"


class MaterializeFailed(RuntimeError):
    """A unit's prompt could not be produced, or is not the one that was asked for."""


@dataclass
class Prompt:
    """One unit's prompt, on the validator host."""

    unit_id: str
    path: Path
    sha256: str
    #: Set when the unit this prompt belongs to replaced one whose expert never succeeded.
    substituted_from: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"unit_id": self.unit_id, "sha256": self.sha256}
        if self.substituted_from:
            out["substituted_from"] = self.substituted_from
        return out


@dataclass
class Materialized:
    """What a duel's materialize phase produced."""

    root: Path
    prompts: dict[str, Prompt] = field(default_factory=dict)

    def manifest(self) -> list[dict[str, Any]]:
        return [self.prompts[k].as_dict() for k in sorted(self.prompts)]


def needed(spec: Spec, track: str) -> bool:
    """Whether this field's prompts are produced per duel rather than drawn from a pool."""
    return spec.prompts(track) == FROM_MATERIALIZE


def materialize(
    spec: Spec,
    track: str,
    units: list[dict[str, Any]],
    root: Path,
    *,
    benchmark: Any,
    timeout_s: float | None = None,
    runner: Any = None,
) -> Materialized:
    """Produce and verify every unit's prompt, before either side of the duel starts.

    `benchmark` is the plugin for this field: it builds the argv that does the work, so nothing
    here imports a simulator and the work can run in another image or on another host.
    """
    root.mkdir(parents=True, exist_ok=True)
    run = runner or _run
    out = Materialized(root=root)
    for unit in units:
        unit_id = unit["unit_id"]
        out_dir = root / unit_id
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = [str(a) for a in benchmark.materialize_command(unit=unit, out_dir=str(out_dir))]
        log.info("materialize %s: %s", unit_id, " ".join(argv))
        code, err = run(argv, timeout_s)
        if code != 0:
            raise MaterializeFailed(f"{unit_id}: materialize exited {code}: {err[:300]}")
        verdict = benchmark.verify_prompt(path=str(out_dir), unit=unit)
        if not verdict.get("ok"):
            raise MaterializeFailed(
                f"{unit_id}: the prompt is not the one the unit asked for: "
                + "; ".join(verdict.get("problems") or ["no reason given"])
            )
        out.prompts[unit_id] = Prompt(
            unit_id=unit_id,
            path=out_dir,
            sha256=str(verdict["sha256"]),
            substituted_from=unit.get("substituted_from"),
        )
    (root / "prompts.json").write_text(json.dumps(out.manifest(), indent=1))
    return out


def verify_against(root: Path, manifest: list[dict[str, Any]]) -> list[str]:
    """Re-check a materialized directory against what was published. Pure; no simulator."""
    problems: list[str] = []
    for entry in manifest:
        path = root / entry["unit_id"]
        if not path.exists():
            problems.append(f"{entry['unit_id']}: missing")
            continue
        files = sorted(p for p in path.rglob("*") if p.is_file())
        if not files:
            problems.append(f"{entry['unit_id']}: empty")
    return problems


def _run(argv: list[str], timeout_s: float | None) -> tuple[int, str]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    return proc.returncode, proc.stderr or ""


def prompt_sha256(path: Path) -> str:
    """The content address of a prompt file."""
    return sha256_file(path)
