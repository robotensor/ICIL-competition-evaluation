"""The materializing phase: every unit's prompt demonstration is generated before either side
runs, so both sides see the same prompts.

For each scored unit the skill's simulator generates a prompt from the unit's seeds
(`generation.prompt_seed`, `max_attempts` of them); when none succeeds the unit takes the next
task in the duel's shuffled task order (the order unit derivation spread the units over),
records the task it was first derived for as `substituted_from`, and tries again with the next
block of seeds. Once a prompt exists the simulator finalizes the unit against it (a drawing
unit's board keeps the demonstration's angle or pen start, by its change kind). Prompts land in
`<run>/assets/<unit_id>.npz`, their sha256 on the unit.

Generation runs in spawned worker processes: LIBERO reads its config when it is imported, and a
fresh interpreter imports it under the validator's config (`libero_config`) rather than
`~/.libero`; a forked child would inherit the parent's EGL context and hang in the renderer.
Diagnostic units are prompted with stored demonstrations and are left alone.
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ids import prompt_seed
from ..pools.demos import load_demo
from ..pools.schema import Pool
from ..rng import HashRng
from ..simulators import for_skill
from ..spec import Spec

log = logging.getLogger(__name__)

MAX_SUBSTITUTIONS = 3


@dataclass(frozen=True)
class GenerationContext:
    """What generating needs beyond the catalogue: the vendored BPP checkout and the raw cache
    (the grasp-source files), from which a worker builds LIBERO's config."""

    bpp_root: Path
    raw_root: Path
    libero_datasets: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "bpp_root": str(self.bpp_root),
            "raw_root": str(self.raw_root),
            "libero_datasets": str(self.libero_datasets),
        }


@dataclass
class UnitGeneration:
    unit_id: str
    skill: str
    task: str
    success: bool
    attempts: int
    substitutions: int
    sha256: str | None = None
    steps: int = 0
    wall_s: float = 0.0
    error: str | None = None
    substituted_from: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "skill": self.skill,
            "task": self.task,
            "success": self.success,
            "attempts": self.attempts,
            "substitutions": self.substitutions,
            "sha256": self.sha256,
            "steps": self.steps,
            "wall_s": round(self.wall_s, 2),
            "error": self.error,
            "substituted_from": self.substituted_from,
        }


@dataclass
class MaterializeReport:
    units: list[UnitGeneration] = field(default_factory=list)
    wall_s: float = 0.0

    def summary(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for u in self.units:
            s = out.setdefault(
                u.skill, {"generated": 0, "failed": 0, "substituted": 0, "attempts": 0}
            )
            s["generated" if u.success else "failed"] += 1
            s["substituted"] += int(u.substituted_from is not None)
            s["attempts"] += u.attempts
        return out

    def notes(self) -> list[str]:
        notes = []
        for skill, s in sorted(self.summary().items()):
            notes.append(
                f"{skill}: {s['generated']} prompts generated in {s['attempts']} attempts, "
                f"{s['substituted']} units substituted, {s['failed']} without a prompt"
            )
        for u in self.units:
            if u.substituted_from:
                notes.append(f"{u.unit_id}: {u.substituted_from} -> {u.task}")
            if not u.success:
                notes.append(f"{u.unit_id}: no prompt ({u.error})")
        notes.append(f"materializing took {self.wall_s:.0f} s")
        return notes


# ---------------------------------------------------------------- workers
def worker_init(ctx: dict[str, str], config_dir: str) -> None:
    """Runs first in every worker process: LIBERO's config before libero is imported."""
    from ..simulators.libero.generate import libero_config

    libero_config(Path(config_dir), Path(ctx["bpp_root"]), Path(ctx["libero_datasets"]))
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def generate_one(
    spec: Spec, pool: Pool, ctx: GenerationContext, job: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The skill's simulator generates the prompt for one unit; never raises."""
    unit = job["unit"]
    t0 = time.monotonic()
    try:
        sim = for_skill(spec, unit["skill"])
        if sim.generate_prompt is None:
            raise RuntimeError(f"simulator {sim.name} cannot generate prompts")
        out = sim.generate_prompt(
            unit,
            pool.tasks[unit["task"]],
            pool,
            spec,
            list(job["seeds"]),
            Path(job["out_npz"]),
            ctx,
        )
    except Exception as exc:  # noqa: BLE001 - reported per unit, the caller substitutes
        out = {"success": False, "attempts": [], "error": f"{type(exc).__name__}: {exc}"[:300]}
    out["wall_s"] = time.monotonic() - t0
    return job, out


def _run_jobs(
    jobs: list[dict[str, Any]],
    spec: Spec,
    pool: Pool,
    ctx: GenerationContext,
    *,
    workers: int,
    inline: bool,
    config_dir: Path,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    if inline:
        for job in jobs:
            yield generate_one(spec, pool, ctx, job)
        return
    # spawned, not forked: the parent may hold an EGL / CUDA context and has imported libero
    # under another config; a fresh interpreter has neither
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, workers),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=worker_init,
        initargs=(ctx.as_dict(), str(config_dir)),
    ) as ex:
        futures = [ex.submit(generate_one, spec, pool, ctx, job) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            yield fut.result()


# ---------------------------------------------------------------- substitution
def task_order(pool: Pool, spec: Spec, duel_id: str, skill: str) -> list[str]:
    """The order unit derivation spread the skill's units over: eligible tasks shuffled by the
    duel id (the first draw of `HashRng(duel_id, skill)`, as in `pools.units._spread`)."""
    return HashRng(duel_id, skill).shuffled(pool.eligible(skill))


def substitute(
    unit: dict[str, Any], pool: Pool, spec: Spec, duel_id: str, tried: set[str], k: int
) -> dict[str, Any] | None:
    """The unit re-derived for the next task in the duel's order that it has not tried; None
    when every task was tried."""
    skill = unit["skill"]
    order = task_order(pool, spec, duel_id, skill)
    origin = unit.get("substituted_from") or unit["task"]
    start = order.index(origin) if origin in order else 0
    candidate = None
    for step in range(1, len(order) + 1):
        cand = order[(start + step) % len(order)]
        if cand not in tried:
            candidate = cand
            break
    if candidate is None:
        return None
    task = pool.tasks[candidate]
    rng = HashRng(duel_id, unit["unit_id"], "substitute", k)
    fresh = for_skill(spec, skill).make_unit(spec, skill, unit["index"], task, unit["seed"], rng)
    doc = fresh.as_dict()
    doc["unit_id"] = unit["unit_id"]
    doc["demo"] = unit["demo"]
    doc["substituted_from"] = origin
    return doc


# ---------------------------------------------------------------- the phase
def seeds_for(spec: Spec, duel_id: str, unit: dict[str, Any], block: int) -> list[int]:
    """The unit's prompt seeds for its `block`-th task (0: the task it was derived for)."""
    n = int(spec.generation["max_attempts"])
    return [
        prompt_seed(duel_id, unit["skill"], int(unit["index"]), block * n + a) for a in range(n)
    ]


def materialize(
    units: Iterable[dict[str, Any]],
    pool: Pool,
    spec: Spec,
    duel_id: str,
    assets_dir: Path,
    ctx: GenerationContext,
    *,
    workers: int = 1,
    inline: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], MaterializeReport]:
    """Generate every scored unit's prompt into `assets_dir`; returns the units as they must be
    run (finalized against their prompts, substituted where generation failed) and a report."""
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    config_dir = assets_dir.parent / "libero-config"
    out_units = [dict(u) for u in units]
    todo = list(range(len(out_units)))
    tried: dict[int, set[str]] = {i: {out_units[i]["task"]} for i in todo}
    subs: dict[int, int] = dict.fromkeys(todo, 0)
    attempts: dict[int, int] = dict.fromkeys(todo, 0)
    report = MaterializeReport()
    t0 = time.monotonic()
    done = 0

    def job_for(i: int) -> dict[str, Any]:
        u = out_units[i]
        return {
            "index": i,
            "unit": u,
            "seeds": seeds_for(spec, duel_id, u, subs[i]),
            "out_npz": str(assets_dir / f"{u['unit_id']}.npz"),
        }

    pending = [job_for(i) for i in todo]
    while pending:
        next_round: list[dict[str, Any]] = []
        for job, out in _run_jobs(
            pending, spec, pool, ctx, workers=workers, inline=inline, config_dir=config_dir
        ):
            i = int(job["index"])
            unit = out_units[i]
            attempts[i] += len(out.get("attempts", [])) or 1
            if out.get("success"):
                sim = for_skill(spec, unit["skill"])
                demo = load_demo(job["out_npz"])
                final = dict(unit)
                if sim.finalize_unit is not None:
                    final = sim.finalize_unit(final, demo, spec)
                final["prompt_sha256"] = out["sha256"]
                final["generation"] = {
                    "attempts": attempts[i],
                    "substitutions": subs[i],
                    "steps": int(out.get("steps", 0)),
                }
                out_units[i] = final
                report.units.append(
                    UnitGeneration(
                        unit["unit_id"],
                        unit["skill"],
                        unit["task"],
                        True,
                        attempts[i],
                        subs[i],
                        out["sha256"],
                        int(out.get("steps", 0)),
                        float(out.get("wall_s", 0.0)),
                        substituted_from=unit.get("substituted_from"),
                    )
                )
                done += 1
                if on_progress:
                    on_progress(done, len(todo))
                log.info(
                    "%s: prompt %s (%d attempts)", unit["unit_id"], out["sha256"][:12], attempts[i]
                )
                continue
            error = str(out.get("error") or "no attempt succeeded")
            replacement = None
            if subs[i] < MAX_SUBSTITUTIONS:
                replacement = substitute(unit, pool, spec, duel_id, tried[i], subs[i] + 1)
            if replacement is None:
                unit["generation"] = {
                    "failed": True,
                    "attempts": attempts[i],
                    "substitutions": subs[i],
                    "error": error,
                }
                unit["prompt_sha256"] = None
                report.units.append(
                    UnitGeneration(
                        unit["unit_id"],
                        unit["skill"],
                        unit["task"],
                        False,
                        attempts[i],
                        subs[i],
                        error=error,
                        substituted_from=unit.get("substituted_from"),
                    )
                )
                done += 1
                if on_progress:
                    on_progress(done, len(todo))
                log.warning(
                    "%s: no prompt after %d attempts: %s", unit["unit_id"], attempts[i], error
                )
                continue
            subs[i] += 1
            tried[i].add(replacement["task"])
            out_units[i] = replacement
            log.info(
                "%s: %s -> %s (substitution %d)",
                unit["unit_id"],
                unit["task"],
                replacement["task"],
                subs[i],
            )
            next_round.append(job_for(i))
        pending = next_round
    report.wall_s = time.monotonic() - t0
    return out_units, report
