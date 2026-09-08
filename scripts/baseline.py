"""Organizer sweep: run a model over every eligible task of a skill and report success rates.

    PYTHONPATH=src MUJOCO_GL=egl SDL_VIDEODRIVER=dummy $BP scripts/baseline.py \
        --pool pools/2026.09-v3 --model-dir models/genesis --skill pick_and_place \
        --instances 2 --out runs/baseline-v3-pp

Every eligible task (or `--max-tasks` of them, spread evenly over the list) gets `--instances`
initial states, spread over its usable ones, each with a demonstration that did not start from
that state; seeds and picks are a pure function of the
task id, so the sweep is reproducible. Writes `<out>/units.jsonl` (one line per episode, from
`side_runner`) and `<out>/baseline.json`: overall, per task and per `meta.source_split`. The
numbers quoted in docs/pools.md come from here. No clips are recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icilval.pools.demos import select_indices  # noqa: E402
from icilval.pools.schema import Pool  # noqa: E402
from icilval.pools.units import Unit, draw_instance  # noqa: E402
from icilval.rng import HashRng  # noqa: E402
from icilval.spec import _repo_root, load_spec  # noqa: E402


def sweep_units(
    pool: Pool,
    spec,
    skill: str,
    per_task: int,
    task_filter: str | None,
    max_tasks: int | None = None,
) -> list[Unit]:
    units: list[Unit] = []
    code = spec.skill_code(skill)
    env = spec.env(skill)
    index = 0
    task_ids = [t for t in pool.eligible(skill) if not task_filter or task_filter in t]
    if max_tasks is not None and max_tasks < len(task_ids):
        task_ids = [task_ids[i] for i in select_indices(len(task_ids), max_tasks)]
    for tid in task_ids:
        task = pool.tasks[tid]
        valid = task.valid_instances
        for k in select_indices(len(valid), per_task):
            instance = valid[k]
            rng = HashRng("baseline", tid, instance)
            candidates = [d for d in task.demos if task.demo_init_index.get(d) != instance]
            demo = (candidates or task.demos)[rng.below(len(candidates or task.demos))]
            params = {}
            if spec.simulator(skill) == "draw":
                state = draw_instance(tid, instance, env)
                params = {
                    **state,
                    "demo_angle_rad": float(task.meta.get("demo_angles", {}).get(demo, 0.0)),
                }
            units.append(
                Unit(
                    unit_id=f"{code}-{index:04d}",
                    skill=skill,
                    index=index,
                    task=tid,
                    task_label=task.label,
                    instance=instance,
                    demo=demo,
                    seed=rng.below(2**31),
                    instance_params=params,
                    max_steps=min(task.max_steps, spec.max_steps(skill)),
                    bddl=task.bddl,
                    init=task.init,
                    goal=task.goal,
                    steps=task.steps,
                )
            )
            index += 1
    return units


def summarize(pool: Pool, units: list[Unit], out: Path, skill: str) -> dict:
    from icilval.duel.side_runner import read_results

    results = read_results(out / "units.jsonl")
    by_unit = {u.unit_id: u for u in units}
    per_task: dict[str, dict[str, int]] = {}
    per_split: dict[str, dict[str, int]] = {}
    scored = successes = void = 0
    for uid, rec in results.items():
        u = by_unit.get(uid)
        if u is None:
            continue
        split = str(pool.tasks[u.task].meta.get("source_split", pool.tasks[u.task].suite))
        for bucket in (per_task.setdefault(u.task, {}), per_split.setdefault(split, {})):
            bucket["episodes"] = bucket.get("episodes", 0) + 1
            if rec.get("void"):
                bucket["void"] = bucket.get("void", 0) + 1
            else:
                bucket["scored"] = bucket.get("scored", 0) + 1
                bucket["success"] = bucket.get("success", 0) + int(bool(rec.get("success")))
        if rec.get("void"):
            void += 1
        else:
            scored += 1
            successes += int(bool(rec.get("success")))

    def rate(b: dict[str, int]) -> float | None:
        return round(b["success"] / b["scored"], 4) if b.get("scored") else None

    summary = {
        "skill": skill,
        "pool_id": pool.pool_id,
        "tasks": len(per_task),
        "episodes": len(results),
        "scored": scored,
        "void": void,
        "success_rate": round(successes / scored, 4) if scored else None,
        "per_split": {s: {**b, "success_rate": rate(b)} for s, b in sorted(per_split.items())},
        "per_task": {t: {**b, "success_rate": rate(b)} for t, b in sorted(per_task.items())},
    }
    (out / "baseline.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model-dir", required=True, help="one subdirectory per skill")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--instances", type=int, default=2, help="initial states per task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--task-filter", default=None, help="only tasks whose id contains this")
    ap.add_argument("--max-tasks", type=int, default=None, help="tasks spread evenly over the list")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    spec = load_spec()
    pool = Pool.load(args.pool)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    units = sweep_units(pool, spec, args.skill, args.instances, args.task_filter, args.max_tasks)
    if not args.summarize_only:
        from icilval.duel.side_runner import run_side

        arch = Path(args.arch) if args.arch else (_repo_root() or Path.cwd()) / "arch"
        run_side(
            side="king",
            model_dir=Path(args.model_dir),
            arch_dir=arch,
            pool=pool,
            units=[u.as_dict() for u in units],
            spec=spec,
            out_dir=out,
            device=args.device,
            record_video=False,
        )
    summary = summarize(pool, units, out, args.skill)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
