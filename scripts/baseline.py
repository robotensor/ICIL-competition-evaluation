"""Organizer sweep: generate prompts and score a model over a skill's catalogue tasks, per change
kind and per drawing family, the way a duel would.

    PYTHONPATH=src MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYNPUT_BACKEND=dummy $BP scripts/baseline.py \\
        --pool catalogues/2026.09-v4 --model-dir models/genesis --skill pick_and_place \\
        --instances 1 --out runs/baseline-v4-pp [--change displace] [--family glyph] [--workers 4]

Every eligible task of the skill (or `--max-tasks` of them, spread evenly over the list) gets
`--instances` units, each derived with the skill's own unit builder from a seed that is a pure
function of the task id and the instance, so the sweep is reproducible; `--change` forces one
change kind on every LIBERO unit and `--family` keeps only one drawing family, which is how the
change ranges are calibrated. The units are materialized (prompts generated into `<out>/assets`)
and run by `side_runner`. Writes `<out>/units.jsonl`, `<out>/materialize.json` and
`<out>/baseline.json`: overall, per change kind, per family, per task and per source split. The
numbers quoted in docs/pools.md come from here. No clips are recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icilval.duel.materialize import GenerationContext, materialize  # noqa: E402
from icilval.duel.score import lookup  # noqa: E402
from icilval.pools.demos import select_indices  # noqa: E402
from icilval.pools.schema import Pool  # noqa: E402
from icilval.pools.sources import DEFAULT_CACHE, Sources  # noqa: E402
from icilval.rng import HashRng  # noqa: E402
from icilval.simulators import for_skill  # noqa: E402
from icilval.spec import _repo_root, load_spec  # noqa: E402


def sweep_units(
    pool: Pool,
    spec,
    skill: str,
    per_task: int,
    task_filter: str | None,
    max_tasks: int | None = None,
    change: str | None = None,
    family: str | None = None,
) -> list[dict]:
    from icilval.simulators.libero.changes import sample_change

    sim = for_skill(spec, skill)
    task_ids = [t for t in pool.eligible(skill) if not task_filter or task_filter in t]
    if family:
        task_ids = [t for t in task_ids if pool.tasks[t].meta.get("family") == family]
    if max_tasks is not None and max_tasks < len(task_ids):
        task_ids = [task_ids[i] for i in select_indices(len(task_ids), max_tasks)]
    units: list[dict] = []
    index = 0
    for tid in task_ids:
        task = pool.tasks[tid]
        for k in range(per_task):
            rng = HashRng("baseline", tid, k)
            unit = sim.make_unit(spec, skill, index, task, rng.below(2**31), rng).as_dict()
            unit["unit_id"] = f"{spec.skill_code(skill)}-{index:04d}"
            unit["demo"] = f"generated/{unit['unit_id']}"
            if change and spec.simulator(skill) == "libero":
                unit["change"] = sample_change(
                    spec, skill, task.steps, HashRng("baseline-change", tid, k), kind=change
                )
            elif change:
                unit["change"] = {"kind": change}
            units.append(unit)
            index += 1
    return units


def summarize(pool: Pool, units: list[dict], out: Path, skill: str, spec) -> dict:
    from icilval.duel.side_runner import read_results

    results = read_results(out / "units.jsonl")
    by_unit = {u["unit_id"]: u for u in units}
    buckets: dict[str, dict[str, dict[str, int]]] = {
        "per_change": {},
        "per_family": {},
        "per_task": {},
        "per_split": {},
    }
    scored = successes = void = 0
    for uid, rec in results.items():
        u = by_unit.get(uid)
        if u is None:
            continue
        task = pool.tasks[u["task"]]
        keys = {
            "per_change": str(lookup(u, "change.kind")),
            "per_family": str(lookup(u, "instance_params.family") or "-"),
            "per_task": u["task"],
            "per_split": str(task.meta.get("source_split", task.suite)),
        }
        for name, key in keys.items():
            b = buckets[name].setdefault(key, {})
            b["episodes"] = b.get("episodes", 0) + 1
            if rec.get("void"):
                b["void"] = b.get("void", 0) + 1
            else:
                b["scored"] = b.get("scored", 0) + 1
                b["success"] = b.get("success", 0) + int(bool(rec.get("success")))
        if rec.get("void"):
            void += 1
        else:
            scored += 1
            successes += int(bool(rec.get("success")))

    def rate(b: dict[str, int]) -> float | None:
        return round(b["success"] / b["scored"], 4) if b.get("scored") else None

    generation = {}
    if (out / "materialize.json").exists():
        gen = json.loads((out / "materialize.json").read_text())
        generation = {
            "units": len(gen),
            "generated": sum(1 for g in gen if g["success"]),
            "attempts": sum(g["attempts"] for g in gen),
            "substituted": sum(1 for g in gen if g.get("substituted_from")),
            "wall_s": round(sum(g.get("wall_s", 0.0) for g in gen), 1),
        }
    summary = {
        "skill": skill,
        "pool_id": pool.pool_id,
        "tasks": len(buckets["per_task"]),
        "episodes": len(results),
        "scored": scored,
        "void": void,
        "success_rate": round(successes / scored, 4) if scored else None,
        "generation": generation,
        **{
            name: {k: {**b, "success_rate": rate(b)} for k, b in sorted(bs.items())}
            for name, bs in buckets.items()
        },
    }
    (out / "baseline.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", required=True, help="the catalogue directory")
    ap.add_argument("--model-dir", required=True, help="one subdirectory per skill")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--instances", type=int, default=1, help="units per task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--task-filter", default=None, help="only tasks whose id contains this")
    ap.add_argument("--max-tasks", type=int, default=None, help="tasks spread evenly over the list")
    ap.add_argument("--change", default=None, help="force this change kind on every unit")
    ap.add_argument("--family", default=None, help="only drawing tasks of this family")
    ap.add_argument("--workers", type=int, default=None, help="generation processes (spec default)")
    ap.add_argument("--raw", default=None, help="raw cache with the grasp-source files")
    ap.add_argument("--bpp-root", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--materialize-only", action="store_true")
    args = ap.parse_args()

    spec = load_spec()
    pool = Pool.load(args.pool)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = _repo_root() or Path.cwd()
    units = sweep_units(
        pool,
        spec,
        args.skill,
        args.instances,
        args.task_filter,
        args.max_tasks,
        args.change,
        args.family,
    )
    units_path = out / "units.json"
    if not args.summarize_only:
        bpp_root = Path(args.bpp_root) if args.bpp_root else root / "vendor" / "behavior_prompting"
        raw = Path(args.raw) if args.raw else DEFAULT_CACHE / "raw"
        grasp = spec.tasks(args.skill).get("grasp_sources") or {}
        datasets = (
            Sources(raw=raw, bpp_root=bpp_root).dataset_root(str(grasp["dataset"]))
            if grasp
            else raw
        )
        ctx = GenerationContext(bpp_root=bpp_root, raw_root=raw, libero_datasets=datasets)
        duel = HashRng("baseline", args.skill).shuffled(["baseline"])[0] + "-" + pool.pool_id[:16]
        units, report = materialize(
            units,
            pool,
            spec,
            duel,
            out / "assets",
            ctx,
            workers=int(args.workers or spec.generation["workers"]),
            on_progress=lambda d, t: print(f"generated {d}/{t}", flush=True),
        )
        (out / "materialize.json").write_text(
            json.dumps([g.as_dict() for g in report.units], indent=1)
        )
        units_path.write_text(json.dumps(units, indent=1))
        for note in report.notes():
            print(note)
    elif units_path.exists():
        units = json.loads(units_path.read_text())
    if not args.summarize_only and not args.materialize_only:
        from icilval.duel.side_runner import run_side

        arch = Path(args.arch) if args.arch else root / "arch"
        run_side(
            side="king",
            model_dir=Path(args.model_dir),
            arch_dir=arch,
            pool=pool,
            units=units,
            spec=spec,
            out_dir=out,
            device=args.device,
            record_video=False,
            assets_dir=out / "assets",
        )
    summary = summarize(pool, units, out, args.skill, spec)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
