"""Run one side of a duel over a unit list. Resumable; runs where the simulators and torch live.

Units are grouped by skill (spec order). For each skill the side's checkpoint for that skill
is loaded from `<model_dir>/<skill>`, its units run by its simulator's `run_units` (see
`icilval.simulators`), and the model is unloaded before the next skill's is loaded. The
simulator gets a `SideContext`: the time budget, the executor, and how to record a unit.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..canon import sha256_file
from ..pools.schema import Pool
from ..simulators import for_skill, make_policy
from ..spec import Spec
from ..video import VideoWriter

log = logging.getLogger(__name__)


def read_results(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        out[rec["unit_id"]] = rec
    return out


def run_side(
    *,
    side: str,
    model_dir: Path,
    arch_dir: Path,
    pool: Pool,
    units: list[dict[str, Any]],
    spec: Spec,
    track: str,
    out_dir: Path,
    device: str = "cuda",
    on_unit: Callable[[dict[str, Any]], None] | None = None,
    record_video: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    media_dir.mkdir(exist_ok=True)
    results_path = out_dir / "units.jsonl"
    done = read_results(results_path)
    t_start = time.monotonic()
    side_wall = float(spec.budgets["side_wall_seconds"])
    summary: dict[str, Any] = {
        "side": side,
        "units": len(units),
        "ran": 0,
        "resumed": len(done),
        "void": 0,
        "success": 0,
        "errors": 0,
        "load_seconds": {},
        "skills": {},
    }

    class SideContext:
        """What a simulator's `run_units` needs from the side runner."""

        def __init__(self) -> None:
            self.t_start = t_start
            self.side_wall = side_wall
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def out_of_time(self) -> bool:
            return time.monotonic() - self.t_start > self.side_wall

        def record(self, unit: dict[str, Any], res: Any, clip: Path) -> dict[str, Any]:
            return _record(unit, res, clip, out_dir, record_video)

        @staticmethod
        def timed_out(unit: dict[str, Any]) -> dict[str, Any]:
            return _timed_out(unit)

        @staticmethod
        def close_writer(writer: VideoWriter | None, unit: dict[str, Any]) -> None:
            _close_writer(writer, unit)

        def finish(self, unit: dict[str, Any], rec: dict[str, Any], res: Any | None) -> None:
            _append(results_path, rec)
            done[rec["unit_id"]] = rec
            skill_summary = summary["skills"].setdefault(
                unit["skill"], {"ran": 0, "void": 0, "success": 0}
            )
            summary["ran"] += 1
            skill_summary["ran"] += 1
            summary["void"] += int(bool(rec.get("void")))
            skill_summary["void"] += int(bool(rec.get("void")))
            summary["success"] += int(bool(rec.get("success")))
            skill_summary["success"] += int(bool(rec.get("success")))
            if res is not None:
                summary["errors"] += res.model_errors
                log.info(
                    "%s %s: success=%s metric=%s steps=%d wall=%.1fs%s",
                    side,
                    unit["unit_id"],
                    res.success,
                    res.metric,
                    res.steps,
                    res.wall_s,
                    f" error={res.error}" if res.error else "",
                )
            if on_unit is not None:
                on_unit(rec)

    ctx = SideContext()
    try:
        for skill in spec.skills(track):
            todo = [u for u in units if u["skill"] == skill and u["unit_id"] not in done]
            if not todo:
                continue
            policy = make_policy(model_dir / skill, arch_dir, spec, skill, device=device)
            policy.load()
            summary["load_seconds"][skill] = round(policy.load_seconds, 1)
            try:
                for_skill(spec, skill).run_units(
                    ctx, skill, policy, pool, todo, spec, media_dir, record_video
                )
            finally:
                policy.unload()
    finally:
        ctx.executor.shutdown(wait=False)
    summary["wall_seconds"] = round(time.monotonic() - t_start, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _record(unit: dict[str, Any], res: Any, clip: Path, out_dir: Path, record_video: bool) -> dict:
    rec = {
        "unit_id": unit["unit_id"],
        "skill": unit["skill"],
        "success": None if res.void else bool(res.success),
        "progress": res.progress,
        "metric": res.metric,
        "first_step_done_at": res.first_step_done_at,
        "steps": res.steps,
        "model_errors": res.model_errors,
        "wall_s": round(res.wall_s, 2),
        "error": res.error,
        "void": res.void,
        "prompt_steps": res.prompt_steps,
        "prompt_chunks": res.prompt_chunks,
        "instance_applied": res.instance_applied,
        "video": None,
        "video_sha256": None,
    }
    if record_video and clip.exists() and clip.stat().st_size > 0 and not res.void:
        rec["video"] = str(clip.relative_to(out_dir))
        rec["video_sha256"] = sha256_file(clip)
    return rec


def _timed_out(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit_id": unit["unit_id"],
        "skill": unit["skill"],
        "void": True,
        "error": "side wall time exceeded",
        "success": None,
        "metric": None,
    }


def _close_writer(writer: VideoWriter | None, unit: dict[str, Any]) -> None:
    if writer is None:
        return
    try:
        writer.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("video encode failed for %s: %s", unit["unit_id"], exc)


def _append(path: Path, rec: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
