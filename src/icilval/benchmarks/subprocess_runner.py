"""Running a plugged benchmark's units, without importing it.

The ABI splits a benchmark in two: a pure half the orchestrator calls directly, and command
builders that return an **argv**. This module is the other end of the second half - the thing that
takes `run_command`'s argv, runs it, and turns what it wrote back into the record a duel scores.

Why it exists at all: before it, `simulators.adapt` wired a plugged benchmark's `run_units` to a
refusal, so the only benchmarks that could actually run were the two shipped in this repository.
An orchestration layer that can only run its own benchmarks is not one.

Three properties it must hold, each of which is the reason for a piece of the code below:

- **Nothing here imports the benchmark's simulator.** The plugin object is a pure Python object;
  everything needing SAPIEN, MuJoCo, assets or a GPU happens in the subprocess. So a validator
  host can drive a benchmark it could not itself import.
- **A unit that goes wrong is void, not fatal.** A subprocess that crashes, times out, writes
  nothing, or writes something unreadable produces a void unit with the reason on it. One bad unit
  must not lose the other seventeen, and `max_void_fraction` is what decides whether too many of
  them invalidates the duel - that judgement belongs to scoring, not here.
- **The benchmark says what happened; the orchestrator decides what it means.** `read_result` is
  read for the fields the ABI promises and nothing more, and anything missing takes a default
  rather than raising.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: What a benchmark's result file is read for. Anything else it carries is the benchmark's own
#: business and is left where it is rather than half-understood here.
RESULT_FIELDS = ("success", "void", "steps", "error")


@dataclass
class Outcome:
    """One unit's result, in the shape `duel.side_runner._record` reads.

    A dataclass rather than the plugin's dict because the record builder reads attributes, and
    because the defaults belong in one place: a benchmark that reports only what the ABI requires
    must still produce a complete record.
    """

    success: bool | None
    void: bool
    steps: int | None
    error: str | None
    wall_s: float
    progress: float | None = None
    metric: float | None = None
    first_step_done_at: float | None = None
    model_errors: int = 0
    prompt_steps: int | None = None
    prompt_chunks: int | None = None
    instance_applied: bool | None = None


def outcome_from(result: dict[str, Any], *, wall_s: float) -> Outcome:
    """Read a plugin's `read_result` mapping.

    `success is None` exactly when the unit is void - the ABI says so, and the two disagreeing is
    the kind of thing that turns into a wrong score rather than an error, so it is reconciled here
    in favour of void.
    """
    void = bool(result.get("void"))
    success = result.get("success")
    if success is None:
        void = True
    if void:
        success = None
    return Outcome(
        success=None if success is None else bool(success),
        void=void,
        steps=_int_or_none(result.get("steps")),
        error=_str_or_none(result.get("error")),
        wall_s=wall_s,
        progress=_float_or_none(result.get("progress")),
        metric=_float_or_none(result.get("metric")),
        first_step_done_at=_float_or_none(result.get("first_step_done_at")),
        model_errors=_int_or_none(result.get("model_errors")) or 0,
        prompt_steps=_int_or_none(result.get("prompt_steps")),
        prompt_chunks=_int_or_none(result.get("prompt_chunks")),
        instance_applied=(
            None if result.get("instance_applied") is None else bool(result["instance_applied"])
        ),
    )


def voided(reason: str, *, wall_s: float = 0.0) -> Outcome:
    return Outcome(success=None, void=True, steps=None, error=reason, wall_s=wall_s)


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def make_run_units(name: str, benchmark: Any):
    """A `Simulator.run_units` that drives `benchmark` through its command builders.

    Matches the in-repo simulators' signature exactly, so `duel.side_runner` does not learn
    whether the benchmark it is running lives in this repository or another one.
    """

    def run_units(
        ctx: Any,
        skill: str,
        policy: Any,
        pool: Any,
        units: list[dict[str, Any]],
        spec: Any,
        media_dir: Path,
        record_video: bool,
    ) -> None:
        address = getattr(policy, "address", None)
        unit_wall = float(spec.budgets["unit_wall_seconds"])
        work_root = Path(media_dir).parent / "units"

        for unit in units:
            if ctx.out_of_time():
                ctx.finish(unit, ctx.timed_out(unit), None)
                continue

            unit_id = str(unit["unit_id"])
            out_dir = work_root / unit_id
            out_dir.mkdir(parents=True, exist_ok=True)
            clip = Path(media_dir) / f"{unit_id}.mp4"

            started = time.monotonic()
            outcome = _run_one(
                name=name,
                benchmark=benchmark,
                unit=unit,
                out_dir=out_dir,
                address=address,
                timeout_s=unit_wall,
                record_video=record_video,
            )
            outcome.wall_s = round(time.monotonic() - started, 2)
            _collect_clip(out_dir, clip, record_video)
            ctx.finish(unit, ctx.record(unit, outcome, clip), outcome)

    return run_units


def _run_one(
    *,
    name: str,
    benchmark: Any,
    unit: dict[str, Any],
    out_dir: Path,
    address: str | None,
    timeout_s: float,
    record_video: bool,
) -> Outcome:
    """One unit, start to finish. Every failure comes back as a void outcome, never an exception."""
    prompt = unit.get("prompt_dir") or unit.get("prompt")
    if not prompt:
        return voided(f"{name}: the unit carries no materialized prompt")

    try:
        argv = [
            str(a)
            for a in benchmark.run_command(
                unit=unit,
                prompt=str(prompt),
                out_dir=str(out_dir),
                policy_address=address or "",
                record_video=record_video,
            )
        ]
    except Exception as exc:  # noqa: BLE001 - a plugin's own failure is this unit's, not the duel's
        return voided(f"{name}: run_command failed: {exc}")

    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return voided(f"{name}: unit exceeded its {timeout_s:.0f}s budget")
    except OSError as exc:
        return voided(f"{name}: could not start the benchmark: {exc}")

    if completed.returncode != 0:
        return voided(f"{name}: exited {completed.returncode}: {_tail(completed.stderr)}")

    try:
        result = benchmark.read_result(out_dir=str(out_dir))
    except Exception as exc:  # noqa: BLE001
        return voided(f"{name}: read_result failed: {exc}")
    if not isinstance(result, dict):
        return voided(f"{name}: read_result returned {type(result).__name__}, not a mapping")

    return outcome_from(result, wall_s=0.0)


def _collect_clip(out_dir: Path, clip: Path, record_video: bool) -> None:
    """Move whatever clip the benchmark wrote to where the side runner expects it.

    The benchmark names its own files; the record addresses clips by unit. Absent is ordinary -
    a void unit has nothing to show - so a miss is silent.
    """
    if not record_video:
        return
    for candidate in sorted(out_dir.glob("*.mp4")):
        if candidate.stat().st_size > 0:
            clip.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate), str(clip))
            return


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text[-limit:] if len(text) > limit else text


def read_result_file(out_dir: str | Path, name: str = "result.json") -> dict[str, Any]:
    """The conventional `read_result` for a benchmark that writes one JSON file.

    Here rather than in every plugin because it is the orchestrator's own convention; a benchmark
    that writes something else implements `read_result` itself, which is why it is on the ABI.
    """
    path = Path(out_dir) / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"success": None, "void": True, "steps": None, "error": f"unreadable {path.name}"}


def demo_frames_from(demo: dict[str, Any], channels: dict[str, tuple[str, ...]]) -> list[Any]:
    """Frames for the demonstration clip, from whatever arrays the benchmark calls video.

    A plugged benchmark's demonstration used to render as an empty clip, because `adapt` handed
    back `lambda demo: []`. The video channel already says which arrays hold pictures - including
    as a prefix, since one array per camera cannot be enumerated here - so the clip can be built
    without knowing anything else about the benchmark.

    Several cameras are laid side by side in one frame, in sorted name order, which is the
    convention the in-repo simulators already follow for a multi-camera demonstration.
    """
    import numpy as np

    names, prefixes = [], []
    for entry in channels.get("video", ()):
        (prefixes if entry.endswith("*") else names).append(entry.rstrip("*"))

    arrays = [
        value
        for key, value in sorted(demo.items())
        if (key in names or (prefixes and key.startswith(tuple(prefixes))))
        and hasattr(value, "shape")
        and getattr(value, "ndim", 0) == 4
    ]
    if not arrays:
        return []
    steps = min(len(a) for a in arrays)
    return [np.concatenate([np.asarray(a[i]) for a in arrays], axis=1) for i in range(steps)]
