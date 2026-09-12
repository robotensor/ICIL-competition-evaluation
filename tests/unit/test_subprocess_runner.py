"""Driving a benchmark this process cannot import.

The point of the argv split is that the orchestrator runs a benchmark whose simulator it could
never load. These tests use a fake benchmark whose `run_command` is a real subprocess writing a
real result file, so the path under test is the one a duel takes - not a mock of it.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from icilval.benchmarks import subprocess_runner as runner

# A subprocess that writes whatever result it was told to, the way a benchmark would.
WRITER = (
    "import json,sys,pathlib;"
    "d=pathlib.Path(sys.argv[1]);d.mkdir(parents=True,exist_ok=True);"
    "(d/'result.json').write_text(sys.argv[2]);"
    "open(d/'rollout.mp4','wb').write(b'\\x00\\x00\\x00\\x18ftypmp42'+b'x'*64)"
)


class FakeBenchmark:
    """A benchmark whose simulator half is a python one-liner."""

    id = "fakesim"
    api_version = 1

    def __init__(self, result: dict[str, Any] | None = None, *, argv: list[str] | None = None):
        self._result = (
            result if result is not None else {"success": True, "void": False, "steps": 7}
        )
        self._argv = argv
        self.seen: list[dict[str, Any]] = []

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "api_version": 1, "demo_channels": {"video": ["frames_*"]}}

    def run_command(self, *, unit, prompt, out_dir, policy_address, **extra):
        self.seen.append({"unit": dict(unit), "prompt": prompt, "policy_address": policy_address})
        if self._argv is not None:
            return self._argv
        return [sys.executable, "-c", WRITER, out_dir, json.dumps(self._result)]

    def read_result(self, *, out_dir: str) -> dict[str, Any]:
        return runner.read_result_file(out_dir)


class Ctx:
    """The slice of `SideContext` a simulator's `run_units` is given."""

    def __init__(self, out_of_time: bool = False):
        self._out_of_time = out_of_time
        self.finished: list[tuple[dict, dict, Any]] = []

    def out_of_time(self) -> bool:
        return self._out_of_time

    def record(self, unit, res, clip):
        return {
            "unit_id": unit["unit_id"],
            "success": res.success,
            "void": res.void,
            "steps": res.steps,
            "error": res.error,
            "video": str(clip) if clip.exists() else None,
        }

    @staticmethod
    def timed_out(unit):
        return {"unit_id": unit["unit_id"], "void": True, "error": "side wall time exceeded"}

    def finish(self, unit, rec, res):
        self.finished.append((unit, rec, res))


class Spec:
    budgets = {"unit_wall_seconds": 30, "side_wall_seconds": 600}


class Policy:
    address = "unix:///tmp/policy.sock"


def drive(benchmark, units, tmp_path, *, ctx=None, record_video=True):
    ctx = ctx or Ctx()
    media = tmp_path / "media"
    media.mkdir(parents=True, exist_ok=True)
    run_units = runner.make_run_units("fakesim", benchmark)
    run_units(ctx, "rt_pick_and_place", Policy(), None, units, Spec(), media, record_video)
    return ctx


def unit(n: int = 0, prompt: str = "/prompts/u0") -> dict[str, Any]:
    return {"unit_id": f"u{n}", "skill": "rt_pick_and_place", "prompt_dir": prompt}


def test_a_unit_runs_in_a_subprocess_and_its_result_comes_back(tmp_path):
    bench = FakeBenchmark({"success": True, "void": False, "steps": 7})
    ctx = drive(bench, [unit(0)], tmp_path)

    assert len(ctx.finished) == 1
    _, rec, res = ctx.finished[0]
    assert rec["success"] is True and rec["void"] is False and rec["steps"] == 7
    assert res.wall_s >= 0
    # The benchmark was handed the prompt and the served policy's address.
    assert bench.seen[0]["prompt"] == "/prompts/u0"
    assert bench.seen[0]["policy_address"] == "unix:///tmp/policy.sock"


def test_the_clip_the_benchmark_wrote_lands_where_the_record_addresses_it(tmp_path):
    ctx = drive(FakeBenchmark(), [unit(0)], tmp_path)
    _, rec, _ = ctx.finished[0]
    assert rec["video"] is not None
    assert (tmp_path / "media" / "u0.mp4").exists(), (
        "the clip is addressed by unit, not by the name the benchmark chose"
    )


def test_no_clip_is_collected_when_video_is_off(tmp_path):
    ctx = drive(FakeBenchmark(), [unit(0)], tmp_path, record_video=False)
    _, rec, _ = ctx.finished[0]
    assert rec["video"] is None
    assert not (tmp_path / "media" / "u0.mp4").exists()


@pytest.mark.parametrize(
    "broken, reason",
    [
        ({"argv": [sys.executable, "-c", "raise SystemExit(3)"]}, "exited 3"),
        ({"argv": [sys.executable, "-c", "import time; time.sleep(60)"]}, "budget"),
        ({"argv": ["/nonexistent/benchmark"]}, "could not start"),
        ({"argv": [sys.executable, "-c", "pass"]}, "unreadable"),
    ],
)
def test_a_unit_that_goes_wrong_is_void_and_says_why(tmp_path, broken, reason):
    """One bad unit must not lose the other seventeen, and the reason must survive to the record."""
    spec = Spec()
    spec.budgets = {**Spec.budgets, "unit_wall_seconds": 2}
    bench = FakeBenchmark(argv=broken["argv"])
    ctx = Ctx()
    run_units = runner.make_run_units("fakesim", bench)
    media = tmp_path / "media"
    media.mkdir(parents=True)
    run_units(ctx, "s", Policy(), None, [unit(0)], spec, media, True)

    _, rec, _ = ctx.finished[0]
    assert rec["void"] is True
    assert rec["success"] is None
    assert reason in rec["error"], rec["error"]


def test_one_bad_unit_does_not_stop_the_rest(tmp_path):
    class Flaky(FakeBenchmark):
        def run_command(self, *, unit, prompt, out_dir, policy_address, **extra):
            if unit["unit_id"] == "u1":
                return [sys.executable, "-c", "raise SystemExit(1)"]
            return super().run_command(
                unit=unit, prompt=prompt, out_dir=out_dir, policy_address=policy_address
            )

    ctx = drive(Flaky(), [unit(0), unit(1), unit(2)], tmp_path)
    assert [r["unit_id"] for _, r, _ in ctx.finished] == ["u0", "u1", "u2"]
    assert [r["void"] for _, r, _ in ctx.finished] == [False, True, False]


def test_a_unit_with_no_materialized_prompt_is_void_rather_than_run(tmp_path):
    bench = FakeBenchmark()
    ctx = drive(bench, [{"unit_id": "u0", "skill": "s"}], tmp_path)
    _, rec, _ = ctx.finished[0]
    assert rec["void"] is True and "no materialized prompt" in rec["error"]
    assert bench.seen == [], "run_command was called without a prompt to run"


def test_the_side_wall_clock_stops_the_run(tmp_path):
    bench = FakeBenchmark()
    ctx = drive(bench, [unit(0), unit(1)], tmp_path, ctx=Ctx(out_of_time=True))
    assert [r["error"] for _, r, _ in ctx.finished] == ["side wall time exceeded"] * 2
    assert bench.seen == []


def test_success_and_void_cannot_disagree():
    """The ABI says `success is None` exactly when void. Reconciled toward void, because the
    alternative is a wrong score rather than an error."""
    assert runner.outcome_from({"success": None, "void": False}, wall_s=1).void is True
    voided = runner.outcome_from({"success": True, "void": True}, wall_s=1)
    assert voided.void is True and voided.success is None


def test_a_result_carrying_only_what_the_abi_requires_still_makes_a_record():
    out = runner.outcome_from(
        {"success": False, "void": False, "steps": 3, "error": None}, wall_s=2
    )
    assert (out.success, out.void, out.steps) == (False, False, 3)
    assert out.progress is None and out.model_errors == 0


def test_a_plugin_that_raises_voids_its_unit_rather_than_the_duel(tmp_path):
    class Exploding(FakeBenchmark):
        def run_command(self, **kwargs):
            raise RuntimeError("boom")

    ctx = drive(Exploding(), [unit(0)], tmp_path)
    _, rec, _ = ctx.finished[0]
    assert rec["void"] is True and "run_command failed: boom" in rec["error"]


def test_demo_frames_come_from_whatever_the_benchmark_calls_video():
    """Before this, a plugged benchmark's demonstration clip encoded from an empty frame list."""
    import numpy as np

    demo = {
        "frames_head_camera": np.zeros((3, 4, 4, 3), dtype=np.uint8),
        "frames_far_side_camera": np.ones((3, 4, 4, 3), dtype=np.uint8),
        "qpos": np.zeros((3, 14)),
        "times": np.zeros(3),
    }
    frames = runner.demo_frames_from(demo, {"video": ("frames_*",)})

    assert len(frames) == 3
    # Two cameras side by side, and nothing but cameras.
    assert frames[0].shape == (4, 8, 3)
    assert runner.demo_frames_from(demo, {"video": ()}) == []
