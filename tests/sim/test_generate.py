"""Generating a LIBERO prompt for a catalogue task with BPP's generator."""

import numpy as np
import pytest

from icilval.pools.demos import load_demo
from icilval.simulators.libero.env import LiberoEnv
from icilval.simulators.libero.generate import generate_prompt, grasp_sources

from .conftest import bpp_root, grasp_sources_root

pytestmark = [pytest.mark.sim, pytest.mark.slow]

SEEDS = [11, 12, 13, 14, 15, 16, 17, 18]


def first_libero_task(pool):
    return next(t for _, t in sorted(pool.tasks.items()) if t.skill == "pick_and_place")


def test_generate_prompt_and_replay_it(spec, smoke_pool, tmp_path):
    task = first_libero_task(smoke_pool)
    sources = grasp_sources(smoke_pool, "pick_and_place", spec)
    root = grasp_sources_root()
    missing = [n for n in sources["files"] if not (root / sources["prefix"] / n).exists()]
    if missing:
        pytest.skip(f"grasp-source files not in the raw cache: {missing[:2]}…")
    res = generate_prompt(
        task,
        smoke_pool,
        spec,
        "pick_and_place",
        SEEDS,
        tmp_path / "prompt.npz",
        bpp_root=bpp_root(),
        work_dir=tmp_path / "work",
        demo_id="generated/pp-000",
        keep_work_dir=True,
    )
    assert res.success, res.as_dict()
    assert res.npz == tmp_path / "prompt.npz" and len(res.sha256) == 64
    assert all(not a.success for a in res.attempts[:-1]) and res.attempts[-1].success
    demo = load_demo(res.npz)
    n = demo["actions"].shape[0]
    assert n == res.steps and demo["actions"].shape == (n, 7)
    assert demo["agentview"].shape == (n, 128, 128, 3) and demo["eye_in_hand"].shape == (
        n,
        128,
        128,
        3,
    )
    assert demo["ee_pos"].shape == (n, 3) and demo["gripper"].shape == (n, 2)
    assert demo["meta"]["demo_id"] == "generated/pp-000"
    assert demo["meta"]["seed"] == res.attempts[-1].seed and demo["meta"]["task"] == task.task_id
    assert (tmp_path / "work" / "generator.log").exists()
    # replaying the generated actions from the recorded initial state reproduces success
    env = LiberoEnv(smoke_pool.path(task.bddl), spec, skill="pick_and_place")
    env.reset(7, demo["init_state"])
    ok = False
    for a in demo["actions"]:
        env.step(np.asarray(a, dtype=np.float64))
        if env.success():
            ok = True
            break
    env.close()
    assert ok
