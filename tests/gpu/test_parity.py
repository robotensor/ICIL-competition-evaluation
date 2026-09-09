"""The converted genesis checkpoints must reproduce BPP's published performance."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from icilval.model.fingerprint import check_submission
from icilval.pools.demos import load_demo
from icilval.simulators.draw.env import DrawBoard
from icilval.simulators.draw.episode import run_draw_episode
from icilval.simulators.draw.policy import DrawPolicy
from icilval.simulators.libero.env import LiberoEnv
from icilval.simulators.libero.episode import run_episode
from icilval.simulators.libero.policy import BPPPolicy
from icilval.spec import _repo_root

from .conftest import bpp_root

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

N_TASKS = int(os.environ.get("ICILVAL_PARITY_TASKS", "10"))
N_INIT = int(os.environ.get("ICILVAL_PARITY_INITS", "5"))
FLOOR = float(os.environ.get("ICILVAL_PARITY_FLOOR", "0.80"))
DRAW_TASKS = int(os.environ.get("ICILVAL_PARITY_DRAW_TASKS", "10"))
DRAW_FLOOR = float(os.environ.get("ICILVAL_PARITY_DRAW_FLOOR", "0.5"))


def arch_dir() -> Path:
    return (_repo_root() or Path.cwd()) / "arch"


def test_genesis_passes_fingerprint(spec, genesis_dir):
    rep = check_submission(genesis_dir, spec, arch_dir())
    assert rep.ok, rep.errors
    assert set(rep.skills) == set(spec.skills)


def test_parity_libero_generated_prompts(spec, genesis_dir, smoke_pool_or_skip, tmp_path):
    """The LIBERO checkpoint, prompted with a demonstration generated for the task, does the
    task from fresh numbered resets on at least FLOOR of the episodes."""
    from icilval.rng import HashRng
    from icilval.simulators.libero.generate import generate_prompt

    pool = smoke_pool_or_skip
    tasks = [t for _, t in sorted(pool.tasks.items()) if t.skill == "pick_and_place"][:N_TASKS]
    if not tasks:
        pytest.skip("catalogue has no pick_and_place tasks")
    attempts = int(spec.generation["max_attempts"])
    prompts = {}
    for task in tasks:
        rng = HashRng("parity", task.task_id)
        seeds = [rng.below(1 << 31) for _ in range(attempts)]
        res = generate_prompt(
            task,
            pool,
            spec,
            "pick_and_place",
            seeds,
            tmp_path / f"{task.task_id.replace('/', '__')}.npz",
            bpp_root=bpp_root(),
            work_dir=tmp_path / "work" / task.task_id.replace("/", "__"),
            demo_id=f"generated/parity-{task.task_id}",
        )
        if res.success:
            prompts[task.task_id] = load_demo(res.npz)
    assert prompts, "no prompt could be generated"
    policy = BPPPolicy(genesis_dir / "pick_and_place", arch_dir(), spec, "pick_and_place")
    policy.load()
    successes, n = 0, 0
    for task in tasks:
        if task.task_id not in prompts:
            continue
        env = LiberoEnv(pool.path(task.bddl), spec, skill="pick_and_place")
        for i in range(N_INIT):
            unit = {
                "unit_id": f"pp-{i:03d}",
                "skill": "pick_and_place",
                "task": task.task_id,
                "seed": 1000 + i,
                "max_steps": task.max_steps,
                "instance": i,
                "instance_params": {},
            }
            res = run_episode(env, policy, unit, None, prompts[task.task_id], spec)
            assert not res.void, res.error
            successes += int(res.success)
            n += 1
        env.close()
    policy.unload()
    assert successes / n >= FLOOR, f"success {successes}/{n}"


def test_parity_draw_anything(spec, genesis_dir, smoke_pool_or_skip):
    """The BPP drawing checkpoint, prompted with one stored human demonstration (the handmade
    diagnostic's units), redraws it on a turned board within the success threshold on at least
    DRAW_FLOOR of the units."""
    from icilval.ids import ModelRef, duel_id
    from icilval.pools.demos import load_demo
    from icilval.pools.units import derive_units

    pool = smoke_pool_or_skip
    if not pool.tasks_of("draw_anything"):
        pytest.skip("pool has no drawing tasks")
    policy = DrawPolicy(genesis_dir / "draw_anything", arch_dir(), spec, "draw_anything")
    policy.load()
    did = duel_id(spec.version, spec.track_id, ModelRef.make("parity/draw", "1" * 40), None)
    units = [
        u.as_dict()
        for u in derive_units(pool, spec, did, "heavy")
        if u.skill == "draw_anything" and u.diagnostic
    ][:DRAW_TASKS]
    if not units:
        pytest.skip("catalogue has no diagnostic drawing tasks")
    board = DrawBoard(spec, "draw_anything")
    successes, metrics = 0, []
    for unit in units:
        demo = load_demo(pool.path("demos") / f"{unit['demo']}.npz")
        res = run_draw_episode(board, policy, unit, demo, spec)
        assert not res.void, res.error
        successes += int(res.success)
        metrics.append(res.metric)
    board.close()
    policy.unload()
    assert successes / len(units) >= DRAW_FLOOR, (
        f"success {successes}/{len(units)} chamfer {metrics}"
    )
