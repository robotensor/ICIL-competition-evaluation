"""The converted genesis checkpoints must reproduce BPP's published performance."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from icilval.model.fingerprint import check_submission
from icilval.simulators.draw.env import DrawBoard
from icilval.simulators.draw.episode import run_draw_episode
from icilval.simulators.draw.policy import DrawPolicy
from icilval.simulators.libero.env import LiberoEnv, load_init_states
from icilval.simulators.libero.episode import run_episode
from icilval.simulators.libero.policy import BPPPolicy
from icilval.spec import _repo_root

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

N_TASKS = int(os.environ.get("ICILVAL_PARITY_TASKS", "10"))
N_INIT = int(os.environ.get("ICILVAL_PARITY_INITS", "5"))
FLOOR = float(os.environ.get("ICILVAL_PARITY_FLOOR", "0.90"))
DRAW_TASKS = int(os.environ.get("ICILVAL_PARITY_DRAW_TASKS", "10"))
DRAW_FLOOR = float(os.environ.get("ICILVAL_PARITY_DRAW_FLOOR", "0.5"))


def arch_dir() -> Path:
    return (_repo_root() or Path.cwd()) / "arch"


def test_genesis_passes_fingerprint(spec, genesis_dir):
    rep = check_submission(genesis_dir, spec, arch_dir())
    assert rep.ok, rep.errors
    assert set(rep.skills) == set(spec.all_skills)


def test_parity_libero_spatial(spec, genesis_dir, smoke_pool_or_skip):
    pool = smoke_pool_or_skip
    tasks = [t for t in pool.tasks.values() if t.suite == "libero_spatial"][:N_TASKS]
    if not tasks:
        pytest.skip("pool has no libero_spatial tasks")
    policy = BPPPolicy(genesis_dir / "pick_and_place", arch_dir(), spec, "pick_and_place")
    policy.load()
    from icilval.pools.demos import load_demo

    successes, n = 0, 0
    for task in tasks:
        env = LiberoEnv(pool.path(task.bddl), spec, skill="pick_and_place")
        states = load_init_states(pool.path(task.init))
        demo = load_demo(pool.path("demos") / f"{task.demos[0]}.npz")
        for i in range(min(N_INIT, len(states))):
            unit = {
                "unit_id": f"pp-{i:03d}",
                "skill": "pick_and_place",
                "seed": 1000 + i,
                "max_steps": task.max_steps,
                "instance": i,
                "instance_params": {},
            }
            res = run_episode(env, policy, unit, np.asarray(states[i]), demo, spec)
            assert not res.void, res.error
            successes += int(res.success)
            n += 1
        env.close()
    policy.unload()
    assert successes / n >= FLOOR, f"success {successes}/{n}"


def test_parity_draw_anything(spec, genesis_dir, smoke_pool_or_skip):
    """The BPP drawing checkpoint, prompted with one human demonstration, redraws it on a
    turned board within the success threshold on at least DRAW_FLOOR of the units."""
    from icilval.ids import ModelRef, duel_id
    from icilval.pools.demos import load_demo
    from icilval.pools.units import derive_units

    pool = smoke_pool_or_skip
    if not pool.tasks_of("draw_anything"):
        pytest.skip("pool has no drawing tasks")
    policy = DrawPolicy(genesis_dir / "draw_anything", arch_dir(), spec, "draw_anything")
    policy.load()
    did = duel_id(spec.version, "sensorimotor", ModelRef.make("parity/draw", "1" * 40), None)
    units = [
        u.as_dict() for u in derive_units(pool, spec, did, "heavy") if u.skill == "draw_anything"
    ][:DRAW_TASKS]
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
