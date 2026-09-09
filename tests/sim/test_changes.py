"""Scene changes apply to a reset LIBERO scene, and a reset restores the model."""

import numpy as np
import pytest

from icilval.rng import HashRng
from icilval.simulators.libero.changes import Infeasible, apply_change, sample_change
from icilval.simulators.libero.env import LiberoEnv
from icilval.simulators.libero.episode import instance_seed

pytestmark = pytest.mark.sim


def first_libero_task(pool):
    return next(t for _, t in sorted(pool.tasks.items()) if t.skill == "pick_and_place")


def test_changes_apply_and_reset_restores(spec, smoke_pool):
    task = first_libero_task(smoke_pool)
    env = LiberoEnv(smoke_pool.path(task.bddl), spec, skill="pick_and_place")
    seed = instance_seed({"task": task.task_id, "instance": 0})
    obs0 = env.reset(seed, None)
    m = env.mj_model
    cam_pos, light_diffuse = m.cam_pos.copy(), m.light_diffuse.copy()
    target = task.steps[0][1]
    before = env.object_position(target).copy()

    # displace: the first feasible candidate moves the target by at least min_delta_m
    cfg = spec.changes("pick_and_place")["displace"]
    change = {
        "kind": "displace",
        "target": target,
        "min_delta_m": cfg["min_delta_m"],
        "clearance_m": cfg["clearance_m"],
        "candidates": [
            {"delta_xy": [0.5, 0.5], "yaw": 0.0},  # off the table
            {"delta_xy": [0.04, 0.0], "yaw": 0.3},
            {"delta_xy": [0.0, -0.04], "yaw": 0.0},
        ],
    }
    applied = apply_change(env, change)
    assert applied["kind"] == "displace" and applied["candidate"] >= 1
    assert applied["distance_m"] >= cfg["min_delta_m"] - 1e-3
    assert np.linalg.norm(env.object_position(target)[:2] - before[:2]) >= cfg["min_delta_m"] - 1e-3
    assert not env.success()
    moved = env.current_observation()
    assert not np.array_equal(moved["agentview"], obs0["agentview"])
    with pytest.raises(Infeasible):
        apply_change(env, {**change, "candidates": [{"delta_xy": [0.9, 0.9], "yaw": 0.0}]})
    # every sampled draw of the catalogue's tasks has a feasible candidate
    for t in [x for _, x in sorted(smoke_pool.tasks.items()) if x.skill == "pick_and_place"]:
        e = LiberoEnv(smoke_pool.path(t.bddl), spec, skill="pick_and_place")
        for i in range(3):
            e.reset(instance_seed({"task": t.task_id, "instance": i}), None)
            draw = sample_change(
                spec, "pick_and_place", t.steps, HashRng("feasible", t.task_id, i), kind="displace"
            )
            assert apply_change(e, draw)["kind"] == "displace"
        e.close()

    # camera: the agentview camera moves; reset puts it back
    env.reset(seed, None)
    assert np.allclose(m.cam_pos, cam_pos)
    cam = env.camera_id("agentview")
    applied = apply_change(
        env,
        {
            "kind": "camera",
            "camera": "agentview",
            "pos_delta": [0.03, 0.0, 0.02],
            "angles": [0.05, 0.0, 0.0],
        },
    )
    assert applied["kind"] == "camera" and not np.allclose(m.cam_pos[cam], cam_pos[cam])
    assert not np.array_equal(env.current_observation()["agentview"], obs0["agentview"])
    env.reset(seed, None)
    assert np.allclose(m.cam_pos, cam_pos)

    # lighting: diffuse terms change; reset puts them back
    light = {
        "active": True,
        "diffuse_scale": 0.5,
        "ambient_add": 0.1,
        "specular_scale": 1.0,
        "pos_jitter": [0.0, 0.0, 0.0],
        "dir_jitter": [0.0, 0.0, 0.0],
    }
    applied = apply_change(
        env, {"kind": "lighting", "lights": [light, light], "headlight_scale": 0.9}
    )
    assert applied["kind"] == "lighting" and applied["lights_applied"] == [0, 1]
    assert not np.allclose(m.light_diffuse, light_diffuse)
    env.reset(seed, None)
    assert np.allclose(m.light_diffuse, light_diffuse)

    # a sampled change of every kind applies without error
    kinds = set()
    for i in range(40):
        change = sample_change(spec, "pick_and_place", task.steps, HashRng("sim", task.task_id, i))
        if change["kind"] in kinds:
            continue
        env.reset(seed, None)
        applied = apply_change(env, change)
        assert applied["kind"] == change["kind"]
        kinds.add(change["kind"])
    assert kinds == set(spec.changes("pick_and_place"))
    env.close()


def test_scene_builds_with_joint_noise(spec, smoke_pool):
    task = first_libero_task(smoke_pool)
    magnitude = spec.changes("pick_and_place")["robot_pose"]["init_noise_magnitude"]
    env = LiberoEnv(
        smoke_pool.path(task.bddl), spec, skill="pick_and_place", init_noise_magnitude=magnitude
    )
    assert env.init_noise_magnitude == magnitude
    obs = env.reset(3, None)
    assert obs["agentview"].shape == (128, 128, 3) and not env.success()
    env.close()


def test_episode_applies_the_units_change(spec, smoke_pool):
    """The episode loop applies the unit's change after the reset and reports it."""
    from icilval.model.prompt import PromptInfo
    from icilval.simulators.libero.episode import run_episode

    task = first_libero_task(smoke_pool)
    env = LiberoEnv(smoke_pool.path(task.bddl), spec, skill="pick_and_place")
    exec_h = int(spec.env("pick_and_place")["exec_horizon"])

    class Hold:
        def seed(self, seed):
            return None

        def set_prompt(self, demo):
            return PromptInfo(0, 0, 0)

        def act(self, history):
            return np.tile(np.array([0, 0, 0, 0, 0, 0, -1.0]), (exec_h, 1))

    change = sample_change(spec, "pick_and_place", task.steps, HashRng("episode", task.task_id, 3))
    unit = {
        "unit_id": "pp-000",
        "skill": "pick_and_place",
        "task": task.task_id,
        "seed": 5,
        "instance": 1,
        "max_steps": exec_h,
        "instance_params": {},
        "change": change,
    }
    res = run_episode(env, Hold(), unit, None, {"actions": np.zeros((0, 7))}, spec)
    assert not res.void, res.error
    assert res.change_applied["kind"] == change["kind"] and res.steps == exec_h
    assert res.instance_applied == {"reset_seed": instance_seed(unit)}
    env.close()
