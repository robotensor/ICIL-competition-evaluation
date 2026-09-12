import numpy as np
import pytest

from icilval.simulators.libero.env import LiberoEnv, load_init_states
from icilval.video import VideoWriter, is_faststart

pytestmark = pytest.mark.sim


def first_base_task(pool):
    return next(t for _, t in sorted(pool.tasks.items()) if t.skill == "pick_and_place")


def first_draw_task(pool):
    return next(t for t in pool.tasks.values() if t.skill == "draw_anything")


def test_env_reset_observe_render_and_predicates(spec, smoke_pool):
    task = first_base_task(smoke_pool)
    env = LiberoEnv(smoke_pool.path(task.bddl), spec, skill="pick_and_place")
    states = load_init_states(smoke_pool.path(task.init))
    obs = env.reset(7, states[0])
    assert obs["agentview"].shape == (128, 128, 3) and obs["agentview"].dtype == np.uint8
    assert obs["ee_quat"].shape == (4,) and obs["gripper"].shape == (2,)
    assert env.success() is False and env.goal_status() == [False] * len(task.goal)
    frame = env.render()
    assert frame.shape == (spec.media["video"]["resolution"], spec.media["video"]["resolution"], 3)
    # restoring the same state twice gives the same observation
    again = env.reset(7, states[0])
    assert np.array_equal(again["agentview"], obs["agentview"])
    env.close()


def test_video_writer_faststart(spec, smoke_pool, tmp_path):
    task = first_base_task(smoke_pool)
    env = LiberoEnv(smoke_pool.path(task.bddl), spec, skill="pick_and_place")
    states = load_init_states(smoke_pool.path(task.init))
    env.reset(7, states[0])
    out = tmp_path / "clip.mp4"
    with VideoWriter(out, int(spec.media["video"]["fps"]), spec.media["video"]) as w:
        for _ in range(10):
            env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
            w.write(env.render())
    assert out.stat().st_size > 1000 and is_faststart(out)
    env.close()


def test_replay_demo_actions_succeed(spec, smoke_pool):
    """Executing a demonstration's own actions from its own initial state reproduces success."""
    task = first_base_task(smoke_pool)
    from icilval.pools.demos import load_demo

    demo = load_demo(smoke_pool.path("demos") / f"{task.demos[0]}.npz")
    env = LiberoEnv(smoke_pool.path(task.bddl), spec, skill="pick_and_place")
    env.reset(7, demo["init_state"])
    ok = False
    for a in demo["actions"]:
        env.step(a)
        if env.success():
            ok = True
            break
    env.close()
    assert ok


# ---------------------------------------------------------------- the drawing board
def test_draw_board_replays_demo_to_zero_chamfer(spec, smoke_pool):
    """A demonstration's own actions on its own board redraw its strokes exactly."""
    from icilval.pools.demos import load_demo
    from icilval.simulators.draw.env import DrawBoard

    task = first_draw_task(smoke_pool)
    demo = load_demo(smoke_pool.path("demos") / f"{task.demos[0]}.npz")
    board = DrawBoard(spec, "draw_anything")
    angle = float(demo["boundary_angle"])
    cursor = (int(demo["agent_pos"][0][0]), int(demo["agent_pos"][0][1]))
    obs = board.reset(3, angle=angle, cursor=cursor, target=demo["drawing"], target_angle=angle)
    size = spec.env("draw_anything")["policy_image_resolution"]
    assert (
        obs["image"].shape == (3, size, size)
        and 0.0 <= obs["image"].min() <= obs["image"].max() <= 1.0
    )
    assert obs["agent_pos"].shape == (2,) and obs["pen_down"].shape == (1,)
    assert board.chamfer() == float("inf")  # nothing drawn yet
    for a in demo["actions"]:
        _, chamfer = board.step(a)
    assert chamfer < 0.5
    res = spec.env("draw_anything")["render_resolution"]
    assert board.render().shape == (res, res, 3)
    # the same strokes on a board turned by a quarter turn are far from the target
    board.reset(3, angle=angle + 0.7, cursor=cursor, target=demo["drawing"], target_angle=angle)
    for a in demo["actions"]:
        _, chamfer = board.step(a)
    assert chamfer > 10.0
    board.close()


def test_draw_episode_with_replaying_policy(spec, smoke_pool, tmp_path):
    """The episode loop scores a policy that replays the demonstration rotated into the unit's board."""
    from icilval.ids import ModelRef, duel_id
    from icilval.model.prompt import PromptInfo
    from icilval.pools.demos import load_demo
    from icilval.pools.units import derive_units
    from icilval.simulators.draw.env import DrawBoard
    from icilval.simulators.draw.episode import run_draw_episode

    did = duel_id(spec.version, spec.sole_track, ModelRef.make("a/b", "1" * 40), None)
    unit = next(
        u for u in derive_units(smoke_pool, spec, did, "smoke") if u.skill == "draw_anything"
    ).as_dict()
    demo = load_demo(smoke_pool.path("demos") / f"{unit['demo']}.npz")
    theta = float(unit["instance_params"]["angle_rad"]) - float(demo["boundary_angle"])
    c, s = np.cos(theta), np.sin(theta)
    rotated = demo["actions"].copy()
    rotated[:, 0] = (demo["actions"][:, 0] - 256) * c - (demo["actions"][:, 1] - 256) * s + 256
    rotated[:, 1] = (demo["actions"][:, 0] - 256) * s + (demo["actions"][:, 1] - 256) * c + 256

    class Replay:
        def __init__(self):
            self.t = 0

        def seed(self, seed):
            return None

        def set_prompt(self, demo):
            return PromptInfo(len(demo["actions"]), 1, len(demo["actions"]))

        def act(self, history):
            h = spec.env("draw_anything")["exec_horizon"]
            chunk = rotated[self.t : self.t + h]
            self.t += h
            if len(chunk) < h:  # done: hold the pen up where the drawing ended
                last = rotated[-1:].copy()
                last[:, 2] = 0
                chunk = np.concatenate([chunk, np.repeat(last, h - len(chunk), axis=0)])
            return chunk

    board = DrawBoard(spec, "draw_anything")
    out = tmp_path / "clip.mp4"
    with VideoWriter(out, spec.env("draw_anything")["control_freq"], spec.media["video"]) as w:
        res = run_draw_episode(board, Replay(), unit, demo, spec, video=w)
    board.close()
    assert not res.void, res.error
    assert res.metric is not None and res.metric < spec.success("draw_anything")["threshold"]
    assert res.success and res.steps >= len(demo["actions"])
    assert out.stat().st_size > 1000 and is_faststart(out)
