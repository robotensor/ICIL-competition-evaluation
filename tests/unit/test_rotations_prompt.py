import numpy as np
import pytest

from icilval.model.prompt import chunk_layout
from icilval.simulators.draw.prompt import build_draw_prompt
from icilval.simulators.libero import rotations as R
from icilval.simulators.libero.prompt import build_prompt


def test_rot6d_identity_and_roundtrip():
    assert np.allclose(R.axis_angle_to_rotation_6d(np.zeros(3)), [1, 0, 0, 0, 1, 0])
    rng = np.random.default_rng(0)
    aa = rng.normal(size=(200, 3)) * 1.5
    back = R.rotation_6d_to_axis_angle(R.axis_angle_to_rotation_6d(aa))
    # same rotation (axis-angle is unique up to 2pi wraps; keep angles < pi)
    small = np.linalg.norm(aa, axis=-1) < np.pi
    assert np.allclose(back[small], aa[small], atol=1e-6)
    m = R.axis_angle_to_matrix(aa)
    assert np.allclose(m @ np.transpose(m, (0, 2, 1)), np.eye(3), atol=1e-9)
    assert np.allclose(R.quaternion_to_matrix(R.matrix_to_quaternion(m)), m, atol=1e-9)


def test_quat_and_actions():
    q_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(R.quat_xyzw_to_rotation_6d(q_xyzw), [1, 0, 0, 0, 1, 0])
    a7 = np.array([[0.1, -0.2, 0.3, 0.0, 0.0, 0.5, -1.0]])
    a10 = R.actions_7_to_10(a7)
    assert a10.shape == (1, 10) and a10.dtype == np.float32
    assert np.allclose(R.actions_10_to_7(a10), a7, atol=1e-6)
    yaw = R.yaw_quaternion_wxyz(np.pi / 2)
    assert np.allclose(R.quaternion_to_matrix(yaw) @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    assert np.allclose(R.quat_multiply_wxyz(yaw, [1, 0, 0, 0]), yaw)


@pytest.mark.parametrize(
    "steps,chunks,padded",
    [
        (135, 7, 140),
        (20, 1, 20),
        (40, 2, 40),
        (15, 1, 20),
        (1, 1, 20),
        (1000, 50, 1000),
        (1001, 51, 1020),
    ],
)
def test_chunk_layout_zeros(steps, chunks, padded):
    assert chunk_layout(steps, 20) == (chunks, padded)
    assert chunk_layout(steps, 20, "repeat") == (chunks, padded)


@pytest.mark.parametrize(
    "steps,chunks,padded",
    [(135, 13, 130), (10, 1, 10), (40, 4, 40), (5, 1, 10), (1, 1, 10), (396, 39, 390)],
)
def test_chunk_layout_no_padding(steps, chunks, padded):
    """BPP's `pad_end_prompt_actions: no` (the drawing skill): a partial chunk is dropped."""
    assert chunk_layout(steps, 10, "no") == (chunks, padded)


def test_chunk_layout_rejects_bad_mode():
    with pytest.raises(ValueError):
        chunk_layout(10, 5, "maybe")


def fake_demo(T):
    return {
        "agentview": np.zeros((T, 128, 128, 3), np.uint8),
        "eye_in_hand": np.zeros((T, 128, 128, 3), np.uint8),
        "ee_pos": np.arange(T * 3, dtype=np.float32).reshape(T, 3),
        "ee_ori": np.zeros((T, 3), np.float32),
        "gripper": np.zeros((T, 2), np.float32),
        "actions": np.ones((T, 7), np.float32),
    }


def test_build_prompt_shapes_and_padding():
    prompt, info = build_prompt(fake_demo(135), 20, max_chunks=50)
    assert (info.steps, info.chunks, info.padded_steps) == (135, 7, 140)
    assert prompt["action"].shape == (7, 20, 10)
    assert np.all(prompt["action"][6, 15:] == 0) and np.all(prompt["action"][6, :15, 0] == 1)
    assert prompt["obs"]["agentview_rgb"].shape == (7, 128, 128, 3)
    assert prompt["obs"]["ee_ori"].shape == (7, 6) and np.allclose(
        prompt["obs"]["ee_ori"][0], [1, 0, 0, 0, 1, 0]
    )
    assert np.allclose(prompt["obs"]["ee_pos"][1], [60, 61, 62])  # frame 20
    assert prompt["mask"].shape == (7,) and not prompt["mask"].any()
    with pytest.raises(ValueError):
        build_prompt(fake_demo(1200), 20, max_chunks=50)
    short, info = build_prompt(fake_demo(5), 20)
    assert info.chunks == 1 and short["obs"]["ee_pos"].shape == (1, 3)


def fake_draw_demo(T):
    return {
        "image": np.zeros((T, 224, 224, 3), np.uint8),
        "agent_pos": np.arange(T * 2, dtype=np.float32).reshape(T, 2),
        "pen_down": (np.arange(T) % 2).astype(np.float32),
        "actions": np.tile(np.array([[100.0, 200.0, 1.0]], np.float32), (T, 1)),
    }


def test_build_draw_prompt_drops_partial_chunk():
    prompt, info = build_draw_prompt(fake_draw_demo(135), 10, max_chunks=80)
    assert (info.steps, info.chunks, info.padded_steps) == (135, 13, 130)
    assert prompt["action"].shape == (13, 10, 3) and np.all(prompt["action"][:, :, 2] == 1)
    assert prompt["obs"]["image"].shape == (13, 224, 224, 3)
    assert prompt["obs"]["agent_pos"].shape == (13, 2) and np.allclose(
        prompt["obs"]["agent_pos"][1], [20, 21]
    )
    assert prompt["obs"]["pen_down"].shape == (13, 1)
    assert prompt["mask"].shape == (13,) and not prompt["mask"].any()
    short, info = build_draw_prompt(fake_draw_demo(4), 10)
    assert info.chunks == 1 and short["action"].shape == (1, 10, 3)
    assert np.all(short["action"][0, 4:] == 0) and np.all(short["action"][0, :4, 0] == 100)
    with pytest.raises(ValueError):
        build_draw_prompt(fake_draw_demo(900), 10, max_chunks=80)
