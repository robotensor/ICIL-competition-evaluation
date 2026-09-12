"""The `bpp_robotwin_v1` conversion, on synthetic arrays, in the pure environment.

Every test here runs with numpy alone: the conversion is the half of the policy that has to be
checkable without torch, without a simulator and without the benchmark. What it is checked
against is not itself - it is the properties that would go wrong silently. A tool offset applied
in the world frame instead of the tool's own still produces a plausible pose; a gripper scale off
by its origin still produces a number in range; a crop centred on the wrong camera still produces
a 224-pixel image. None of those would fail a shape assertion, and all of them would quietly
change what the network was asked.

Numeric agreement with the benchmark's own adapter is checked separately, outside this suite,
because it needs an environment where both are importable.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from icilval.model.bpp_robotwin import conversion as C
from icilval.model.bpp_robotwin import images as I
from icilval.model.bpp_robotwin import rotations as R
from icilval.model.bpp_robotwin.settings import (
    ACTION_DIM,
    BIMANUAL_EE_DIM,
    MODEL_FINGER_RANGE_M,
    OSC_POSITION_SCALE_M,
    OSC_ROTATION_SCALE_RAD,
    ROBOT_FINGER_RANGE_M,
    TCP_OFFSET_M,
    ConversionError,
    Settings,
    from_env,
)
from icilval.model.prompt import chunk_actions

AGENTVIEW = "far_side_camera"
WRISTS = ("left_camera", "right_camera")
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


# ------------------------------------------------------------------ synthetic arrays


def _quat(axis, angle: float) -> np.ndarray:
    return R.axis_angle_to_quat(np.asarray(axis, dtype=np.float64) * angle)


def demo_arrays(steps: int = 45, arm: str = "left", uneven: bool = True, seed: int = 3) -> dict:
    """A demonstration in exactly the arrays the benchmark publishes."""
    rng = np.random.default_rng(seed)
    gaps = rng.choice([0.02, 0.05, 0.11], size=steps - 1)
    times = np.concatenate([[0.0], np.cumsum(gaps if uneven else np.full(steps - 1, 0.05))])

    start = C.arm_base(arm) + np.array([0.0, 0.25, 0.08])
    path = start + np.cumsum(rng.normal(0.0, 0.004, size=(steps, 3)), axis=0)
    angles = np.cumsum(rng.normal(0.0, 0.05, size=steps))
    quats = np.stack([_quat([0.0, 0.0, 1.0], a) for a in angles])
    idle = C.arm_base(C.other_arm(arm)) + np.array([0.0, 0.2, 0.1])
    idle_flange = C.flange_from_tcp(np.concatenate([idle, IDENTITY]))

    grip = np.ones(steps)
    grip[steps // 2 :] = 0.0
    active = C.flange_from_tcp(np.concatenate([path, quats], axis=1))

    rows = np.zeros((steps, BIMANUAL_EE_DIM))
    a, b = (0, 8) if arm == "left" else (8, 0)
    rows[:, a : a + 7] = active
    rows[:, a + 7] = grip
    rows[:, b : b + 7] = idle_flange
    rows[:, b + 7] = 1.0

    low, high = ROBOT_FINGER_RANGE_M
    fingers = np.zeros((steps, 2, 2))
    index = 0 if arm == "left" else 1
    fingers[:, index, :] = (low + grip * (high - low))[:, None]
    fingers[:, 1 - index, :] = high

    arrays = {
        "times": times,
        "qpos": np.tile(np.concatenate([np.full(7, 0.1), np.full(7, 0.2)]), (steps, 1)),
        "endpose": rows,
        "gripper_joints": fingers,
    }
    for name, shape in ((AGENTVIEW, (180, 320)), (WRISTS[0], (240, 320)), (WRISTS[1], (240, 320))):
        block = np.zeros((steps, *shape, 3), dtype=np.uint8)
        block[:] = np.arange(shape[1], dtype=np.uint8)[None, None, :, None]
        block += np.arange(steps, dtype=np.uint8)[:, None, None, None]
        arrays[f"frames_{name}"] = block
    return arrays


def observation(arrays: dict, index: int) -> dict:
    """One step of a demonstration, in the shape an observation arrives in."""
    return {key: value[index] for key, value in arrays.items() if key != "times"}


# ----------------------------------------------------------------------- rotations


def test_a_6d_rotation_round_trips():
    """rot6d is what the network emits and what `ee_ori` carries, in both directions."""
    rng = np.random.default_rng(11)
    axes = rng.normal(size=(64, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = rng.uniform(0.0, np.pi, size=(64, 1))
    matrices = R.axis_angle_to_matrix(axes * angles)
    back = R.rot6d_to_matrix(R.matrix_to_rot6d(matrices))
    np.testing.assert_allclose(back, matrices, atol=1e-12)


def test_rot6d_is_the_first_two_rows_not_the_first_two_columns():
    """The two conventions differ by a transpose, and both decode to a rotation."""
    matrix = R.axis_angle_to_matrix(np.array([0.3, -0.7, 1.1]))
    np.testing.assert_allclose(R.matrix_to_rot6d(matrix), matrix[:2, :].reshape(6), atol=0)
    assert not np.allclose(R.matrix_to_rot6d(matrix), matrix[:, :2].T.reshape(6))


def test_six_arbitrary_numbers_decode_to_a_rotation():
    """A network's raw output is not orthonormal; Gram-Schmidt is what makes it one."""
    m = R.rot6d_to_matrix(np.array([2.0, 0.1, -0.3, 0.05, 1.5, 0.2]))
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(m) == pytest.approx(1.0)


def test_a_quaternion_round_trips_through_a_matrix_near_half_a_turn():
    """Shepperd's method exists for this case: w near zero, where the trace formula loses it."""
    q = _quat([0.0, 0.0, 1.0], np.pi - 1e-7)
    np.testing.assert_allclose(R.matrix_to_quat(R.quat_to_matrix(q)), q, atol=1e-9)


# --------------------------------------------------------------------------- frames


def test_the_tool_centre_offset_is_applied_in_the_tool_frame_not_the_world_frame():
    """0.12 m along the flange's *own* +x. A flange turned to face world -y puts its tool centre
    0.12 m along -y, not along +x - which is the whole difference between grasping the object and
    grasping the air beside it."""
    position = np.array([0.1, -0.4, 0.8])
    turned = np.concatenate([position, _quat([0.0, 0.0, 1.0], np.pi / 2)])
    tool = C.tcp_from_flange(turned)
    np.testing.assert_allclose(tool[:3], position + [0.0, TCP_OFFSET_M, 0.0], atol=1e-12)
    # The world-frame mistake would have put it here instead.
    assert not np.allclose(tool[:3], position + [TCP_OFFSET_M, 0.0, 0.0])
    # And the orientation is untouched.
    np.testing.assert_allclose(tool[3:], turned[3:], atol=0)


def test_the_flange_and_the_tool_centre_are_inverses():
    rng = np.random.default_rng(5)
    poses = np.concatenate(
        [rng.normal(size=(32, 3)), R.canonical_quat(rng.normal(size=(32, 4)))], axis=1
    )
    np.testing.assert_allclose(C.flange_from_tcp(C.tcp_from_flange(poses)), poses, atol=1e-12)


def test_the_model_frame_is_the_robot_base_frame():
    """A vector of the training world is `M v` in the robot's world, and M is the root's own
    rotation - so the training domain's +x is the robot's forward, which is world +y here."""
    np.testing.assert_allclose(C.model_to_world([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        C.world_to_model(C.model_to_world([0.3, -0.2, 0.5])), [0.3, -0.2, 0.5]
    )


def test_a_pose_at_the_arm_base_lands_on_the_declared_model_base():
    """`p_model = M.T (p_world - b_arm) + b_base`: the anchor, with nothing else in the way."""
    settings = Settings()
    for arm in ("left", "right"):
        flange = C.flange_from_tcp(np.concatenate([C.arm_base(arm), IDENTITY]))
        position, _ = C.model_tool_pose(flange, settings, arm)
        np.testing.assert_allclose(position, settings.model_base_m, atol=1e-12)


# -------------------------------------------------------------------------- gripper


def test_gripper_scaling_maps_the_documented_open_and_closed_range():
    """The robot's finger joints run -0.01 to 0.045 m; the network's `gripper_states` runs
    0 to 0.04, as `[finger, -finger]`."""
    closed, opened = ROBOT_FINGER_RANGE_M
    model_closed, model_open = MODEL_FINGER_RANGE_M
    np.testing.assert_allclose(C.gripper_states(closed), [model_closed, -model_closed], atol=1e-12)
    np.testing.assert_allclose(C.gripper_states(opened), [model_open, -model_open], atol=1e-12)
    half = C.gripper_states((closed + opened) / 2)
    np.testing.assert_allclose(half, [(model_closed + model_open) / 2] * 1 + [-0.02], atol=1e-12)
    assert half[1] == -half[0]


def test_the_measurement_is_read_and_not_the_command():
    """The gripper inside `endpose` reads fully closed while the fingers rest on an object, so a
    conversion that used it would tell the network the object is not there."""
    arrays = demo_arrays()
    arrays["gripper_joints"][:] = ROBOT_FINGER_RANGE_M[1]  # fingers wide open
    arrays["endpose"][:, 7] = 0.0  # commanded shut
    state = C.observation_state(observation(arrays, 0), Settings(), "left", 32)
    np.testing.assert_allclose(state["gripper_states"], [MODEL_FINGER_RANGE_M[1], -0.04], atol=1e-9)


def test_a_demonstration_without_the_measurement_is_refused():
    arrays = demo_arrays()
    del arrays["gripper_joints"]
    with pytest.raises(ConversionError, match="command"):
        C.observation_state(observation(arrays, 0), Settings(), "left", 32)


# -------------------------------------------------------------------------- cameras


def test_the_camera_chosen_is_the_one_the_settings_name():
    """Two cameras of the same shape are trivially swapped, and nothing downstream would notice:
    the third-person view is the one the settings name, and the wrist view the acting arm's."""
    arrays = demo_arrays(arm="left")
    settings = Settings()
    obs = observation(arrays, 0)
    obs[f"frames_{settings.agentview_camera}"][:] = 10
    obs[f"frames_{settings.wrist_cameras[0]}"][:] = 20
    obs[f"frames_{settings.wrist_cameras[1]}"][:] = 30

    left = C.observation_state(obs, settings, "left", 32)
    assert left["agentview_rgb"].mean() == pytest.approx(10 / 255, abs=1e-4)
    assert left["eye_in_hand_rgb"].mean() == pytest.approx(20 / 255, abs=1e-4)
    # The other arm takes the other wrist, and the same third-person camera.
    right = C.observation_state(obs, settings, "right", 32)
    assert right["eye_in_hand_rgb"].mean() == pytest.approx(30 / 255, abs=1e-4)


def test_a_missing_camera_names_what_it_wanted():
    arrays = demo_arrays()
    del arrays[f"frames_{Settings().agentview_camera}"]
    with pytest.raises(ConversionError, match="far_side_camera"):
        C.observation_state(observation(arrays, 0), Settings(), "left", 32)


def test_the_crop_follows_the_acting_arm_across_the_image_centre():
    """The arms sit either side of the centre line and the camera looks back at them, so the two
    crops fall on opposite sides of a 320-wide frame - which is the reason the crop is arm-centred
    at all."""
    settings = Settings()
    width = settings.agentview_geometry()[0]
    left, right = C.crop_column(settings, "left"), C.crop_column(settings, "right")
    assert right < width / 2 < left
    # `left` names the image side the *camera* sees; with the camera facing the robot they swap.
    assert I.crop_start(left, settings.crop, width) != I.crop_start(right, settings.crop, width)


def test_the_crop_column_can_be_pinned_instead_of_projected():
    settings = from_env(None, crop_column={"left": 160.0, "right": 100.0})
    assert C.crop_column(settings, "left") == 160.0


def test_the_view_is_a_128_pixel_frame_brought_up_and_not_a_sharp_one():
    """The checkpoint saw 128-pixel renders. Going straight to 224 would hand it detail its
    training distribution never had, so the 128 stage is not an optimisation to skip."""
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[:, ::2] = 255  # a one-pixel comb the 128 stage must average away
    view = I.arm_centred_view(frame, centre_col=160.0, crop=180, output=224)
    assert view.shape == (224, 224, 3)
    assert 0.4 < float(view.mean()) < 0.6
    assert float(view.max()) < 1.0  # nothing survives at full white


# --------------------------------------------------------------------- the prompt


def test_the_demonstration_is_resampled_on_time_not_on_the_frame_index():
    """The expert's frames are unevenly spaced, so an index-based resampling would stretch the
    prompt wherever it paused."""
    arrays = demo_arrays(uneven=True)
    settings = Settings()
    resampled = C.resample(arrays, settings, rate_hz=settings.rate_hz)
    spacing = np.diff(resampled.times)
    np.testing.assert_allclose(spacing, 1.0 / settings.rate_hz, atol=1e-12)
    assert len(resampled) != len(arrays["times"])
    assert resampled.times[-1] >= arrays["times"][-1] - 1e-9


def test_a_demonstration_without_times_is_refused_rather_than_guessed():
    arrays = demo_arrays()
    del arrays["times"]
    with pytest.raises(ConversionError, match="times"):
        C.resample(arrays, Settings(), rate_hz=20.0)
    uniform = from_env(None, assume_uniform_times=True)
    assert len(C.resample(arrays, uniform, rate_hz=20.0)) == len(arrays["endpose"])


def test_a_gripper_value_is_held_and_never_interpolated():
    """Half a gripper command is not a command the robot ever issued."""
    arrays = demo_arrays()
    resampled = C.resample(arrays, Settings(), rate_hz=7.0)
    values = set(np.round(resampled.endposes[:, 7], 12))
    assert values <= set(np.round(arrays["endpose"][:, 7], 12))


def test_the_prompt_actions_have_the_shape_and_the_gripper_sign_the_network_expects():
    arrays = demo_arrays(arm="left")
    settings = Settings()
    prompt = C.build_prompt(arrays, settings, "left")
    assert prompt.actions.ndim == 2 and prompt.actions.shape[1] == ACTION_DIM
    assert set(np.unique(prompt.actions[:, 9])) <= {-1.0, 1.0}
    # The demonstration opens, then closes: negative is open, positive is close.
    assert prompt.actions[0, 9] == -1.0
    assert prompt.actions[-1, 9] == 1.0


def test_a_prompt_observation_is_read_at_the_step_the_chunker_asks_for():
    arrays = demo_arrays()
    settings = Settings()
    prompt = C.build_prompt(arrays, settings, "left")
    chunks, idx, info = chunk_actions(prompt.actions, 20, "zeros", None)
    state = C.prompt_observations(arrays, prompt, idx, settings, 32)
    assert chunks.shape == (info.chunks, 20, ACTION_DIM)
    for key, shape in (
        ("agentview_rgb", (3, 32, 32)),
        ("eye_in_hand_rgb", (3, 32, 32)),
        ("ee_pos", (3,)),
        ("ee_ori", (6,)),
        ("gripper_states", (2,)),
    ):
        assert state[key].shape == (info.chunks, *shape), key
    # One observation per chunk, and the first is the demonstration's own first step.
    single = C.observation_state(observation(arrays, 0), settings, "left", 32)
    np.testing.assert_allclose(state["ee_pos"][0], single["ee_pos"], atol=1e-12)


def test_the_arm_that_travels_further_is_the_one_driven():
    settings = Settings()
    for arm in ("left", "right"):
        choice = C.choose_arm(demo_arrays(arm=arm), settings)
        assert (choice.arm, choice.rule) == (arm, "path")
        assert choice.idle == C.other_arm(arm)


# ------------------------------------------------------------------- the action path


def test_the_action_decode_inverts_the_encode():
    """The prompt encodes a measured tool delta into an action; execution decodes an action back
    into a tool delta. They are one transformation used in both directions, so a gain applied on
    one side only would show up here and nowhere else."""
    settings = Settings()
    arrays = demo_arrays(arm="left", seed=9)
    prompt = C.build_prompt(arrays, settings, "left")
    poses = prompt.resampled.endposes[:, 0:7]
    tools = C.tcp_from_flange(poses)
    positions = tools[:, :3]
    rotations = R.quat_to_matrix(tools[:, 3:])

    # Only where nothing hit the clip: a clipped action is deliberately not invertible, and the
    # clip is the reason the rotation is re-encoded from the clipped axis-angle.
    moves = C.world_to_model(np.diff(positions, axis=0)) / (settings.alpha_p * OSC_POSITION_SCALE_M)
    spins = C.matrix_to_axis_angle(
        C.MODEL_TO_WORLD.T
        @ np.einsum("tij,tkj->tik", rotations[1:], rotations[:-1])
        @ C.MODEL_TO_WORLD
    ) / (settings.alpha_r * OSC_ROTATION_SCALE_RAD)
    unclipped = np.all(np.abs(moves) < 1.0, axis=1) & np.all(np.abs(spins) < 1.0, axis=1)
    assert unclipped.sum() > 5

    for step in np.flatnonzero(unclipped):
        move, turn, _ = C.decode_action(prompt.actions[step], settings)
        np.testing.assert_allclose(move, positions[step + 1] - positions[step], atol=1e-12)
        np.testing.assert_allclose(turn @ rotations[step], rotations[step + 1], atol=1e-9)


def test_one_action_unit_is_one_full_scale_goal_offset_times_the_gain():
    """A delta is a goal offset the arm covers `alpha` of in one step, not a displacement."""
    settings = Settings()
    action = np.zeros(ACTION_DIM)
    action[0] = 1.0  # full scale along the training domain's +x
    action[3:9] = R.matrix_to_rot6d(np.eye(3))
    move, turn, gripper = C.decode_action(action, settings)
    expected = settings.alpha_p * OSC_POSITION_SCALE_M
    np.testing.assert_allclose(move, C.model_to_world([expected, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(turn, np.eye(3), atol=1e-12)
    assert gripper == 0.0  # a non-negative gripper action closes


def test_a_negative_gripper_action_opens():
    settings = Settings()
    action = np.zeros(ACTION_DIM)
    action[3:9] = R.matrix_to_rot6d(np.eye(3))
    action[9] = -1.0
    assert C.decode_action(action, settings)[2] == 1.0


def test_a_full_scale_rotation_turns_by_the_gain_times_half_a_radian():
    settings = Settings()
    action = np.zeros(ACTION_DIM)
    action[3:9] = R.matrix_to_rot6d(R.axis_angle_to_matrix([0.0, 0.0, 1.0]))
    _, turn, _ = C.decode_action(action, settings)
    angle = np.linalg.norm(R.matrix_to_axis_angle(turn))
    assert angle == pytest.approx(settings.alpha_r * OSC_ROTATION_SCALE_RAD)


def test_an_executed_action_is_one_ee_row_with_the_idle_arm_held():
    arrays = demo_arrays(arm="left")
    settings = Settings()
    execution = C.Execution(settings, "left")
    first = observation(arrays, 0)
    action = np.zeros(ACTION_DIM)
    action[3:9] = R.matrix_to_rot6d(np.eye(3))
    row = execution.act(action, first)
    assert row.shape == (1, BIMANUAL_EE_DIM)
    # The right arm's eight slots are exactly what the first observation reported.
    np.testing.assert_allclose(row[0, 8:], arrays["endpose"][0, 8:], atol=1e-12)
    # Held, not echoed: the idle arm does not follow the observation it is given later.
    moved = observation(arrays, 5)
    moved["endpose"] = moved["endpose"].copy()
    moved["endpose"][8:11] += 0.05
    later = execution.act(action, moved)
    np.testing.assert_allclose(later[0, 8:], arrays["endpose"][0, 8:], atol=1e-12)


def test_the_virtual_target_keeps_sub_tolerance_motion():
    """Ten commands of a millimetre must add up to a centimetre of target, not ten targets a
    millimetre from wherever the arm happened to be."""
    settings = Settings()
    target = C.VirtualTarget(settings.max_position_error_m, settings.max_rotation_error_rad)
    measured, rotation = np.zeros(3), np.eye(3)
    for _ in range(10):
        position, _ = target.step(measured, rotation, np.array([0.001, 0.0, 0.0]))
    np.testing.assert_allclose(position, [0.01, 0.0, 0.0], atol=1e-12)
    assert target.reanchors["tracking"] == 0
    # Once the arm is further off than the bound, the target is re-anchored to it.
    target.step(np.array([1.0, 0.0, 0.0]), rotation, np.zeros(3))
    assert target.reanchors["tracking"] == 1


def test_the_re_anchoring_bound_must_exceed_one_commanded_step():
    """A bound below one step re-anchors on every call, which is the bug the bound exists to
    prevent, so it is refused at construction rather than discovered in a rollout."""
    with pytest.raises(ConversionError, match="full-scale commanded step"):
        from_env(None, max_position_error_m=0.001)


def test_a_stall_is_counted_when_the_tool_was_told_to_move_and_did_not():
    detector = C.StallDetector(window=3, min_motion_m=0.002)
    for _ in range(5):
        detector.update(np.zeros(3), commanded_m=0.001)
    assert detector.events == 1 and detector.stalled


# ------------------------------------------------------------------- the boundaries


def _module_paths() -> list[Path]:
    package = Path(C.__file__).resolve().parent
    return sorted(package.glob("*.py"))


def _module_scope_imports(tree: ast.AST) -> list[str]:
    """Every module named by an import that runs at import time.

    Bodies of functions are skipped and nothing else is: a class body, a `try` and an `if` all
    run when the module is loaded, so an import hidden in one of them is a module-scope import.
    """
    found: list[str] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(child, ast.Import):
                found.extend(a.name for a in child.names)
            elif isinstance(child, ast.ImportFrom):
                found.append(child.module or "")
            walk(child)

    walk(tree)
    return found


def test_nothing_in_the_package_imports_torch_or_the_benchmark_at_module_scope():
    """The conversion has to be checkable in the validator's pure environment, and this
    repository must never import the benchmark it scores - not even transitively. `policy.py` may
    use torch, but only inside the functions that need it, so importing the package costs
    nothing."""
    forbidden = {"torch", "robotwin_icil", "icil_policies", "behavior_prompting", "sapien"}
    offenders = [
        f"{path.name}: {module}"
        for path in _module_paths()
        for module in _module_scope_imports(ast.parse(path.read_text()))
        if module.split(".")[0] in forbidden
    ]
    assert offenders == [], offenders


def test_the_guard_would_catch_a_module_scope_import():
    """The test above passes trivially if `_module_scope_imports` finds nothing, so make it find
    the thing it is there to find."""
    tree = ast.parse("import numpy\nclass A:\n    import torch\ndef f():\n    import torch\n")
    assert _module_scope_imports(tree) == ["numpy", "torch"]


def test_only_the_policy_module_imports_torch():
    for path in _module_paths():
        imports = _module_scope_imports(ast.parse(path.read_text()))
        assert "torch" not in imports, path.name
        if path.name != "policy.py":
            assert "import torch" not in path.read_text(), path.name


def test_the_conversion_imports_with_numpy_alone():
    """A smoke test of the import graph: nothing optional, nothing heavy."""
    import importlib

    for name in ("conversion", "images", "rotations", "settings"):
        importlib.import_module(f"icilval.model.bpp_robotwin.{name}")


# ------------------------------------------------------------------- the policy shell


class _Tensor:
    """Just enough of a torch tensor for `_plan` to unwrap one, so the plumbing is testable
    without torch. The numbers it carries are numpy's."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, index):
        return _Tensor(self.array[index])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class _Network:
    """A stand-in for the BPP network: records what it was prompted with, returns a fixed chunk."""

    def __init__(self, chunk):
        self.chunk = np.asarray(chunk)
        self.prompts: list[dict] = []
        self.calls = 0
        self.exec_horizon = None

    def reset(self, action_exec_horizon=None):
        self.exec_horizon = action_exec_horizon

    def prompt(self, prompt_dict):
        self.prompts.append(prompt_dict)

    def predict_action(self, obs_dict):
        self.calls += 1
        self.last_obs = obs_dict
        return {"action": _Tensor(self.chunk[None])}


def _policy(spec, tmp_path, monkeypatch, chunk):
    """A `BPPRoboTwinPolicy` wired to a fake network, with torch stubbed out of the two places
    that reach for it.

    Built on `pick_and_place` because what `PolicyBase` reads from a skill is the BPP cadence -
    the chunk size, the padding, the two horizons - and that is the skill in the contract today
    that declares all of it. Nothing here touches its simulator.
    """
    import sys
    import types
    from contextlib import nullcontext

    from icilval.model.bpp_robotwin.policy import BPPRoboTwinPolicy

    stub = types.ModuleType("torch")
    stub.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", stub)

    policy = BPPRoboTwinPolicy(tmp_path, tmp_path, spec, "pick_and_place", device="cpu")
    monkeypatch.setattr(policy, "_lowdim", lambda a: np.asarray(a, dtype=np.float32))
    monkeypatch.setattr(policy, "_mask", lambda a: np.asarray(a, dtype=bool))
    policy.policy = _Network(chunk)
    policy.max_chunks = None
    policy.image_size = 32
    return policy


def _chunk(rows: int) -> np.ndarray:
    chunk = np.zeros((rows, ACTION_DIM))
    chunk[:, 0] = 0.5
    chunk[:, 3:9] = R.matrix_to_rot6d(np.eye(3))
    chunk[:, 9] = -1.0
    return chunk


def test_the_policy_prompts_the_network_once_with_the_five_keys(spec, tmp_path, monkeypatch):
    policy = _policy(spec, tmp_path, monkeypatch, _chunk(12))
    arrays = demo_arrays(arm="left")
    info = policy.set_prompt(arrays)

    assert len(policy.policy.prompts) == 1
    prompt = policy.policy.prompts[0]
    assert set(prompt["obs"]) == set(C.OBSERVATION_KEYS)
    assert prompt["action"].shape == (info.chunks, policy.chunk_n, ACTION_DIM)
    assert prompt["metadata"]["mask"].shape == (info.chunks,)
    assert info.steps == len(policy._prompt.actions)
    assert policy.policy.exec_horizon == policy.exec_horizon


def test_one_action_leaves_the_queue_per_act_and_a_chunk_is_planned_once(
    spec, tmp_path, monkeypatch
):
    """The benchmark's step limit counts model steps, and each delta is applied to the pose
    measured at the step it is executed at - so a chunk cannot be handed back all at once."""
    policy = _policy(spec, tmp_path, monkeypatch, _chunk(12))
    arrays = demo_arrays(arm="left")
    policy.set_prompt(arrays)

    rows = [policy.act([observation(arrays, i)]) for i in range(policy.exec_horizon)]
    assert all(row.shape == (1, BIMANUAL_EE_DIM) for row in rows)
    assert policy.policy.calls == 1
    assert policy.plans == 1
    # The queue is empty again, so the next call re-plans.
    policy.act([observation(arrays, 0)])
    assert policy.policy.calls == 2


def test_reset_forgets_the_episode(spec, tmp_path, monkeypatch):
    policy = _policy(spec, tmp_path, monkeypatch, _chunk(12))
    arrays = demo_arrays(arm="left")
    policy.set_prompt(arrays)
    policy.act([observation(arrays, 0)])
    policy.reset()
    assert not policy._queue and policy._prompt is None and policy.episode_info() == {}
    with pytest.raises(ConversionError, match="set_prompt"):
        policy.act([observation(arrays, 0)])


def test_the_episode_record_says_which_arm_was_driven(spec, tmp_path, monkeypatch):
    policy = _policy(spec, tmp_path, monkeypatch, _chunk(12))
    arrays = demo_arrays(arm="right")
    policy.set_prompt(arrays)
    policy.act([observation(arrays, 0)])
    info = policy.episode_info()
    assert info["active_arm"] == "right" and info["arm_rule"] == "path"
    assert info["plans"] == 1 and info["calls"] == 1
    assert "clipped_action_fraction" in info and "reanchors" in info
