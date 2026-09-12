"""The `bpp_robotwin_v1` conversion: a bimanual robot's arrays as the network's, and back.

**numpy only.** Nothing here imports torch, and nothing here imports the benchmark - not its
core, not its policy toolkit. The conversion stands alone, so it is testable in the validator's
pure environment and so removing the in-repo benchmarks changes nothing about it. The policy
class that drives it is `policy.py`, which imports torch lazily.

The array-name contract
-----------------------

A demonstration reaches `set_prompt` as the named arrays the benchmark wrote, after the field's
`demoview` has dropped whatever that field withholds. An observation reaches `act` as the same
names, one step at a time, so a single mapping serves both and neither side has to translate:

| name | demonstration | one observation | what |
| --- | --- | --- | --- |
| `frames_<camera>` | (T, h, w, 3) uint8 | (h, w, 3) uint8 | one array per camera |
| `qpos` | (T, 14) | (14,) | joint positions, 6 joints + gripper per arm |
| `endpose` | (T, 16) | (16,) | per arm, flange pose `[x, y, z, qw, qx, qy, qz]` then the gripper command |
| `gripper_joints` | (T, 2, J) | (2, J) | the **measured** finger joints, left arm then right, in metres |
| `times` | (T,) | - | seconds since the expert started; metadata, not a channel |
| `actions` | (T-1, 14) | - | the expert's own actions; this conversion does not use them |

Confirmed against the benchmark's `competition/src/icil_benchmark_robotwin/prompt.py`, whose
`CHANNELS` publishes exactly `frames_` (a prefix), `qpos`, `endpose`, `gripper_joints` and
`actions`, with `times` as metadata. Two things the benchmark has to keep matching, because
nothing else would catch them:

- `gripper_joints` must be handed over. The gripper inside `qpos` and `endpose` is a *command*,
  which reads fully closed while the fingers rest on an object; the network was trained on where
  the fingers actually are, so the measurement is the only usable source.
- `times` must reach the policy. It says when a frame was taken, never what the robot did, so it
  belongs in a metadata channel that every view keeps. Without it a demonstration cannot be
  resampled onto the checkpoint's control rate, and this refuses rather than resampling on the
  frame index, which distorts the prompt by up to a fifth of a second per motion primitive.

What the network consumes
-------------------------

Per the architecture template's `shape_meta`: `agentview_rgb` and `eye_in_hand_rgb`, each
(3, 224, 224) float in [0, 1]; `ee_pos` (3,), the tool centre point in the training domain's
world frame; `ee_ori` (6,), the rot6d of the tool's orientation; `gripper_states` (2,),
`[finger, -finger]` in metres; and a (10,) action `[dx, dy, dz, rot6d(6), gripper]`, `rep:
delta`. One action is a 20 Hz OSC command in [-1, 1]: position in units of 0.05 m of goal
offset, rotation as an axis-angle in units of 0.5 rad re-encoded as that axis-angle's rot6d
(exactly as the checkpoint's own dataset encodes it), and a gripper command that is **negative
to open**.

An OSC delta is a goal *offset*, not a displacement: the arm covers about `alpha` of it in one
control step, so prompt deltas are divided by the gain and executed deltas multiplied by it.

Frames: a vector v of the training world is `M v` in the robot's world (`MODEL_TO_WORLD`), and
the controller applies position deltas at the tool centre point and rotations on the left in the
world frame, so a rotation delta in the training axes is `M.T dR M` in the robot's. Orientations
also take a fixed tool-frame correction C, since this robot approaches along the flange's +x and
the training domain's end-effector along +z.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import images as image_path
from .rotations import (
    axis_angle_to_matrix,
    canonical_quat,
    matrix_to_axis_angle,
    matrix_to_quat,
    matrix_to_rot6d,
    quat_to_matrix,
    relative_angle,
    rot6d_to_matrix,
    slerp,
)
from .settings import (
    ACTION_DIM,
    ARM_BASE_IN_ROOT,
    ARMS,
    BIMANUAL_EE_DIM,
    CAMERA_POSES,
    EE_POSE_DIM,
    EE_SLICES,
    GRIPPER_CLOSED_BELOW,
    GRIPPER_FLAT,
    MODEL_FINGER_RANGE_M,
    MODEL_TO_WORLD,
    OSC_POSITION_SCALE_M,
    OSC_ROTATION_SCALE_RAD,
    ROBOT_FINGER_RANGE_M,
    ROBOT_ROOT_POSITION,
    ROBOT_ROOT_QUAT,
    TCP_OFFSET_M,
    ConversionError,
    Settings,
)

#: The prefix the benchmark publishes one camera array under.
FRAMES_PREFIX = "frames_"
#: The names the network's observation dict carries, in the order this module builds them.
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori", "gripper_states")


# --------------------------------------------------------------------------------- frames


def check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ConversionError(f"arm must be one of {ARMS}, got {arm!r}")
    return arm


def other_arm(arm: str) -> str:
    return ARMS[1 - ARMS.index(check_arm(arm))]


def root_rotation() -> np.ndarray:
    """(3, 3) the robot root's rotation in the world."""
    return quat_to_matrix(canonical_quat(np.asarray(ROBOT_ROOT_QUAT, dtype=np.float64)))


def arm_base(arm: str) -> np.ndarray:
    """(3,) where an arm's base link sits in the world: about (-0.30, -0.42, 0.78) for the left."""
    offset = np.asarray(ARM_BASE_IN_ROOT[check_arm(arm)], dtype=np.float64)
    return root_rotation() @ offset + np.asarray(ROBOT_ROOT_POSITION, dtype=np.float64)


def world_to_model(vectors: np.ndarray) -> np.ndarray:
    """(..., 3) world vectors in the training domain's axes: `M.T v`."""
    return np.asarray(vectors, dtype=np.float64) @ MODEL_TO_WORLD


def model_to_world(vectors: np.ndarray) -> np.ndarray:
    """(..., 3) training-domain vectors in the world: `M v`."""
    return np.asarray(vectors, dtype=np.float64) @ MODEL_TO_WORLD.T


def tcp_from_flange(pose: np.ndarray) -> np.ndarray:
    """(..., 7) flange pose -> tool centre pose: 0.12 m along the flange's own +x, same rotation.

    Along the flange's **own** axis, not the world's: the offset is `R[:, 0] * 0.12`, where R is
    the pose's rotation, so a rotated flange puts its tool centre somewhere else entirely.
    """
    return _along_own_x(pose, TCP_OFFSET_M)


def flange_from_tcp(pose: np.ndarray) -> np.ndarray:
    """(..., 7) tool centre pose -> the flange pose an `ee` action carries."""
    return _along_own_x(pose, -TCP_OFFSET_M)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """(..., 7) -> (..., 4, 4) homogeneous transforms."""
    pose = _poses(pose)
    out = np.zeros(pose.shape[:-1] + (4, 4))
    out[..., :3, :3] = quat_to_matrix(pose[..., 3:])
    out[..., :3, 3] = pose[..., :3]
    out[..., 3, 3] = 1.0
    return out


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    """(..., 4, 4) -> (..., 7), the quaternion with `w >= 0`."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[..., :3, 3], matrix_to_quat(matrix[..., :3, :3])], axis=-1)


def _along_own_x(pose: np.ndarray, distance: float) -> np.ndarray:
    pose = _poses(pose)
    x_axis = quat_to_matrix(pose[..., 3:])[..., :, 0]
    return np.concatenate([pose[..., :3] + distance * x_axis, pose[..., 3:]], axis=-1)


def _poses(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 0 or pose.shape[-1] != EE_POSE_DIM:
        raise ConversionError(f"expected poses of shape (..., {EE_POSE_DIM}), got {pose.shape}")
    return pose


def tool_correction(settings: Settings) -> np.ndarray:
    """(3, 3) C, the fixed tool-frame correction."""
    return axis_angle_to_matrix(np.asarray(settings.tool_correction_axis_angle))


# -------------------------------------------------------------------------------- cameras


def camera_pose(settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of the third-person camera in the world, from the rig's own numbers.

    The simulator builds a static camera's pose from `forward`, `left` and their cross product,
    so the camera's own axes are x forward, y left, z up and R's columns are exactly those.
    """
    entry = CAMERA_POSES[settings.agentview_camera]
    forward = np.asarray(entry["forward"], dtype=np.float64)
    left = np.asarray(entry["left"], dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    left = left / np.linalg.norm(left)
    rotation = np.stack([forward, left, np.cross(forward, left)], axis=1)
    return rotation, np.asarray(entry["position"], dtype=np.float64)


def project(point: np.ndarray, settings: Settings) -> tuple[float, float]:
    """(column, row) where a world point lands in the third-person frame.

    A pinhole camera with the simulator's `fovy` (the vertical field): the focal length in pixels
    is `(height / 2) / tan(fovy / 2)`, 217.3 px for the 45-degree camera at 320x180. The camera
    looks along its own +x, with +y to image left and +z to image top.
    """
    width, height, fovy_deg = settings.agentview_geometry()
    rotation, position = camera_pose(settings)
    camera = rotation.T @ (np.asarray(point, dtype=np.float64) - position)
    if camera[0] <= 1e-6:
        raise ConversionError(
            f"the point {list(np.asarray(point))} is not in front of {settings.agentview_camera}"
        )
    focal = (height / 2) / math.tan(math.radians(fovy_deg) / 2)
    return width / 2 - focal * camera[1] / camera[0], height / 2 - focal * camera[2] / camera[0]


def workspace_centre(settings: Settings, arm: str) -> np.ndarray:
    """(3,) the world point the crop is centred on: the arm's base, offset towards the table.

    A constant of the camera rig and the embodiment. Never scene information: no object pose, no
    task, nothing an episode could differ in.
    """
    return arm_base(check_arm(arm)) + np.asarray(settings.workspace_centre_offset_m)


def crop_column(settings: Settings, arm: str) -> float:
    """The column the arm-centred crop is centred on, before clamping.

    `Settings.crop_column` pins it; otherwise it is the projection of the arm's workspace centre.
    `images.crop_start` then clamps the window inside the frame, which for a 180-wide crop of a
    320-wide frame holds the centre within columns 90-230.
    """
    check_arm(arm)
    if settings.crop_column is not None:
        try:
            return float(settings.crop_column[arm])
        except (KeyError, TypeError) as exc:
            raise ConversionError(f"crop_column has no {arm!r} entry") from exc
    return project(workspace_centre(settings, arm), settings)[0]


def agentview_view(image: np.ndarray, settings: Settings, arm: str, output: int) -> np.ndarray:
    """(output, output, 3) float32 from the third-person frame: arm-centred crop, 128, then out."""
    return image_path.arm_centred_view(
        image, crop_column(settings, arm), crop=settings.crop, output=output
    )


def wrist_view(image: np.ndarray, settings: Settings, output: int) -> np.ndarray:
    """(output, output, 3) float32 from a wrist frame: centre square, the fixed roll, 128, out."""
    image = np.asarray(image)
    square = image_path.crop_square(image, settings.wrist_crop, centre_col=image.shape[1] / 2)
    turned = np.ascontiguousarray(np.rot90(square, k=settings.wrist_roll_quarter_turns))
    return image_path.arm_centred_view(
        turned, centre_col=settings.wrist_crop / 2, crop=settings.wrist_crop, output=output
    )


def views(
    images: Mapping[str, np.ndarray], settings: Settings, arm: str, output: int
) -> dict[str, np.ndarray]:
    """The network's two image keys, each (3, output, output) float32.

    Nothing is flipped. BPP's loader flips the rows of its stored training frames, which are
    upside down, and its own runner flips the simulator's frames the same way before the policy
    sees them; both leave the model an upright image, which is what this benchmark renders.
    """
    wanted = (settings.agentview_camera, settings.wrist_camera(arm))
    missing = [name for name in wanted if name not in images]
    if missing:
        raise ConversionError(
            f"the demonstration must carry {[FRAMES_PREFIX + m for m in missing]}; "
            f"it has {sorted(FRAMES_PREFIX + name for name in images)}"
        )
    return {
        "agentview_rgb": _chw(agentview_view(images[wanted[0]], settings, arm, output)),
        "eye_in_hand_rgb": _chw(wrist_view(images[wanted[1]], settings, output)),
    }


def _chw(view: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.moveaxis(view, -1, 0))


# ------------------------------------------------------------------------- proprioception


def gripper_states(finger_m: float) -> np.ndarray:
    """(2,) `[finger, -finger]` in metres, from one measured finger joint of this robot.

    Affine from the robot's own `gripper_scale` range onto the training domain's, so a closed
    gripper here reads as a closed gripper there. The measurement, never the commanded value,
    which reads fully closed while the fingers rest on an object.
    """
    low, high = ROBOT_FINGER_RANGE_M
    model_low, model_high = MODEL_FINGER_RANGE_M
    finger = model_low + (float(finger_m) - low) * (model_high - model_low) / (high - low)
    return np.array([finger, -finger], dtype=np.float64)


def model_tool_pose(
    flange: np.ndarray, settings: Settings, arm: str
) -> tuple[np.ndarray, np.ndarray]:
    """(position (3,), rotation (3, 3)) of the tool centre point in the training domain's frame.

    `p_model = M.T (p_world - b_arm) + b_base` and `R_model = M.T R_world C`.
    """
    pose = np.asarray(flange, dtype=np.float64)
    if pose.shape != (EE_POSE_DIM,):
        raise ConversionError(f"expected a flange pose of shape ({EE_POSE_DIM},), got {pose.shape}")
    tool = tcp_from_flange(pose)
    position = world_to_model(tool[:3] - arm_base(check_arm(arm))) + np.asarray(
        settings.model_base_m
    )
    rotation = MODEL_TO_WORLD.T @ quat_to_matrix(tool[3:]) @ tool_correction(settings)
    return position, rotation


def proprioception(
    flange: np.ndarray, finger_m: float, settings: Settings, arm: str
) -> dict[str, np.ndarray]:
    """The network's three proprioception keys for one arm at one instant."""
    position, rotation = model_tool_pose(flange, settings, arm)
    return {
        "ee_pos": position,
        "ee_ori": matrix_to_rot6d(rotation),
        "gripper_states": gripper_states(finger_m),
    }


# ------------------------------------------------------------------ reading the arrays


def cameras_in(demo: Mapping[str, Any]) -> tuple[str, ...]:
    """The camera names a demonstration or observation carries, from its `frames_*` arrays."""
    return tuple(sorted(key[len(FRAMES_PREFIX) :] for key in demo if key.startswith(FRAMES_PREFIX)))


def images_at(demo: Mapping[str, Any], index: int | None = None) -> dict[str, np.ndarray]:
    """camera -> one frame. `index` picks a step out of a demonstration; None takes it whole."""
    out = {}
    for camera in cameras_in(demo):
        block = demo[FRAMES_PREFIX + camera]
        out[camera] = block if index is None else block[index]
    return out


def endposes_of(demo: Mapping[str, Any]) -> np.ndarray:
    """(T, 16) the `ee` rows of a demonstration, or (1, 16) for a single observation."""
    rows = np.asarray(_require(demo, "endpose"), dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != BIMANUAL_EE_DIM:
        raise ConversionError(f"endpose has shape {rows.shape}, expected (..., {BIMANUAL_EE_DIM})")
    return rows


def fingers_of(demo: Mapping[str, Any], arm: str) -> np.ndarray:
    """(T,) one arm's measured base finger joint, in metres, over a demonstration.

    The base joint is entry 0, as the benchmark records it; the rest of the chain follows it and
    the network was trained on a single number.
    """
    joints = np.asarray(_require(demo, "gripper_joints", _NO_FINGERS), dtype=np.float64)
    if joints.ndim == 2:  # one observation: (2, J)
        joints = joints[None, ...]
    if joints.ndim != 3 or joints.shape[1] != len(ARMS) or joints.shape[2] < 1:
        raise ConversionError(
            f"gripper_joints has shape {joints.shape}, expected (T, {len(ARMS)}, joints)"
        )
    return joints[:, ARMS.index(check_arm(arm)), 0]


def times_of(demo: Mapping[str, Any], settings: Settings) -> np.ndarray:
    """(T,) seconds of each frame since the expert started.

    Required, and refused rather than guessed: see the module docstring. A caller whose frames
    really are evenly spaced says so with `Settings.assume_uniform_times`.
    """
    if "times" in demo:
        times = np.asarray(demo["times"], dtype=np.float64).reshape(-1)
        if np.any(np.diff(times) < 0):
            raise ConversionError("times must be non-decreasing")
        return times
    steps = len(endposes_of(demo))
    if settings.assume_uniform_times:
        return np.arange(steps, dtype=np.float64) / settings.rate_hz
    raise ConversionError(
        "the demonstration carries no `times`, so it cannot be resampled onto the checkpoint's "
        f"{settings.rate_hz:g} Hz control rate. The expert's frames are not evenly spaced, so "
        "resampling on the frame index would distort the prompt; the benchmark must publish "
        "`times` in a channel every demonstration view keeps"
    )


_NO_FINGERS = (
    "this conversion needs the measured gripper joints: the gripper inside `qpos` and `endpose` "
    "is a command, which reads closed while the fingers rest on an object, and the network was "
    "trained on where the fingers actually are"
)


def _require(demo: Mapping[str, Any], key: str, why: str = "") -> Any:
    if key not in demo:
        detail = f"; {why}" if why else ""
        raise ConversionError(
            f"the demonstration has no {key!r}; it carries {sorted(demo)}{detail}"
        )
    return demo[key]


# ----------------------------------------------------------------------------- resampling


def time_slack(t: np.ndarray | float) -> np.ndarray | float:
    """How far apart two times near `t` may be and still be one instant, in seconds.

    Frame times are step counts times a float32 physics timestep, so a frame at step 25 sits
    nanoseconds off the 0.1 s grid point - an error that grows with the time. This is far below a
    physics step for any demonstration's length.
    """
    return 1e-9 + 1e-6 * np.abs(t)


def sample_times(times: np.ndarray, rate_hz: float, include_end: bool = True) -> np.ndarray:
    """The grid `t0 + k / rate_hz` covering `times`.

    It runs to the last grid point at or before the final frame; with `include_end`, one more
    point when that falls short, so the final frame - where the expert succeeded - is always
    sampled.
    """
    if not (math.isfinite(rate_hz) and rate_hz > 0):
        raise ConversionError(f"rate_hz must be positive and finite, got {rate_hz}")
    times = np.asarray(times, dtype=np.float64)
    start, end = float(times[0]), float(times[-1])
    slack = time_slack(end)
    count = math.floor((end - start + slack) * rate_hz) + 1
    if include_end and start + (count - 1) / rate_hz < end - slack:
        count += 1
    return start + np.arange(count) / rate_hz


@dataclass(frozen=True)
class Resampled:
    """A demonstration's state at each sample of a fixed-rate grid."""

    rate_hz: float
    times: np.ndarray  # (N,) seconds, on the grid t0 + k / rate_hz
    source: np.ndarray  # (N,) the frame whose image each sample shows
    held: np.ndarray  # (N,) the last frame at or before each sample, whose grippers it holds
    endposes: np.ndarray  # (N, 16)
    fingers: dict[str, np.ndarray]  # arm -> (N,) the measured base finger joint

    def __len__(self) -> int:
        return len(self.times)


def resample(demo: Mapping[str, Any], settings: Settings, rate_hz: float) -> Resampled:
    """The demonstration on a `rate_hz` grid over its own times.

    Positions interpolate linearly and orientations by slerp; every gripper value, commanded or
    measured, is held from the last frame at or before the sample, so a gripper command is never
    half-issued; images are the nearest frame's, the earlier one on a tie.
    """
    times = times_of(demo, settings)
    poses = endposes_of(demo)
    if len(times) != len(poses):
        raise ConversionError(f"{len(times)} times but {len(poses)} endpose rows")
    if len(times) < 2:
        raise ConversionError(f"a demonstration needs at least two frames, got {len(times)}")
    fingers = {arm: fingers_of(demo, arm) for arm in ARMS}

    grid = sample_times(times, rate_hz)
    last = len(times) - 1
    slack = time_slack(grid)
    before = np.clip(np.searchsorted(times, grid + slack, side="right") - 1, 0, last)
    after = np.minimum(before + 1, last)
    span = times[after] - times[before]
    moving = (after > before) & (span > 0)
    fraction = np.where(moving, (grid - times[before]) / np.where(moving, span, 1.0), 0.0)
    fraction = np.clip(fraction, 0.0, 1.0)  # a snapped sample sits a hair before its frame
    nearest = np.where(grid - times[before] <= times[after] - grid + slack, before, after)

    endposes = poses[before].copy()  # grippers held; poses overwritten below
    for arm in ARMS:
        start = EE_SLICES[arm].start
        position = slice(start, start + 3)
        quat = slice(start + 3, start + EE_POSE_DIM)
        endposes[:, position] = poses[before][:, position] + fraction[:, None] * (
            poses[after][:, position] - poses[before][:, position]
        )
        endposes[:, quat] = slerp(poses[before][:, quat], poses[after][:, quat], fraction)

    return Resampled(
        rate_hz=float(rate_hz),
        times=grid,
        source=nearest,
        held=before,
        endposes=endposes,
        fingers={arm: values[before] for arm, values in fingers.items()},
    )


# ---------------------------------------------------------------------------- arm choice


@dataclass(frozen=True)
class ArmChoice:
    """The arm a single-arm policy drives for one demonstration, and why.

    Read off the demonstration's end-effector poses and times alone - nothing privileged, nothing
    a real robot could not observe.
    """

    arm: str
    rule: str  # "path", "first_move" or "default": which step of the rule decided
    path_m: dict[str, float]
    first_move_s: dict[str, float | None]

    @property
    def idle(self) -> str:
        return other_arm(self.arm)

    def info(self) -> dict[str, Any]:
        return {
            "active_arm": self.arm,
            "arm_rule": self.rule,
            "tcp_path_m": {arm: round(self.path_m[arm], 6) for arm in ARMS},
            "first_move_s": self.first_move_s,
        }


def tool_positions(demo: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """arm -> (T, 3) the tool centre point's world position in every frame."""
    poses = endposes_of(demo)
    return {
        arm: tcp_from_flange(poses[:, EE_SLICES[arm].start : EE_SLICES[arm].start + EE_POSE_DIM])[
            :, :3
        ]
        for arm in ARMS
    }


def choose_arm(demo: Mapping[str, Any], settings: Settings) -> ArmChoice:
    """The arm whose tool centre travels further over the demonstration.

    Paths within `arm_tie_m` of each other are a tie, broken by whichever arm's tool centre first
    moves more than `arm_move_m` from where it started; if neither ever does, or both do at the
    same instant, the left arm.
    """
    times = times_of(demo, settings)
    paths: dict[str, float] = {}
    first_move: dict[str, float | None] = {}
    for arm, tool in tool_positions(demo).items():
        paths[arm] = float(np.linalg.norm(np.diff(tool, axis=0), axis=1).sum())
        away = np.flatnonzero(np.linalg.norm(tool - tool[0], axis=1) > settings.arm_move_m)
        first_move[arm] = float(times[away[0]]) if away.size else None

    left, right = ARMS
    if abs(paths[left] - paths[right]) > settings.arm_tie_m:
        arm, rule = max(ARMS, key=paths.__getitem__), "path"
    elif first_move[left] != first_move[right] and (first_move[left], first_move[right]) != (
        None,
        None,
    ):
        started = {a: t for a, t in first_move.items() if t is not None}
        arm, rule = min(started, key=started.__getitem__), "first_move"
    else:
        arm, rule = left, "default"
    return ArmChoice(arm=arm, rule=rule, path_m=paths, first_move_s=first_move)


# -------------------------------------------------------------------------------- prompt


@dataclass(frozen=True)
class Prompt:
    """One demonstration as the network's prompt.

    `actions` is (T, 10) at `rate_hz`, one row per control step. Which of the resampled steps
    the prompt's observations are read at is the chunker's to say (`model.prompt.chunk_actions`),
    so it is not stored here.
    """

    arm: str
    rate_hz: float
    actions: np.ndarray
    clipped: float  # the fraction of action entries that hit +-1
    resampled: Resampled = field(repr=False, compare=False)


def prompt_actions(resampled: Resampled, arm: str, settings: Settings) -> tuple[np.ndarray, float]:
    """((T, 10), clipped fraction) the demonstration's own motion in the network's representation.

    Tool-centre deltas between consecutive samples, in the training domain's frame, divided by
    the gains and clipped to [-1, 1]; the rotation is then re-encoded as the rot6d of the clipped
    axis-angle, as the checkpoint's dataset does. The gripper label follows the onset of the
    commanded change - +1 (close) while the commanded value falls or is closed, -1 while it rises
    or is open - so a label never lags the robot's slow gripper ramp the way a threshold on the
    value would.
    """
    check_arm(arm)
    start = EE_SLICES[arm].start
    poses = resampled.endposes[:, start : start + EE_POSE_DIM]
    grips = resampled.endposes[:, start + EE_POSE_DIM]
    tools = tcp_from_flange(poses)
    positions, rotations = tools[:, :3], quat_to_matrix(tools[:, 3:])

    moves = world_to_model(np.diff(positions, axis=0)) / (settings.alpha_p * OSC_POSITION_SCALE_M)
    turns = np.einsum("tij,tkj->tik", rotations[1:], rotations[:-1])  # dR in the world
    turns = MODEL_TO_WORLD.T @ turns @ MODEL_TO_WORLD  # M.T dR M: the same delta in the model
    spins = matrix_to_axis_angle(turns) / (settings.alpha_r * OSC_ROTATION_SCALE_RAD)

    clipped_moves, clipped_spins = np.clip(moves, -1.0, 1.0), np.clip(spins, -1.0, 1.0)
    clipped = float(np.mean(np.concatenate([np.abs(moves) > 1.0, np.abs(spins) > 1.0], axis=1)))
    change = np.diff(grips)
    closing = (change < -GRIPPER_FLAT) | (
        (np.abs(change) <= GRIPPER_FLAT) & (grips[1:] < GRIPPER_CLOSED_BELOW)
    )
    actions = np.concatenate(
        [
            clipped_moves,
            matrix_to_rot6d(axis_angle_to_matrix(clipped_spins)),
            np.where(closing, 1.0, -1.0)[:, None],
        ],
        axis=1,
    )
    return actions, clipped


def build_prompt(demo: Mapping[str, Any], settings: Settings, arm: str) -> Prompt:
    """One demonstration resampled on its own times and converted into the network's actions."""
    rate = settings.rate_hz * settings.stretch
    resampled = resample(demo, settings, rate_hz=rate)
    if len(resampled) < 2:
        raise ConversionError(f"the demonstration is {len(resampled)} samples long at {rate} Hz")
    actions, clipped = prompt_actions(resampled, arm, settings)
    return Prompt(arm=arm, rate_hz=rate, actions=actions, clipped=clipped, resampled=resampled)


def prompt_observations(
    demo: Mapping[str, Any],
    prompt: Prompt,
    frames: np.ndarray,
    settings: Settings,
    output: int,
) -> dict[str, np.ndarray]:
    """The prompt's observations at `frames`, stacked: (L, 3, S, S) images and (L, d) proprio.

    One entry per chunk, taken at the resampled steps the chunker asks for, which is where the
    checkpoint's own sampler reads them.
    """
    resampled, arm = prompt.resampled, prompt.arm
    start = EE_SLICES[arm].start
    rows = []
    for step in (int(i) for i in frames):
        rows.append(
            {
                **views(images_at(demo, int(resampled.source[step])), settings, arm, output),
                **proprioception(
                    resampled.endposes[step, start : start + EE_POSE_DIM],
                    float(resampled.fingers[arm][step]),
                    settings,
                    arm,
                ),
            }
        )
    return {key: np.stack([row[key] for row in rows]) for key in OBSERVATION_KEYS}


def observation_state(
    observation: Mapping[str, Any], settings: Settings, arm: str, output: int
) -> dict[str, np.ndarray]:
    """The network's five observation keys for one step: two images, three proprio arrays."""
    start = EE_SLICES[arm].start
    flange = endposes_of(observation)[0, start : start + EE_POSE_DIM]
    finger = float(fingers_of(observation, arm)[0])
    return {
        **views(images_at(observation), settings, arm, output),
        **proprioception(flange, finger, settings, arm),
    }


# ----------------------------------------------------------------------------- execution


def decode_action(action: np.ndarray, settings: Settings) -> tuple[np.ndarray, np.ndarray, float]:
    """One network action as (world position delta, world rotation delta, gripper command).

    The inverse of the prompt's encoding, times the gains: the tool centre moves
    `alpha_p * 0.05 * a` metres and turns by `alpha_r * 0.5 * a` radians. The gripper command is
    the robot's own, 1 (open) when the network's gripper action is negative, else 0.
    """
    row = np.asarray(action, dtype=np.float64)
    if row.shape != (ACTION_DIM,):
        raise ConversionError(f"expected an action of shape ({ACTION_DIM},), got {row.shape}")
    move = MODEL_TO_WORLD @ (row[:3] * settings.alpha_p * OSC_POSITION_SCALE_M)
    spin = (
        matrix_to_axis_angle(rot6d_to_matrix(row[3:9])) * settings.alpha_r * OSC_ROTATION_SCALE_RAD
    )
    turn = MODEL_TO_WORLD @ axis_angle_to_matrix(spin) @ MODEL_TO_WORLD.T
    return move, turn, 1.0 if row[9] < 0 else 0.0


class VirtualTarget:
    """A tool-centre target integrated from deltas, re-anchored only when tracking breaks down.

    The checkpoint commands about a millimetre of tool motion per control step, under the
    planner's goal tolerance, so re-basing every command on the measured pose could stall: each
    plan would count the arm as already there. Integrating on a target of our own lets
    sub-tolerance residuals add up, and the target is re-anchored to the measured pose only when
    the tracking error exceeds its bound or a plan failed.

    Deltas are applied as the controller applies OSC deltas: the translation added to the
    position, the rotation multiplied on the left in the world frame.
    """

    def __init__(self, max_position_error_m: float, max_rotation_error_rad: float) -> None:
        if not (max_position_error_m > 0 and max_rotation_error_rad > 0):
            raise ConversionError("the re-anchoring bounds must be positive")
        self.max_position_error_m = max_position_error_m
        self.max_rotation_error_rad = max_rotation_error_rad
        self.reset()

    def reset(self) -> None:
        self.position: np.ndarray | None = None
        self.rotation: np.ndarray | None = None
        self.reanchors = {"tracking": 0, "plan_failed": 0}

    def tracking_error(
        self, measured_position: np.ndarray, measured_rotation: np.ndarray
    ) -> tuple[float, float]:
        """(metres, radians) between the target and the measured pose."""
        if self.position is None or self.rotation is None:
            raise ConversionError("no target yet: step() anchors the first one")
        distance = float(np.linalg.norm(self.position - np.asarray(measured_position)))
        return distance, float(relative_angle(self.rotation, measured_rotation))

    def step(
        self,
        measured_position: np.ndarray,
        measured_rotation: np.ndarray,
        delta_position: np.ndarray,
        delta_rotation: np.ndarray | None = None,
        plan_failed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply one delta and return the new target (position, rotation matrix), as copies."""
        measured_position = np.asarray(measured_position, dtype=np.float64)
        measured_rotation = np.asarray(measured_rotation, dtype=np.float64)
        if self.position is None or self.rotation is None:
            self._anchor(measured_position, measured_rotation)
        elif plan_failed:
            self._anchor(measured_position, measured_rotation)
            self.reanchors["plan_failed"] += 1
        else:
            distance, angle = self.tracking_error(measured_position, measured_rotation)
            if distance > self.max_position_error_m or angle > self.max_rotation_error_rad:
                self._anchor(measured_position, measured_rotation)
                self.reanchors["tracking"] += 1
        self.position = self.position + np.asarray(delta_position, dtype=np.float64)
        if delta_rotation is not None:
            self.rotation = np.asarray(delta_rotation, dtype=np.float64) @ self.rotation
        return self.position.copy(), self.rotation.copy()

    def info(self) -> dict[str, Any]:
        return {"reanchors": dict(self.reanchors)}

    def _anchor(self, position: np.ndarray, rotation: np.ndarray) -> None:
        self.position, self.rotation = position.copy(), rotation.copy()


class StallDetector:
    """Counts stalls: `window` calls in which the tool, told to move, did not.

    It reports; what to do about a stall is the caller's, and the count goes into the episode's
    record.
    """

    def __init__(self, window: int, min_motion_m: float) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ConversionError(f"window must be a positive integer, got {window!r}")
        if not min_motion_m > 0:
            raise ConversionError(f"min_motion_m must be positive, got {min_motion_m!r}")
        self.window = window
        self.min_motion_m = min_motion_m
        self.reset()

    def reset(self) -> None:
        self._positions: deque[np.ndarray] = deque(maxlen=self.window + 1)
        self._commanded: deque[float] = deque(maxlen=self.window)
        self.stalled = False
        self.events = 0

    def update(self, position: np.ndarray, commanded_m: float = 0.0) -> bool:
        """Record the tool's measured position and the motion commanded since the last call."""
        self._positions.append(np.asarray(position, dtype=np.float64))
        if len(self._positions) > 1:
            self._commanded.append(float(commanded_m))
        stalled = False
        if len(self._positions) == self.window + 1:
            first = self._positions[0]
            moved = max(float(np.linalg.norm(p - first)) for p in self._positions)
            stalled = moved < self.min_motion_m <= sum(self._commanded)
        if stalled and not self.stalled:
            self.events += 1
        self.stalled = stalled
        return stalled

    def info(self) -> dict[str, Any]:
        return {"stall_events": self.events}


@dataclass(frozen=True)
class IdleArmHold:
    """The idle arm's fixed target for a whole episode, taken from its first observation.

    Fixed rather than echoing the current pose: every successful plan re-bases on the measured
    joints within the planner's tolerance, which could let a loaded arm creep.
    """

    arm: str
    ee: np.ndarray  # (8,) flange pose then the gripper command

    @classmethod
    def from_observation(cls, observation: Mapping[str, Any], arm: str) -> IdleArmHold:
        row = endposes_of(observation)[0]
        return cls(arm=check_arm(arm), ee=np.array(row[EE_SLICES[arm]], dtype=np.float64))

    def apply(self, action: np.ndarray) -> np.ndarray:
        """(..., 16) `ee` actions with the idle arm's slots set to the hold; a copy."""
        out = np.array(action, dtype=np.float64)
        if out.ndim == 0 or out.shape[-1] != BIMANUAL_EE_DIM:
            raise ConversionError(f"expected an action of shape (..., {BIMANUAL_EE_DIM})")
        out[..., EE_SLICES[self.arm]] = self.ee
        return out


class Execution:
    """Turns the network's actions into the robot's, for one arm, over one episode."""

    def __init__(self, settings: Settings, arm: str) -> None:
        self.settings = settings
        self.arm = check_arm(arm)
        self.target = VirtualTarget(
            max_position_error_m=settings.max_position_error_m,
            max_rotation_error_rad=settings.max_rotation_error_rad,
        )
        self.stall = StallDetector(
            window=settings.stall_window, min_motion_m=settings.stall_motion_m
        )
        self.reset()

    def reset(self) -> None:
        self.target.reset()
        self.stall.reset()
        self.idle: IdleArmHold | None = None
        self.calls = 0

    def start(self, observation: Mapping[str, Any]) -> None:
        """Take the idle arm's fixed hold from the episode's first observation."""
        if self.idle is None:
            self.idle = IdleArmHold.from_observation(observation, other_arm(self.arm))

    def measured(self, observation: Mapping[str, Any]) -> np.ndarray:
        """(7,) the active arm's flange pose."""
        start = EE_SLICES[self.arm].start
        return endposes_of(observation)[0, start : start + EE_POSE_DIM]

    def act(self, action: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray:
        """(1, 16): the `ee` action row this network action becomes."""
        self.start(observation)
        flange = self.measured(observation)
        tool = tcp_from_flange(flange)
        move, turn, gripper = decode_action(action, self.settings)
        position, rotation = self.target.step(tool[:3], quat_to_matrix(tool[3:]), move, turn)
        self.stall.update(tool[:3], float(np.linalg.norm(move)))
        self.calls += 1

        target = np.eye(4)
        target[:3, :3], target[:3, 3] = rotation, position
        flange_target = flange_from_tcp(matrix_to_pose(target))
        row = np.concatenate([flange_target, [gripper]])
        ordered = np.zeros(BIMANUAL_EE_DIM)
        ordered[EE_SLICES[self.arm]] = row
        assert self.idle is not None
        return self.idle.apply(ordered)[None, :]

    def info(self) -> dict[str, Any]:
        return {"calls": self.calls, **self.target.info(), **self.stall.info()}
