"""Every constant the `bpp_robotwin_v1` conversion runs on, in one place.

Ported from the benchmark adapter's `icil_policies/bpp/settings.py` and the YAML it ships
(`policies/configs/bpp_liberogen_combination.yaml`), whose values are the defaults below. They
live here, not in `spec.json`: none of them is part of the competition contract. What *is*
contract - the image resolution, the observation history, the action horizon, the chunk size and
the padding - is read from `spec.json` through `PolicyBase`, and no number of it is repeated here.

The fixed numbers belong to the checkpoint's training domain and to its controller, not to this
competition:

- One action unit is 0.05 m of position goal and 0.5 rad of rotation goal: robosuite 1.4's
  `controllers/config/osc_pose.json` `output_max`, the controller the checkpoint's domain loads.
- `rate_hz` 20 is the rate the checkpoint was trained at, which is what a demonstration is
  resampled to. It is not the benchmark's `control_freq`, which happens to be 20 as well: one is
  a property of the weights, the other of the simulator, and they are free to differ.
- The training domain's `gripper_states` is `[finger, -finger]` in metres over roughly
  [0, 0.04]; the robot's own finger joints run -0.01 to 0.045 m (its embodiment's
  `gripper_scale`), and `gripper_states()` maps one range affinely onto the other.
- The training domain's Panda base stands at `(-0.16 - table_length / 2, 0, table_offset_z)` =
  `(-0.66, 0, 0.90)`. The z is the table top rather than the mount's own frame: an assumption
  the reference records as one, cross-checked against the checkpoint's normalizer.

The camera block is the one place a *scene* constant appears, and it is a constant of the camera
rig rather than of any episode: the far-side third-person camera's pose, copied from the
benchmark's `cameras.yml`. It is used only to project the acting arm's workspace centre, which
is where the 180-pixel crop is centred - never to read anything about the objects on the table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

#: robosuite 1.4 OSC_POSE `output_max`: one action unit of position, and one of rotation.
OSC_POSITION_SCALE_M = 0.05
OSC_ROTATION_SCALE_RAD = 0.5

#: `[x, y, z, qw, qx, qy, qz]`: the flange pose the robot reports, per arm.
EE_POSE_DIM = 7
#: The `ee` action row: a flange pose and a gripper command, per arm, left arm first.
BIMANUAL_EE_DIM = 2 * (EE_POSE_DIM + 1)
ARMS = ("left", "right")
#: Each arm's slice of that row.
EE_SLICES = {
    "left": slice(0, EE_POSE_DIM + 1),
    "right": slice(EE_POSE_DIM + 1, BIMANUAL_EE_DIM),
}

#: The network's action: position 3, rot6d 6, gripper 1. The architecture template declares it
#: as well (`action.shape`); this is what the conversion is written against.
ACTION_DIM = 10

#: The robot's root pose, from its embodiment config (`robot_pose`): a quarter turn about z, so
#: it faces world +y. 0.707 is the file's own rounding of sqrt(1/2).
ROBOT_ROOT_POSITION = (0.0, -0.65, 0.0)
ROBOT_ROOT_QUAT = (0.707, 0.0, 0.0, 0.707)

#: Each arm's base link in the root frame, from the URDF's `fl_base_joint` / `fr_base_joint`
#: origins. Their small yaws turn the base links, not these points.
ARM_BASE_IN_ROOT = {
    "left": (0.2305, 0.297, 0.782),
    "right": (0.2315, -0.3063, 0.781),
}

#: The tool centre point sits this far along the flange's own +x axis (`gripper_bias`). The
#: controller the checkpoint was trained with applies its deltas at the tool centre, so the
#: conversion integrates there and converts back to the flange for an `ee` action.
TCP_OFFSET_M = 0.12

#: A vector v of the checkpoint's training world is `MODEL_TO_WORLD @ v` in the robot's world.
#: It is the robot root's own rotation, so the training domain's axes are the robot's base axes:
#: x forward, y left, z up. The reference calls this matrix LIBERO_TO_WORLD.
MODEL_TO_WORLD = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

#: The finger ranges, both in metres: the checkpoint's training domain, then the robot's.
MODEL_FINGER_RANGE_M = (0.0, 0.04)
ROBOT_FINGER_RANGE_M = (-0.01, 0.045)

#: The simulator's camera types (`_camera_config.yml`): width, height, vertical field of view.
CAMERA_GEOMETRY = {
    "L515": (320, 180, 45.0),
    "Large_L515": (640, 360, 45.0),
    "D435": (320, 240, 37.0),
    "Large_D435": (640, 480, 37.0),
}

#: Static camera poses, copied from the benchmark's `cameras.yml`. A camera's own axes are x
#: forward, y left, z up, which is how the simulator builds a static camera's pose
#: (`stack([forward, left, up], axis=1)`), so a rotation matrix's columns are exactly those.
CAMERA_POSES = {
    "far_side_camera": {
        "type": "L515",
        "position": (0.0, 0.90, 1.45),
        "forward": (0.0, -0.778, -0.628),
        "left": (1.0, 0.0, 0.0),
    }
}

#: A commanded gripper value moves by more than this between samples to count as rising or
#: falling; below it the value is flat and the label follows whether it is open or closed.
GRIPPER_FLAT = 1e-6
#: The robot's commanded gripper is 0 (closed) to 1 (open); the halfway point splits the two.
GRIPPER_CLOSED_BELOW = 0.5


class ConversionError(ValueError):
    """A demonstration or observation this conversion cannot make sense of."""


@dataclass(frozen=True)
class Settings:
    """The conversion's constants. Frozen: nothing mutates them once a policy is built."""

    # Tracking gains: an OSC delta is a goal offset, and the arm covers `alpha` of it in one
    # control step, so prompt deltas are divided by the gain and executed deltas multiplied by
    # it. Fitted by the benchmark's `icil-bpp calibrate` on the checkpoint's own training data.
    alpha_p: float = 0.241057
    alpha_r: float = 0.203565
    #: Samples per demonstration second are `stretch * rate_hz`: above 1 slows the demonstration
    #: down, so an expert that moves faster than one training step does not clip.
    stretch: float = 1.0
    #: The checkpoint's control rate; see the module docstring on why it is not `control_freq`.
    rate_hz: float = 20.0

    #: The tool-frame correction C of `R_model = M.T R_world C`: the robot approaches along the
    #: flange's +x and the training domain's end-effector along +z, so the default is a quarter
    #: turn about the flange's y.
    tool_correction_axis_angle: tuple[float, float, float] = (0.0, math.pi / 2, 0.0)
    #: Quarter turns applied to the wrist image so it reaches the network the way the training
    #: domain's did: the two rigs need not agree on which way is up.
    wrist_roll_quarter_turns: int = 0

    # Cameras.
    agentview_camera: str = "far_side_camera"
    wrist_cameras: tuple[str, str] = ("left_camera", "right_camera")
    wrist_type: str = "D435"
    crop: int = 180
    wrist_crop: int = 240
    #: Where the acting arm's workspace centre sits relative to its base, in world axes: 0.25 m
    #: towards the table and down to the table top. Its projection centres the crop.
    workspace_centre_offset_m: tuple[float, float, float] = (0.0, 0.25, -0.04)
    #: Set to pin the crop's centre column per arm instead of projecting the workspace centre.
    crop_column: dict[str, float] | None = None

    #: The training domain's Panda base, the anchor of `p_model = M.T (p_world - b_arm) + b_base`.
    model_base_m: tuple[float, float, float] = (-0.66, 0.0, 0.90)

    # Execution. The virtual target is re-anchored to the measured pose once tracking is this
    # far off. The bound must exceed one full-scale commanded step, or every call would
    # re-anchor and sub-tolerance motion could never add up - which is the point of the target.
    max_position_error_m: float = 0.025
    max_rotation_error_rad: float = 0.25
    stall_window: int = 10
    stall_motion_m: float = 0.002

    # Which arm a single-arm policy drives: the arm whose tool centre travels further, ties
    # broken by whichever moved first.
    arm_tie_m: float = 0.01
    arm_move_m: float = 0.005

    #: Whether a demonstration handed over without its frame timestamps may be treated as evenly
    #: spaced at `rate_hz`. Off by default, and it should stay off: the expert records a frame
    #: before each motion primitive and after every physics step of a run of them, so its frames
    #: are not evenly spaced, and resampling on the index instead of on time distorts a 20 Hz
    #: prompt by up to 0.2 s per primitive. On only for a caller that knows its frames are
    #: uniform, such as a synthetic one.
    assume_uniform_times: bool = False

    def __post_init__(self) -> None:
        for name in ("alpha_p", "alpha_r", "stretch", "rate_hz"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ConversionError(f"{name} must be positive and finite, got {value!r}")
        if self.agentview_camera not in CAMERA_POSES:
            raise ConversionError(
                f"no pose for the camera {self.agentview_camera!r}; "
                f"this conversion knows {sorted(CAMERA_POSES)}"
            )
        if self.wrist_type not in CAMERA_GEOMETRY:
            raise ConversionError(f"wrist_type must be one of {sorted(CAMERA_GEOMETRY)}")
        step_m = self.alpha_p * OSC_POSITION_SCALE_M
        step_rad = self.alpha_r * OSC_ROTATION_SCALE_RAD
        if self.max_position_error_m <= step_m or self.max_rotation_error_rad <= step_rad:
            raise ConversionError(
                "the re-anchoring bounds must exceed one full-scale commanded step "
                f"({step_m:.4f} m, {step_rad:.4f} rad at these gains), or the virtual target is "
                "re-anchored at every call and sub-tolerance motion is lost"
            )
        width, height, _ = self.agentview_geometry()
        if not 0 < self.crop <= min(width, height):
            raise ConversionError(f"crop {self.crop} does not fit a {width}x{height} frame")
        wrist_width, wrist_height, _ = CAMERA_GEOMETRY[self.wrist_type]
        if not 0 < self.wrist_crop <= min(wrist_width, wrist_height):
            raise ConversionError(f"wrist_crop {self.wrist_crop} does not fit the wrist frame")
        object.__setattr__(self, "wrist_cameras", tuple(self.wrist_cameras))
        if len(self.wrist_cameras) != 2:
            raise ConversionError(f"wrist_cameras must name two, got {self.wrist_cameras}")
        for name in ("tool_correction_axis_angle", "workspace_centre_offset_m", "model_base_m"):
            object.__setattr__(self, name, _triple(getattr(self, name), name))

    def agentview_geometry(self) -> tuple[int, int, float]:
        """(width, height, vertical field of view) of the third-person camera."""
        return CAMERA_GEOMETRY[CAMERA_POSES[self.agentview_camera]["type"]]

    def wrist_camera(self, arm: str) -> str:
        return self.wrist_cameras[0] if arm == "left" else self.wrist_cameras[1]

    def describe(self) -> dict[str, Any]:
        """Every constant, JSON-ready, for a run record."""
        plain: dict[str, Any] = {}
        for entry in fields(self):
            value = getattr(self, entry.name)
            plain[entry.name] = list(value) if isinstance(value, tuple) else value
        return plain


#: The keys of `spec.json`'s `skills.<skill>.environment` a run may use to override a default.
#: Everything else there belongs to `PolicyBase`, and an unknown key is left alone rather than
#: guessed at - a typo in a config must not silently become a different conversion.
SPEC_OVERRIDES = tuple(f.name for f in fields(Settings))


def from_env(env: dict[str, Any] | None = None, **overrides: Any) -> Settings:
    """The settings, with anything a skill's `environment` block declares on top.

    A skill that says nothing gets the defaults, which are the values the reference adapter runs
    with. Only the names above are read from the environment block; the rest of it is the
    architecture's and `PolicyBase` reads it.
    """
    values: dict[str, Any] = {
        key: value for key, value in (env or {}).items() if key in SPEC_OVERRIDES
    }
    values.update(overrides)
    for key in ("tool_correction_axis_angle", "workspace_centre_offset_m", "model_base_m"):
        if key in values and values[key] is not None:
            values[key] = tuple(values[key])
    if "wrist_cameras" in values and values["wrist_cameras"] is not None:
        values["wrist_cameras"] = tuple(values["wrist_cameras"])
    try:
        return Settings(**values)
    except TypeError as exc:  # pragma: no cover - only a bad keyword reaches this
        raise ConversionError(str(exc)) from exc


def _triple(value: Any, where: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ConversionError(f"{where} must be three finite numbers, got {value!r}")
    return (float(array[0]), float(array[1]), float(array[2]))


#: The network action's three parts, named once so a reader sees the whole layout at once.
ACTION_POSITION = slice(0, 3)
ACTION_ROTATION = slice(3, 9)
ACTION_GRIPPER = 9
