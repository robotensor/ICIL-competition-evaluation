"""The one change a scored LIBERO scene carries: sampled from the skill's menu when the unit is
derived (pure, from the unit's RNG), applied to the environment after the scored reset, and
recorded exactly as applied.

Sampling lives here so that unit derivation stays a pure function of the catalogue and the duel
id; applying lives here so that every mutation of the MuJoCo model has one owner. A `displace`
draw carries several candidate moves; the first one the scene accepts (an object that started on
the table stays on it, the object does not close in on another one to within the clearance, the
goal is still unsatisfied) is the one applied, and its index is recorded. `LiberoEnv.reset` restores the model's lights and cameras before each unit, so a
change never leaks into the next one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...rng import HashRng
from ...spec import Spec
from .env import LiberoEnv
from .rotations import quat_multiply_wxyz, yaw_quaternion_wxyz

DISPLACE_CANDIDATES = 16
SETTLE_STEPS = 20
MAX_DROP_M = 0.05
N_LIGHTS = 4


class Infeasible(Exception):
    """The change left the scene in a state that must not be scored."""


# ---------------------------------------------------------------- sampling (pure)
def target_object(steps: list[list[str]]) -> str | None:
    """The object the task grasps: the first Grasp step's object."""
    for step in steps:
        if step and step[0].lower() == "grasp" and len(step) > 1:
            return step[1]
    return None


def sample_change(
    spec: Spec, skill: str, steps: list[list[str]], rng: HashRng, kind: str | None = None
) -> dict[str, Any]:
    """One entry of the skill's menu, uniform over the entries (or `kind`, for a calibration
    sweep), with its parameters drawn uniformly in the published ranges."""
    menu = spec.changes(skill)
    if kind is None:
        kind = sorted(menu)[rng.below(len(menu))]
    elif kind not in menu:
        raise ValueError(f"{kind!r} is not in skills.{skill}.changes")
    cfg = menu[kind]
    if kind == "displace":
        radius, lo = float(cfg["radius_m"]), float(cfg["min_delta_m"])
        yaw_max = float(cfg.get("yaw_max_rad", 0.0))
        candidates = []
        for _ in range(DISPLACE_CANDIDATES):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            dist = rng.uniform(lo, radius)
            candidates.append(
                {
                    "delta_xy": [
                        round(dist * math.cos(angle), 4),
                        round(dist * math.sin(angle), 4),
                    ],
                    "yaw": round(rng.uniform(-yaw_max, yaw_max), 4) if yaw_max else 0.0,
                }
            )
        return {
            "kind": kind,
            "target": target_object(steps),
            "min_delta_m": lo,
            "clearance_m": float(cfg["clearance_m"]),
            "candidates": candidates,
        }
    if kind == "camera":
        pos = float(cfg["pos_jitter_m"])
        ang = float(cfg["angle_jitter_rad"])
        return {
            "kind": kind,
            "camera": str(spec.media["video"]["camera"]),
            "pos_delta": [round(rng.uniform(-pos, pos), 4) for _ in range(3)],
            "angles": [round(rng.uniform(-ang, ang), 4) for _ in range(3)],
        }
    if kind == "lighting":
        lo_d, hi_d = (float(x) for x in cfg["diffuse_scale"])
        lo_a, hi_a = (float(x) for x in cfg["ambient_add"])
        lo_s, hi_s = (float(x) for x in cfg["specular_scale"])
        lo_h, hi_h = (float(x) for x in cfg["headlight_scale"])
        lights = []
        for _ in range(N_LIGHTS):
            lights.append(
                {
                    "active": not rng.chance(float(cfg["light_drop_prob"])),
                    "diffuse_scale": round(rng.uniform(lo_d, hi_d), 4),
                    "ambient_add": round(rng.uniform(lo_a, hi_a), 4),
                    "specular_scale": round(rng.uniform(lo_s, hi_s), 4),
                    "pos_jitter": [
                        round(rng.uniform(-1, 1) * float(cfg["pos_jitter_m"]), 4) for _ in range(3)
                    ],
                    "dir_jitter": [
                        round(rng.uniform(-1, 1) * float(cfg["dir_jitter_rad"]), 4)
                        for _ in range(3)
                    ],
                }
            )
        if all(not light["active"] for light in lights):
            lights[0]["active"] = True
        return {
            "kind": kind,
            "lights": lights,
            "headlight_scale": round(rng.uniform(lo_h, hi_h), 4),
        }
    if kind == "observation":
        lo_b, hi_b = (float(x) for x in cfg["brightness_scale"])
        lo_n, hi_n = (float(x) for x in cfg["noise_sigma"])
        return {
            "kind": kind,
            "brightness_scale": round(rng.uniform(lo_b, hi_b), 4),
            "noise_sigma": round(rng.uniform(lo_n, hi_n), 4),
            "noise_seed": rng.below(1 << 31),
        }
    if kind == "robot_pose":
        return {"kind": kind, "init_noise_magnitude": float(cfg["init_noise_magnitude"])}
    raise ValueError(f"unknown change kind {kind!r} in skills.{skill}.changes")


# ---------------------------------------------------------------- applying (simulator)
def init_noise_of(change: dict[str, Any] | None) -> float | None:
    """The joint initialization noise a unit's environment must be built with, if any."""
    if change and change.get("kind") == "robot_pose":
        return float(change["init_noise_magnitude"])
    return None


def apply_change(env: LiberoEnv, change: dict[str, Any] | None) -> dict[str, Any]:
    """Apply a sampled change to a freshly reset environment. Returns what was applied."""
    kind = (change or {}).get("kind", "none")
    if kind in ("none", "observation", "robot_pose"):
        return {"kind": kind}
    if kind == "displace":
        return displace(env, change)
    if kind == "camera":
        return move_camera(env, change)
    if kind == "lighting":
        return apply_lighting(env, change)
    raise ValueError(f"unknown change kind {kind!r}")


def _table_bounds(env: LiberoEnv, margin_m: float = 0.04) -> tuple[np.ndarray, np.ndarray] | None:
    raw = env.raw
    size = next((getattr(raw, a) for a in dir(raw) if a.endswith("table_full_size")), None)
    offset = next((getattr(raw, a) for a in dir(raw) if a.endswith("table_offset")), None)
    if size is None or offset is None:
        return None
    half = np.asarray(size[:2], dtype=np.float64) / 2.0 - margin_m
    center = np.asarray(offset[:2], dtype=np.float64)
    return center - half, center + half


def displace(env: LiberoEnv, change: dict[str, Any]) -> dict[str, Any]:
    """Move the target object by the first candidate the scene accepts."""
    name = change.get("target")
    if not name:
        raise Infeasible("the task names no object to displace")
    state = env.sim_state()
    others = {o: env.object_position(o)[:2] for o in env.movable_objects() if o != name}
    bounds = _table_bounds(env)
    clearance = float(change.get("clearance_m", 0.0))
    min_delta = float(change.get("min_delta_m", 0.0))
    start = env.object_position(name).copy()
    # an object that starts on a fixture (the cabinet top, the stove) is not bound to the table
    on_table = bounds is not None and bool(
        np.all(start[:2] >= bounds[0]) and np.all(start[:2] <= bounds[1])
    )
    reasons = []
    for index, cand in enumerate(change["candidates"]):
        env.set_sim_state(state)
        before = env.object_position(name).copy()
        target = before[:2] + np.asarray(cand["delta_xy"], dtype=np.float64)
        if (
            on_table
            and bounds is not None
            and (np.any(target < bounds[0]) or np.any(target > bounds[1]))
        ):
            reasons.append(f"{index}: off the table")
            continue
        # it may not close in on another object to within the clearance; an object it already
        # sat closer to than that (a cluttered scene) is fine as long as it does not get nearer
        crowding = [
            o
            for o, xy in others.items()
            if np.linalg.norm(target - xy) < clearance
            and np.linalg.norm(target - xy) < np.linalg.norm(before[:2] - xy)
        ]
        if crowding:
            reasons.append(f"{index}: closes in on {', '.join(crowding)}")
            continue
        joint = env.object_joint(name)
        qpos = np.array(env.sim.data.get_joint_qpos(joint), dtype=np.float64).reshape(-1)
        if qpos.shape != (7,):
            raise Infeasible(f"{name} is not a free-joint object")
        qpos[0] += float(cand["delta_xy"][0])
        qpos[1] += float(cand["delta_xy"][1])
        qpos[2] += 0.005  # a hair up so a turned mesh does not start inside the table
        yaw = float(cand.get("yaw", 0.0))
        if yaw:
            qpos[3:7] = quat_multiply_wxyz(yaw_quaternion_wxyz(yaw), qpos[3:7])
        env.sim.data.set_joint_qpos(joint, qpos)
        env.forward()
        env.settle(SETTLE_STEPS)
        after = env.object_position(name)
        dist = float(np.linalg.norm(after[:2] - before[:2]))
        dropped = float(before[2] - after[2])
        if dropped > MAX_DROP_M:
            reasons.append(f"{index}: fell {dropped:.3f} m")
            continue
        if dist + 1e-6 < min_delta:
            reasons.append(f"{index}: moved {dist:.3f} m, below {min_delta:.3f} m")
            continue
        if env.success():
            reasons.append(f"{index}: goal already satisfied")
            continue
        return {
            "kind": "displace",
            "target": name,
            "candidate": index,
            "delta_xy": [
                round(float(after[0] - before[0]), 4),
                round(float(after[1] - before[1]), 4),
            ],
            "yaw": yaw,
            "distance_m": round(dist, 4),
        }
    env.set_sim_state(state)
    raise Infeasible(f"no displacement of {name} was feasible: " + "; ".join(reasons))


def _rotate_small(v: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate a direction by small angles about x, y, z (in that order)."""
    ax, ay, az = (float(a) for a in angles)
    cx, sx, cy, sy, cz, sz = (
        math.cos(ax),
        math.sin(ax),
        math.cos(ay),
        math.sin(ay),
        math.cos(az),
        math.sin(az),
    )
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    out = rz @ ry @ rx @ np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(out)
    return out / n if n > 1e-9 else out


def _small_rotation_quat_wxyz(angles: np.ndarray) -> np.ndarray:
    """The quaternion of the same small rotation `_rotate_small` applies (z after y after x)."""

    def axis_quat(axis: int, a: float) -> np.ndarray:
        q = np.zeros(4)
        q[0] = math.cos(a / 2)
        q[1 + axis] = math.sin(a / 2)
        return q

    ax, ay, az = (float(a) for a in angles)
    return quat_multiply_wxyz(
        axis_quat(2, az), quat_multiply_wxyz(axis_quat(1, ay), axis_quat(0, ax))
    )


def move_camera(env: LiberoEnv, change: dict[str, Any]) -> dict[str, Any]:
    """Shift and turn a fixed camera of the model."""
    m = env.mj_model
    cam = env.camera_id(str(change["camera"]))
    m.cam_pos[cam] = m.cam_pos[cam] + np.asarray(change["pos_delta"], dtype=np.float64)
    m.cam_quat[cam] = quat_multiply_wxyz(
        _small_rotation_quat_wxyz(np.asarray(change["angles"], dtype=np.float64)), m.cam_quat[cam]
    )
    env.forward()
    return {
        "kind": "camera",
        "camera": str(change["camera"]),
        "pos": [round(float(x), 4) for x in m.cam_pos[cam]],
        "quat": [round(float(x), 5) for x in m.cam_quat[cam]],
    }


def apply_lighting(env: LiberoEnv, lighting: dict[str, Any]) -> dict[str, Any]:
    """Rescale and jitter the lights present in the model, and the headlight."""
    m = env.mj_model
    n = int(m.nlight)
    applied = []
    for i, light in enumerate(lighting.get("lights", [])[:n]):
        m.light_active[i] = 1 if light.get("active", True) else 0
        m.light_diffuse[i] = np.clip(m.light_diffuse[i] * float(light["diffuse_scale"]), 0, 2)
        m.light_ambient[i] = np.clip(m.light_ambient[i] + float(light["ambient_add"]), 0, 1)
        m.light_specular[i] = np.clip(m.light_specular[i] * float(light["specular_scale"]), 0, 2)
        m.light_pos[i] = m.light_pos[i] + np.asarray(light["pos_jitter"], dtype=np.float64)
        m.light_dir[i] = _rotate_small(
            m.light_dir[i], np.asarray(light["dir_jitter"], dtype=np.float64)
        )
        applied.append(i)
    scale = float(lighting.get("headlight_scale", 1.0))
    m.vis.headlight.diffuse[:] = np.clip(np.asarray(m.vis.headlight.diffuse) * scale, 0, 2)
    env.forward()
    return {
        "kind": "lighting",
        "lights_applied": applied,
        "model_lights": n,
        "headlight_scale": scale,
    }


# ---------------------------------------------------------------- observations
class ObservationChange:
    """Brightness and Gaussian pixel noise on the images the policy sees, seeded so both sides
    see the same pixels; clips keep the rendered frame."""

    IMAGE_KEYS = ("agentview", "eye_in_hand")

    def __init__(self, change: dict[str, Any] | None):
        active = bool(change) and change.get("kind") == "observation"
        self.scale = float(change["brightness_scale"]) if active else 1.0
        self.sigma = float(change["noise_sigma"]) if active else 0.0
        self.rng = np.random.default_rng(int(change["noise_seed"])) if active else None
        self.active = active and (self.scale != 1.0 or self.sigma > 0.0)

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not self.active:
            return obs
        out = dict(obs)
        for key in self.IMAGE_KEYS:
            img = np.asarray(obs[key], dtype=np.float32) * self.scale
            if self.sigma > 0.0 and self.rng is not None:
                img = img + self.rng.normal(0.0, self.sigma, size=img.shape).astype(np.float32)
            out[key] = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        return out
