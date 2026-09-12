"""Rotation conversions with pytorch3d semantics (what BPP's RotationTransformer uses), in numpy.

rot6d = the first two rows of the rotation matrix, flattened. Axis-angle follows
pytorch3d: quaternion with non-negative real part, angle = 2*atan2(|xyz|, w).
"""

from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def axis_angle_to_quaternion(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64)
    angles = np.linalg.norm(aa, axis=-1, keepdims=True)
    half = angles * 0.5
    small = np.abs(angles) < 1e-6
    sin_half_over_angle = np.where(
        small, 0.5 - (angles * angles) / 48.0, np.sin(half) / np.maximum(angles, 1e-30)
    )
    return np.concatenate([np.cos(half), aa * sin_half_over_angle], axis=-1)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / np.sum(q * q, axis=-1)
    m = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    """Shepperd's method; returns (w, x, y, z) with w >= 0."""
    m = np.asarray(m, dtype=np.float64)
    batch = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    out = np.empty((m.shape[0], 4))
    for n, r in enumerate(m):
        t = np.trace(r)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q = [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = [
                (r[2, 1] - r[1, 2]) / s,
                0.25 * s,
                (r[0, 1] + r[1, 0]) / s,
                (r[0, 2] + r[2, 0]) / s,
            ]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = [
                (r[0, 2] - r[2, 0]) / s,
                (r[0, 1] + r[1, 0]) / s,
                0.25 * s,
                (r[1, 2] + r[2, 1]) / s,
            ]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = [
                (r[1, 0] - r[0, 1]) / s,
                (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s,
                0.25 * s,
            ]
        q = np.asarray(q)
        if q[0] < 0:
            q = -q
        out[n] = q / np.linalg.norm(q)
    return out.reshape(batch + (4,))


def quaternion_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    half = np.arctan2(norms, q[..., :1])
    angles = 2 * half
    small = np.abs(angles) < 1e-6
    sin_half_over_angle = np.where(
        small, 0.5 - (angles * angles) / 48.0, np.sin(half) / np.maximum(angles, 1e-30)
    )
    return q[..., 1:] / sin_half_over_angle


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    return quaternion_to_matrix(axis_angle_to_quaternion(aa))


def matrix_to_axis_angle(m: np.ndarray) -> np.ndarray:
    return quaternion_to_axis_angle(matrix_to_quaternion(m))


def matrix_to_rotation_6d(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return m[..., :2, :].reshape(m.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def axis_angle_to_rotation_6d(aa: np.ndarray) -> np.ndarray:
    return matrix_to_rotation_6d(axis_angle_to_matrix(aa))


def rotation_6d_to_axis_angle(d6: np.ndarray) -> np.ndarray:
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def quat_xyzw_to_rotation_6d(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64)
    wxyz = q[..., [3, 0, 1, 2]]
    return matrix_to_rotation_6d(quaternion_to_matrix(wxyz))


def actions_7_to_10(actions: np.ndarray) -> np.ndarray:
    """(…,7) pos3 + axis-angle3 + gripper1 -> (…,10) pos3 + rot6d + gripper1, float32."""
    a = np.asarray(actions, dtype=np.float64)
    return np.concatenate(
        [a[..., :3], axis_angle_to_rotation_6d(a[..., 3:6]), a[..., 6:7]], axis=-1
    ).astype(np.float32)


def actions_10_to_7(actions: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float64)
    return np.concatenate(
        [a[..., :3], rotation_6d_to_axis_angle(a[..., 3:9]), a[..., 9:10]], axis=-1
    ).astype(np.float32)


def yaw_quaternion_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )
