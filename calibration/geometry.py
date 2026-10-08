"""Unambiguous standard homogeneous-transform helpers.

Every name ``a_from_b`` maps a point expressed in frame B into frame A.
Matrices use column vectors: ``p_a = a_from_b @ p_b``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def transform(rotation: Any, translation: Any) -> np.ndarray:
    rotation_array = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation_array = np.asarray(translation, dtype=np.float64).reshape(3)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_array
    result[:3, 3] = translation_array
    validate_transform(result)
    return result


def validate_transform(value: Any, *, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant is not +1")
    return matrix


def inverse(value: Any) -> np.ndarray:
    matrix = validate_transform(value)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return transform(rotation.T, -rotation.T @ translation)


def euler_xyz_to_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """Match the legacy code's ``Rz @ Ry @ Rx`` Euler convention, in radians."""

    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rx_matrix = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    ry_matrix = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rz_matrix = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return rz_matrix @ ry_matrix @ rx_matrix


def pose_xyz_euler_to_transform(pose: Iterable[float]) -> np.ndarray:
    values = tuple(float(value) for value in pose)
    if len(values) != 6:
        raise ValueError("pose must contain xyz and Euler rx/ry/rz")
    return transform(euler_xyz_to_rotation(*values[3:]), values[:3])


def legacy_link_backward_transform(rotation: Any, translation: Any) -> np.ndarray:
    """Represent the legacy ``Link.backward`` child-to-parent mapping.

    The upstream implementation uses ``einsum('ji,...j->...i', R.T, p)``.
    Despite the transposed operand, that index expression evaluates to ``R @ p``.
    Keeping this conversion explicit prevents a second accidental transpose.
    """

    return transform(rotation, translation)


def rotation_error_deg(left: Any, right: Any) -> float:
    relative = np.asarray(left)[:3, :3].T @ np.asarray(right)[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def translation_error_m(left: Any, right: Any) -> float:
    return float(np.linalg.norm(np.asarray(left)[:3, 3] - np.asarray(right)[:3, 3]))


def mean_transform(values: Iterable[Any]) -> np.ndarray:
    matrices = [validate_transform(value) for value in values]
    if not matrices:
        raise ValueError("at least one transform is required")
    mean_rotation = np.mean([matrix[:3, :3] for matrix in matrices], axis=0)
    u, _, vt = np.linalg.svd(mean_rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = np.mean([matrix[:3, 3] for matrix in matrices], axis=0)
    return transform(rotation, translation)


def matrix_json(value: Any) -> list[list[float]]:
    return validate_transform(value).tolist()


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty series")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }
