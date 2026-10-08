"""Head-camera eye-to-hand solver with method comparison and residuals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import cv2
import numpy as np

from .geometry import (
    inverse,
    matrix_json,
    mean_transform,
    percentile_summary,
    rotation_error_deg,
    transform,
    translation_error_m,
    validate_transform,
)


@dataclass(frozen=True)
class EyeToHandSample:
    sample_id: str
    base_from_tool: np.ndarray
    camera_from_board: np.ndarray
    reprojection_rms_px: float = 0.0


@dataclass(frozen=True)
class EyeToHandResult:
    method: str
    base_from_camera: np.ndarray
    tool_from_board: np.ndarray
    translation_residual_mm: dict[str, float]
    rotation_residual_deg: dict[str, float]
    reprojection_rms_px: dict[str, float]
    sample_count: int

    def canonical(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "base_from_camera": matrix_json(self.base_from_camera),
            "tool_from_board": matrix_json(self.tool_from_board),
            "translation_residual_mm": self.translation_residual_mm,
            "rotation_residual_deg": self.rotation_residual_deg,
            "reprojection_rms_px": self.reprojection_rms_px,
            "sample_count": self.sample_count,
        }


METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _validate_motion_diversity(samples: list[EyeToHandSample]) -> None:
    if len(samples) < 8:
        raise ValueError("at least 8 valid poses are required; 20-30 are recommended")
    reference = validate_transform(samples[0].base_from_tool)[:3, :3]
    rotation_vectors: list[np.ndarray] = []
    for sample in samples[1:]:
        relative = reference.T @ validate_transform(sample.base_from_tool)[:3, :3]
        vector, _ = cv2.Rodrigues(relative)
        rotation_vectors.append(vector.reshape(3))
    values = np.asarray(rotation_vectors)
    singular_values = np.linalg.svd(values - np.mean(values, axis=0), compute_uv=False)
    maximum_rotation = max(float(np.linalg.norm(value)) for value in values)
    if maximum_rotation < math.radians(15.0):
        raise ValueError("robot poses do not contain at least 15 degrees of rotational diversity")
    if len(singular_values) < 2 or singular_values[1] < math.radians(3.0):
        raise ValueError("robot pose rotations are nearly parallel/degenerate")


def _solve_method(samples: list[EyeToHandSample], name: str, method: int) -> EyeToHandResult:
    # For fixed-camera/moving-target eye-to-hand, feed tool_from_base rather
    # than base_from_tool. OpenCV's returned cam-to-gripper slot then represents
    # base_from_camera. The synthetic self-test verifies this convention.
    tool_from_base = [inverse(sample.base_from_tool) for sample in samples]
    camera_from_board = [validate_transform(sample.camera_from_board) for sample in samples]
    rotation, translation = cv2.calibrateHandEye(
        [value[:3, :3] for value in tool_from_base],
        [value[:3, 3] for value in tool_from_base],
        [value[:3, :3] for value in camera_from_board],
        [value[:3, 3] for value in camera_from_board],
        method=method,
    )
    base_from_camera = transform(rotation, translation)
    if not np.isfinite(base_from_camera).all():
        raise ValueError(f"{name} produced non-finite values")

    tool_from_boards = [
        inverse(sample.base_from_tool) @ base_from_camera @ sample.camera_from_board
        for sample in samples
    ]
    tool_from_board = mean_transform(tool_from_boards)
    translation_mm = [
        translation_error_m(value, tool_from_board) * 1000.0 for value in tool_from_boards
    ]
    rotation_deg = [rotation_error_deg(value, tool_from_board) for value in tool_from_boards]
    reprojection = [sample.reprojection_rms_px for sample in samples]
    return EyeToHandResult(
        method=name,
        base_from_camera=base_from_camera,
        tool_from_board=tool_from_board,
        translation_residual_mm=percentile_summary(translation_mm),
        rotation_residual_deg=percentile_summary(rotation_deg),
        reprojection_rms_px=percentile_summary(reprojection),
        sample_count=len(samples),
    )


def solve_eye_to_hand(samples: Iterable[EyeToHandSample]) -> tuple[EyeToHandResult, list[dict[str, Any]]]:
    selected = list(samples)
    _validate_motion_diversity(selected)
    results: list[EyeToHandResult] = []
    failures: list[dict[str, str]] = []
    for name, method in METHODS.items():
        try:
            results.append(_solve_method(selected, name, method))
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            failures.append({"method": name, "error": str(error)})
    if not results:
        raise ValueError(f"every eye-to-hand method failed: {failures}")
    results.sort(
        key=lambda value: (
            value.translation_residual_mm["median"],
            value.rotation_residual_deg["median"],
        )
    )
    comparisons = [result.canonical() for result in results]
    comparisons.extend(failures)
    return results[0], comparisons
