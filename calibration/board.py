"""Checkerboard specification, detection, PnP, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import transform


@dataclass(frozen=True)
class BoardSpec:
    inner_corners_columns: int
    inner_corners_rows: int
    square_size_m: float

    def __post_init__(self) -> None:
        if self.inner_corners_columns < 3 or self.inner_corners_rows < 3:
            raise ValueError("checkerboard needs at least 3x3 inner corners")
        if not 0.001 <= self.square_size_m <= 0.2:
            raise ValueError("square_size_m is outside a plausible 1-200 mm range")

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.inner_corners_columns, self.inner_corners_rows

    @property
    def object_points(self) -> np.ndarray:
        points = np.zeros(
            (self.inner_corners_columns * self.inner_corners_rows, 3),
            dtype=np.float64,
        )
        grid = np.mgrid[
            0 : self.inner_corners_columns,
            0 : self.inner_corners_rows,
        ].T.reshape(-1, 2)
        points[:, :2] = grid * self.square_size_m
        return points

    def canonical(self) -> dict[str, float | int]:
        return {
            "inner_corners_columns": self.inner_corners_columns,
            "inner_corners_rows": self.inner_corners_rows,
            "square_size_m": self.square_size_m,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "BoardSpec":
        return cls(
            int(value["inner_corners_columns"]),
            int(value["inner_corners_rows"]),
            float(value["square_size_m"]),
        )


@dataclass(frozen=True)
class PnpObservation:
    camera_from_board: np.ndarray
    corners_px: np.ndarray
    reprojection_errors_px: np.ndarray

    @property
    def reprojection_rms_px(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.reprojection_errors_px))))


def camera_matrix(intrinsics: dict[str, Any]) -> np.ndarray:
    return np.array(
        (
            (float(intrinsics["fx"]), 0.0, float(intrinsics["ppx"])),
            (0.0, float(intrinsics["fy"]), float(intrinsics["ppy"])),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def distortion_coefficients(intrinsics: dict[str, Any]) -> np.ndarray:
    coefficients = intrinsics.get("coeffs", (0.0, 0.0, 0.0, 0.0, 0.0))
    result = np.asarray(coefficients, dtype=np.float64).reshape(-1, 1)
    model = str(intrinsics.get("distortion_model", "none")).lower()
    if "inverse_brown" in model and not np.allclose(result, 0.0, atol=1e-12):
        raise ValueError(
            "inverse Brown-Conrady image points are not directly compatible with OpenCV PnP"
        )
    return result


def depth_corner_metrics(
    depth_raw: np.ndarray,
    depth_scale_m_per_unit: float,
    board: BoardSpec,
    observation: PnpObservation,
    *,
    radius_px: int = 2,
) -> dict[str, float]:
    if depth_raw.ndim != 2 or depth_raw.size == 0:
        raise ValueError("aligned depth must be a non-empty 2D array")
    if depth_scale_m_per_unit <= 0:
        raise ValueError("depth scale must be positive")
    predicted = (
        observation.camera_from_board[:3, :3] @ board.object_points.T
        + observation.camera_from_board[:3, 3:4]
    ).T[:, 2]
    errors_mm: list[float] = []
    height, width = depth_raw.shape
    for corner, expected_z in zip(observation.corners_px, predicted):
        u, v = (int(round(float(value))) for value in corner)
        u0, u1 = max(0, u - radius_px), min(width, u + radius_px + 1)
        v0, v1 = max(0, v - radius_px), min(height, v + radius_px + 1)
        valid = depth_raw[v0:v1, u0:u1]
        valid = valid[valid > 0]
        if valid.size:
            measured_z = float(np.median(valid)) * depth_scale_m_per_unit
            errors_mm.append(abs(measured_z - float(expected_z)) * 1000.0)
    valid_fraction = len(errors_mm) / len(observation.corners_px)
    if not errors_mm:
        return {"valid_fraction": 0.0, "median_mm": float("inf"), "p95_mm": float("inf")}
    return {
        "valid_fraction": float(valid_fraction),
        "median_mm": float(np.median(errors_mm)),
        "p95_mm": float(np.percentile(errors_mm, 95)),
    }


def detect_corners(image_bgr: np.ndarray, board: BoardSpec) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("empty checkerboard image")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
    flags |= getattr(cv2, "CALIB_CB_ACCURACY", 0)
    found, corners = cv2.findChessboardCornersSB(gray, board.pattern_size, flags=flags)
    if not found or corners is None:
        raise ValueError(
            f"checkerboard {board.pattern_size[0]}x{board.pattern_size[1]} inner corners not found"
        )
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(corners) != len(board.object_points):
        raise ValueError("checkerboard detector returned an unexpected corner count")
    return corners


def solve_pnp(
    image_bgr: np.ndarray,
    board: BoardSpec,
    intrinsics: dict[str, Any],
) -> PnpObservation:
    corners = detect_corners(image_bgr, board)
    matrix = camera_matrix(intrinsics)
    distortion = distortion_coefficients(intrinsics)
    ok, rotation_vector, translation = cv2.solvePnP(
        board.object_points,
        corners,
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnP failed")
    if hasattr(cv2, "solvePnPRefineLM"):
        rotation_vector, translation = cv2.solvePnPRefineLM(
            board.object_points,
            corners,
            matrix,
            distortion,
            rotation_vector,
            translation,
        )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    projected, _ = cv2.projectPoints(
        board.object_points,
        rotation_vector,
        translation,
        matrix,
        distortion,
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)
    return PnpObservation(transform(rotation, translation), corners, errors)


def write_diagnostic(
    path: str | Path,
    image_bgr: np.ndarray,
    board: BoardSpec,
    observation: PnpObservation,
    intrinsics: dict[str, Any],
) -> None:
    output = image_bgr.copy()
    cv2.drawChessboardCorners(
        output,
        board.pattern_size,
        observation.corners_px.reshape(-1, 1, 2).astype(np.float32),
        True,
    )
    rotation_vector, _ = cv2.Rodrigues(observation.camera_from_board[:3, :3])
    cv2.drawFrameAxes(
        output,
        camera_matrix(intrinsics),
        distortion_coefficients(intrinsics),
        rotation_vector,
        observation.camera_from_board[:3, 3],
        board.square_size_m * 2.0,
        2,
    )
    if not cv2.imwrite(str(path), output):
        raise OSError(f"failed to write diagnostic image: {path}")
