"""Synthetic calibration test; requires no camera or robot connection."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .board import BoardSpec, depth_corner_metrics, distortion_coefficients, solve_pnp
from .eye_to_hand import EyeToHandSample, solve_eye_to_hand
from .geometry import (
    euler_xyz_to_rotation,
    inverse,
    legacy_link_backward_transform,
    rotation_error_deg,
    transform,
    translation_error_m,
)


def synthetic_samples() -> tuple[list[EyeToHandSample], np.ndarray, np.ndarray]:
    base_from_camera = transform(
        euler_xyz_to_rotation(math.radians(8), math.radians(-18), math.radians(172)),
        (0.32, -0.08, 0.86),
    )
    tool_from_board = transform(
        euler_xyz_to_rotation(math.radians(2), math.radians(88), math.radians(-4)),
        (0.03, 0.01, 0.16),
    )
    samples: list[EyeToHandSample] = []
    for index in range(24):
        angle = 2.0 * math.pi * index / 24.0
        base_from_tool = transform(
            euler_xyz_to_rotation(
                math.radians(25) * math.sin(angle),
                math.radians(22) * math.cos(angle * 1.7),
                math.radians(30) * math.sin(angle * 0.7),
            ),
            (
                0.38 + 0.08 * math.cos(angle),
                0.04 + 0.12 * math.sin(angle),
                0.30 + 0.07 * math.sin(angle * 1.3),
            ),
        )
        camera_from_board = inverse(base_from_camera) @ base_from_tool @ tool_from_board
        samples.append(
            EyeToHandSample(
                f"synthetic_{index:03d}",
                base_from_tool,
                camera_from_board,
                0.1,
            )
        )
    return samples, base_from_camera, tool_from_board


def main() -> int:
    legacy_rotation = euler_xyz_to_rotation(0.0, 0.0, math.radians(90.0))
    legacy_translation = np.array((0.1, -0.2, 0.3))
    point_child = np.array((0.4, 0.2, -0.1))
    expected_point_parent = (
        np.einsum("ji,j->i", legacy_rotation.T, point_child) + legacy_translation
    )
    converted_point_parent = (
        legacy_link_backward_transform(legacy_rotation, legacy_translation)
        @ np.append(point_child, 1.0)
    )[:3]
    if not np.allclose(converted_point_parent, expected_point_parent, atol=1e-12):
        raise AssertionError("legacy Link.backward conversion has an extra transpose")
    print("legacy non-identity base transform: PASS")

    samples, expected_camera, expected_board = synthetic_samples()
    result, comparisons = solve_eye_to_hand(samples)
    camera_translation_mm = translation_error_m(
        result.base_from_camera, expected_camera
    ) * 1000.0
    camera_rotation_deg = rotation_error_deg(result.base_from_camera, expected_camera)
    board_translation_mm = translation_error_m(result.tool_from_board, expected_board) * 1000.0
    board_rotation_deg = rotation_error_deg(result.tool_from_board, expected_board)
    print(f"OpenCV {cv2.__version__}, Numpy {np.__version__}")
    print(f"selected method: {result.method}; compared methods: {len(comparisons)}")
    print(
        f"base_from_camera error: {camera_translation_mm:.9f}mm, "
        f"{camera_rotation_deg:.9f}deg"
    )
    print(
        f"tool_from_board error: {board_translation_mm:.9f}mm, "
        f"{board_rotation_deg:.9f}deg"
    )
    if camera_translation_mm > 0.01 or camera_rotation_deg > 0.001:
        raise AssertionError("eye-to-hand transform convention failed synthetic recovery")
    if board_translation_mm > 0.01 or board_rotation_deg > 0.001:
        raise AssertionError("tool-to-board transform failed synthetic recovery")

    degenerate = []
    for index in range(8):
        base_from_tool = transform(np.eye(3), (0.3 + index * 0.01, 0.0, 0.3))
        camera_from_board = inverse(expected_camera) @ base_from_tool @ expected_board
        degenerate.append(
            EyeToHandSample(str(index), base_from_tool, camera_from_board, 0.1)
        )
    try:
        solve_eye_to_hand(degenerate)
    except ValueError as error:
        print(f"degenerate-motion rejection: PASS ({error})")
    else:
        raise AssertionError("degenerate motion was not rejected")

    board = BoardSpec(9, 6, 0.025)
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    square_px = 40
    origin_u, origin_v = 120, 100
    for row in range(board.inner_corners_rows + 1):
        for column in range(board.inner_corners_columns + 1):
            if (row + column) % 2 == 0:
                u0 = origin_u + column * square_px
                v0 = origin_v + row * square_px
                image[v0 : v0 + square_px, u0 : u0 + square_px] = 0
    intrinsics = {
        "fx": 600.0,
        "fy": 600.0,
        "ppx": 320.0,
        "ppy": 240.0,
        "distortion_model": "distortion.inverse_brown_conrady",
        "coeffs": [0.0] * 5,
    }
    distortion_coefficients(intrinsics)
    pnp = solve_pnp(image, board, intrinsics)
    if pnp.reprojection_rms_px > 0.2:
        raise AssertionError(f"synthetic PnP RMS too high: {pnp.reprojection_rms_px}")
    depth = np.zeros((480, 640), dtype=np.uint16)
    for corner in pnp.corners_px:
        u, v = (int(round(float(value))) for value in corner)
        depth[v - 2 : v + 3, u - 2 : u + 3] = int(
            round(pnp.camera_from_board[2, 3] / 0.001)
        )
    depth_metric = depth_corner_metrics(depth, 0.001, board, pnp)
    if depth_metric["valid_fraction"] < 0.99 or depth_metric["p95_mm"] > 0.6:
        raise AssertionError(f"synthetic aligned-depth check failed: {depth_metric}")
    print(
        f"synthetic checkerboard PnP/depth: PASS "
        f"(RMS={pnp.reprojection_rms_px:.6f}px, depth_p95={depth_metric['p95_mm']:.3f}mm)"
    )
    print("synthetic eye-to-hand self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
