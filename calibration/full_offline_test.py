"""End-to-end synthetic test of datasets, intrinsics, hand-eye, and alignment.

All files are generated in an isolated temporary directory. No RealSense or
robot SDK is imported and no network or hardware connection is made.
"""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .board import BoardSpec, camera_matrix
from .geometry import (
    euler_xyz_to_rotation,
    inverse,
    matrix_json,
    rotation_error_deg,
    transform,
    translation_error_m,
)
from .io import SCHEMA_VERSION, sha256_file, write_json
from .solve_head_eye_to_hand import solve_dataset
from .validate_intrinsics import validate_intrinsics
from .validate_three_camera_alignment import validate_alignment


INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 602.0,
    "fy": 604.0,
    "ppx": 319.5,
    "ppy": 239.5,
    "distortion_model": "distortion.inverse_brown_conrady",
    "coeffs": [0.0] * 5,
}
SERIALS = {"left": "synthetic-left", "head": "synthetic-head", "right": "synthetic-right"}


def visible_camera_from_board(
    board: BoardSpec,
    *,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    center_u: float,
    center_v: float,
    center_z: float,
) -> np.ndarray:
    rotation = euler_xyz_to_rotation(
        math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    )
    center_board = np.array(
        (
            (board.inner_corners_columns - 1) * board.square_size_m / 2.0,
            (board.inner_corners_rows - 1) * board.square_size_m / 2.0,
            0.0,
        )
    )
    rotated_center = rotation @ center_board
    target_center = np.array(
        (
            (center_u - INTRINSICS["ppx"]) / INTRINSICS["fx"] * center_z,
            (center_v - INTRINSICS["ppy"]) / INTRINSICS["fy"] * center_z,
            center_z,
        )
    )
    return transform(rotation, target_center - rotated_center)


def render_board(
    board: BoardSpec, camera_from_board: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    square_pixels = 80
    squares_columns = board.inner_corners_columns + 1
    squares_rows = board.inner_corners_rows + 1
    texture = np.full(
        (squares_rows * square_pixels + 1, squares_columns * square_pixels + 1),
        255,
        dtype=np.uint8,
    )
    for row in range(squares_rows):
        for column in range(squares_columns):
            if (row + column) % 2 == 0:
                texture[
                    row * square_pixels : (row + 1) * square_pixels,
                    column * square_pixels : (column + 1) * square_pixels,
                ] = 0

    square = board.square_size_m
    outer_board = np.array(
        (
            (-square, -square, 0.0),
            (board.inner_corners_columns * square, -square, 0.0),
            (
                board.inner_corners_columns * square,
                board.inner_corners_rows * square,
                0.0,
            ),
            (-square, board.inner_corners_rows * square, 0.0),
        ),
        dtype=np.float64,
    )
    rotation_vector, _ = cv2.Rodrigues(camera_from_board[:3, :3])
    projected_outer, _ = cv2.projectPoints(
        outer_board,
        rotation_vector,
        camera_from_board[:3, 3],
        camera_matrix(INTRINSICS),
        np.zeros((5, 1)),
    )
    source_outer = np.array(
        (
            (0.0, 0.0),
            (squares_columns * square_pixels, 0.0),
            (squares_columns * square_pixels, squares_rows * square_pixels),
            (0.0, squares_rows * square_pixels),
        ),
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        source_outer, projected_outer.reshape(-1, 2).astype(np.float32)
    )
    gray = cv2.warpPerspective(
        texture,
        homography,
        (INTRINSICS["width"], INTRINSICS["height"]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    depth = np.zeros((INTRINSICS["height"], INTRINSICS["width"]), dtype=np.uint16)
    projected_inner, _ = cv2.projectPoints(
        board.object_points,
        rotation_vector,
        camera_from_board[:3, 3],
        camera_matrix(INTRINSICS),
        np.zeros((5, 1)),
    )
    points_camera = (
        camera_from_board[:3, :3] @ board.object_points.T
        + camera_from_board[:3, 3:4]
    ).T
    for pixel, point in zip(projected_inner.reshape(-1, 2), points_camera):
        u, v = (int(round(float(value))) for value in pixel)
        depth[max(0, v - 4) : v + 5, max(0, u - 4) : u + 5] = int(
            round(float(point[2]) / 0.001)
        )
    return image, depth


def camera_record(sample_directory: Path, name: str, camera_from_board: np.ndarray) -> dict:
    image, depth = render_board(BOARD, camera_from_board)
    rgb_name = f"{name}_rgb.png"
    depth_name = f"{name}_depth.npy"
    if not cv2.imwrite(str(sample_directory / rgb_name), image):
        raise OSError(rgb_name)
    np.save(sample_directory / depth_name, depth, allow_pickle=False)
    return {
        "serial_number": SERIALS[name],
        "depth_scale_m_per_unit": 0.001,
        "intrinsics": INTRINSICS,
        "rgb_file": rgb_name,
        "depth_file": depth_name,
        "rgb_sha256": sha256_file(sample_directory / rgb_name),
        "depth_sha256": sha256_file(sample_directory / depth_name),
    }


def write_manifest(root: Path, mode: str) -> None:
    write_json(
        root / "dataset.json",
        {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "arm": "left",
            "board": BOARD.canonical(),
            "camera_serials": SERIALS,
            "world_from_arm_base": matrix_json(np.eye(4)),
        },
    )


def build_hand_eye_dataset(root: Path) -> tuple[np.ndarray, np.ndarray]:
    write_manifest(root, "head_eye_to_hand")
    base_from_head = transform(
        euler_xyz_to_rotation(math.radians(5), math.radians(-12), math.radians(8)),
        (0.28, -0.06, 0.82),
    )
    tool_from_board = transform(
        euler_xyz_to_rotation(math.radians(2), math.radians(7), math.radians(-3)),
        (0.03, -0.01, 0.15),
    )
    for index in range(24):
        phase = 2.0 * math.pi * index / 24.0
        camera_from_board = visible_camera_from_board(
            BOARD,
            rx_deg=20.0 * math.sin(phase),
            ry_deg=18.0 * math.cos(phase * 1.7),
            rz_deg=12.0 * math.sin(phase * 0.8),
            center_u=320.0 + 90.0 * math.cos(phase * 1.3),
            center_v=240.0 + 65.0 * math.sin(phase),
            center_z=0.72 + 0.16 * (0.5 + 0.5 * math.sin(phase * 1.1)),
        )
        base_from_tool = base_from_head @ camera_from_board @ inverse(tool_from_board)
        sample_directory = root / "samples" / f"sample_{index:03d}"
        sample_directory.mkdir(parents=True)
        write_json(
            sample_directory / "sample.json",
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"sample_{index:03d}",
                "cameras": {
                    "head": camera_record(sample_directory, "head", camera_from_board)
                },
                "robot": {"base_from_tool": matrix_json(base_from_tool)},
            },
        )
    return base_from_head, tool_from_board


def build_alignment_dataset(
    root: Path, world_from_head: np.ndarray
) -> np.ndarray:
    write_manifest(root, "alignment")
    head_from_board = visible_camera_from_board(
        BOARD,
        rx_deg=7.0,
        ry_deg=-9.0,
        rz_deg=3.0,
        center_u=330.0,
        center_v=235.0,
        center_z=0.82,
    )
    world_from_board = world_from_head @ head_from_board
    for index in range(10):
        phase = 2.0 * math.pi * index / 10.0
        left_from_board = visible_camera_from_board(
            BOARD,
            rx_deg=12.0 * math.sin(phase),
            ry_deg=10.0 * math.cos(phase),
            rz_deg=-5.0,
            center_u=300.0 + 35.0 * math.cos(phase),
            center_v=240.0 + 25.0 * math.sin(phase),
            center_z=0.63 + 0.05 * math.sin(phase),
        )
        right_from_board = visible_camera_from_board(
            BOARD,
            rx_deg=-10.0 * math.cos(phase),
            ry_deg=13.0 * math.sin(phase),
            rz_deg=6.0,
            center_u=340.0 - 30.0 * math.sin(phase),
            center_v=245.0 + 20.0 * math.cos(phase),
            center_z=0.67 + 0.04 * math.cos(phase),
        )
        world_from_left = world_from_board @ inverse(left_from_board)
        world_from_right = world_from_board @ inverse(right_from_board)
        sample_directory = root / "samples" / f"sample_{index:03d}"
        sample_directory.mkdir(parents=True)
        write_json(
            sample_directory / "sample.json",
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"sample_{index:03d}",
                "cameras": {
                    "left": camera_record(sample_directory, "left", left_from_board),
                    "head": camera_record(sample_directory, "head", head_from_board),
                    "right": camera_record(sample_directory, "right", right_from_board),
                },
                "robot": {
                    "base_from_tool": matrix_json(np.eye(4)),
                    "dynamic_world_from_cameras": {
                        "left": matrix_json(world_from_left),
                        "right": matrix_json(world_from_right),
                    },
                },
            },
        )
    return world_from_board


BOARD = BoardSpec(9, 6, 0.025)


def main() -> int:
    with TemporaryDirectory(prefix="robot-calibration-offline-") as temporary:
        root = Path(temporary)
        hand_eye_root = root / "head_dataset"
        expected_head, _ = build_hand_eye_dataset(hand_eye_root)
        intrinsics_report = validate_intrinsics(
            hand_eye_root, "head", root / "head_intrinsics.json"
        )
        if intrinsics_report["status"] != "consistent_with_factory_intrinsics":
            raise AssertionError(intrinsics_report["metrics"])
        head_report = solve_dataset(hand_eye_root, root / "head_calibration.json")
        translation_mm = (
            translation_error_m(head_report["world_from_head_camera"], expected_head)
            * 1000.0
        )
        rotation_deg = rotation_error_deg(
            head_report["world_from_head_camera"], expected_head
        )
        # Raster warping and sub-pixel corner extraction add a small quantization
        # error. This remains five times stricter than the 10 mm live p95 gate.
        if translation_mm > 2.0 or rotation_deg > 0.1:
            raise AssertionError(
                f"dataset hand-eye error is {translation_mm:.3f}mm/{rotation_deg:.4f}deg"
            )

        alignment_root = root / "alignment_dataset"
        build_alignment_dataset(alignment_root, expected_head)
        alignment_report = validate_alignment(
            alignment_root,
            root / "head_calibration.json",
            root / "alignment_report.json",
        )
        if alignment_report["status"] != "approved":
            raise AssertionError(alignment_report["metrics"])
        print(
            "intrinsics dataset validation: PASS "
            f"(RMS={intrinsics_report['metrics']['calibration_rms_px']:.4f}px)"
        )
        print(
            "head eye-to-hand dataset solve: PASS "
            f"({translation_mm:.3f}mm, {rotation_deg:.4f}deg)"
        )
        metrics = alignment_report["metrics"]
        print(
            "three-camera fixed-board validation: PASS "
            f"(translation p95={metrics['three_camera_translation_consensus_mm']['p95']:.3f}mm, "
            f"rotation p95={metrics['three_camera_rotation_consensus_deg']['p95']:.4f}deg)"
        )
    print("full offline calibration test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
