"""Validate that left/head/right observe one fixed board at one world pose."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from itertools import combinations

import cv2
import numpy as np

from .board import depth_corner_metrics, solve_pnp, write_diagnostic
from .geometry import (
    matrix_json,
    mean_transform,
    percentile_summary,
    rotation_error_deg,
    translation_error_m,
    validate_transform,
)
from .io import load_dataset, load_sample, read_json, sample_directories, write_json


CAMERAS = ("left", "head", "right")


def validate_alignment(
    dataset_path: str | Path,
    head_calibration_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root, manifest, board = load_dataset(dataset_path, expected_mode="alignment")
    head_calibration = read_json(head_calibration_path)
    expected_serial = manifest["camera_serials"]["head"]
    if head_calibration.get("camera_serial") != expected_serial:
        raise ValueError("head calibration serial does not match the alignment dataset")
    world_from_head = validate_transform(
        head_calibration["world_from_head_camera"], name="world_from_head_camera"
    )
    diagnostics = Path(output_path).resolve().parent / "alignment_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    translation_differences_mm: list[float] = []
    rotation_differences_deg: list[float] = []
    reprojection_rms_px: list[float] = []
    depth_errors_p95_mm: list[float] = []
    depth_valid_fractions: list[float] = []
    all_world_from_boards = []
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []

    for sample_directory in sample_directories(root):
        try:
            sample = load_sample(sample_directory)
            dynamic = sample["robot"]["dynamic_world_from_cameras"]
            world_from_camera = {
                "left": validate_transform(dynamic["left"]),
                "head": world_from_head,
                "right": validate_transform(dynamic["right"]),
            }
            world_from_boards = {}
            sample_reprojection: list[float] = []
            sample_depth_errors: list[float] = []
            sample_depth_fractions: list[float] = []
            for camera_name in CAMERAS:
                camera = sample["cameras"][camera_name]
                image = cv2.imread(
                    str(sample_directory / camera["rgb_file"]), cv2.IMREAD_COLOR
                )
                if image is None:
                    raise ValueError(f"{camera_name} RGB could not be decoded")
                observation = solve_pnp(image, board, camera["intrinsics"])
                sample_reprojection.append(observation.reprojection_rms_px)
                depth = np.load(
                    sample_directory / camera["depth_file"], allow_pickle=False
                )
                depth_metric = depth_corner_metrics(
                    depth,
                    float(camera["depth_scale_m_per_unit"]),
                    board,
                    observation,
                )
                if depth_metric["valid_fraction"] < 0.5:
                    raise ValueError(
                        f"{camera_name}: fewer than 50% of checkerboard corners "
                        "have valid aligned depth"
                    )
                sample_depth_errors.append(depth_metric["p95_mm"])
                sample_depth_fractions.append(depth_metric["valid_fraction"])
                world_from_boards[camera_name] = (
                    world_from_camera[camera_name] @ observation.camera_from_board
                )
                write_diagnostic(
                    diagnostics / f"{sample_directory.name}_{camera_name}.png",
                    image,
                    board,
                    observation,
                    camera["intrinsics"],
                )
            # Publish metrics only after all three cameras passed this sample.
            reprojection_rms_px.extend(sample_reprojection)
            depth_errors_p95_mm.extend(sample_depth_errors)
            depth_valid_fractions.extend(sample_depth_fractions)
            all_world_from_boards.extend(world_from_boards.values())
            for left_name, right_name in combinations(CAMERAS, 2):
                left = world_from_boards[left_name]
                right = world_from_boards[right_name]
                translation_differences_mm.append(
                    translation_error_m(left, right) * 1000.0
                )
                rotation_differences_deg.append(rotation_error_deg(left, right))
            accepted.append(sample["sample_id"])
        except (KeyError, ValueError, OSError, cv2.error) as error:
            rejected.append({"sample": sample_directory.name, "error": str(error)})

    if len(accepted) < 5:
        raise ValueError("at least five valid three-camera alignment samples are required")
    translation = percentile_summary(translation_differences_mm)
    rotation = percentile_summary(rotation_differences_deg)
    reprojection = percentile_summary(reprojection_rms_px)
    board_consensus = mean_transform(all_world_from_boards)
    fixed_board_translation = percentile_summary(
        translation_error_m(value, board_consensus) * 1000.0
        for value in all_world_from_boards
    )
    fixed_board_rotation = percentile_summary(
        rotation_error_deg(value, board_consensus) for value in all_world_from_boards
    )
    depth_p95 = percentile_summary(depth_errors_p95_mm)
    depth_valid_fraction = percentile_summary(depth_valid_fractions)
    passed = (
        translation["p95"] <= 10.0
        and rotation["p95"] <= 1.0
        and reprojection["p95"] <= 1.0
        and fixed_board_translation["p95"] <= 10.0
        and fixed_board_rotation["p95"] <= 1.0
        and depth_p95["p95"] <= 15.0
        and depth_valid_fraction["median"] >= 0.7
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "approved" if passed else "rejected",
        "source_dataset": str(root),
        "source_head_calibration": str(Path(head_calibration_path).resolve()),
        "camera_serials": manifest["camera_serials"],
        "accepted_sample_ids": accepted,
        "rejected_samples": rejected,
        "world_from_head_camera": matrix_json(world_from_head),
        "metrics": {
            "three_camera_translation_consensus_mm": translation,
            "three_camera_rotation_consensus_deg": rotation,
            "reprojection_rms_px": reprojection,
            "fixed_board_translation_mm": fixed_board_translation,
            "fixed_board_rotation_deg": fixed_board_rotation,
            "aligned_depth_corner_p95_mm": depth_p95,
            "aligned_depth_valid_fraction": depth_valid_fraction,
        },
        "thresholds": {
            "translation_p95_mm": 10.0,
            "rotation_p95_deg": 1.0,
            "reprojection_p95_px": 1.0,
            "fixed_board_translation_p95_mm": 10.0,
            "fixed_board_rotation_p95_deg": 1.0,
            "aligned_depth_error_p95_mm": 15.0,
            "aligned_depth_valid_fraction_median": 0.7,
        },
        "world_from_board_consensus": matrix_json(board_consensus),
    }
    write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--head-calibration", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = validate_alignment(
        arguments.dataset, arguments.head_calibration, arguments.output
    )
    print(f"alignment status: {report['status']}")
    for name, values in report["metrics"].items():
        print(f"{name}: median={values['median']:.4f}, p95={values['p95']:.4f}")
    return 0 if report["status"] == "approved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
