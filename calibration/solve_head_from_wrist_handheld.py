"""Estimate a fixed head camera from an arbitrarily hand-held checkerboard.

The checkerboard pose may change between samples. Within each sample it must be
still and visible in head plus both wrist cameras. Known dynamic wrist-camera
world poses cancel the unknown board pose and yield two independent estimates
of ``world_from_head`` per sample.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .board import depth_corner_metrics, solve_pnp, write_diagnostic
from .geometry import (
    inverse,
    matrix_json,
    mean_transform,
    percentile_summary,
    rotation_error_deg,
    translation_error_m,
    validate_transform,
)
from .io import load_dataset, load_sample, sample_directories, write_json


CAMERAS = ("left", "head", "right")


def solve_handheld_dataset(
    dataset_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    root, manifest, board = load_dataset(
        dataset_path, expected_mode="head_from_wrist_handheld"
    )
    output = Path(output_path).resolve()
    diagnostics = output.parent / "handheld_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []

    for sample_directory in sample_directories(root):
        try:
            sample = load_sample(sample_directory)
            dynamic = sample["robot"]["dynamic_world_from_cameras"]
            world_from_wrist = {
                "left": validate_transform(dynamic["left"], name="world_from_left"),
                "right": validate_transform(dynamic["right"], name="world_from_right"),
            }
            observations = {}
            depth_metrics = {}
            for camera_name in CAMERAS:
                camera = sample["cameras"][camera_name]
                if str(camera["serial_number"]) != str(
                    manifest["camera_serials"][camera_name]
                ):
                    raise ValueError(f"{camera_name} serial differs from dataset manifest")
                image = cv2.imread(
                    str(sample_directory / camera["rgb_file"]), cv2.IMREAD_COLOR
                )
                if image is None:
                    raise ValueError(f"{camera_name} RGB could not be decoded")
                observation = solve_pnp(image, board, camera["intrinsics"])
                if observation.reprojection_rms_px > 1.5:
                    raise ValueError(
                        f"{camera_name} reprojection RMS "
                        f"{observation.reprojection_rms_px:.3f}px exceeds 1.5px"
                    )
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
                        f"{camera_name}: fewer than 50% of board corners have valid depth"
                    )
                observations[camera_name] = observation
                depth_metrics[camera_name] = depth_metric
                write_diagnostic(
                    diagnostics / f"{sample_directory.name}_{camera_name}.png",
                    image,
                    board,
                    observation,
                    camera["intrinsics"],
                )

            world_from_board = {
                wrist: world_from_wrist[wrist]
                @ observations[wrist].camera_from_board
                for wrist in ("left", "right")
            }
            world_from_head_candidates = {
                wrist: world_from_board[wrist]
                @ inverse(observations["head"].camera_from_board)
                for wrist in ("left", "right")
            }
            wrist_board_translation_mm = (
                translation_error_m(
                    world_from_board["left"], world_from_board["right"]
                )
                * 1000.0
            )
            wrist_board_rotation_deg = rotation_error_deg(
                world_from_board["left"], world_from_board["right"]
            )
            if wrist_board_translation_mm > 50.0 or wrist_board_rotation_deg > 5.0:
                raise ValueError(
                    "left/right wrist estimates disagree by "
                    f"{wrist_board_translation_mm:.1f}mm/{wrist_board_rotation_deg:.2f}deg"
                )
            accepted.append(
                {
                    "sample_id": sample["sample_id"],
                    "observations": observations,
                    "depth_metrics": depth_metrics,
                    "world_from_board": world_from_board,
                    "world_from_head_candidates": world_from_head_candidates,
                    "wrist_board_translation_mm": wrist_board_translation_mm,
                    "wrist_board_rotation_deg": wrist_board_rotation_deg,
                }
            )
        except (KeyError, ValueError, OSError, cv2.error) as error:
            rejected.append({"sample": sample_directory.name, "error": str(error)})

    if len(accepted) < 8:
        raise ValueError("at least 8 valid handheld three-camera samples are required")

    # Deterministic alternating split: calibration and validation use different photos.
    calibration_samples = accepted[::2]
    validation_samples = accepted[1::2]
    calibration_candidates = [
        sample["world_from_head_candidates"][wrist]
        for sample in calibration_samples
        for wrist in ("left", "right")
    ]
    world_from_head_split = mean_transform(calibration_candidates)
    all_candidates = [
        sample["world_from_head_candidates"][wrist]
        for sample in accepted
        for wrist in ("left", "right")
    ]
    # This is the best estimate available from the existing run. It is reported
    # separately from the alternating split estimate used by the strict gate.
    world_from_head = mean_transform(all_candidates)

    head_translation_errors_mm = []
    head_rotation_errors_deg = []
    board_pair_translation_mm = []
    board_pair_rotation_deg = []
    reprojection_rms_px = []
    depth_p95_mm = []
    depth_valid_fraction = []
    for sample in validation_samples:
        observations = sample["observations"]
        board_estimates = {
            **sample["world_from_board"],
            "head": world_from_head_split @ observations["head"].camera_from_board,
        }
        for wrist in ("left", "right"):
            candidate = sample["world_from_head_candidates"][wrist]
            head_translation_errors_mm.append(
                translation_error_m(candidate, world_from_head_split) * 1000.0
            )
            head_rotation_errors_deg.append(
                rotation_error_deg(candidate, world_from_head_split)
            )
        for first, second in combinations(CAMERAS, 2):
            board_pair_translation_mm.append(
                translation_error_m(board_estimates[first], board_estimates[second])
                * 1000.0
            )
            board_pair_rotation_deg.append(
                rotation_error_deg(board_estimates[first], board_estimates[second])
            )
        for camera_name in CAMERAS:
            reprojection_rms_px.append(
                observations[camera_name].reprojection_rms_px
            )
            depth_p95_mm.append(sample["depth_metrics"][camera_name]["p95_mm"])
            depth_valid_fraction.append(
                sample["depth_metrics"][camera_name]["valid_fraction"]
            )

    all_head_translation_errors_mm = []
    all_head_rotation_errors_deg = []
    all_board_pair_translation_mm = []
    all_board_pair_rotation_deg = []
    for sample in accepted:
        observations = sample["observations"]
        board_estimates = {
            **sample["world_from_board"],
            "head": world_from_head @ observations["head"].camera_from_board,
        }
        for wrist in ("left", "right"):
            candidate = sample["world_from_head_candidates"][wrist]
            all_head_translation_errors_mm.append(
                translation_error_m(candidate, world_from_head) * 1000.0
            )
            all_head_rotation_errors_deg.append(
                rotation_error_deg(candidate, world_from_head)
            )
        for first, second in combinations(CAMERAS, 2):
            all_board_pair_translation_mm.append(
                translation_error_m(board_estimates[first], board_estimates[second])
                * 1000.0
            )
            all_board_pair_rotation_deg.append(
                rotation_error_deg(board_estimates[first], board_estimates[second])
            )

    metrics = {
        "heldout_head_translation_error_mm": percentile_summary(
            head_translation_errors_mm
        ),
        "heldout_head_rotation_error_deg": percentile_summary(head_rotation_errors_deg),
        "heldout_three_camera_board_translation_mm": percentile_summary(
            board_pair_translation_mm
        ),
        "heldout_three_camera_board_rotation_deg": percentile_summary(
            board_pair_rotation_deg
        ),
        "all_sample_head_translation_error_mm": percentile_summary(
            all_head_translation_errors_mm
        ),
        "all_sample_head_rotation_error_deg": percentile_summary(
            all_head_rotation_errors_deg
        ),
        "all_sample_three_camera_board_translation_mm": percentile_summary(
            all_board_pair_translation_mm
        ),
        "all_sample_three_camera_board_rotation_deg": percentile_summary(
            all_board_pair_rotation_deg
        ),
        "left_right_board_translation_mm_all_samples": percentile_summary(
            sample["wrist_board_translation_mm"] for sample in accepted
        ),
        "left_right_board_rotation_deg_all_samples": percentile_summary(
            sample["wrist_board_rotation_deg"] for sample in accepted
        ),
        "reprojection_rms_px": percentile_summary(reprojection_rms_px),
        "aligned_depth_corner_p95_mm": percentile_summary(depth_p95_mm),
        "aligned_depth_valid_fraction": percentile_summary(depth_valid_fraction),
    }
    passed = (
        metrics["heldout_head_translation_error_mm"]["p95"] <= 15.0
        and metrics["heldout_head_rotation_error_deg"]["p95"] <= 1.5
        and metrics["heldout_three_camera_board_translation_mm"]["p95"] <= 15.0
        and metrics["heldout_three_camera_board_rotation_deg"]["p95"] <= 1.5
        and metrics["reprojection_rms_px"]["p95"] <= 1.0
        and metrics["aligned_depth_corner_p95_mm"]["p95"] <= 15.0
        and metrics["aligned_depth_valid_fraction"]["median"] >= 0.7
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_needs_task_validation" if passed else "rejected",
        "method": "arbitrary_handheld_board_anchored_by_both_wrist_cameras",
        "source_dataset": str(root),
        "camera_serial": manifest["camera_serials"]["head"],
        "camera_serials": manifest["camera_serials"],
        "world_from_head_camera": matrix_json(world_from_head),
        "world_from_head_camera_calibration_split": matrix_json(world_from_head_split),
        "calibration_sample_ids": [sample["sample_id"] for sample in calibration_samples],
        "heldout_validation_sample_ids": [
            sample["sample_id"] for sample in validation_samples
        ],
        "rejected_samples": rejected,
        "metrics": metrics,
        "thresholds": {
            "heldout_translation_p95_mm": 15.0,
            "heldout_rotation_p95_deg": 1.5,
            "reprojection_p95_px": 1.0,
            "aligned_depth_error_p95_mm": 15.0,
            "aligned_depth_valid_fraction_median": 0.7,
        },
        "notes": [
            f"{len(accepted)} valid samples were captured in this run; 20-30 are preferred.",
            "The board pose may differ between samples.",
            "The board must be stationary during each concurrent three-camera capture.",
            "The estimate inherits error from both existing wrist-camera extrinsics.",
            "Hardware task execution remains disabled until independent task validation.",
        ],
    }
    write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = solve_handheld_dataset(arguments.dataset, arguments.output)
    print(f"head calibration status: {report['status']}")
    for name, values in report["metrics"].items():
        print(f"{name}: median={values['median']:.4f}, p95={values['p95']:.4f}")
    return 0 if report["status"] != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
