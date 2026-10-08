"""Solve one wrist camera's end-relative extrinsic from a fixed board.

This is the standard eye-in-hand setup: the board stays rigidly fixed while the
selected robot arm changes pose.  The head camera is used as an independent
check that the board did not move, and is not required as a calibration anchor.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .board import solve_pnp, write_diagnostic
from .geometry import (
    inverse,
    matrix_json,
    mean_transform,
    percentile_summary,
    rotation_error_deg,
    translation_error_m,
    validate_transform,
)
from .io import load_dataset, load_sample, read_json, sample_directories, write_json
from .solve_wrist_eye_in_hand_via_head import (
    _calibrate_method,
    _comparison,
    _legacy_transform,
    _pairwise_motion_metrics,
)


MINIMUM_SAMPLES = 8
MINIMUM_ROTATION_SPAN_DEG = 12.0
BOARD_TRANSLATION_SPAN_LIMIT_MM = 10.0
BOARD_ROTATION_SPAN_LIMIT_DEG = 1.0


def solve_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    arm: str,
    current_config_path: str | Path | None = None,
    previous_config_path: str | Path | None = None,
) -> dict[str, Any]:
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    root, manifest, board = load_dataset(
        dataset_path, expected_mode="wrist_eye_in_hand_fixed_board"
    )
    output = Path(output_path).resolve()
    diagnostics = output.parent / f"wrist_fixed_{arm}_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    ends: list[np.ndarray] = []
    camera_from_boards: list[np.ndarray] = []
    head_from_boards: list[np.ndarray] = []
    accepted_ids: list[str] = []
    rejected: list[dict[str, str]] = []

    for sample_directory in sample_directories(root):
        try:
            sample = load_sample(sample_directory)
            dynamic_ends = sample["robot"]["dynamic_world_from_arm_ends"]
            observations: dict[str, Any] = {}
            for camera_name in ("left", "head", "right"):
                camera = sample["cameras"][camera_name]
                if str(camera["serial_number"]) != str(
                    manifest["camera_serials"][camera_name]
                ):
                    raise ValueError(f"{camera_name} serial differs from manifest")
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
                observations[camera_name] = observation
                write_diagnostic(
                    diagnostics / f"{sample_directory.name}_{camera_name}.png",
                    image,
                    board,
                    observation,
                    camera["intrinsics"],
                )
            ends.append(validate_transform(dynamic_ends[arm], name="world_from_end"))
            camera_from_boards.append(observations[arm].camera_from_board)
            head_from_boards.append(observations["head"].camera_from_board)
            accepted_ids.append(sample["sample_id"])
        except (KeyError, ValueError, OSError, cv2.error) as error:
            rejected.append({"sample": sample_directory.name, "error": str(error)})

    if len(ends) < MINIMUM_SAMPLES:
        error = f"at least {MINIMUM_SAMPLES} valid samples are required; got {len(ends)}"
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "rejected",
            "method": "fixed_board_standard_eye_in_hand",
            "arm": arm,
            "source_dataset": str(root),
            "accepted_sample_ids": accepted_ids,
            "rejected_samples": rejected,
            "error": error,
        }
        write_json(output, report)
        return report

    motion = _pairwise_motion_metrics(ends)
    board_reference = mean_transform(head_from_boards)
    board_translation_errors = [
        translation_error_m(value, board_reference) * 1000.0
        for value in head_from_boards
    ]
    board_rotation_errors = [
        rotation_error_deg(value, board_reference) for value in head_from_boards
    ]
    board_metrics = {
        "translation_mm": percentile_summary(board_translation_errors),
        "rotation_deg": percentile_summary(board_rotation_errors),
    }
    failures: dict[str, str] = {}
    candidates: list[
        tuple[float, str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
    ] = []
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    for name, method in methods.items():
        try:
            end_from_camera = _calibrate_method(ends, camera_from_boards, method)
            world_from_board_estimates = [
                # ``camera_from_board`` is the PnP transform board -> camera.
                # Therefore the fixed board pose is world <- end <- camera
                # <- board; it must be multiplied directly, not inverted.
                world_from_end @ end_from_camera @ camera_from_board
                for world_from_end, camera_from_board in zip(ends, camera_from_boards)
            ]
            world_from_board = mean_transform(world_from_board_estimates)
            world_from_head_estimates = [
                world_board @ inverse(head_from_board)
                for world_board, head_from_board in zip(
                    world_from_board_estimates, head_from_boards
                )
            ]
            world_from_head = mean_transform(world_from_head_estimates)
            translation_errors = [
                translation_error_m(value, world_from_board) * 1000.0
                for value in world_from_board_estimates
            ]
            rotation_errors = [
                rotation_error_deg(value, world_from_board)
                for value in world_from_board_estimates
            ]
            head_translation_errors = [
                translation_error_m(value, world_from_head) * 1000.0
                for value in world_from_head_estimates
            ]
            head_rotation_errors = [
                rotation_error_deg(value, world_from_head)
                for value in world_from_head_estimates
            ]
            residuals = {
                "world_from_board_translation_error_mm": percentile_summary(
                    translation_errors
                ),
                "world_from_board_rotation_error_deg": percentile_summary(
                    rotation_errors
                ),
                "world_from_head_translation_error_mm": percentile_summary(
                    head_translation_errors
                ),
                "world_from_head_rotation_error_deg": percentile_summary(
                    head_rotation_errors
                ),
            }
            score = (
                residuals["world_from_board_translation_error_mm"]["p95"]
                + residuals["world_from_head_translation_error_mm"]["p95"]
                + 5.0
                * (
                    residuals["world_from_board_rotation_error_deg"]["p95"]
                    + residuals["world_from_head_rotation_error_deg"]["p95"]
                )
            )
            candidates.append(
                (score, name, end_from_camera, world_from_board, world_from_head, residuals)
            )
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            failures[name] = str(error)

    motion_error = None
    if motion["pairwise_rotation_deg"]["max"] < MINIMUM_ROTATION_SPAN_DEG:
        motion_error = (
            "insufficient arm motion: maximum end-orientation separation is "
            f"{motion['pairwise_rotation_deg']['max']:.3f}deg; require at least "
            f"{MINIMUM_ROTATION_SPAN_DEG:.1f}deg"
        )
    if motion_error is None and not candidates:
        motion_error = f"all OpenCV hand-eye methods failed: {failures}"
    if motion_error is None and (
        board_metrics["translation_mm"]["p95"] > BOARD_TRANSLATION_SPAN_LIMIT_MM
        or board_metrics["rotation_deg"]["p95"] > BOARD_ROTATION_SPAN_LIMIT_DEG
    ):
        motion_error = (
            "fixed-board check failed: head-camera board pose changed by "
            f"{board_metrics['translation_mm']['p95']:.3f}mm/"
            f"{board_metrics['rotation_deg']['p95']:.3f}deg"
        )

    result_fields: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "fixed_board_standard_eye_in_hand",
        "arm": arm,
        "source_dataset": str(root),
        "accepted_sample_ids": accepted_ids,
        "rejected_samples": rejected,
        "motion_excitation": motion,
        "fixed_board_check": board_metrics,
        "method_failures": failures,
        "thresholds": {
            "minimum_samples": MINIMUM_SAMPLES,
            "minimum_rotation_span_deg": MINIMUM_ROTATION_SPAN_DEG,
            "fixed_board_translation_p95_mm": BOARD_TRANSLATION_SPAN_LIMIT_MM,
            "fixed_board_rotation_p95_deg": BOARD_ROTATION_SPAN_LIMIT_DEG,
            "world_from_board_residual_translation_p95_mm": 15.0,
            "world_from_board_residual_rotation_p95_deg": 1.5,
            "world_from_head_residual_translation_p95_mm": 15.0,
            "world_from_head_residual_rotation_p95_deg": 1.5,
        },
        "notes": [
            "end_from_camera is directly compatible with move cam_extrinsic R/t.",
            "The head camera checks that the board remained fixed; it is not used as a calibration anchor.",
            "No source configuration is modified by this solver.",
        ],
    }
    if motion_error is not None:
        result_fields.update({"status": "rejected", "error": motion_error})
    else:
        _, method_name, end_from_camera, world_from_board, world_from_head, residuals = min(
            candidates, key=lambda candidate: candidate[0]
        )
        result_fields.update(
            {
                "status": (
                    "candidate_needs_task_validation"
                    if residuals["world_from_board_translation_error_mm"]["p95"] <= 15.0
                    and residuals["world_from_board_rotation_error_deg"]["p95"] <= 1.5
                    and residuals["world_from_head_translation_error_mm"]["p95"] <= 15.0
                    and residuals["world_from_head_rotation_error_deg"]["p95"] <= 1.5
                    else "rejected"
                ),
                "method_selected": method_name,
                "end_from_camera": matrix_json(end_from_camera),
                "world_from_board": matrix_json(world_from_board),
                "world_from_head_camera": matrix_json(world_from_head),
                "residuals": residuals,
                "legacy_cam_extrinsic_candidate": {
                    "R": end_from_camera[:3, :3].tolist(),
                    "t": end_from_camera[:3, 3].tolist(),
                },
            }
        )
        references: list[tuple[str, dict[str, Any]]] = []
        if current_config_path is not None:
            references.append(("current_move_config", read_json(current_config_path)))
        if previous_config_path is not None:
            references.append(("previous_measured_config", read_json(previous_config_path)))
        result_fields["reference_comparisons"] = {
            name: _comparison(end_from_camera, _legacy_transform(config, arm))
            for name, config in references
        }
    write_json(output, result_fields)
    return result_fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--current-config")
    parser.add_argument("--previous-config")
    arguments = parser.parse_args()
    report = solve_dataset(
        arguments.dataset,
        arguments.output,
        arm=arguments.arm,
        current_config_path=arguments.current_config,
        previous_config_path=arguments.previous_config,
    )
    print(f"fixed-board {arguments.arm} wrist calibration status: {report['status']}")
    if report["status"] == "rejected":
        print(report.get("error", "residual threshold failed"))
    else:
        print(f"end_from_camera={report['end_from_camera']}")
        for name, comparison in report.get("reference_comparisons", {}).items():
            print(
                f"vs {name}: {comparison['translation_difference_mm']:.3f}mm / "
                f"{comparison['rotation_difference_deg']:.3f}deg"
            )
    return 0 if report["status"] != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
