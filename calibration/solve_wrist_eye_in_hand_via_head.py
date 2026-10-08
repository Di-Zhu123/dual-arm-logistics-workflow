"""Calibrate wrist cameras relative to robot arm ends using the fixed head camera.

The checkerboard may be moved arbitrarily between samples.  At each sample all
three cameras observe the same stationary board.  The head camera therefore
provides the board-independent relative pose between each wrist camera and the
fixed head camera.  Diverse robot arm poses then provide the motion excitation
needed to solve each eye-in-hand transform.

All transform names use ``parent_from_child`` and column-vector convention.
The result ``end_from_camera`` is directly compatible with the legacy
``cam_extrinsic`` R/t fields.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .board import solve_pnp, write_diagnostic
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
from .io import load_dataset, load_sample, read_json, sample_directories, write_json


ARMS = ("left", "right")
CAMERAS = ("left", "head", "right")
MINIMUM_SAMPLES = 8
MINIMUM_ROTATION_SPAN_DEG = 12.0
MINIMUM_EXCITED_ROTATION_AXES = 2


def _pairwise_motion_metrics(world_from_ends: list[np.ndarray]) -> dict[str, Any]:
    translations_mm: list[float] = []
    rotations_deg: list[float] = []
    axes: list[np.ndarray] = []
    for first_index, first in enumerate(world_from_ends):
        for second in world_from_ends[first_index + 1 :]:
            relative = inverse(first) @ second
            translations_mm.append(translation_error_m(first, second) * 1000.0)
            angle_deg = rotation_error_deg(np.eye(4), relative)
            rotations_deg.append(angle_deg)
            if angle_deg >= 2.0:
                vector, _ = cv2.Rodrigues(relative[:3, :3])
                vector = np.asarray(vector, dtype=np.float64).reshape(3)
                norm = float(np.linalg.norm(vector))
                if norm > 1e-9:
                    axes.append(vector / norm)

    if axes:
        singular_values = np.linalg.svd(np.asarray(axes), compute_uv=False)
        normalized = singular_values / max(float(singular_values[0]), 1e-12)
        axis_rank = int(np.sum(normalized >= 0.15))
    else:
        singular_values = np.zeros(3, dtype=np.float64)
        axis_rank = 0
    return {
        "pairwise_translation_mm": percentile_summary(translations_mm),
        "pairwise_rotation_deg": percentile_summary(rotations_deg),
        "rotation_axis_singular_values": [float(value) for value in singular_values],
        "excited_rotation_axis_rank": axis_rank,
    }


def _calibrate_method(
    world_from_ends: list[np.ndarray],
    wrist_from_heads: list[np.ndarray],
    method: int,
) -> np.ndarray:
    rotation, translation = cv2.calibrateHandEye(
        [matrix[:3, :3] for matrix in world_from_ends],
        [matrix[:3, 3] for matrix in world_from_ends],
        [matrix[:3, :3] for matrix in wrist_from_heads],
        [matrix[:3, 3] for matrix in wrist_from_heads],
        method=method,
    )
    return transform(rotation, np.asarray(translation).reshape(3))


def solve_transform_pair(
    world_from_ends: Iterable[Any], wrist_from_heads: Iterable[Any]
) -> dict[str, Any]:
    """Solve one arm's end-from-camera and world-from-head transforms."""

    ends = [validate_transform(value, name="world_from_end") for value in world_from_ends]
    observations = [
        validate_transform(value, name="wrist_from_head") for value in wrist_from_heads
    ]
    if len(ends) != len(observations):
        raise ValueError("world_from_ends and wrist_from_heads must have equal length")
    if len(ends) < MINIMUM_SAMPLES:
        raise ValueError(f"at least {MINIMUM_SAMPLES} valid samples are required")

    motion = _pairwise_motion_metrics(ends)
    maximum_rotation = motion["pairwise_rotation_deg"]["max"]
    axis_rank = motion["excited_rotation_axis_rank"]
    if maximum_rotation < MINIMUM_ROTATION_SPAN_DEG:
        raise ValueError(
            "insufficient arm motion: maximum end-orientation separation is "
            f"{maximum_rotation:.3f}deg; require at least "
            f"{MINIMUM_ROTATION_SPAN_DEG:.1f}deg"
        )
    if axis_rank < MINIMUM_EXCITED_ROTATION_AXES:
        raise ValueError(
            "degenerate arm motion: rotate the arm end about at least two different axes"
        )

    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    candidates: list[tuple[float, str, np.ndarray, np.ndarray, dict[str, Any]]] = []
    failures: dict[str, str] = {}
    for name, method in methods.items():
        try:
            end_from_camera = _calibrate_method(ends, observations, method)
            head_estimates = [
                world_from_end @ end_from_camera @ wrist_from_head
                for world_from_end, wrist_from_head in zip(ends, observations)
            ]
            world_from_head = mean_transform(head_estimates)
            translation_errors = [
                translation_error_m(value, world_from_head) * 1000.0
                for value in head_estimates
            ]
            rotation_errors = [
                rotation_error_deg(value, world_from_head) for value in head_estimates
            ]
            residuals = {
                "head_translation_error_mm": percentile_summary(translation_errors),
                "head_rotation_error_deg": percentile_summary(rotation_errors),
            }
            score = (
                residuals["head_translation_error_mm"]["p95"]
                + 5.0 * residuals["head_rotation_error_deg"]["p95"]
            )
            candidates.append(
                (score, name, end_from_camera, world_from_head, residuals)
            )
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            failures[name] = str(error)

    if not candidates:
        raise ValueError(f"all OpenCV hand-eye methods failed: {failures}")
    _, method_name, end_from_camera, world_from_head, residuals = min(
        candidates, key=lambda candidate: candidate[0]
    )
    return {
        "method": method_name,
        "sample_count": len(ends),
        "end_from_camera": matrix_json(end_from_camera),
        "world_from_head_camera": matrix_json(world_from_head),
        "motion_excitation": motion,
        "residuals": residuals,
        "method_failures": failures,
    }


def _legacy_transform(config: dict[str, Any], arm: str) -> np.ndarray:
    section = config[f"{arm}_arm_config"]["cam_extrinsic"]
    return transform(section["R"], section["t"])


def _comparison(result: Any, reference: Any) -> dict[str, float]:
    return {
        "translation_difference_mm": translation_error_m(result, reference) * 1000.0,
        "rotation_difference_deg": rotation_error_deg(result, reference),
    }


def solve_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    current_config_path: str | Path | None = None,
    previous_config_path: str | Path | None = None,
) -> dict[str, Any]:
    root, manifest, board = load_dataset(
        dataset_path, expected_mode="wrist_eye_in_hand_via_head"
    )
    output = Path(output_path).resolve()
    diagnostics = output.parent / "wrist_handeye_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, list[np.ndarray]]] = {
        arm: {"world_from_ends": [], "wrist_from_heads": []} for arm in ARMS
    }
    accepted_ids: list[str] = []
    rejected: list[dict[str, str]] = []

    for sample_directory in sample_directories(root):
        try:
            sample = load_sample(sample_directory)
            ends = sample["robot"]["dynamic_world_from_arm_ends"]
            observations = {}
            for camera_name in CAMERAS:
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

            head_from_board = observations["head"].camera_from_board
            for arm in ARMS:
                wrist_from_board = observations[arm].camera_from_board
                wrist_from_head = wrist_from_board @ inverse(head_from_board)
                records[arm]["world_from_ends"].append(
                    validate_transform(ends[arm], name=f"world_from_{arm}_end")
                )
                records[arm]["wrist_from_heads"].append(wrist_from_head)
            accepted_ids.append(sample["sample_id"])
        except (KeyError, ValueError, OSError, cv2.error) as error:
            rejected.append({"sample": sample_directory.name, "error": str(error)})

    arm_results: dict[str, Any] = {}
    for arm in ARMS:
        try:
            arm_results[arm] = solve_transform_pair(
                records[arm]["world_from_ends"], records[arm]["wrist_from_heads"]
            )
            arm_results[arm]["status"] = "solved"
        except ValueError as error:
            arm_results[arm] = {
                "status": "rejected",
                "sample_count": len(records[arm]["world_from_ends"]),
                "error": str(error),
            }

    references: list[tuple[str, dict[str, Any]]] = []
    if current_config_path is not None:
        references.append(("current_move_config", read_json(current_config_path)))
    if previous_config_path is not None:
        references.append(("previous_measured_config", read_json(previous_config_path)))
    for arm in ARMS:
        if arm_results[arm]["status"] != "solved":
            continue
        result = arm_results[arm]["end_from_camera"]
        arm_results[arm]["legacy_cam_extrinsic_candidate"] = {
            "R": np.asarray(result)[:3, :3].tolist(),
            "t": np.asarray(result)[:3, 3].tolist(),
        }
        arm_results[arm]["reference_comparisons"] = {
            name: _comparison(result, _legacy_transform(config, arm))
            for name, config in references
        }

    solved = all(arm_results[arm]["status"] == "solved" for arm in ARMS)
    head_consistency = None
    if solved:
        left_head = arm_results["left"]["world_from_head_camera"]
        right_head = arm_results["right"]["world_from_head_camera"]
        head_consistency = _comparison(left_head, right_head)
        passed = (
            arm_results["left"]["residuals"]["head_translation_error_mm"]["p95"]
            <= 15.0
            and arm_results["right"]["residuals"]["head_translation_error_mm"]["p95"]
            <= 15.0
            and arm_results["left"]["residuals"]["head_rotation_error_deg"]["p95"]
            <= 1.5
            and arm_results["right"]["residuals"]["head_rotation_error_deg"]["p95"]
            <= 1.5
            and head_consistency["translation_difference_mm"] <= 15.0
            and head_consistency["rotation_difference_deg"] <= 1.5
        )
    else:
        passed = False

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_needs_task_validation" if passed else "rejected",
        "method": "eye_in_hand_from_fixed_head_camera_and_arbitrary_board",
        "source_dataset": str(root),
        "accepted_sample_ids": accepted_ids,
        "rejected_samples": rejected,
        "arms": arm_results,
        "left_right_head_solution_consistency": head_consistency,
        "thresholds": {
            "minimum_samples": MINIMUM_SAMPLES,
            "minimum_rotation_span_deg": MINIMUM_ROTATION_SPAN_DEG,
            "minimum_excited_rotation_axis_rank": MINIMUM_EXCITED_ROTATION_AXES,
            "residual_translation_p95_mm": 15.0,
            "residual_rotation_p95_deg": 1.5,
            "left_right_head_translation_mm": 15.0,
            "left_right_head_rotation_deg": 1.5,
        },
        "notes": [
            "end_from_camera is directly compatible with move cam_extrinsic R/t.",
            "The checkerboard may move between samples because the fixed head camera removes board motion.",
            "Both robot ends must have diverse rotations about at least two axes.",
            "No source configuration is modified by this solver.",
        ],
    }
    write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--current-config")
    parser.add_argument("--previous-config")
    arguments = parser.parse_args()
    report = solve_dataset(
        arguments.dataset,
        arguments.output,
        current_config_path=arguments.current_config,
        previous_config_path=arguments.previous_config,
    )
    print(f"wrist eye-in-hand calibration status: {report['status']}")
    for arm in ARMS:
        result = report["arms"][arm]
        if result["status"] != "solved":
            print(f"{arm}: REJECTED: {result['error']}")
            continue
        print(f"{arm}: solved with {result['method']}")
        print(f"  end_from_camera={result['end_from_camera']}")
        for name, comparison in result["reference_comparisons"].items():
            print(
                f"  vs {name}: "
                f"{comparison['translation_difference_mm']:.3f}mm / "
                f"{comparison['rotation_difference_deg']:.3f}deg"
            )
    return 0 if report["status"] != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
