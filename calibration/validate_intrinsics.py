"""Offline checkerboard validation of recorded factory color intrinsics.

This never writes calibration into a RealSense device. It estimates an
independent checkerboard model from a saved dataset and reports the difference
from the per-frame factory intrinsics recorded during capture.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .board import camera_matrix, detect_corners
from .geometry import percentile_summary
from .io import load_dataset, load_sample, sample_directories, write_json


def validate_intrinsics(
    dataset_path: str | Path,
    camera_name: str,
    output_path: str | Path,
) -> dict[str, Any]:
    root, _, board = load_dataset(dataset_path)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    factory_matrices: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []

    for sample_directory in sample_directories(root):
        try:
            sample = load_sample(sample_directory)
            camera = sample["cameras"][camera_name]
            image = cv2.imread(
                str(sample_directory / camera["rgb_file"]), cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError("RGB image could not be decoded")
            current_size = (int(image.shape[1]), int(image.shape[0]))
            if image_size is not None and image_size != current_size:
                raise ValueError("image resolution differs from earlier samples")
            image_size = current_size
            corners = detect_corners(image, board)
            object_points.append(board.object_points.astype(np.float32))
            image_points.append(corners.reshape(-1, 1, 2).astype(np.float32))
            factory_matrices.append(camera_matrix(camera["intrinsics"]))
            accepted.append(sample["sample_id"])
        except (KeyError, ValueError, OSError, cv2.error) as error:
            rejected.append({"sample": sample_directory.name, "error": str(error)})

    if len(accepted) < 8 or image_size is None:
        raise ValueError(
            "at least 8 valid, diverse checkerboard views are required for intrinsics validation"
        )
    factory = np.median(np.stack(factory_matrices), axis=0)
    rms, estimated, distortion, _, _, _, _, per_view_errors = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        factory.copy(),
        np.zeros((5, 1), dtype=np.float64),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )
    focal_relative_percent = {
        "fx": abs(float(estimated[0, 0] / factory[0, 0] - 1.0)) * 100.0,
        "fy": abs(float(estimated[1, 1] / factory[1, 1] - 1.0)) * 100.0,
    }
    principal_delta_px = {
        "ppx": abs(float(estimated[0, 2] - factory[0, 2])),
        "ppy": abs(float(estimated[1, 2] - factory[1, 2])),
    }
    per_view = percentile_summary(np.asarray(per_view_errors).reshape(-1))
    metrics_passed = (
        float(rms) <= 1.0
        and max(focal_relative_percent.values()) <= 3.0
        and max(principal_delta_px.values()) <= 20.0
        and per_view["p95"] <= 1.5
    )
    sample_count_sufficient = len(accepted) >= 12
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "consistent_with_factory_intrinsics"
            if metrics_passed and sample_count_sufficient
            else "insufficient_sample_count"
            if metrics_passed
            else "review_required"
        ),
        "note": (
            "Read-only validation only; these estimated values are not written to the "
            "camera and are not automatically used by the workflow."
        ),
        "source_dataset": str(root),
        "camera": camera_name,
        "image_size": list(image_size),
        "accepted_sample_ids": accepted,
        "rejected_samples": rejected,
        "factory_camera_matrix": factory.tolist(),
        "checkerboard_estimated_camera_matrix": estimated.tolist(),
        "checkerboard_estimated_distortion": np.asarray(distortion).reshape(-1).tolist(),
        "metrics": {
            "calibration_rms_px": float(rms),
            "per_view_error_px": per_view,
            "focal_relative_difference_percent": focal_relative_percent,
            "principal_point_difference_px": principal_delta_px,
        },
        "thresholds": {
            "calibration_rms_px": 1.0,
            "per_view_error_p95_px": 1.5,
            "focal_relative_difference_percent": 3.0,
            "principal_point_difference_px": 20.0,
            "minimum_valid_views": 8,
            "recommended_valid_views": 20,
        },
        "sample_count_sufficient_for_recommended_check": sample_count_sufficient,
    }
    write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--camera", choices=("left", "head", "right"), required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = validate_intrinsics(arguments.dataset, arguments.camera, arguments.output)
    print(f"intrinsics status: {report['status']}")
    print(f"calibration RMS: {report['metrics']['calibration_rms_px']:.4f}px")
    print("No RealSense calibration was changed.")
    return 0 if report["status"] == "consistent_with_factory_intrinsics" else 2


if __name__ == "__main__":
    raise SystemExit(main())
