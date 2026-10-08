"""Solve head-camera eye-to-hand extrinsics from a captured dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .board import depth_corner_metrics, solve_pnp, write_diagnostic
from .eye_to_hand import EyeToHandSample, solve_eye_to_hand
from .geometry import matrix_json, validate_transform
from .io import load_dataset, load_sample, sample_directories, write_json


def solve_dataset(dataset_path: str | Path, output_path: str | Path) -> dict:
    root, manifest, board = load_dataset(dataset_path, expected_mode="head_eye_to_hand")
    diagnostics = Path(output_path).resolve().parent / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    observations: list[EyeToHandSample] = []
    depth_metrics: list[dict[str, float]] = []
    rejected: list[dict[str, str]] = []

    for sample_directory in sample_directories(root):
        try:
            sample = load_sample(sample_directory)
            camera = sample["cameras"]["head"]
            image = cv2.imread(str(sample_directory / camera["rgb_file"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("head RGB image could not be decoded")
            observation = solve_pnp(image, board, camera["intrinsics"])
            if observation.reprojection_rms_px > 1.5:
                raise ValueError(
                    f"PnP reprojection RMS {observation.reprojection_rms_px:.3f}px exceeds 1.5px"
                )
            write_diagnostic(
                diagnostics / f"{sample_directory.name}_head.png",
                image,
                board,
                observation,
                camera["intrinsics"],
            )
            depth = np.load(sample_directory / camera["depth_file"], allow_pickle=False)
            depth_metric = depth_corner_metrics(
                depth,
                float(camera["depth_scale_m_per_unit"]),
                board,
                observation,
            )
            if depth_metric["valid_fraction"] < 0.5:
                raise ValueError(
                    "fewer than 50% of checkerboard corners have valid aligned depth"
                )
            depth_metrics.append({"sample_id": sample["sample_id"], **depth_metric})
            observations.append(
                EyeToHandSample(
                    sample_id=sample["sample_id"],
                    base_from_tool=validate_transform(sample["robot"]["base_from_tool"]),
                    camera_from_board=observation.camera_from_board,
                    reprojection_rms_px=observation.reprojection_rms_px,
                )
            )
        except (KeyError, ValueError, OSError, cv2.error) as error:
            rejected.append({"sample": sample_directory.name, "error": str(error)})

    result, method_comparison = solve_eye_to_hand(observations)
    world_from_base = validate_transform(
        manifest["world_from_arm_base"], name="world_from_arm_base"
    )
    world_from_head = world_from_base @ result.base_from_camera
    report = {
        "schema_version": 1,
        "status": "candidate_not_yet_alignment_validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(root),
        "arm": manifest["arm"],
        "board": board.canonical(),
        "camera_serial": manifest["camera_serials"]["head"],
        "accepted_sample_ids": [sample.sample_id for sample in observations],
        "rejected_samples": rejected,
        "world_from_arm_base": matrix_json(world_from_base),
        "arm_base_from_head_camera": matrix_json(result.base_from_camera),
        "world_from_head_camera": matrix_json(world_from_head),
        "tool_from_board": matrix_json(result.tool_from_board),
        "selected_method": result.method,
        "metrics": {
            "translation_residual_mm": result.translation_residual_mm,
            "rotation_residual_deg": result.rotation_residual_deg,
            "reprojection_rms_px": result.reprojection_rms_px,
            "aligned_depth_corner_error": {
                "valid_fraction_median": float(
                    np.median([value["valid_fraction"] for value in depth_metrics])
                ),
                "median_mm": float(np.median([value["median_mm"] for value in depth_metrics])),
                "p95_mm": float(np.percentile([value["p95_mm"] for value in depth_metrics], 95)),
            },
        },
        "per_sample_depth_metrics": depth_metrics,
        "method_comparison": method_comparison,
        "coordinate_convention": "p_parent = parent_from_child @ p_child; column vectors",
    }
    write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = solve_dataset(arguments.dataset, arguments.output)
    metrics = report["metrics"]
    print(f"selected method: {report['selected_method']}")
    print(
        "translation residual: "
        f"median={metrics['translation_residual_mm']['median']:.3f}mm, "
        f"p95={metrics['translation_residual_mm']['p95']:.3f}mm"
    )
    print(
        "rotation residual: "
        f"median={metrics['rotation_residual_deg']['median']:.3f}deg, "
        f"p95={metrics['rotation_residual_deg']['p95']:.3f}deg"
    )
    print("candidate only: run three-camera alignment validation before approval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
